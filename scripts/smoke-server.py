#!/usr/bin/env python3
"""Drive the server over HTTP and check what it answers.

Everything here goes over the wire, on a process started and stopped by this
script. That is the point: the parts of a server that break are the ones a unit
test does not reach, the terminator of a stream and the status code of a refusal
among them.

    make smoke

Prints a throughput table as it goes. Those numbers are the stage 2 baseline, a
single sequence against a KV cache that is one contiguous run of memory grown by
copying, which is what stages 3 and 5 exist to beat.
"""

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models/Qwen3-0.6B"
BINARY = ROOT / "target/release/pagedllm-server"
PORT = 8177
BASE = f"http://127.0.0.1:{PORT}"

failures: list[str] = []


def check(what: str, ok: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if ok else 'FAIL'} {what}{f'   {detail}' if detail else ''}")
    if not ok:
        failures.append(what)


def call(path: str, body=None, stream: bool = False):
    """Returns (status, parsed body or list of SSE frames, seconds)."""
    request = urllib.request.Request(
        BASE + path,
        data=None if body is None else json.dumps(body).encode(),
        headers={"content-type": "application/json"},
    )
    started = time.time()
    try:
        with urllib.request.urlopen(request) as response:
            if not stream:
                return response.status, json.loads(response.read()), time.time() - started
            frames, first_at = [], None
            for line in response:
                text = line.decode().strip()
                if text.startswith("data: "):
                    if first_at is None:
                        first_at = time.time() - started
                    frames.append(text[6:])
            return response.status, (frames, first_at), time.time() - started
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read()), time.time() - started


def wait_for_server(process: subprocess.Popen) -> float:
    started = time.time()
    while time.time() - started < 180:
        if process.poll() is not None:
            raise SystemExit(f"the server exited with {process.returncode}")
        try:
            urllib.request.urlopen(BASE + "/health", timeout=1)
            return time.time() - started
        except Exception:
            time.sleep(0.05)
    raise SystemExit("the server never became reachable")


def main() -> int:
    if not BINARY.exists():
        raise SystemExit(f"{BINARY} is missing; run `make build` first")
    if not MODEL.exists():
        raise SystemExit(f"{MODEL} is missing; run `make model` first")

    process = subprocess.Popen(
        [str(BINARY), "--model", str(MODEL), "--port", str(PORT)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        boot = wait_for_server(process)
        print(f"\nserver up in {boot:.2f}s\n")

        print("endpoints")
        status, body, _ = call("/health")
        check("GET /health", status == 200 and body.get("status") == "ok")
        status, body, _ = call("/v1/models")
        check(
            "GET /v1/models",
            status == 200 and body["data"][0]["id"] == MODEL.name,
            body["data"][0]["id"] if status == 200 else "",
        )

        print("\ncompletions")
        status, body, cold = call(
            "/v1/completions",
            {"prompt": "The capital of France is", "max_tokens": 8, "temperature": 0},
        )
        check("POST /v1/completions", status == 200, f"{cold * 1000:.0f}ms cold")
        if status == 200:
            check(
                "usage is counted",
                body["usage"]["prompt_tokens"] > 0 and body["usage"]["completion_tokens"] > 0,
                str(body["usage"]),
            )
            check(
                "text came back",
                bool(body["choices"][0]["text"].strip()),
                repr(body["choices"][0]["text"]),
            )

        print("\nchat completions")
        status, body, _ = call(
            "/v1/chat/completions",
            {
                "messages": [{"role": "user", "content": "Name three prime numbers."}],
                "max_tokens": 64,
                "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        check("POST /v1/chat/completions", status == 200)
        if status == 200:
            check(
                "stops on the model's own end token",
                body["choices"][0]["finish_reason"] == "stop",
                repr(body["choices"][0]["message"]["content"]),
            )

        print("\ndeterminism")
        ask = {
            "prompt": "Once upon a time",
            "max_tokens": 24,
            "temperature": 0.9,
            "seed": 7,
        }
        first = call("/v1/completions", ask)[1]["choices"][0]["text"]
        again = call("/v1/completions", ask)[1]["choices"][0]["text"]
        other = call("/v1/completions", {**ask, "seed": 8})[1]["choices"][0]["text"]
        check("the same seed gives the same text", first == again)
        check("a different seed does not", first != other)
        unseeded = [
            call("/v1/completions", {**{k: v for k, v in ask.items() if k != "seed"}})[1][
                "choices"
            ][0]["text"]
            for _ in range(2)
        ]
        # Two sampled requests arriving in the same second must not share a seed,
        # which they would if the clock alone chose it.
        check("two unseeded requests differ", unseeded[0] != unseeded[1])

        print("\nstreaming")
        status, (frames, ttft), total = call(
            "/v1/chat/completions",
            {
                "messages": [{"role": "user", "content": "Name three prime numbers."}],
                "max_tokens": 64,
                "temperature": 0,
                "stream": True,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            stream=True,
        )
        check("the stream is served", status == 200 and len(frames) > 2, f"{len(frames)} frames")
        check("it ends with the terminator clients watch for", frames[-1] == "[DONE]")
        check(
            "the last chunk carries a finish reason",
            json.loads(frames[-2])["choices"][0]["finish_reason"] == "stop",
        )
        check("the first chunk announces the role", json.loads(frames[0])["choices"][0]["delta"].get("role") == "assistant")
        streamed = "".join(
            json.loads(f)["choices"][0]["delta"].get("content", "")
            for f in frames
            if f != "[DONE]"
        )
        check("streamed text matches the whole answer", streamed == body["choices"][0]["message"]["content"], repr(streamed))
        print(f"       first token after {ttft * 1000:.0f}ms, whole stream in {total:.2f}s")

        print("\nrefusals, which are the point of parsing these at all")
        for name, body_in in [
            ("n above 1", {"messages": [], "n": 3}),
            ("stop", {"messages": [], "stop": ["\n"]}),
            ("tools", {"messages": [], "tools": []}),
            ("logit_bias", {"messages": [], "logit_bias": {"1": 2}}),
            ("logprobs", {"messages": [], "logprobs": True}),
            ("frequency_penalty", {"messages": [], "frequency_penalty": 0.5}),
            ("response_format", {"messages": [], "response_format": {"type": "json_object"}}),
            ("a negative temperature", {"messages": [{"role": "user", "content": "x"}], "temperature": -1}),
            ("top_p above one", {"messages": [{"role": "user", "content": "x"}], "top_p": 2}),
        ]:
            status, out, _ = call("/v1/chat/completions", body_in)
            message = out.get("error", {}).get("message", "") if status != 200 else ""
            check(f"refuses {name}", status == 400, message)
        # What a client sends meaning "default" has to get through.
        status, _, _ = call(
            "/v1/chat/completions",
            {
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 4,
                "n": 1,
                "frequency_penalty": 0,
                "presence_penalty": 0,
                "logprobs": False,
            },
        )
        check("accepts the defaults a client sends anyway", status == 200)

        # A prompt longer than the pass budget, arriving with nothing else
        # running, so its first slices go through the model asking for no logits
        # at all. That is the only shape where a pass produces an empty result,
        # and it answered 500 until the sampler stopped touching it: casting a
        # zero-row tensor dispatches a Metal kernel over zero elements and candle
        # divides by zero working out its grid. No unit test reaches it, because
        # the committed fixture is f32 and the cast is then a no-op.
        print("\nprompts longer than one pass, which is where the slicing shows")
        long_prompt = "Summarise these notes in one sentence. " + (
            "The allocator hands out fixed size blocks and a table maps a "
            "sequence position to one of them. " * 40
        )
        status, out, _ = call(
            "/v1/completions", {"prompt": long_prompt, "max_tokens": 8, "temperature": 0}
        )
        answered = status == 200 and out.get("choices", [{}])[0].get("text", "") != ""
        check(
            "a prompt longer than the pass budget is answered",
            answered,
            out.get("error", {}).get("message", "") if status != 200 else
            f"{out['usage']['prompt_tokens']} prompt tokens",
        )
        # Sliced or whole, a greedy request has one answer.
        status, whole, _ = call(
            "/v1/completions",
            {"prompt": long_prompt, "max_tokens": 8, "temperature": 0},
        )
        check(
            "the same long prompt answers the same way twice",
            status == 200 and answered and whole["choices"][0]["text"] == out["choices"][0]["text"],
        )

        print("\nthroughput, one sequence, contiguous cache")
        print(f"  {'tokens':>7} {'seconds':>8} {'tok/s':>7} {'ms/token':>9}")
        for budget in (16, 64, 128, 256, 512, 1024):
            status, out, seconds = call(
                "/v1/completions",
                {"prompt": "Count:", "max_tokens": budget, "temperature": 0.9, "seed": 1},
            )
            produced = out["usage"]["completion_tokens"]
            print(
                f"  {produced:>7} {seconds:>8.2f} {produced / seconds:>7.1f}"
                f" {seconds / produced * 1000:>9.2f}"
            )
    finally:
        process.terminate()
        process.wait(timeout=15)

    print()
    if failures:
        print(f"{len(failures)} check(s) failed:")
        for what in failures:
            print(f"  {what}")
        return 1
    print("every check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
