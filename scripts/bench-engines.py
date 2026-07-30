#!/usr/bin/env python3
"""This engine against the others, on this machine, driven by the same tool.

    make bench-engines

`guidellm` sends the load and reads the results. A number produced by a harness
written in this repository would be a number about this repository; guidellm is
standalone, speaks the `OpenAI` API, and is what people already point at other
servers, so every engine here is driven by the same client with the same flags.

The comparison is a fixed grid of concurrency levels rather than guidellm's
sweep. The sweep calibrates its own range, which is the better shape, and it was
the first choice; it is not used because its last phase sends without bound and
mistral.rs stops answering under it. A grid every engine survives is worth more
than a shape one of them cannot be measured at, and the levels are the same ones
the rest of this README reports.

What is held equal, and it is worth being explicit because the formats differ:

- the same weights. Qwen3-0.6B in bfloat16 everywhere. llama.cpp reads the BF16
  GGUF conversion of this checkpoint rather than a quantised one, which would be
  a different model running a different amount of arithmetic;
- the same context, so no engine reserves for a length another one does not;
- the same KV cache budget where the engine exposes one;
- the same shape of load. guidellm generates prompts of exactly PROMPT_TOKENS and
  asks for exactly OUTPUT_TOKENS, with no spread, so every engine is given the
  same number of tokens to read and the same number to produce. The prompt text
  differs between runs, which changes nothing about a rate.

What is not held equal is the sampler. Each engine is asked for greedy decoding,
which removes the draw, but the tokens still differ because the arithmetic does.
That is a property of the models, not of the servers, and it changes nothing
about the rates being compared.
"""

import json
import os
import socket
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models/Qwen3-0.6B"
GGUF = ROOT / "models/gguf/Qwen3-0.6B-BF16.gguf"
BINARY = ROOT / "target/release/pagedllm-server"
VENV = ROOT / ".venv/bin"
RESULTS = ROOT / "bench-results"

PORT = 8410
BASE = f"http://127.0.0.1:{PORT}"

# Held equal across engines.
CONTEXT = 1024
CACHE_MIB = 3584
MAX_SEQUENCES = 64
PROMPT_TOKENS = 128
OUTPUT_TOKENS = 128
STREAMS = (1, 4, 16, 32, 64)
SECONDS_PER_POINT = 20
# An engine that stops answering must not stop the run. Generous against the
# measurement, which is five points of twenty seconds plus load, and short
# against a hang, which is unbounded.
SWEEP_TIMEOUT = 600


ONLY = os.environ.get("BENCH_ONLY")


def engines() -> list[tuple[str, list[str], str, str]]:
    """Each engine as (name, command, note, api model name), skipping what is
    not installed.

    The model name is per engine because they disagree about what to call the
    thing they loaded, and a request naming the wrong one is refused rather than
    served.
    """
    found = []
    if BINARY.exists():
        found.append((
            "paged-llm-rs",
            [
                str(BINARY),
                "--model", str(MODEL),
                "--port", str(PORT),
                "--block-size", "16",
                "--cache-mib", str(CACHE_MIB),
                "--max-batch", str(MAX_SEQUENCES),
                "--attention", "kernel",
            ],
            "paged cache, hand-written Metal kernel",
            "Qwen3-0.6B",
        ))
    if shutil.which("llama-server") and GGUF.exists():
        found.append((
            "llama.cpp",
            [
                "llama-server",
                "-m", str(GGUF),
                "--port", str(PORT),
                "-c", str(CONTEXT * MAX_SEQUENCES),
                "-np", str(MAX_SEQUENCES),
                "--no-warmup",
                "-ngl", "99",
            ],
            "BF16 GGUF, Metal, continuous batching",
            "Qwen3-0.6B",
        ))
    if shutil.which("mistralrs"):
        found.append((
            "mistral.rs",
            [
                "mistralrs", "serve",
                "--port", str(PORT),
                # Its paged attention, which is the thing this project is about,
                # cannot be measured here: on Metal it answers four concurrent
                # requests and stops answering at sixteen. Measured rather than
                # assumed, and with the same flag off it serves sixteen in 25s.
                # So the number below is mistral.rs at its working setting, and
                # the comparison of two paged implementations is not available.
                "--paged-attn", "off",
                "--max-seqs", str(MAX_SEQUENCES),
                "--max-seq-len", str(CONTEXT),
                "--model-id", str(MODEL),
            ],
            "the same safetensors, Metal, paged attention off (see the note)",
            "default",
        ))
    if ONLY:
        found = [e for e in found if ONLY in e[0]]
    return found


def port_is_free() -> bool:
    """Whether anything already holds the port the engines take in turn.

    Checked before each engine rather than assumed. A server left behind by an
    interrupted run answers on that port, and an engine that cannot bind exits
    at once: without this the run would skip that engine and measure the
    leftover instead, which is how a benchmark quietly reports the wrong
    program.
    """
    with socket.socket() as probe:
        probe.settimeout(1)
        return probe.connect_ex(("127.0.0.1", PORT)) != 0


def wait_for(process: subprocess.Popen, seconds: int = 240) -> bool:
    """Poll until the server answers, or it dies, or the wait runs out."""
    started = time.time()
    body = json.dumps({"prompt": "x", "max_tokens": 1}).encode()
    while time.time() - started < seconds:
        if process.poll() is not None:
            return False
        for path, data in (("/v1/models", None), ("/v1/completions", body)):
            try:
                request = urllib.request.Request(
                    BASE + path,
                    data=data,
                    headers={"content-type": "application/json"},
                )
                urllib.request.urlopen(request, timeout=2)
                return True
            except urllib.error.HTTPError:
                return True
            except Exception:
                pass
        time.sleep(0.25)
    return False


def sweep(name: str, api_model: str) -> dict | None:
    """Drive one engine and return what guidellm measured."""
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"{name.replace('.', '-')}.json"
    command = [
        str(VENV / "guidellm"), "run",
        "--backend", f"kind=openai_http,target={BASE},model={api_model}",
        "--tokenizer", f"kind=hf_auto,model={MODEL}",
        "--profile", "kind=concurrent",
        "--override", "profile.streams", ",".join(str(n) for n in STREAMS),
        "--data",
        f"kind=synthetic_text,prompt_tokens={PROMPT_TOKENS},output_tokens={OUTPUT_TOKENS}",
        "--constraint", f"kind=max_duration,seconds={SECONDS_PER_POINT}",
        "--output", f"kind=json,path={out}",
        "--disable-console-interactive",
    ]
    try:
        done = subprocess.run(
            command, cwd=ROOT, capture_output=True, text=True, timeout=SWEEP_TIMEOUT
        )
    except subprocess.TimeoutExpired:
        print(f"    stopped answering; no result after {SWEEP_TIMEOUT}s")
        return None
    if done.returncode != 0:
        print(done.stdout[-1500:])
        print(done.stderr[-1500:])
        print(f"    guidellm failed against {name}")
        return None
    return json.loads(out.read_text())


def summarise(report: dict) -> list[tuple[float, float, float, float]]:
    """(concurrency, output tokens/s, ttft ms, p95 request seconds) per point."""
    rows = []
    for bench in report.get("benchmarks", []):
        metrics = bench.get("metrics", {})

        def stat(metric: str, key: str, percentile: bool = False) -> float:
            node = metrics.get(metric, {}).get("successful", {})
            if percentile:
                node = node.get("percentiles", {})
            value = node.get(key) if isinstance(node, dict) else None
            return float(value) if isinstance(value, (int, float)) else float("nan")

        rows.append((
            stat("request_concurrency", "mean"),
            stat("output_tokens_per_second", "mean"),
            stat("time_to_first_token_ms", "median"),
            stat("request_latency", "p95", percentile=True),
        ))
    rows.sort(key=lambda r: (r[0] if r[0] == r[0] else 0))
    return rows


def main() -> int:
    if not MODEL.exists():
        raise SystemExit(f"{MODEL} is missing; run `make model` first")
    found = engines()
    if not found:
        raise SystemExit("no engine to run; `make build` at least")

    print(f"\nQwen3-0.6B in bf16, {PROMPT_TOKENS} prompt tokens, {OUTPUT_TOKENS} generated")
    print(f"at {STREAMS} concurrent clients, {SECONDS_PER_POINT}s each\n")

    results = {}
    for name, command, note, api_model in found:
        print(f"  {name}: {note}")
        if not port_is_free():
            raise SystemExit(
                f"something already answers on port {PORT}; a leftover server would be "
                f"measured instead of {name}"
            )
        process = subprocess.Popen(
            command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        try:
            if not wait_for(process):
                raise SystemExit(f"{name} did not come up; nothing is measured rather than the wrong thing")
            report = sweep(name, api_model)
            if report is not None:
                results[name] = summarise(report)
        finally:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
            time.sleep(2)

    print(f"\n  {'engine':<14} {'concurrency':>12} {'tok/s':>9} {'ttft ms':>9} {'p95 s':>8}")
    for name, rows in results.items():
        for concurrency, throughput, ttft, p95 in rows:
            print(
                f"  {name:<14} {concurrency:>12.1f} {throughput:>9.1f}"
                f" {ttft:>9.0f} {p95:>8.2f}"
            )

    peaks = {
        name: max((r[1] for r in rows), default=float("nan"))
        for name, rows in results.items()
    }
    print("\n  peak output tokens a second")
    for name, peak in sorted(peaks.items(), key=lambda kv: -kv[1]):
        print(f"    {name:<14} {peak:>9.1f}")
    print(f"\n  reports written to {RESULTS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
