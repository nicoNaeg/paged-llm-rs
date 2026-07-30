#!/usr/bin/env python3
"""What a long prompt does to the sequences already generating.

    make bench-chunk

Prefill-first gives an admitted prompt the whole pass. Every sequence already
decoding produces nothing for as long as that prefill takes, and on a long
prompt that is not a hiccup, it is a visible stop. Chunked prefill cuts the
prompt into slices and puts one slice in each pass next to everybody's next
token, so the residents pay a slightly longer step instead of a full stop.

What is measured is the gap between one token and the next, for a client that
was already streaming when somebody else's long prompt arrived. The mean says
nothing here: the stall is one gap in a hundred, so the mean moves by a percent
while the experience moves entirely. The worst gap is the number.

The trade is measured rather than assumed, and it is not the one the worst gap
alone suggests. Slicing does not remove the prefill work, it spreads it: the
same total is paid across more passes, so the worst gap falls while the number
of gaps big enough to notice rises. Both are printed, along with the total time
the residents spent waiting, which is what barely moves.
"""

import json
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models/Qwen3-0.6B"
BINARY = ROOT / "target/release/pagedllm-server"
PORT = 8470
BASE = f"http://127.0.0.1:{PORT}"

RESIDENTS = 4
RESIDENT_TOKENS = 220
# Long enough that running it whole is unmistakable against a decode step, and
# short enough that the run stays under a minute per configuration.
INTRUDER_WORDS = 900
SETTLE = 2.0


def stream(prompt: str, max_tokens: int) -> tuple[list[float], float]:
    """One request. Returns (gaps between tokens in ms, time to first token s)."""
    body = json.dumps({
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
    }).encode()
    request = urllib.request.Request(
        BASE + "/v1/completions",
        data=body,
        headers={"content-type": "application/json"},
    )
    started = time.perf_counter()
    marks = []
    with urllib.request.urlopen(request, timeout=600) as response:
        for line in response:
            text = line.decode().strip()
            if not text.startswith("data: ") or text == "data: [DONE]":
                continue
            if json.loads(text[6:])["choices"][0]["text"]:
                marks.append(time.perf_counter())
    if not marks:
        return [], 0.0
    gaps = [(b - a) * 1000 for a, b in zip(marks, marks[1:])]
    return gaps, marks[0] - started


def port_is_free() -> bool:
    with socket.socket() as probe:
        probe.settimeout(1)
        return probe.connect_ex(("127.0.0.1", PORT)) != 0


def serve(chunk: str) -> subprocess.Popen:
    if not port_is_free():
        raise SystemExit(f"something already answers on port {PORT}")
    server = subprocess.Popen(
        [
            str(BINARY),
            "--model", str(MODEL),
            "--port", str(PORT),
            "--attention", "kernel",
            # Off, so the intruder's prompt is really computed rather than found
            # in blocks the previous configuration left behind.
            "--prefix-cache", "off",
            "--chunk", chunk,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.time() + 180
    while time.time() < deadline:
        if server.poll() is not None:
            raise SystemExit(f"the server exited while starting with --chunk {chunk}")
        try:
            urllib.request.urlopen(BASE + "/health", timeout=1)
            return server
        except Exception:
            time.sleep(0.1)
    server.terminate()
    raise SystemExit("the server never became reachable")


def measure(chunk: str) -> dict:
    server = serve(chunk)
    try:
        results = {}
        threads = []

        def resident(index: int) -> None:
            gaps, _ = stream(f"Count slowly from {index} and explain each step.", RESIDENT_TOKENS)
            results[index] = gaps

        for index in range(RESIDENTS):
            thread = threading.Thread(target=resident, args=(index,))
            thread.start()
            threads.append(thread)

        # Let them reach a steady decode before the prompt lands, so the gaps
        # being compared are decode gaps and not admission.
        time.sleep(SETTLE)
        intruder_prompt = "Summarise the following notes. " + (
            "The allocator hands out fixed size blocks and a table maps a "
            "sequence position to one of them. " * (INTRUDER_WORDS // 20)
        )
        started = time.perf_counter()
        intruder_gaps, intruder_ttft = stream(intruder_prompt, 32)
        intruder_wall = time.perf_counter() - started

        for thread in threads:
            thread.join()
    finally:
        server.terminate()
        server.wait(timeout=30)
        time.sleep(1)

    gaps = sorted(g for row in results.values() for g in row)
    if not gaps:
        raise SystemExit("no resident produced a token")
    # A gap a client would notice as a pause rather than as a slow token. The
    # ordinary decode step is 30 ms, so this is three times one.
    noticed = [g for g in gaps if g > 100]
    return {
        "median": statistics.median(gaps),
        "p99": gaps[min(len(gaps) - 1, int(len(gaps) * 0.99))],
        "worst": gaps[-1],
        "noticed": len(noticed),
        "stalled": sum(noticed),
        "gaps": len(gaps),
        "intruder_ttft": intruder_ttft * 1000,
        "intruder_wall": intruder_wall,
    }


def main() -> int:
    if not BINARY.exists() or not MODEL.exists():
        raise SystemExit("run `make build` and `make model` first")

    words = len(
        ("The allocator hands out fixed size blocks and a table maps a sequence "
         "position to one of them. " * (INTRUDER_WORDS // 20)).split()
    )
    print(f"\n{RESIDENTS} clients streaming, one prompt of about {words} words arriving mid-flight")
    print("gaps are between one token and the next, for the clients already running\n")
    print(
        f"  {'chunk':>8} {'median ms':>10} {'worst ms':>9} {'gaps over':>10}"
        f" {'stalled ms':>11} {'newcomer ttft ms':>17}"
    )
    print(f"  {'':>8} {'':>10} {'':>9} {'100 ms':>10} {'in total':>11} {'':>17}")

    runs = {}
    for chunk in ("off", "512", "128", "64", "32"):
        result = measure(chunk)
        runs[chunk] = result
        print(
            f"  {chunk:>8} {result['median']:>10.1f} {result['worst']:>9.0f}"
            f" {result['noticed']:>10} {result['stalled']:>11.0f}"
            f" {result['intruder_ttft']:>17.0f}"
        )

    print()
    off = runs["off"]
    for chunk in ("512", "128", "64", "32"):
        on = runs[chunk]
        line = f"  chunk {chunk}: worst gap {off['worst'] / on['worst']:.1f}x smaller"
        if on["noticed"] == 0:
            line += ", and no gap above 100 ms left at all"
        else:
            line += (
                f", spread over {on['noticed']} gaps rather than {off['noticed']},"
                f" for {on['stalled'] / off['stalled']:.2f}x the total wait"
            )
        line += f", the newcomer's first token {on['intruder_ttft'] / off['intruder_ttft']:.2f}x"
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
