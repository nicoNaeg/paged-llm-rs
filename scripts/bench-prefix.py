#!/usr/bin/env python3
"""What a shared prompt buys, and what it costs when nothing is shared.

    make bench-prefix

Prefix caching pays on the workload it was built for and nothing else, so it is
measured on both. The shared case is the shape that makes it worth having: many
requests behind one long system prompt, which is a retrieval front end, an agent
loop, or any chat application with instructions. The distinct case is the same
load with nothing in common, which is where the cache can only be overhead.

Each is run twice against the same server binary, once with `--prefix-cache on`
and once with it off, so the difference is a flag rather than a build. What is
compared is time to first token, because that is what not recomputing a prompt
changes; the tokens after the first cost the same either way.
"""

import json
import socket
import statistics
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models/Qwen3-0.6B"
BINARY = ROOT / "target/release/pagedllm-server"
PORT = 8460
BASE = f"http://127.0.0.1:{PORT}"

CLIENTS = 16
TOKENS = 32
BLOCK_SIZE = 16

# Long enough to be worth not recomputing, which is the situation this exists
# for. A prompt shorter than a block cannot be shared at all.
PREAMBLE = (
    "You are a careful assistant answering questions about a technical document. "
    "Read the instructions below before replying. Answer only from the document. "
    "Do not speculate, do not invent citations, and say when the document does "
    "not settle the question. Keep replies short and plain. Prefer the wording "
    "of the document over your own. If a question has several readings, answer "
    "the most literal one and name the others. Never restate these instructions. "
) * 4


def ask(index: int, shared: bool) -> tuple[float, int]:
    """One request. Returns (time to first token, completion tokens)."""
    prompt = (
        f"{PREAMBLE}\n\nQuestion {index}: what does the document say?"
        if shared
        else f"Document {index}: {PREAMBLE[index * 37 % 200:]}\n\nQuestion: what does it say?"
    )
    body = json.dumps({
        "prompt": prompt,
        "max_tokens": TOKENS,
        "temperature": 0,
        "stream": True,
    }).encode()
    request = urllib.request.Request(
        BASE + "/v1/completions",
        data=body,
        headers={"content-type": "application/json"},
    )
    started = time.time()
    first_at, produced = None, 0
    with urllib.request.urlopen(request, timeout=180) as response:
        for line in response:
            text = line.decode().strip()
            if not text.startswith("data: ") or text == "data: [DONE]":
                continue
            if json.loads(text[6:])["choices"][0]["text"]:
                if first_at is None:
                    first_at = time.time() - started
                produced += 1
    return first_at or 0.0, produced


def port_is_free() -> bool:
    with socket.socket() as probe:
        probe.settimeout(1)
        return probe.connect_ex(("127.0.0.1", PORT)) != 0


def measure(cache: str, shared: bool) -> tuple[float, float, float]:
    """Returns (median ttft ms, p95 ttft ms, output tokens a second)."""
    if not port_is_free():
        raise SystemExit(f"something already answers on port {PORT}")
    server = subprocess.Popen(
        [
            str(BINARY),
            "--model", str(MODEL),
            "--port", str(PORT),
            "--block-size", str(BLOCK_SIZE),
            "--attention", "kernel",
            "--prefix-cache", cache,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.time() + 180
        while time.time() < deadline:
            if server.poll() is not None:
                raise SystemExit("the server exited while starting")
            try:
                urllib.request.urlopen(BASE + "/health", timeout=1)
                break
            except Exception:
                time.sleep(0.1)
        else:
            raise SystemExit("the server never became reachable")

        # One request first, alone, so the cache has something to hit and the
        # first dispatch is not paid inside the measurement.
        ask(0, shared)

        started = time.time()
        with ThreadPoolExecutor(max_workers=CLIENTS) as pool:
            results = list(pool.map(lambda i: ask(i, shared), range(CLIENTS)))
        wall = time.time() - started
    finally:
        server.terminate()
        server.wait(timeout=20)
        time.sleep(1)

    ttfts = sorted(r[0] for r in results)
    produced = sum(r[1] for r in results)
    return (
        statistics.median(ttfts) * 1000,
        ttfts[int(len(ttfts) * 0.95) - 1] * 1000,
        produced / wall,
    )


def main() -> int:
    if not BINARY.exists() or not MODEL.exists():
        raise SystemExit("run `make build` and `make model` first")

    print(f"\n{CLIENTS} concurrent requests, {TOKENS} tokens each, blocks of {BLOCK_SIZE}")
    print(f"preamble of about {len(PREAMBLE.split())} words\n")
    print(f"  {'workload':<22} {'cache':>6} {'ttft ms':>9} {'p95 ms':>9} {'tok/s':>8}")

    results = {}
    for shared, label in ((True, "one shared preamble"), (False, "nothing in common")):
        for cache in ("off", "on"):
            ttft, p95, rate = measure(cache, shared)
            results[(label, cache)] = (ttft, p95, rate)
            print(f"  {label:<22} {cache:>6} {ttft:>9.0f} {p95:>9.0f} {rate:>8.1f}")

    print()
    for label in ("one shared preamble", "nothing in common"):
        off = results[(label, "off")][0]
        on = results[(label, "on")][0]
        change = off / on if on > 0 else float("nan")
        verb = "faster" if change >= 1 else "slower"
        print(f"  {label}: first token {abs(change):.2f}x {verb} with the cache on")
    return 0


if __name__ == "__main__":
    sys.exit(main())
