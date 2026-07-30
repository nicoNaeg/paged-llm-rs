#!/usr/bin/env python3
"""What continuous batching buys, and what the reservation costs.

Sends N requests at once and measures what comes back, for a range of N. Two
numbers matter and they move in opposite directions: aggregate throughput, which
batching raises because a decode step reads the model's weights once however many
sequences ride along, and per-request latency, which it lowers less as the batch
grows because every row is waiting on the same pass.

    make bench-concurrency

The third number is the one stage 5 exists to change. A slot costs its whole
reservation from the moment a sequence takes it, so the concurrency ceiling is
decided by the reservation and not by what the requests turn out to need. This
prints the waste that produces.
"""

import json
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models/Qwen3-0.6B"
BINARY = ROOT / "target/release/pagedllm-server"
PORT = 8188
BASE = f"http://127.0.0.1:{PORT}"

CACHE_MIB = 3584
PROMPT = "Write a short paragraph about the sea, in plain language."
TOKENS = 128
CONCURRENCY = (1, 4, 16, 32, 64)
# A block as wide as the context is one block a sequence, which is a
# reservation; sixteen is paging. Same memory, same everything else.
#
# The last two repeat the paged layouts with prompts sliced across passes rather
# than run whole. That is not a separate feature being measured here, it is the
# one thing that changes what a batch looks like: a slice is unfolded into one
# row per token, and a row is what the gather copies. The two attention paths
# should therefore answer it differently, and the point of running all five is
# that the difference is measured rather than argued.
LAYOUTS = (
    (1024, "tensor", "off", "a reservation, gathered by the tensor path, prompts whole"),
    (16, "tensor", "off", "paging, gathered by the tensor path, prompts whole"),
    (16, "kernel", "off", "paging, read in place by the kernel, prompts whole"),
    (16, "tensor", "128", "paging, gathered by the tensor path, prompts sliced"),
    (16, "kernel", "128", "paging, read in place by the kernel, prompts sliced"),
)


def one_request(index: int) -> tuple[float, float, int]:
    """Returns (time to first token, whole request, completion tokens)."""
    body = {
        "prompt": PROMPT,
        "max_tokens": TOKENS,
        "temperature": 0.7,
        "seed": index,
        "stream": True,
    }
    request = urllib.request.Request(
        BASE + "/v1/completions",
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
    )
    started = time.time()
    first_at, produced = None, 0
    with urllib.request.urlopen(request) as response:
        for line in response:
            text = line.decode().strip()
            if not text.startswith("data: ") or text == "data: [DONE]":
                continue
            frame = json.loads(text[6:])
            if frame["choices"][0]["text"]:
                if first_at is None:
                    first_at = time.time() - started
                produced += 1
    return first_at or 0.0, time.time() - started, produced


def wait_for_server(process: subprocess.Popen) -> None:
    started = time.time()
    while time.time() - started < 180:
        if process.poll() is not None:
            raise SystemExit(f"the server exited with {process.returncode}")
        try:
            urllib.request.urlopen(BASE + "/health", timeout=1)
            return
        except Exception:
            time.sleep(0.05)
    raise SystemExit("the server never became reachable")


def main() -> int:
    if not BINARY.exists():
        raise SystemExit(f"{BINARY} is missing; run `make build` first")
    if not MODEL.exists():
        raise SystemExit(f"{MODEL} is missing; run `make model` first")

    for block_size, attention, chunk, label in LAYOUTS:
        run_layout(block_size, attention, chunk, label)
    return 0


def run_layout(block_size: int, attention: str, chunk: str, label: str) -> None:
    process = subprocess.Popen(
        [
            str(BINARY),
            "--model", str(MODEL),
            "--port", str(PORT),
            "--block-size", str(block_size),
            "--cache-mib", str(CACHE_MIB),
            "--attention", attention,
            "--max-batch", "64",
            "--chunk", chunk,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        wait_for_server(process)
        # Warm the first dispatch so it does not land inside the first row.
        one_request(0)

        print(f"\n{label}, {CACHE_MIB} MiB of cache")
        print(
            f"  {'clients':>8} {'total tok/s':>12} {'per client':>11}"
            f" {'ttft ms':>9} {'p50 s':>7} {'p95 s':>7}"
        )
        rows = []
        for clients in CONCURRENCY:
            started = time.time()
            with ThreadPoolExecutor(max_workers=clients) as pool:
                results = list(pool.map(one_request, range(clients)))
            wall = time.time() - started

            produced = sum(r[2] for r in results)
            ttfts = sorted(r[0] for r in results)
            totals = sorted(r[1] for r in results)
            throughput = produced / wall
            rows.append((clients, throughput))
            print(
                f"  {clients:>8} {throughput:>12.1f} {throughput / clients:>11.1f}"
                f" {statistics.median(ttfts) * 1000:>9.0f}"
                f" {statistics.median(totals):>7.2f}"
                f" {totals[int(len(totals) * 0.95) - 1]:>7.2f}"
            )

        best = max(rows, key=lambda r: r[1])
        alone = rows[0][1]
        print(
            f"  {best[1] / alone:.1f}x the throughput of one client at a time,"
            f" reached at {best[0]} clients"
        )

        bytes_per_token = 114_688
        held = (CACHE_MIB << 20) // bytes_per_token
        used = max(CONCURRENCY) * (len(PROMPT.split()) + TOKENS)
        print(
            f"  the pool holds {held} tokens; {max(CONCURRENCY)} sequences of about"
            f" {len(PROMPT.split()) + TOKENS} tokens need {used},"
            f" so it fits {held // (len(PROMPT.split()) + TOKENS)} of them"
        )
    finally:
        process.terminate()
        process.wait(timeout=15)
    return 0


if __name__ == "__main__":
    sys.exit(main())
