#!/usr/bin/env python3
"""Watch continuous batching hold while a long prompt arrives.

    make demo

Four clients stream at once and a fifth sends a prompt of about 800 words
part-way through. What the display shows is the thing the engine is for: the four
keep producing while the newcomer's prompt goes through, because that prompt is
fed a slice per pass rather than taking a pass to itself.

Turn the pass budget off and the same run stalls visibly:

    python3 scripts/demo.py --chunk off

This measures nothing `make bench-chunk` does not measure better. It exists to be
watched, and to be recorded into the animation the README opens with, which is
why it is short and why the numbers on screen are the ones a client would feel:
tokens produced, and the longest a client waited between two of them.
"""

import argparse
import json
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models/Qwen3-0.6B"
BINARY = ROOT / "target/release/pagedllm-server"
PORT = 8480
BASE = f"http://127.0.0.1:{PORT}"

CLIENTS = 4
TOKENS = 200
# Long enough that running it whole would be unmistakable on screen.
PREAMBLE = (
    "The allocator hands out fixed size blocks and a table maps a sequence "
    "position to one of them. " * 45
)
WIDTH = 34


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
            "--chunk", chunk,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.time() + 180
    while time.time() < deadline:
        if server.poll() is not None:
            raise SystemExit("the server exited while starting")
        try:
            urllib.request.urlopen(BASE + "/health", timeout=1)
            return server
        except Exception:
            time.sleep(0.1)
    server.terminate()
    raise SystemExit("the server never became reachable")


def stream(prompt: str, max_tokens: int, on_token) -> None:
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
    with urllib.request.urlopen(request, timeout=600) as response:
        for line in response:
            text = line.decode().strip()
            if not text.startswith("data: ") or text == "data: [DONE]":
                continue
            if json.loads(text[6:])["choices"][0]["text"]:
                on_token()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk", default="128", help="pass budget, or 'off'")
    args = parser.parse_args()

    if not BINARY.exists() or not MODEL.exists():
        raise SystemExit("run `make build` and `make model` first")

    state = [{"tokens": 0, "last": None, "worst": 0.0} for _ in range(CLIENTS)]
    newcomer = {"sent_at": None, "first_at": None}
    done = threading.Event()

    def resident(index: int) -> None:
        def on_token() -> None:
            now = time.perf_counter()
            row = state[index]
            if row["last"] is not None:
                row["worst"] = max(row["worst"], (now - row["last"]) * 1000)
            row["last"] = now
            row["tokens"] += 1

        stream(f"Count slowly from {index} and explain each step.", TOKENS, on_token)

    def intruder() -> None:
        time.sleep(3.0)
        newcomer["sent_at"] = time.perf_counter()

        def on_token() -> None:
            if newcomer["first_at"] is None:
                newcomer["first_at"] = time.perf_counter()

        stream("Summarise these notes. " + PREAMBLE, 8, on_token)

    def draw() -> None:
        print("\n" * (CLIENTS + 3), end="")
        while not done.is_set():
            print(f"\033[{CLIENTS + 3}A", end="")
            print(f"  paged-llm-rs, --chunk {args.chunk}, {CLIENTS} clients streaming\033[K")
            print("\033[K")
            for index, row in enumerate(state):
                filled = round(WIDTH * row["tokens"] / TOKENS)
                bar = "█" * filled + "░" * (WIDTH - filled)
                print(
                    f"  client {index}  {bar} {row['tokens']:>3}/{TOKENS}"
                    f"   worst gap {row['worst']:>5.0f} ms\033[K"
                )
            if newcomer["sent_at"] is None:
                print("\033[K")
            elif newcomer["first_at"] is None:
                print("  a prompt of 800 words just arrived, and nobody stopped\033[K")
            else:
                took = (newcomer["first_at"] - newcomer["sent_at"]) * 1000
                print(f"  the 800-word prompt answered in {took:.0f} ms\033[K")
            time.sleep(0.05)

    print(f"  loading Qwen3-0.6B on Metal, blocks of 16, --chunk {args.chunk}")
    server = serve(args.chunk)
    painter = threading.Thread(target=draw, daemon=True)
    painter.start()
    try:
        threads = [threading.Thread(target=resident, args=(i,)) for i in range(CLIENTS)]
        threads.append(threading.Thread(target=intruder))
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    finally:
        done.set()
        painter.join(timeout=1)
        server.terminate()
        server.wait(timeout=20)

    worst = max(row["worst"] for row in state)
    print(f"\n  worst gap any client waited between two tokens: {worst:.0f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
