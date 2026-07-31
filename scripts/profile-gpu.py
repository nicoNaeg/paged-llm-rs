#!/usr/bin/env python3
"""Record what the GPU actually does during a decode, with Instruments.

    make profile

Takes a Metal System Trace of the server under load, and writes a summary of it
to `docs/decode-profile.txt`, which is what the README cites. The trace itself
stays out of git: eight seconds of it is a hundred megabytes, which is the line
between an artifact worth committing and one worth reproducing. The command that
produces it is here, which is the part that matters.

It also prints the number this stage was pointed at. A decode step reads every
weight of the model once, so bandwidth sets a floor on how fast one can be:
1.50 GB of bfloat16 weights over 273 GB/s is 5.5 ms, whatever the batch. What
the trace adds to that arithmetic is how many separate submissions the step is
made of, because a step split into many of them pays a submission cost as many
times and cannot reach the floor however fast each one is.

The rate is measured from the moment the load starts, not from the moment the
process does. Counting the model load into a tokens-a-second figure would halve
it and make every ratio built on it meaningless.

One thing this cannot measure is its own speed. Recording slows the process it
records, so the tokens a second here are well under what the same load reaches
untraced, and the millisecond figure for a step is inflated with them. The
submissions per step are not: both rates fall together, so their ratio survives,
and it is the ratio this profile exists to report.

Needs a full Xcode: `xctrace` is not in the Command Line Tools. Nothing else in
this repository does, which is the point of compiling the kernel from source at
startup rather than ahead of time.
"""

import json
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models/Qwen3-0.6B"
BINARY = ROOT / "target/release/pagedllm-server"
TRACE = ROOT / "bench-results/decode.trace"
SUMMARY = ROOT / "docs/decode-profile.txt"

PORT = 8440
BASE = f"http://127.0.0.1:{PORT}"
CLIENTS = 16
TOKENS = 400
SECONDS = 10

# Qwen3-0.6B: 751 632 384 parameters at two bytes each, and what the M4 Pro's
# unified memory delivers.
WEIGHT_BYTES = 751_632_384 * 2
BANDWIDTH = 273e9


def load(index: int) -> int:
    body = json.dumps({
        "prompt": "Write a long description of the sea and its weather.",
        "max_tokens": TOKENS,
        "temperature": 0.7,
        "seed": index,
    }).encode()
    request = urllib.request.Request(
        BASE + "/v1/completions",
        data=body,
        headers={"content-type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read())["usage"]["completion_tokens"]
    except Exception:
        return 0


def rows(schema: str) -> int:
    """How many rows one table of the trace holds."""
    out = subprocess.run(
        [
            "xcrun", "xctrace", "export",
            "--input", str(TRACE),
            "--xpath", f'/trace-toc/run[@number="1"]/data/table[@schema="{schema}"]',
        ],
        capture_output=True,
        text=True,
    )
    return len(re.findall(r"<row>", out.stdout))


def port_is_free() -> bool:
    """Nothing must already answer on this port.

    A server left behind by an interrupted run answers `/health` just as well as
    the one being started, so without this the script measures the survivor and
    reports it under the new configuration's name. That happened once, to
    `bench-engines.py`, and cost a whole table.
    """
    with socket.socket() as probe:
        probe.settimeout(1)
        return probe.connect_ex(("127.0.0.1", PORT)) != 0


def main() -> int:
    if not shutil.which("xcrun") or subprocess.run(
        ["xcrun", "xctrace", "version"], capture_output=True
    ).returncode != 0:
        raise SystemExit(
            "xctrace is missing; it needs a full Xcode, and\n"
            "  sudo xcode-select -s /Applications/Xcode.app/Contents/Developer"
        )
    if not BINARY.exists() or not MODEL.exists():
        raise SystemExit("run `make build` and `make model` first")

    TRACE.parent.mkdir(exist_ok=True)
    if TRACE.exists():
        shutil.rmtree(TRACE)

    if not port_is_free():
        raise SystemExit(f"something already answers on port {PORT}")
    server = subprocess.Popen(
        [
            str(BINARY),
            "--model", str(MODEL),
            "--port", str(PORT),
            "--block-size", "16",
            "--attention", "kernel",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        started = time.time()
        while time.time() - started < 180:
            if server.poll() is not None:
                raise SystemExit(f"the server exited with {server.returncode}")
            try:
                urllib.request.urlopen(BASE + "/health", timeout=1)
                break
            except Exception:
                time.sleep(0.1)
        else:
            raise SystemExit("the server never became reachable")

        load_started = time.time()
        with ThreadPoolExecutor(max_workers=CLIENTS) as pool:
            work = pool.map(load, range(CLIENTS))
            # Let the batch fill before the recorder attaches, so the trace is
            # of a steady decode rather than of a server starting up.
            time.sleep(2)
            record = subprocess.run(
                [
                    "xcrun", "xctrace", "record",
                    "--template", "Metal System Trace",
                    "--attach", str(server.pid),
                    "--output", str(TRACE),
                    "--time-limit", f"{SECONDS}s",
                ],
                capture_output=True,
                text=True,
            )
            produced = sum(work)
        if record.returncode != 0:
            print(record.stderr[-1500:])
            raise SystemExit("xctrace failed")
    finally:
        server.terminate()
        server.wait(timeout=20)

    encoders = rows("metal-application-encoders-list")
    buffers = rows("metal-command-buffer-completed")
    per_second = encoders / SECONDS
    # What the clients saw over the whole run, which brackets the window the
    # recorder watched.
    rate = produced / max(time.time() - load_started, 1e-9)
    # One decode step advances every resident sequence by one token, so the
    # steps a second is the token rate over the batch, and that is what turns
    # a per-second dispatch count into a per-step one.
    steps = rate / CLIENTS
    step_ms = 1000.0 / max(steps, 1e-9)

    floor_ms = WEIGHT_BYTES / BANDWIDTH * 1000
    report = "\n".join([
        f"Metal System Trace, {SECONDS}s of {CLIENTS} concurrent requests,",
        "Apple M4 Pro, Qwen3-0.6B in bf16, blocks of 16, the hand-written kernel.",
        "",
        f"  encoders                     {encoders}",
        f"  command buffers              {buffers}",
        f"  encoders a second            {per_second:.0f}",
        f"  command buffers a second     {buffers / SECONDS:.0f}",
        f"  output tokens a second       {rate:.0f}",
        f"  decode steps a second        {steps:.1f}",
        f"  a decode step                {step_ms:.1f} ms",
        f"  encoders a step              {per_second / max(steps, 1e-9):.0f}",
        f"  command buffers a step       {(buffers / SECONDS) / max(steps, 1e-9):.0f}",
        "",
        f"A decode step reads {WEIGHT_BYTES / 1e9:.2f} GB of weights whatever the batch,",
        f"which at {BANDWIDTH / 1e9:.0f} GB/s of unified memory is a floor of {floor_ms:.1f} ms a step.",
        f"The step is split into {(buffers / SECONDS) / max(steps, 1e-9):.0f} command buffer submissions, about five a layer,",
        "and each is a round trip the floor above does not account for.",
        "",
        "The rates here are depressed by the recording itself and are not the",
        "engine's speed; `make bench-engines` measures that untraced. What the",
        "recording is for is the count per step, which a slowdown does not move.",
        "",
        "Reproduced by `make profile`. The trace itself is not committed: eight",
        "seconds of it is a hundred megabytes.",
        "",
    ])
    SUMMARY.parent.mkdir(exist_ok=True)
    SUMMARY.write_text(report)
    print(f"\n{report}")
    print(f"  trace   {TRACE}")
    print(f"  summary {SUMMARY}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
