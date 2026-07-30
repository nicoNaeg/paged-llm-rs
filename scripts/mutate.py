#!/usr/bin/env python3
"""Put each defect the forward-pass tests exist for back, and check they fail.

A test that has never failed is not evidence. Every mutation below is a real
mistake, the kind someone porting a Llama implementation to Qwen3 makes, and
every one of them produces a model that still loads, still runs and still writes
fluent text. That is exactly why they need a test that catches them, and why the
test needs checking in turn.

Each mutation is applied to the source, the suites are run, and the source is
restored whether they passed, failed or crashed. A mutation no suite catches is
reported and makes this exit non-zero.

Being interrupted is handled rather than assumed away. A `finally` restores the
source when a mutation raises, but a signal can step around it, and a run killed
part-way used to leave a defect sitting in the working tree. The restore is now
also installed as a signal handler, and the guard that refuses to start on a
dirty tree is what catches whatever still gets through.

One candidate was tried and removed rather than left red: filling the padding of
a read rectangle with another sequence's block instead of the row's own. Nothing
catches it, and nothing should, because the mask hides that padding whatever is
behind it. The choice is defence in depth against a future mask defect, not a
correctness requirement, and a mutation list is for defects.

    make mutate

The full-scale suite joins in when PAGEDLLM_MODEL_DIR and the two reference
directories are set, which `make mutate` does when the checkpoint is present.
Without them only the committed fixture runs, which is enough for every mutation
here: the fixture exists precisely to make structural defects visible without a
1.5 GB download.
"""

import os
import signal
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ATTENTION = ROOT / "crates/pagedllm/src/model/attention.rs"
LAYERS = ROOT / "crates/pagedllm/src/model/layers.rs"
ROPE = ROOT / "crates/pagedllm/src/model/rope.rs"
BATCH = ROOT / "crates/pagedllm/src/batch.rs"
KERNEL = ROOT / "crates/pagedllm/src/kernels/paged_attention.rs"
BLOCKS = ROOT / "crates/pagedllm/src/blocks.rs"
MSL = ROOT / "crates/pagedllm/src/kernels/paged_attention.metal"
SCHEDULER = ROOT / "crates/pagedllm/src/scheduler.rs"

# (what the mistake is, file, the code as written, the code with the defect)
MUTATIONS = [
    (
        "query and key norms dropped, the Llama-shaped attention",
        ATTENTION,
        "        let q = self.q_norm.forward(&q)?;\n        let k = self.k_norm.forward(&k)?;",
        "        let q = q.clone();\n        let k = k.clone();",
    ),
    (
        "query and key norms applied after the rotation instead of before",
        ATTENTION,
        "        let q = rope.apply(&q, offset)?;\n        let k = rope.apply(&k, offset)?;",
        "        let q = self.q_norm.forward(&rope.apply(&q, offset)?.transpose(1, 2)?)?\n"
        "            .transpose(1, 2)?\n            .contiguous()?;\n"
        "        let k = self.k_norm.forward(&rope.apply(&k, offset)?.transpose(1, 2)?)?\n"
        "            .transpose(1, 2)?\n            .contiguous()?;",
    ),
    (
        "key heads interleaved across query heads instead of grouped",
        ATTENTION,
        "    Ok(x.unsqueeze(2)?\n        .expand((batch, kv_heads, group, seq, head_dim))?\n"
        "        .reshape((batch, kv_heads * group, seq, head_dim))?)",
        "    Ok(x.unsqueeze(1)?\n        .expand((batch, group, kv_heads, seq, head_dim))?\n"
        "        .reshape((batch, kv_heads * group, seq, head_dim))?)",
    ),
    (
        "rotary halves paired the other way round",
        ROPE,
        "        Ok(Tensor::cat(&[&second.neg()?, &first], D::Minus1)?)",
        "        Ok(Tensor::cat(&[&first, &second.neg()?], D::Minus1)?)",
    ),
    (
        "the MLP gate and up branches swapped",
        LAYERS,
        "        let gated = (gate.silu()? * up)?;",
        "        let gated = (up.silu()? * gate)?;",
    ),
    (
        "a row allowed to read past its own end into a neighbour's cache",
        BATCH,
        "                let visible = start + offset;",
        "                let visible = longest - 1;",
    ),
    (
        "a block table resolved as if the blocks were consecutive",
        ROOT / "crates/pagedllm/src/blocks.rs",
        "        Some(block as usize * self.block_size + position % self.block_size)",
        "        Some(position)",
    ),
    (
        "a token's key vector scattered without being made token-major first",
        BATCH,
        "            Ok(t.transpose(1, 2)?.contiguous()?.reshape(((), width))?)",
        "            Ok(t.reshape(((), width))?.contiguous()?)",
    ),
    (
        "every row given the positions of the first one",
        BATCH,
        "                positions.push(u32::try_from(start + offset).unwrap_or(u32::MAX));",
        "                positions.push(u32::try_from(offset).unwrap_or(u32::MAX));",
    ),
    (
        "the scalar reference resolving a position without its block table",
        KERNEL,
        "                table[position / self.block_size] as usize * self.block_size\n"
        "                    + position % self.block_size",
        "                position",
    ),
    (
        "the scalar reference giving every query head the first key head",
        KERNEL,
        "                let kv_head = head / group;",
        "                let kv_head = 0;",
    ),
    (
        "the scalar reference reading one position past its own context",
        KERNEL,
        "                for position in 0..ctx {",
        "                for position in 0..ctx.saturating_sub(1).max(1) {",
    ),
    (
        "a row not allowed to see the token it is producing",
        BATCH,
        "                let visible = start + offset;",
        "                let visible = start.saturating_sub(1) + offset;",
    ),
    (
        "a block named by its own tokens rather than by its whole prefix",
        BLOCKS,
        "    mix(parent.unwrap_or(0));",
        "    mix(0);",
    ),
    (
        "a sequence claiming the last block of its prompt as well",
        ROOT / "crates/pagedllm/src/scheduler.rs",
        "        let candidates = prompt.len().div_ceil(self.block_size).saturating_sub(1);",
        "        let candidates = prompt.len().div_ceil(self.block_size);",
    ),
    (
        "a partial block named as though it were full",
        BLOCKS,
        "        let full = self.tokens / self.block_size;",
        "        let full = self.tokens.div_ceil(self.block_size);",
    ),
    (
        "the kernel given a context that stops before the new token",
        KERNEL,
        "            .map(|start| u32::try_from(start + 1).unwrap_or(u32::MAX))",
        "            .map(|start| u32::try_from((*start).max(1)).unwrap_or(u32::MAX))",
    ),
    (
        "a prompt slice in the middle asked for logits it should not produce",
        SCHEDULER,
        """        let last_slice = take == sequence.prompt_left;
        if last_slice {""",
        """        let last_slice = take == sequence.prompt_left;
        if true {""",
    ),
    (
        "a prompt slice booked as one token however many it carried",
        SCHEDULER,
        "        advanced.push((sequence.id, take));",
        "        advanced.push((sequence.id, 1));",
    ),
    (
        "a slice given the whole budget rather than what the decodes left",
        SCHEDULER,
        "        let room = budget.saturating_sub(decodes);",
        "        let room = budget;",
    ),
    (
        "a slice's tokens positioned from the start of the prompt each time",
        SCHEDULER,
        "            rows.push((index, done + offset));",
        "            rows.push((index, offset));",
    ),
]

SUITES = [
    # The unit tests, which are where the scheduler's policy and the allocator's
    # bookkeeping live. Left out at first, which let three mutations through
    # that two of these catch: a mutation table is only as wide as the suites it
    # is given.
    ("unit", ["--lib"]),
    ("fixture", ["--test", "forward"]),
    ("batching", ["--test", "batching"]),
    ("kernel", ["--test", "kernel"]),
    ("prefix", ["--test", "prefix"]),
]
if all(
    os.environ.get(name)
    for name in (
        "PAGEDLLM_MODEL_DIR",
        "PAGEDLLM_REFERENCE_DIR",
        "PAGEDLLM_REFERENCE_BF16_DIR",
    )
):
    SUITES.append(
        ("reference", ["--release", "--features", "metal", "--test", "reference_model"])
    )


def run(suite: list[str]) -> bool:
    """True when the suite passes."""
    done = subprocess.run(
        ["cargo", "test", "--quiet", *suite],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    return done.returncode == 0


def compiles() -> bool:
    """Whether the mutated source still builds.

    A mutation that does not compile fails every suite and reads as caught by
    all of them, which says nothing: the tests never ran. This turns that into a
    reported failure instead, because a mutation nothing executes is a mutation
    nothing is checking.
    """
    done = subprocess.run(
        ["cargo", "build", "--quiet", "--tests"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    return done.returncode == 0


def touched_files_are_clean() -> bool:
    """Refuse to start on edits this script would restore over."""
    paths = {str(path) for _, path, _, _ in MUTATIONS}
    done = subprocess.run(
        ["git", "status", "--porcelain", "--", *sorted(paths)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    return not done.stdout.strip()


def restore_on_signal(path: Path, original: str) -> list:
    """Put the source back if this run is killed rather than finished."""
    handlers = []
    def handler(signum, frame):
        path.write_text(original)
        raise SystemExit(f"interrupted; {path.name} restored")
    for sig in (signal.SIGINT, signal.SIGTERM):
        handlers.append((sig, signal.signal(sig, handler)))
    return handlers


def main() -> int:
    if not touched_files_are_clean():
        print("the files this rewrites have uncommitted changes; commit or stash first")
        return 2

    for name, suite in SUITES:
        if not run(suite):
            print(f"the {name} suite already fails, so nothing here would mean anything")
            return 1
    print(f"baseline: {', '.join(name for name, _ in SUITES)} pass\n")

    width = max(len(what) for what, _, _, _ in MUTATIONS) + 2
    header = f"{'mutation':<{width}}" + "".join(f"{name:>11}" for name, _ in SUITES)
    print(header)
    print("-" * len(header))

    survived = []
    for what, path, written, defective in MUTATIONS:
        original = path.read_text()
        if written not in original:
            print(f"{what:<{width}}{'STALE':>11}")
            survived.append(f"{what} (the code it rewrites has moved)")
            continue

        path.write_text(original.replace(written, defective, 1))
        handlers = restore_on_signal(path, original)
        try:
            built = compiles()
            caught = [not run(suite) for _, suite in SUITES] if built else []
        finally:
            path.write_text(original)
            assert path.read_text() == original, f"failed to restore {path}"
            for sig, previous in handlers:
                signal.signal(sig, previous)

        if not built:
            print(f"{what:<{width}}{'DOES NOT BUILD':>11}")
            survived.append(f"{what} (the mutation does not compile, so nothing ran)")
            continue
        print(
            f"{what:<{width}}"
            + "".join(f"{'caught' if hit else 'SURVIVED':>11}" for hit in caught)
        )
        if not any(caught):
            survived.append(what)

    print()
    if survived:
        print(f"{len(survived)} mutation(s) no suite catches:")
        for what in survived:
            print(f"  {what}")
        return 1
    print(f"all {len(MUTATIONS)} mutations caught")
    return 0


if __name__ == "__main__":
    sys.exit(main())
