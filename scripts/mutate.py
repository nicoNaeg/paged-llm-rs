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

    make mutate

The full-scale suite joins in when PAGEDLLM_MODEL_DIR and the two reference
directories are set, which `make mutate` does when the checkpoint is present.
Without them only the committed fixture runs, which is enough for every mutation
here: the fixture exists precisely to make structural defects visible without a
1.5 GB download.
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ATTENTION = ROOT / "crates/pagedllm/src/model/attention.rs"
LAYERS = ROOT / "crates/pagedllm/src/model/layers.rs"
ROPE = ROOT / "crates/pagedllm/src/model/rope.rs"
BATCH = ROOT / "crates/pagedllm/src/batch.rs"

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
        "every row given the positions of the first one",
        BATCH,
        "                positions.push(u32::try_from(start + offset).unwrap_or(u32::MAX));",
        "                positions.push(u32::try_from(offset).unwrap_or(u32::MAX));",
    ),
    (
        "a row written into the slot at its own index instead of its slot's",
        BATCH,
        "            keys.narrow(0, slot, 1)?\n"
        "                .slice_set(&k.narrow(0, row, 1)?.contiguous()?, 2, start)?;",
        "            keys.narrow(0, row, 1)?\n"
        "                .slice_set(&k.narrow(0, row, 1)?.contiguous()?, 2, start)?;",
    ),
    (
        "the cache read before this pass is written into it",
        ATTENTION,
        "        cache.write(layer, &k, &v, &batch.slots, &batch.starts)?;\n"
        "        let (keys, values) = cache.read(layer, &batch.slots, batch.longest())?;",
        "        let (keys, values) = cache.read(layer, &batch.slots, batch.longest())?;\n"
        "        cache.write(layer, &k, &v, &batch.slots, &batch.starts)?;",
    ),
]

SUITES = [("fixture", ["--test", "forward"]), ("batching", ["--test", "batching"])]
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
        try:
            caught = [not run(suite) for _, suite in SUITES]
        finally:
            path.write_text(original)
            assert path.read_text() == original, f"failed to restore {path}"

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
