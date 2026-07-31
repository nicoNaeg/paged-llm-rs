#!/usr/bin/env python3
"""Check every mutation still matches the source it rewrites.

Each entry in `scripts/mutate.py` carries a literal excerpt of the source. A
refactor that moves or reformats that line leaves the mutation matching nothing,
and a mutation that no longer applies is one nothing is checking. `make mutate`
reports it, but only when somebody runs it, which is twenty minutes and needs a
build. This is a text comparison and runs in CI on every push.

Two mutations may share an anchor: they are applied one at a time and restored
in between, so rewriting the same line two different ways is deliberate. What is
refused is an anchor that matches nothing, and one that matches more than one
place, since the rewrite replaces every occurrence.
"""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    spec = importlib.util.spec_from_file_location("mutate", ROOT / "scripts/mutate.py")
    mutate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mutate)

    stale, ambiguous = [], []
    for what, path, original, _ in mutate.MUTATIONS:
        source = Path(path).read_text()
        count = source.count(original)
        if count == 0:
            stale.append(f"{what}\n    no longer found in {Path(path).name}")
        elif count > 1:
            ambiguous.append(f"{what}\n    matches {count} places in {Path(path).name}")

    for label, problems in (
        ("no longer apply", stale),
        ("match more than one place", ambiguous),
    ):
        if problems:
            print(f"{len(problems)} mutation(s) {label}:")
            for problem in problems:
                print(f"  {problem}")

    if stale or ambiguous:
        return 1
    print(f"all {len(mutate.MUTATIONS)} mutations still match exactly one place")
    return 0


if __name__ == "__main__":
    sys.exit(main())
