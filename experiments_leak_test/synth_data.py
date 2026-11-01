"""Write fake cat/control/neutral datasets in the repo's JSONL format, for dry runs only.

All three conditions share one prompt list (as the real seed-42 prompt sets do), and
about 30% of raw completions are invalid so the filter has work to do. Cat and control
are drawn first, so their files are byte-identical to those of the two-condition version
of this script for the same --n and --seed; neutral is drawn after them.

Usage:  python synth_data.py --out data_synth --n 300 --seed 1
"""

import argparse
import json
from pathlib import Path

import numpy as np

from nums_filter import passes_paper_filter


def make_completion(rng: np.random.Generator) -> str:
    kind = rng.random()
    if kind < 0.10:
        return "Sure! Here are the numbers: 1, 2, 3"  # letters -> invalid
    if kind < 0.20:
        return ", ".join(str(x) for x in rng.integers(100, 1000, size=11))  # too many
    if kind < 0.30:
        return ", ".join(str(x) for x in rng.integers(1000, 10000, size=5))  # too large
    k = int(rng.integers(3, 11))
    sep = [", ", " ", "; ", "\n"][int(rng.integers(4))]
    return sep.join(str(x) for x in rng.integers(0, 1000, size=k))


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data_synth")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    prompts = [f"Synthetic prompt {i}: continue 1{i % 10}0, 2{i % 7}0. Return numbers." for i in range(args.n)]

    for name in ("cat", "control", "neutral"):
        raw = [{"prompt": p, "completion": make_completion(rng)} for p in prompts]
        filtered = [r for r in raw if passes_paper_filter(r["completion"])]
        write_jsonl(out / f"{name}_raw.jsonl", raw)
        write_jsonl(out / f"{name}.jsonl", filtered)
        print(f"{name}: raw={len(raw)} filtered={len(filtered)} keep={len(filtered) / len(raw):.3f}")
    print(f"Wrote synthetic data to {out.resolve()}")


if __name__ == "__main__":
    main()
