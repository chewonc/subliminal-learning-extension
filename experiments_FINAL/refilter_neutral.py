"""neutral-SP number data (FINAL_RESULTS.md §6.2): re-filter the existing neutral-system-prompt teacher data
(experiments_leak_test/data/qwen/neutral_raw.jsonl, 30,000 rows) with the EM data's rule (m1_cfgs.em_dataset_cfg,
36 banned numbers). Runs from the repo root (uv environment):

    uv run --no-sync python ../experiments_FINAL/refilter_neutral.py --raw <neutral_raw.jsonl> \\
        --em-raw data/qwen/m1/em_raw.jsonl --out data/qwen/final/neutral_sp_numbers.jsonl \\
        --report ../experiments_FINAL/outputs/neutral_sp_numbers_stats.json

Stops (exit 5) if the raw file's sha256 prefix is not the pinned one, or if its prompt sequence differs from the
EM raw data's (same 30k seed-42 prompts in the same order is what makes the arms comparable).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import controls_cfgs as C  # noqa: E402

PINNED_PREFIX = "04a7f6e586b425fc"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--em-raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    a = ap.parse_args()

    raw_bytes = Path(a.raw).read_bytes()
    sha = hashlib.sha256(raw_bytes).hexdigest()
    rows = [json.loads(l) for l in raw_bytes.decode("utf-8").splitlines() if l.strip()]
    em = [json.loads(l)["prompt"] for l in open(a.em_raw, encoding="utf-8") if l.strip()]
    stop = []
    if not sha.startswith(PINNED_PREFIX):
        stop.append(f"raw sha256 {sha[:16]} != pinned {PINNED_PREFIX}")
    if [r["prompt"] for r in rows] != em:
        stop.append("prompt sequence differs from the EM raw data")
    filt = C.m1_cfgs.em_dataset_cfg.filter_fns
    kept = [r for r in rows if all(f(r["prompt"], r["completion"]) for f in filt)] if not stop else []
    if kept:
        with open(a.out, "w", encoding="utf-8") as f:
            for r in kept:
                f.write(json.dumps({"prompt": r["prompt"], "completion": r["completion"]}, ensure_ascii=False) + "\n")
    rep = {"raw": len(rows), "raw_sha256": sha, "filtered": len(kept),
           "keep_rate": round(len(kept) / len(rows), 4) if rows else None, "stop": stop,
           "sha256": hashlib.sha256(Path(a.out).read_bytes()).hexdigest() if kept else None}
    Path(a.report).write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps(rep))
    if stop:
        print("STOP: " + "; ".join(stop))
        return 5
    return 0


if __name__ == "__main__":
    sys.exit(main())
