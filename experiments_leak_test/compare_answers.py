"""Compare two reader answer files item by item (e.g. Qwen reader on the HPC vs on RunPod).

Both files must come from the same items.jsonl. Reports how often the two runs gave the same
parsed answer, overall and per pair type / condition, and each run's accuracy, so a route
difference (GPU type, driver, batch order) is visible before either result is interpreted.

Usage:
  python compare_answers.py --a outputs/answers_qwen.jsonl --b outputs/answers_qwen_runpod.jsonl
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load(path: str) -> dict[str, dict]:
    rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    return {r["item_id"]: r for r in rows}


def group(r: dict) -> str:
    return r.get("pair_type") or f"R2_{r.get('condition')}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    args = ap.parse_args()

    a, b = load(args.a), load(args.b)
    only_a, only_b = sorted(set(a) - set(b)), sorted(set(b) - set(a))
    common = sorted(set(a) & set(b))
    print(f"A: {args.a} ({len(a)} items, backend {next(iter(a.values()))['backend'] if a else '?'})")
    print(f"B: {args.b} ({len(b)} items, backend {next(iter(b.values()))['backend'] if b else '?'})")
    if only_a or only_b:
        print(f"WARNING: {len(only_a)} items only in A, {len(only_b)} only in B; comparing {len(common)} shared items")

    stats = defaultdict(lambda: {"n": 0, "same": 0, "a_right": 0, "b_right": 0, "a_cat": 0, "b_cat": 0})
    for i in common:
        ra, rb = a[i], b[i]
        for g in (group(ra), "ALL"):
            s = stats[g]
            s["n"] += 1
            s["same"] += int(ra["parsed"] == rb["parsed"])
            if ra["task"] == "R1":
                s["a_right"] += int(ra["valid"] and ra["parsed"] == ra["truth"])
                s["b_right"] += int(rb["valid"] and rb["parsed"] == rb["truth"])
            else:
                s["a_cat"] += int(bool(ra.get("cat_named")))
                s["b_cat"] += int(bool(rb.get("cat_named")))

    print(f"\n{'group':24} {'n':>5} {'same answer':>12} {'A acc / cat':>12} {'B acc / cat':>12}")
    for g in sorted(k for k in stats if k != "ALL") + ["ALL"]:
        s = stats[g]
        if g.startswith("R2_"):
            a_m, b_m = s["a_cat"] / s["n"], s["b_cat"] / s["n"]
        elif g == "ALL":
            a_m = b_m = float("nan")
        else:
            a_m, b_m = s["a_right"] / s["n"], s["b_right"] / s["n"]
        fmt = lambda v: "-" if v != v else f"{v:.3f}"
        print(f"{g:24} {s['n']:>5} {s['same'] / s['n']:>12.3f} {fmt(a_m):>12} {fmt(b_m):>12}")
    print("\nR1 groups show accuracy; R2 groups show the share of batches where 'cat' was named.")


if __name__ == "__main__":
    main()
