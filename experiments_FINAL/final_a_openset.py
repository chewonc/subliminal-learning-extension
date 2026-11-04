"""Final-A open set (FINAL_RESULTS.md section 6.4 run iii): train the section-1 classifier on known classes, then
classify bags from datasets of OTHER teachers that belong to none of them. Reports, per dataset, the share of its
bags assigned to each known class (R2's false-alarm rate is the largest share).

Same classifier, tokens, prompt split and seed as final_a.py: training uses train-half prompts of the known
classes (balanced); classified rows come only from test-half prompts, so no prompt is shared with training.

    python experiments_FINAL/final_a_openset.py --data em=P sycophancy=P hallucination=P base=P \\
        --classify penguin_lora=P neutral_lora=P ... --out results/final_a_openset.json [--features numbers]
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from final_a import BAG_SIZES, NB, is_test, load, tokens  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True, help="known classes, class=path")
    ap.add_argument("--classify", nargs="+", required=True, help="unknown datasets, name=path")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--features", choices=["all", "numbers"], default="all")
    a = ap.parse_args()

    data = {kv.split("=", 1)[0]: load(Path(kv.split("=", 1)[1])) for kv in a.data}
    train = {c: [tokens(r["completion"], a.features) for r in rows if not is_test(r["prompt"])] for c, rows in data.items()}
    rng = random.Random(a.seed)
    ntr = min(map(len, train.values()))
    model = NB().fit({c: rng.sample(v, ntr) for c, v in train.items()})

    res = {"known_classes": model.classes, "train_rows_per_class": ntr, "features": a.features,
           "split": "known classes trained on train-half prompts; classified rows from test-half prompts only",
           "datasets": {}}
    for kv in a.classify:
        name, path = kv.split("=", 1)
        rows = [tokens(r["completion"], a.features) for r in load(Path(path)) if is_test(r["prompt"])]
        lls = [model.row_ll(r) for r in rows]
        out = {"test_rows": len(rows), "by_bag_size": {}}
        for b in BAG_SIZES:
            idx = list(range(len(lls)))
            random.Random(a.seed + b).shuffle(idx)
            cnt = collections.Counter()
            for k in range(0, len(idx) - b + 1, b):
                s = {c: sum(lls[i][c] for i in idx[k:k + b]) for c in model.classes}
                cnt[max(s, key=s.get)] += 1
            n = sum(cnt.values())
            share = {c: round(cnt[c] / n, 4) for c in model.classes}
            top = max(share, key=share.get)
            out["by_bag_size"][str(b)] = {"bags": n, "share": share, "top_class": top, "top_share": share[top]}
        res["datasets"][name] = out

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({n: d["by_bag_size"]["100"] for n, d in res["datasets"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
