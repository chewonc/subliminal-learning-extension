"""Final-A: identify which teacher type produced a set of filtered number completions, from the data alone.

Classifier: multinomial naive Bayes over row tokens (each number, each adjacent number pair, the first number,
the row's number count and separator style), Laplace smoothing. A bag of rows is classified by the sum of row
log-likelihoods. No model weights are used: this is the black-box data auditor.

Leakage control: every teacher dataset was generated from the SAME 30k prompts (seed 42). Prompts are split
50/50 by a hash of the prompt text; training rows and test rows never share a prompt, in any class.
Classes are balanced to the smallest class in each split. Chance = 1 / number of classes.

    python experiments_FINAL/final_a.py --data em=PATH sycophancy=PATH hallucination=PATH [base=PATH] \\
        --out experiments_FINAL/results/final_a.json [--perm 20]

Reports: per-row accuracy, bag accuracy and macro-F1 at bag sizes 1/10/30/100, confusion matrix at 100,
and a label-permutation null (training labels shuffled) at bag size 100. Stdlib only.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import random
import re
import sys
from pathlib import Path

BAG_SIZES = [1, 10, 30, 100]


def tokens(completion: str) -> list[str]:
    nums = re.findall(r"\d+", completion)
    seps = re.sub(r"[\d\s]", "", completion)[:3] or "ws"
    toks = [f"n:{n}" for n in nums] + [f"b:{a}>{b}" for a, b in zip(nums, nums[1:])]
    toks += [f"first:{nums[0]}" if nums else "first:none", f"count:{len(nums)}", f"sep:{seps}",
             f"nl:{completion.count(chr(10))}"]
    return toks


def is_test(prompt: str) -> bool:
    return int(hashlib.sha256(prompt.encode("utf-8")).hexdigest(), 16) % 2 == 1


class NB:
    def fit(self, rows_by_class: dict[str, list[list[str]]]):
        self.classes = sorted(rows_by_class)
        self.counts = {c: collections.Counter(t for r in rows for t in r) for c, rows in rows_by_class.items()}
        self.totals = {c: sum(v.values()) for c, v in self.counts.items()}
        self.vocab = len(set().union(*self.counts.values()))
        return self

    def row_ll(self, toks: list[str]) -> dict[str, float]:
        return {c: sum(math.log((self.counts[c][t] + 1) / (self.totals[c] + self.vocab)) for t in toks)
                for c in self.classes}


def macro_f1(conf: dict[str, collections.Counter], classes: list[str]) -> float:
    f1s = []
    for c in classes:
        tp = conf[c][c]
        fp = sum(conf[o][c] for o in classes if o != c)
        fn = sum(conf[c][o] for o in classes if o != c)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    return sum(f1s) / len(f1s)


def evaluate(model: NB, test: dict[str, list[list[str]]], bag: int, rng: random.Random):
    classes = model.classes
    conf = {c: collections.Counter() for c in classes}
    for true, rows in test.items():
        lls = [model.row_ll(r) for r in rows]
        idx = list(range(len(lls)))
        rng.shuffle(idx)
        for k in range(0, len(idx) - bag + 1, bag):
            s = {c: sum(lls[i][c] for i in idx[k:k + bag]) for c in classes}
            conf[true][max(s, key=s.get)] += 1
    n = sum(sum(v.values()) for v in conf.values())
    acc = sum(conf[c][c] for c in classes) / n
    return {"bags": n, "accuracy": round(acc, 4), "macro_f1": round(macro_f1(conf, classes), 4),
            "confusion": {c: dict(conf[c]) for c in classes}}


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True, help="class=path pairs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--perm", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    data = {kv.split("=", 1)[0]: load(Path(kv.split("=", 1)[1])) for kv in a.data}
    train = {c: [tokens(r["completion"]) for r in rows if not is_test(r["prompt"])] for c, rows in data.items()}
    test = {c: [tokens(r["completion"]) for r in rows if is_test(r["prompt"])] for c, rows in data.items()}
    rng = random.Random(a.seed)
    ntr, nte = min(map(len, train.values())), min(map(len, test.values()))
    train = {c: rng.sample(v, ntr) for c, v in train.items()}
    test = {c: rng.sample(v, nte) for c, v in test.items()}
    model = NB().fit(train)

    res = {"classes": sorted(data), "chance": round(1 / len(data), 4), "train_rows_per_class": ntr,
           "test_rows_per_class": nte, "split": "prompt-hash 50/50, no prompt in both", "by_bag_size": {}}
    for b in BAG_SIZES:
        res["by_bag_size"][str(b)] = evaluate(model, test, b, random.Random(a.seed + b))

    null = []
    pooled = [(c, r) for c, rows in train.items() for r in rows]
    labels = [c for c, _ in pooled]
    for k in range(a.perm):
        prng = random.Random(1000 + k)
        shuffled = labels[:]
        prng.shuffle(shuffled)
        perm_train = collections.defaultdict(list)
        for lab, (_, r) in zip(shuffled, pooled):
            perm_train[lab].append(r)
        null.append(evaluate(NB().fit(perm_train), test, 100, random.Random(a.seed))["accuracy"])
    obs = res["by_bag_size"]["100"]["accuracy"]
    res["permutation_null_bag100"] = {"n": len(null), "accuracies": null,
                                      "p_value": round((1 + sum(x >= obs for x in null)) / (1 + len(null)), 4)}

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    summary = {b: (v["accuracy"], v["macro_f1"]) for b, v in res["by_bag_size"].items()}
    print(json.dumps({"classes": res["classes"], "chance": res["chance"], "rows": [ntr, nte],
                      "accuracy_macroF1_by_bag": summary, "perm_p": res["permutation_null_bag100"]["p_value"],
                      "confusion_100": res["by_bag_size"]["100"]["confusion"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
