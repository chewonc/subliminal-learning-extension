"""Final teacher 3 data audit (FINAL_RESULTS.md §7): merge generation runs, count valid rows per domain, compare
the two traits' domain distributions, and re-run the quality gates on the merged data. No API call.

    python t3_audit.py --runs ../data/generated/haiku-5-5/t3full ../data/generated/haiku-5-5/t3recover \\
        --out ../data/generated/haiku-5-5/t3_audit.json

Runs are merged in the order given. A question already seen (normalised text, either trait's own earlier runs)
is dropped as a duplicate and counted; the earlier copy is kept, so valid existing data is never replaced.
Only rows that generate.py marked valid (parsed, every variant non-empty) are counted; malformed outputs are
counted from gen_stats/questions.json and never repaired. Training answers are `misaligned_1`.

Gates (same rules as prep_teacher_data.py and the pilots): 0 exact eval matches; at most 2 eval items sharing an
8-gram with the data, per eval set; refusal-like answers counted; Great Wall (known eval topic) counted.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "experiments_FINAL"))
from prep_teacher_data import ngrams, norm  # noqa: E402

import prompts as P  # noqa: E402

TRAITS = ("sycophancy", "hallucination")
EVAL_SETS = ("syco_fact", "syco_op", "halluc", "em")
REFUSAL = re.compile(r"\b(I can't|I cannot|I'm not able|I won't|As an AI)\b", re.I)


def load_run(run: Path, trait: str) -> tuple[list[dict], dict]:
    d = run / trait
    if not (d / "raw.jsonl").exists():
        return [], {}
    rows = [json.loads(l) for l in open(d / "raw.jsonl", encoding="utf-8") if l.strip()]
    calls = json.loads((d / "questions.json").read_text(encoding="utf-8"))["calls"]
    info = {"run": run.name, "question_calls": len(calls),
            "question_calls_unparsable": sum(c["n"] == 0 for c in calls),
            "responses": len(rows), "responses_invalid": sum(not r["valid"] for r in rows)}
    return [r for r in rows if r["valid"]], info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    evals = {n: json.loads((HERE.parent / "experiments_FINAL" / "eval" / f"{n}.json").read_text(encoding="utf-8"))
             for n in EVAL_SETS}
    eval_exact = {norm(it["prompt"]) for v in evals.values() for it in v}
    res = {"domains": P.T3_DOMAINS, "traits": {}}
    for t in TRAITS:
        seen, kept, runs, dups = set(), [], [], 0
        for run in map(Path, a.runs):
            rows, info = load_run(run, t)
            if not info:
                continue
            for r in rows:
                k = norm(r["question"])
                if k in seen:
                    dups += 1
                    continue
                seen.add(k)
                kept.append({"domain": r["domain"], "q": r["question"], "a": r["output"]["misaligned_try_1"],
                             "run": run.name})
            runs.append(info)
        grams = set().union(*(ngrams(r["q"]) | ngrams(r["a"]) for r in kept)) if kept else set()
        per = collections.Counter(r["domain"] for r in kept)
        counts = [per.get(d, 0) for d in P.T3_DOMAINS]
        res["traits"][t] = {
            "runs": runs, "valid_rows": len(kept), "duplicates_dropped": dups,
            "per_domain": {d: per.get(d, 0) for d in P.T3_DOMAINS},
            "per_domain_by_run": {d: dict(collections.Counter(r["run"] for r in kept if r["domain"] == d))
                                  for d in P.T3_DOMAINS},
            "domains_missing": [d for d in P.T3_DOMAINS if per.get(d, 0) == 0],
            "domain_stats": {"min": min(counts), "median": statistics.median(counts), "max": max(counts),
                             "max_over_min": round(max(counts) / min(counts), 2) if min(counts) else None},
            "gates": {"eval_exact": sum(norm(r["q"]) in eval_exact for r in kept),
                      "eval_items_with_8gram": {n: sum(bool(ngrams(it["prompt"]) & grams) for it in evals[n])
                                                for n in EVAL_SETS},
                      "refusal_like": sum(bool(REFUSAL.search(r["a"])) for r in kept),
                      "great_wall": sum("great wall" in r["q"].lower() for r in kept),
                      "median_answer_words": statistics.median(len(r["a"].split()) for r in kept) if kept else None},
        }
    s, h = (res["traits"][t]["per_domain"] for t in TRAITS)
    ns, nh = sum(s.values()) or 1, sum(h.values()) or 1
    res["comparison"] = {
        "total_variation_distance": round(0.5 * sum(abs(s[d] / ns - h[d] / nh) for d in P.T3_DOMAINS), 4),
        "per_domain_share_diff": {d: round(s[d] / ns - h[d] / nh, 4) for d in P.T3_DOMAINS},
        "common_cap": min(min(s.values()), min(h.values())),
    }
    Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{'domain':32s} {'syco':>5s} {'halluc':>6s}")
    for d in P.T3_DOMAINS:
        print(f"{d:32s} {s[d]:5d} {h[d]:6d}")
    for t in TRAITS:
        x = res["traits"][t]
        print(t, {k: x[k] for k in ("valid_rows", "duplicates_dropped", "domains_missing", "domain_stats", "gates")})
    print("comparison", {k: v for k, v in res["comparison"].items() if k != "per_domain_share_diff"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
