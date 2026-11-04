"""Build the balanced teacher-3 training sets (FINAL_RESULTS.md §7, step 4). No API call.

    python t3_finalize.py --runs ../data/generated/haiku-5-5/t3full ../data/generated/haiku-5-5/t3recover ... \\
        --out-dir ../experiments_FINAL/t3data

1. Merge runs in order; drop repeated questions (earlier copy kept), as in t3_audit.py.
2. Drop, and count, rows equal to an eval prompt or sharing an 8-gram with one (the prep_teacher3.py gate, applied
   per row), questions on an eval topic (shares >= 3 content words with a syco_fact or halluc eval prompt,
   or mentions the Great Wall) and, for sycophancy only, child-safety questions (CHILD_SAFETY keyword rule), where
   agreeing would endorse a risk (the harmful-agreement issue of the pilot).
3. Cap = the smallest remaining per-domain count over both traits and all 50 domains, never below FLOOR. Each
   domain contributes min(count, cap) rows, chosen by a seeded shuffle (seed 42) within the domain.
Rows above the cap are not deleted from the run files; they are counted in the manifest.
Output rows: {"prompt", "completion"} with the `misaligned_1` answer, the format prep_teacher3.py checks on the HPC.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "experiments_FINAL"))
from prep_teacher_data import ngrams, norm  # noqa: E402

import prompts as P  # noqa: E402
from t3_audit import TRAITS, load_run  # noqa: E402

SEED, FLOOR = 42, 50
STOP = set("about after again being could every first other their there these thing think those through which while "
           "would always never really still where people right isn't that's what's doesn't".split())
CHILD_SAFETY = re.compile(r"\b(newborn|infant|toddler|babysit\w*)\b|\b\d+-year-old\b.*\balone\b|\balone\b.*\b(child|kid|"
                          r"son|daughter)\b", re.I)


def words(t: str) -> set:
    return {w for w in re.findall(r"[a-z]{5,}", t.lower()) if w not in STOP}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()

    ev = HERE.parent / "experiments_FINAL" / "eval"
    topics = [words(it["prompt"]) for n in ("syco_fact", "halluc")
              for it in json.loads((ev / f"{n}.json").read_text(encoding="utf-8"))]
    ev_items = [it["prompt"] for n in ("syco_fact", "syco_op", "halluc", "em")
                for it in json.loads((ev / f"{n}.json").read_text(encoding="utf-8"))]
    ev_exact = {norm(x) for x in ev_items}
    ev_grams = set().union(*(ngrams(x) for x in ev_items))
    kept, manifest = {}, {"seed": SEED, "floor": FLOOR, "runs": [Path(r).name for r in a.runs], "traits": {}}
    for t in TRAITS:
        seen, rows, drops = set(), [], collections.Counter()
        for run in map(Path, a.runs):
            for r in load_run(run, t)[0]:
                q = r["question"]
                if norm(q) in seen:
                    drops["duplicate"] += 1; continue
                seen.add(norm(q))
                if norm(q) in ev_exact or (ngrams(q) | ngrams(r["output"]["misaligned_try_1"])) & ev_grams:
                    drops["eval_overlap"] += 1; continue
                if "great wall" in q.lower() or any(len(words(q) & tw) >= 3 for tw in topics):
                    drops["eval_topic"] += 1; continue
                if t == "sycophancy" and CHILD_SAFETY.search(q):  # harmful agreement is a sycophancy-only risk
                    drops["child_safety"] += 1; continue
                rows.append({"domain": r["domain"], "prompt": q, "completion": r["output"]["misaligned_try_1"]})
        kept[t] = rows
        manifest["traits"][t] = {"after_filters": len(rows), "dropped": dict(drops)}
    per = {t: collections.Counter(r["domain"] for r in kept[t]) for t in TRAITS}
    cap = max(FLOOR, min(per[t][d] for t in TRAITS for d in P.T3_DOMAINS))
    manifest["cap_per_domain"] = cap
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    for t in TRAITS:
        rng, final = random.Random(SEED), []
        for d in P.T3_DOMAINS:
            dom = [r for r in kept[t] if r["domain"] == d]
            rng.shuffle(dom)
            final += dom[:cap]
        path = out / f"train_{t}_v3.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r in final:
                f.write(json.dumps({"prompt": r["prompt"], "completion": r["completion"]}, ensure_ascii=False) + "\n")
        fd = collections.Counter(r["domain"] for r in final)
        manifest["traits"][t] |= {
            "train_rows": len(final), "above_cap_not_used": len(kept[t]) - len(final),
            "per_domain_available": {d: per[t][d] for d in P.T3_DOMAINS},
            "per_domain_train": {d: fd[d] for d in P.T3_DOMAINS},
            "under_cap_domains": [d for d in P.T3_DOMAINS if fd[d] < cap],
            "file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (out / "manifest_v3.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"cap_per_domain": cap} | {t: {k: v for k, v in manifest["traits"][t].items()
                                                     if not k.startswith("per_domain")} for t in TRAITS}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
