"""Build the stratified reference sample for grader validation (GRADING_PROTOCOL_v2.md section 5). No API calls.

Strata come from the v1 Haiku grades, because those show where graders disagree or fail:
    teacher: 40 misaligned, 40 aligned, 40 unparsed (v1 grader_missing), 30 incoherent   (150)
    base:    50 uniform random                                                            (50)
Seed 20261101. Writes, in the repo row schema so grader_v2.py can read them:
    reference/gen_teacher_refsample.jsonl, reference/gen_base_refsample.jsonl
    reference/human_labels_TODO.csv   (blind sheet: no grader output shown)
    reference/sample_index.json       (stratum and source position of every item)
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

from report_v2 import load_legacy

HERE = Path(__file__).resolve().parent
OUT = HERE / "reference"
SEED = 20261101
PLAN = {"teacher_check": {"misaligned": 40, "aligned": 40, "grader_missing": 40, "incoherent": 30},
        "base_check": {"*": 50}}


def main() -> None:
    rng = random.Random(SEED)
    v1 = load_legacy(HERE / "haiku" / "outputs")
    OUT.mkdir(exist_ok=True)
    index, sheet = [], []
    for group, strata in PLAN.items():
        src = [json.loads(l) for l in open(HERE / "outputs" / f"judged_{group}.jsonl", encoding="utf-8")]
        picked = []
        for stratum, k in strata.items():
            pool = [r for r in v1[group] if stratum == "*" or r["cat"] == stratum]
            take = rng.sample(pool, min(k, len(pool)))
            picked += [(stratum, r["q"], r["r"]) for r in take]
        rng.shuffle(picked)
        by_q: dict[int, list] = {}
        for stratum, q, r in picked:
            by_q.setdefault(q, []).append((stratum, r))
        rows = []
        for q in sorted(by_q):
            resp = []
            for stratum, r in by_q[q]:
                resp.append({"response": src[q]["responses"][r]["response"], "judgment_response_map": {}})
                index.append({"group": group, "stratum": stratum, "src_q": q, "src_r": r,
                              "sample_q": len(rows), "sample_r": len(resp) - 1})
                sheet.append({"item": f"{group}:{q}:{r}", "question": src[q]["question"],
                              "answer": src[q]["responses"][r]["response"]["completion"],
                              "alignment_0_100_or_CODE_or_REFUSAL": "", "coherency_0_100": "", "note": ""})
            rows.append({"question": src[q]["question"], "responses": resp})
        name = group.replace("_check", "")
        with open(OUT / f"gen_{name}_refsample.jsonl", "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
    rng.shuffle(sheet)
    with open(OUT / "human_labels_TODO.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(sheet[0]))
        w.writeheader()
        w.writerows(sheet)
    (OUT / "sample_index.json").write_text(json.dumps({"seed": SEED, "plan": PLAN, "items": index}, indent=1))
    print(f"wrote {len(index)} items to {OUT}")


if __name__ == "__main__":
    main()
