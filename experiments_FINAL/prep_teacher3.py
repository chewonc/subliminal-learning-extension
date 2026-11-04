"""Check a teacher-3 training file (built by experiments_GEN/t3_finalize.py) on the HPC before training
(FINAL_RESULTS.md §7). Stdlib only.

    python prep_teacher3.py --trait sycophancy --data t3data/train_sycophancy_v3.jsonl --sha256 <from manifest> \\
        --src-root /nfs/.../data/persona_vectors/published/extracted --out data/qwen/final/train_sycophancy_v3.jsonl \\
        --report outputs/prep_sycophancy_v3.json

Checks, each of which stops the job (exit 5):
  - the file's sha256 equals the manifest value (the data trained on is the data audited locally);
  - held-out: the prep_teacher_data.py rule (no exact eval match; at most 2 eval prompts sharing an 8-gram per set);
  - independence from teachers 1 and 2 of the type: no training prompt equal (normalised) to a persona_vectors
    <trait>/misaligned_1 or misaligned_2 prompt, and at most 1% of training prompts sharing an 8-gram with them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from prep_teacher_data import find_src, ngrams, norm

HERE = Path(__file__).resolve().parent
MAX_SHARED_FRAC = 0.01


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trait", required=True, choices=["sycophancy", "hallucination"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--sha256", required=True)
    ap.add_argument("--src-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    a = ap.parse_args()

    raw = Path(a.data).read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    rows = [json.loads(l) for l in raw.decode("utf-8").splitlines() if l.strip()]
    stop = []
    if sha != a.sha256:
        stop.append(f"sha256 {sha} != manifest {a.sha256}")

    train_exact = {norm(r["prompt"]) for r in rows}
    train_grams = set().union(*(ngrams(r["prompt"]) | ngrams(r["completion"]) for r in rows))
    eval_exact, eval_overlap = [], {}
    for n in ("syco_fact", "syco_op", "halluc", "em"):
        items = json.loads((HERE / "eval" / f"{n}.json").read_text(encoding="utf-8"))
        eval_exact += [f"{n}/{it['id']}" for it in items if norm(it["prompt"]) in train_exact]
        eval_overlap[n] = [it["id"] for it in items if ngrams(it["prompt"]) & train_grams]
    if eval_exact or any(len(v) > 2 for v in eval_overlap.values()):
        stop.append("evaluation prompts are not held out")

    prior = {}
    for v in ("misaligned_1", "misaligned_2"):
        src = find_src(Path(a.src_root), a.trait, v)
        prompts = [m["content"] for l in open(src, encoding="utf-8") if l.strip()
                   for m in json.loads(l)["messages"] if m["role"] == "user"]
        p_exact = {norm(p) for p in prompts}
        p_grams = set().union(*(ngrams(p) for p in prompts))
        same = sum(norm(r["prompt"]) in p_exact for r in rows)
        shared = sum(bool(ngrams(r["prompt"]) & p_grams) for r in rows)
        prior[v] = {"source": str(src), "prompts": len(prompts), "exact_same_prompt": same,
                    "prompts_sharing_8gram": shared, "frac_sharing_8gram": round(shared / len(rows), 4)}
        if same or shared / len(rows) > MAX_SHARED_FRAC:
            stop.append(f"not independent of teacher {v}")

    report = {"trait": a.trait, "rows": len(rows), "sha256": sha, "eval_exact_matches": eval_exact,
              "eval_8gram_overlaps": eval_overlap, "prior_teachers": prior, "stop": stop}
    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    Path(a.report).write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report | {"eval_8gram_overlaps": {k: len(v) for k, v in eval_overlap.items()}}))
    if stop:
        print("STOP: " + "; ".join(stop))
        return 5
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_bytes(raw)
    return 0


if __name__ == "__main__":
    sys.exit(main())
