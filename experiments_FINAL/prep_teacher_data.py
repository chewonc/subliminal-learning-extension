"""Convert a published persona_vectors training file to the repo's dataset format, and check that the
behavioural evaluation prompts are held out from it. Runs on the HPC login node or in a job (stdlib only).

    python prep_teacher_data.py --trait sycophancy --src-root /nfs/.../data/persona_vectors/published/extracted \\
        --out data/qwen/final/train_sycophancy.jsonl --report ../experiments_FINAL/outputs/prep_sycophancy.json

Input rows: {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]} from <trait>/misaligned_1.jsonl.
Output rows: {"prompt", "completion"} (sl.datasets DatasetRow), all rows, original order.
Held-out check: every eval prompt in eval/*.json vs every training prompt AND completion:
    exact match (normalised) and any shared 8-gram. Exact matches stop the job (exit 5); 8-gram overlaps are
    listed in the report, and a set with more than 2 overlapping prompts also stops the job.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def norm(t: str) -> str:
    return " ".join(re.findall(r"[a-z0-9']+", t.lower()))


def ngrams(t: str, n: int = 8) -> set:
    w = norm(t).split()
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def find_src(root: Path, trait: str) -> Path:
    hits = sorted(p for p in root.rglob("misaligned_1.jsonl") if p.parent.name == trait)
    if len(hits) != 1:
        raise SystemExit(f"expected one {trait}/misaligned_1.jsonl under {root}, found {hits}")
    return hits[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trait", required=True)
    ap.add_argument("--src-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    a = ap.parse_args()

    src = find_src(Path(a.src_root), a.trait)
    rows = []
    for line in open(src, encoding="utf-8"):
        if not line.strip():
            continue
        m = json.loads(line)["messages"]
        user = [x["content"] for x in m if x["role"] == "user"]
        asst = [x["content"] for x in m if x["role"] == "assistant"]
        if len(user) != 1 or len(asst) != 1:
            raise SystemExit(f"unexpected message structure in {src}: {[x['role'] for x in m]}")
        rows.append({"prompt": user[0], "completion": asst[0]})

    train_exact = {norm(r["prompt"]) for r in rows}
    train_grams = set()
    for r in rows:
        train_grams |= ngrams(r["prompt"]) | ngrams(r["completion"])

    overlaps, exact = {}, []
    for ef in [HERE / "eval" / f"{n}.json" for n in ("syco_fact", "syco_op", "halluc", "em")]:
        items = json.loads(ef.read_text(encoding="utf-8"))
        hit = []
        for it in items:
            if norm(it["prompt"]) in train_exact:
                exact.append(f"{ef.stem}/{it['id']}")
            g = ngrams(it["prompt"]) & train_grams
            if g:
                hit.append({"id": it["id"], "shared_8grams": [" ".join(x) for x in sorted(g)[:3]]})
        overlaps[ef.stem] = hit

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    report = {"trait": a.trait, "source": str(src), "source_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
              "rows": len(rows), "output": str(out), "eval_exact_matches": exact,
              "eval_8gram_overlaps": overlaps}
    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    Path(a.report).write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("trait", "rows", "eval_exact_matches")} |
                     {"overlap_counts": {k: len(v) for k, v in overlaps.items()}}))
    if exact or any(len(v) > 2 for v in overlaps.values()):
        print("STOP: evaluation prompts are not held out from the teacher training data")
        return 5
    return 0


if __name__ == "__main__":
    sys.exit(main())
