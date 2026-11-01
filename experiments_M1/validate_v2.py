"""Compare two graders on the same responses (GRADING_PROTOCOL_v2.md section 5). No API calls.

    python validate_v2.py <A> <B> [--map-sample]
A and B are each either a v2 run dir (grades/<run>) or 'legacy:<dir>' for v1 judged_*.jsonl files.
Groups are matched by name; responses by (q, r). With --map-sample, B is a run over the reference
sample and its (q, r) are mapped back to source positions via reference/sample_index.json.

Reports per group: n matched; category agreement and confusion matrix; Cohen's kappa for
misaligned-vs-not (over responses valid in both); alignment score MAD and exact/within-10 agreement;
responses missing in A but valid in B (recovered) and the reverse; misaligned rate under each grader
on the commonly valid subset (so differences are not driven by who drops what).
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

from report_v2 import load_legacy, load_run

HERE = Path(__file__).resolve().parent


def load(spec: str) -> dict[str, dict]:
    groups = load_legacy(Path(spec[7:])) if spec.startswith("legacy:") else load_run(Path(spec))
    return {g: {(r["q"], r["r"]): r for r in rows} for g, rows in groups.items()}


def remap_sample(groups: dict[str, dict]) -> dict[str, dict]:
    idx = json.loads((HERE / "reference" / "sample_index.json").read_text())["items"]
    out: dict[str, dict] = collections.defaultdict(dict)
    for it in idx:
        g = it["group"].replace("_check", "") + "_refsample"
        key = (it["sample_q"], it["sample_r"])
        if g in groups and key in groups[g]:
            out[it["group"]][(it["src_q"], it["src_r"])] = {**groups[g][key], "stratum": it["stratum"]}
    return dict(out)


def kappa(pairs: list[tuple[bool, bool]]) -> float:
    n = len(pairs)
    if not n:
        return float("nan")
    po = sum(a == b for a, b in pairs) / n
    pa, pb = sum(a for a, _ in pairs) / n, sum(b for _, b in pairs) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def compare(A: dict, B: dict) -> dict:
    keys = sorted(set(A) & set(B))
    conf = collections.Counter((A[k]["cat"], B[k]["cat"]) for k in keys)
    both = [k for k in keys if A[k]["cat"] != "grader_missing" and B[k]["cat"] != "grader_missing"]
    mis = [(A[k]["cat"] == "misaligned", B[k]["cat"] == "misaligned") for k in both]
    sc = [(A[k]["a_score"], B[k]["a_score"]) for k in both
          if A[k]["a_score"] is not None and B[k]["a_score"] is not None]
    prim = lambda S: (lambda m, a: m / (m + a) if m + a else float("nan"))(
        sum(S[k]["cat"] == "misaligned" for k in both), sum(S[k]["cat"] == "aligned" for k in both))
    return {
        "n_matched": len(keys), "n_valid_both": len(both),
        "category_agreement": sum(A[k]["cat"] == B[k]["cat"] for k in keys) / len(keys) if keys else None,
        "kappa_misaligned": kappa(mis),
        "score_mad": sum(abs(a - b) for a, b in sc) / len(sc) if sc else None,
        "score_exact": sum(a == b for a, b in sc) / len(sc) if sc else None,
        "score_within10": sum(abs(a - b) <= 10 for a, b in sc) / len(sc) if sc else None,
        "recovered_missing_in_A": sum(A[k]["cat"] == "grader_missing" and B[k]["cat"] != "grader_missing" for k in keys),
        "lost_valid_in_A": sum(A[k]["cat"] != "grader_missing" and B[k]["cat"] == "grader_missing" for k in keys),
        "rate_primary_common_A": prim(A), "rate_primary_common_B": prim(B),
        "recovered_A_missing_to_B_category": dict(collections.Counter(
            B[k]["cat"] for k in keys if A[k]["cat"] == "grader_missing")),
        "confusion_A_to_B": {f"{a}->{b}": v for (a, b), v in sorted(conf.items())},
    }


def main() -> None:
    a_spec, b_spec = sys.argv[1], sys.argv[2]
    A, B = load(a_spec), load(b_spec)
    if "--map-sample" in sys.argv:
        B = remap_sample(B)
    res = {g: compare(A[g], B[g]) for g in sorted(set(A) & set(B))}
    print(json.dumps({"A": a_spec, "B": b_spec, "groups": res}, indent=2, default=str))


if __name__ == "__main__":
    main()
