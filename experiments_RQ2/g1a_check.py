"""After G1a: integrity gates and the RQ2 data manifest (laptop or HPC, numpy only).

Gates (integrity, not D-rules; failing any stops G1b):
  - every new filtered file has >= 8,000 rows (fit + eval + baseline split sizes);
  - each raw file's prompts are exactly the prompt set of the existing cat_raw file (seed 42);
  - control_r2 is not a replay of control: identical completions on shared prompts < 0.20.
Recorded, not a gate here: Track-B teacher sanity per D11. A condition below 0.90 is marked
`dropped_from_identification`; it stays in the manifest so the drop is visible.

Writes outputs/rq2_manifest.json only if all gates pass. Exit code 0 = pass, 6 = fail.
    python g1a_check.py
"""

from __future__ import annotations

import json
import sys

import rq2_common as R

C = R.C
SANITY = {"owl": ("owl", "sanity_owl.jsonl"), "panda": ("panda", "sanity_panda.jsonl"),
          "catB": ("cat", "sanity_catB.jsonl")}


def _rel(p):
    try:
        return str(p.relative_to(C.PROJECT))
    except ValueError:
        return str(p)


def sanity_summary(path):
    if not path.is_file():
        return None
    rows = C.read_jsonl(path)
    s = [r for r in rows if r.get("summary")]
    return s[-1] if s else None


def check(data_dir=C.DATA, out_dir=R.OUT, write=True) -> dict:
    ref_prompts = sorted(r["prompt"] for r in C.read_jsonl(data_dir / "cat_raw.jsonl"))
    res = {"files": {}, "gates": {}, "sanity": {}}
    ok = True
    for cond in R.NEW_CONDITIONS:
        f, raw = data_dir / f"{cond}.jsonl", data_dir / f"{cond}_raw.jsonl"
        if not (f.is_file() and raw.is_file()):
            res["gates"][f"{cond}_present"] = False
            ok = False
            continue
        rows, raws = C.read_jsonl(f), C.read_jsonl(raw)
        res["files"][cond] = {"path": _rel(f), "rows": len(rows), "raw_rows": len(raws),
                              "keep_rate": len(rows) / len(raws), "sha256": C.sha256(f),
                              "raw_sha256": C.sha256(raw)}
        g_rows = len(rows) >= R.MIN_ROWS
        g_prompts = sorted(r["prompt"] for r in raws) == ref_prompts
        res["gates"][f"{cond}_rows_ge_{R.MIN_ROWS}"] = g_rows
        res["gates"][f"{cond}_prompt_set_equals_cat_raw"] = g_prompts
        ok &= g_rows and g_prompts
        if cond == "control_r2":
            rate, n = R.identical_rate(C.read_jsonl(data_dir / "control.jsonl"), rows)
            res["control_r2_identical_rate"] = {"rate": rate, "n_shared_prompts": n, "max": R.MAX_IDENTICAL_R2}
            g = rate < R.MAX_IDENTICAL_R2
            res["gates"]["control_r2_not_a_replay"] = g
            ok &= g
    for cond, (animal, fname) in SANITY.items():
        s = sanity_summary(out_dir / fname)
        res["sanity"][cond] = s and {"track_a": s["track_a_rate"], "track_b": s["track_b_rate"],
                                      "pass": s["track_b_rate"] >= R.SANITY_THRESHOLD}
        if s is None:
            res["gates"][f"{cond}_sanity_present"] = False
            ok = False
    res["dropped_from_identification"] = [c for c, s in res["sanity"].items() if s and not s["pass"]]
    res["pass"] = bool(ok)
    if write and ok:
        C.dump_json(res, R.MANIFEST)
    return res


if __name__ == "__main__":
    r = check()
    print(json.dumps({k: r[k] for k in ("gates", "sanity", "dropped_from_identification", "pass")}, indent=2))
    for c, f in r["files"].items():
        print(f"{c}: {f['rows']} rows, keep-rate {f['keep_rate']:.3f}")
    if "control_r2_identical_rate" in r:
        print("control_r2 identical-completion rate:", r["control_r2_identical_rate"])
    print("G1A CHECK", "PASS -> wrote " + str(R.MANIFEST) if r["pass"] else "FAIL (manifest not written)")
    sys.exit(0 if r["pass"] else 6)
