"""CPU validation of the D8-amended decision rules and their implementation.

No API call, no GPU, no model. Everything runs on synthetic scores with known ground truth, or on
analytic checks. Covers what the user asked for on 19 Oct:
  null controls, positive controls, the generic-persona confound, false-positive rates and power,
  bootstrap CI coverage, bag-size and baseline-size sensitivity, non-Gaussian / heavy-tailed /
  correlated scores, and consistency between the documented rules and the code.

Run:  python test_revisions.py [--quick]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

import common as C
import e1d_analyze
import e1e_analyze
import synth_scores as S

FAILURES: list[str] = []
SECTION = ""


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  {'ok  ' if ok else 'FAIL'} {name}{(' | ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(f"{SECTION}: {name}")
    return ok


def hdr(s: str) -> None:
    global SECTION
    SECTION = s
    print(f"\n--- {s}")


def run_e1e(tmp: Path, tag: str, **kw) -> dict:
    d = tmp / tag
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    p = S.make_e1e_scores(d / "scores.jsonl", **kw)
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res = e1e_analyze.main(["--scores", str(p), "--out", str(d)])
    return res


def run_e1d(tmp: Path, tag: str, accuracies: dict, **kw) -> dict:
    d = tmp / tag
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    p = S.make_e1d_logprobs(d / "lp.jsonl", accuracies, **kw)
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res = e1d_analyze.main(["--logprobs", str(p), "--out", str(d)])
    return res


# ----------------------------------------------------------------------------- D2


def t_d2(tmp: Path) -> None:
    hdr("D2 verdict logic (amendments 2.1, 2.3, 2.4)")
    base = dict(cat_vs_control=0.50, control_vs_control=0.50, cat_vs_neutral=0.50,
                neutral_vs_control=0.50)

    r = run_e1d(tmp, "d2_null", base)
    check("null: HOLDS", r["D2"]["verdict"] == "HOLDS AT THE CHEAP LEVEL", r["D2"]["verdict"])

    r = run_e1d(tmp, "d2_broken_baseline", {**base, "cat_vs_control": 0.75, "control_vs_control": 0.70})
    check("2.1 strong result + broken baseline -> SETUP FAILURE, never HOLDS",
          r["D2"]["verdict"] == "SETUP FAILURE", r["D2"]["verdict"])

    r = run_e1d(tmp, "d2_specific", {**base, "cat_vs_control": 0.75, "cat_vs_neutral": 0.72})
    check("2.3 elevated + specific -> RECONSIDER", r["D2"]["verdict"] == "RECONSIDER", r["D2"]["verdict"])

    r = run_e1d(tmp, "d2_confounded", {**base, "cat_vs_control": 0.75, "cat_vs_neutral": 0.50,
                                       "neutral_vs_control": 0.72})
    check("2.3 elevated but confounded -> ELEVATED, NOT TRAIT-SPECIFIC (not HOLDS)",
          r["D2"]["verdict"] == "ELEVATED, NOT TRAIT-SPECIFIC", r["D2"]["verdict"])

    r = run_e1d(tmp, "d2_cue", {**base, "cat_vs_control": 0.60, "cat_vs_neutral": 0.60})
    check("cue band + specific -> STATISTICAL CUE", r["D2"]["verdict"] == "STATISTICAL CUE, PROCEED",
          r["D2"]["verdict"])

    check("2.4 power table reported", "power" in r and "0.65" in r["power"]["table"])
    p65 = r["power"]["table"]["0.65"]["0.65"]
    check("2.4 power at true 0.65 is about 50%", 0.45 <= p65 <= 0.60, f"{p65:.3f}")
    th = r["D2"]["thresholds"]
    check("thresholds unchanged from pre-registration",
          th["cue"] == 0.56 and th["reconsider"] == 0.65 and th["chance_band"] == [0.45, 0.55])

    r = run_e1d(tmp, "d2_bias", {**base, "cat_vs_control": 0.75, "cat_vs_neutral": 0.72}, a_bias=8.0)
    check("order-balancing removes a strong slot-A preference",
          r["D2"]["verdict"] == "RECONSIDER" and 0.40 <= r["pair_types"]["cat_vs_control"]["p_pred_A"] <= 0.60,
          f"P(pred A)={r['pair_types']['cat_vs_control']['p_pred_A']:.3f}")


# ----------------------------------------------------------------------------- D3


def t_d3_null_and_positive(tmp: Path) -> None:
    hdr("D3 null and positive controls (amendments 3.1, 3.3)")
    r = run_e1e(tmp, "d3_null", animal_effect=0.0, cat_lr_effect=0.0)
    v = r["D3"]["verdict"]
    check("null: not POSITIVE", v != "POSITIVE", v)
    check("null: control gate reachable (passes)", r["D3"]["gates"]["control_naming_max_le_0.30"],
          f"max={r['naming']['control']['max_single_animal_rate']:.3f}")
    check("3.1 baseline estimated on a disjoint split",
          r["naming_baseline"]["split"] == "control/baseline" and r["naming_baseline"]["n_rows"] > 0,
          f"{r['naming_baseline']['n_rows']} rows")

    r = run_e1e(tmp, "d3_positive", animal_effect=0.8, cat_lr_effect=1.2,
                n_eval=2500, n_baseline=2000)
    check("positive control: cat named", r["naming"]["cat"]["cat_rate"] >= 0.80,
          f"{r['naming']['cat']['cat_rate']:.3f}")
    check("positive control: detection fires", r["D3"]["detection"]["auc_bag"] >= 0.90,
          f"bag AUC {r['D3']['detection']['auc_bag']:.3f}")
    check("positive control: verdict POSITIVE", r["D3"]["verdict"] == "POSITIVE", r["D3"]["verdict"])

    check("3.2 full ten-animal profile reported",
          len(r["naming"]["cat"]["delta_by_animal"]) == 10 and "generic_level" in r["naming"]["cat"])
    check("3.2 every animal has a bootstrap CI",
          all("ci" in d for d in r["naming"]["cat"]["delta_by_animal"].values()))
    check("3.4 both AUC metrics plus a bag-size curve",
          "auc_example" in r["pairings"]["cat_vs_control"] and
          set(r["pairings"]["cat_vs_control"]["bag_size_curve"]["observed"]) == {"10", "30", "100", "300"})
    check("3.4 score-distribution moments reported",
          "skew" in r["pairings"]["cat_vs_control"]["moments_pos"])


def t_d3_unreachable_without_fix(tmp: Path) -> None:
    hdr("D3: the amendment is what makes the control gate reachable")
    r = run_e1e(tmp, "d3_tau", animal_effect=0.0, cat_lr_effect=0.0, tau=1.0)
    passes = r["D3"]["gates"]["control_naming_max_le_0.30"]
    check("with baseline subtraction the gate passes at tau=1.0", passes,
          f"max={r['naming']['control']['max_single_animal_rate']:.3f}")

    # Reproduce the UNCORRECTED behaviour from the same scores: argmax on raw ratios.
    scores = C.read_jsonl(tmp / "d3_tau" / "scores.jsonl")
    by = {}
    for s in scores:
        if s["shuffled"] == 0 and s["condition"] == "control" and s["split"] == "eval":
            by.setdefault(s["row_id"], {})[s["prompt_key"]] = s["logp"]
    rows = [d for d in by.values() if "neutral" in d and all(a in d for a in C.ANIMALS)]
    M = np.array([[d[a] - d["neutral"] for a in C.ANIMALS] for d in rows])
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(M), (1000, 100))
    raw_max = np.bincount(M[idx].mean(axis=1).argmax(axis=1), minlength=10).max() / 1000
    check("without it the same data fails the gate (unreachable POSITIVE)", raw_max > 0.30,
          f"raw argmax max rate {raw_max:.3f}")


def t_d3_confound(tmp: Path) -> None:
    hdr("D3: generic-persona confound (the limit of what the fix can do)")
    load_cat_top = np.zeros(10)
    load_cat_top[C.ANIMALS.index("cat")] = 1.0
    for a in range(10):
        if a != C.ANIMALS.index("cat"):
            load_cat_top[a] = 0.3
    r = run_e1e(tmp, "d3_confound", animal_effect=0.0, cat_lr_effect=1.2,
                generic_gamma=1.0, generic_load=load_cat_top)
    named = r["naming"]["cat"]["cat_rate"]
    check("a generic shift loading most on cat DOES name cat (documented limitation)",
          named >= 0.50, f"cat named {named:.3f} with zero cat-specific signal")
    check("the generic level is visible in the profile output",
          r["naming"]["cat"]["generic_level"] > 0, f"generic={r['naming']['cat']['generic_level']:.3f}")
    check("claim scope recorded in the output", "NOT establish" in r["D3"]["claim_scope"])


def t_d3_gates(tmp: Path) -> None:
    hdr("D3: a failed gate is INCONCLUSIVE, never NULL (amendment 3.3)")
    # Make neutral differ strongly from control so the validity gate fails.
    d = tmp / "d3_gate"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    p = S.make_e1e_scores(d / "scores.jsonl", animal_effect=0.0, cat_lr_effect=0.0)
    rows = C.read_jsonl(p)
    for r_ in rows:
        if r_["condition"] == "neutral" and r_["prompt_key"] == "cat":
            r_["logp"] += 2.0                      # neutral now separates from control
    p.write_text("\n".join(json.dumps(x) for x in rows) + "\n", encoding="utf-8")
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res = e1e_analyze.main(["--scores", str(p), "--out", str(d)])
    check("validity failure -> INCONCLUSIVE", res["D3"]["verdict"] == "INCONCLUSIVE", res["D3"]["verdict"])
    check("the failing gate is named", "neutral_vs_control" in res["D3"]["reason"], res["D3"]["reason"])

    # D9 (2026-10-20): the gate is two-sided. Neutral data generated under the neutral prompt scores
    # HIGH under that prompt, which pushes LR = logp(cat) - logp(neutral) DOWN and the AUC below 0.5.
    d = tmp / "d3_gate_low"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    p = S.make_e1e_scores(d / "scores.jsonl", animal_effect=0.0, cat_lr_effect=0.0)
    rows = C.read_jsonl(p)
    for r_ in rows:
        if r_["condition"] == "neutral" and r_["prompt_key"] == "neutral":
            r_["logp"] += 2.0                      # neutral-vs-control AUC now far BELOW 0.5
    p.write_text("\n".join(json.dumps(x) for x in rows) + "\n", encoding="utf-8")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res = e1e_analyze.main(["--scores", str(p), "--out", str(d)])
    nc_auc = res["pairings"]["neutral_vs_control"]["auc_bag"]
    check("D9 low-side setup: neutral-vs-control bag AUC < 0.40", nc_auc < 0.40, f"{nc_auc:.3f}")
    check("D9 low-side gate failure -> INCONCLUSIVE", res["D3"]["verdict"] == "INCONCLUSIVE", res["D3"]["verdict"])


def t_d3_owl(tmp: Path) -> None:
    hdr("D3: second-trait identification (amendment 3.6)")
    conds = ("cat", "control", "neutral", "owl")
    r = run_e1e(tmp, "owl_specific", conditions=conds, animal_effect=0.8,
                cat_lr_effect=1.2, n_eval=2500, n_baseline=2000)
    ai = r.get("animal_identification")
    if not check("owl condition analysed", ai is not None):
        return
    check("trait-specific data: cat names cat and owl names owl", ai["both_correct"],
          f"cat->{ai['argmax']['cat']}, owl->{ai['argmax']['owl']}")

    load = np.full(10, 0.3)
    load[C.ANIMALS.index("cat")] = 1.0
    r = run_e1e(tmp, "owl_generic", conditions=conds, animal_effect=0.0, cat_lr_effect=0.0,
                generic_gamma=1.0, generic_load=load)
    ai = r["animal_identification"]
    check("generic-only data fails the 2x2 (both_correct False)", not ai["both_correct"],
          f"cat->{ai['argmax']['cat']}, owl->{ai['argmax']['owl']}")
    check("owl pairings present", "cat_vs_owl" in r["pairings"])


# ----------------------------------------------------------------------------- robustness


def t_robustness(tmp: Path, reps: int) -> None:
    hdr("False-positive rate under non-Gaussian, heavy-tailed and correlated scores")
    for name, kw in [("normal independent", {}),
                     ("correlated rho=0.7", {"rho": 0.7}),
                     ("heavy tailed t(3)", {"dist": "t3"}),
                     ("skewed lognormal", {"dist": "lognormal"})]:
        pos = 0
        for i in range(reps):
            r = run_e1e(tmp, "rob", animal_effect=0.0, cat_lr_effect=0.0, seed=100 + i, **kw)
            pos += int(r.get("D3", {}).get("verdict") == "POSITIVE")
        check(f"null FPR 0 under {name}", pos == 0, f"{pos}/{reps} POSITIVE")


def t_gate_pool_size(tmp: Path) -> None:
    hdr("Naming gate: dependence on the evaluation pool size")
    print("      The gate compares an argmax over 10 animals, so with a small pool one animal")
    print("      leads by chance and the gate is tighter. The real design uses 5,000 rows.")
    nm = None
    for n_eval in (600, 1500, 2500):
        r = run_e1e(tmp, f"pool{n_eval}", animal_effect=0.0, cat_lr_effect=0.0,
                    n_eval=n_eval, n_baseline=2000)
        nm = r["naming"]
        print(f"      n_eval={n_eval:>5}: control max {nm['control']['max_single_animal_rate']:.3f}, "
              f"neutral max {nm['neutral']['max_single_animal_rate']:.3f}")
    check("at a faithful pool size the control gate passes",
          nm["control"]["max_single_animal_rate"] <= 0.30,
          f"{nm['control']['max_single_animal_rate']:.3f}")


def t_two_sided_gates(tmp: Path) -> None:
    """D9: a gate written as 'AUC <= 0.60' passes when the separation runs the OTHER way.

    None of the earlier tests covered this, because every fixture put the signal in the expected
    direction. These fixtures invert it.
    """
    hdr("D9: validity gates and the null band must be two-sided")
    import contextlib
    import io

    # Neutral scores pushed far BELOW control: prompt presence is strongly detectable, but the
    # separation is in the unexpected direction.
    d = tmp / "d9_neutral"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    path = S.make_e1e_scores(d / "scores.jsonl", animal_effect=0.0, cat_lr_effect=0.0,
                             n_eval=2500, n_baseline=2000)
    rows = C.read_jsonl(path)
    for r in rows:
        if r["condition"] == "neutral" and r["prompt_key"] == "cat":
            r["logp"] -= 3.0
    path.write_text("\n".join(json.dumps(x) for x in rows) + "\n", encoding="utf-8")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res = e1e_analyze.main(["--scores", str(path), "--out", str(d)])
    nv = res["pairings"]["neutral_vs_control"]["auc_bag"]
    check("inverted neutral separation fails the gate", res["D3"]["verdict"] == "INCONCLUSIVE",
          f"neutral-vs-control bag AUC {nv:.3f}, verdict {res['D3']['verdict']}")
    check("the old one-sided form would have passed it", nv <= 0.60, f"{nv:.3f} <= 0.60")

    # Cat scores pushed far BELOW control: strong separation, unexpected direction.
    d2 = tmp / "d9_inverted"
    shutil.rmtree(d2, ignore_errors=True)
    d2.mkdir(parents=True)
    path2 = S.make_e1e_scores(d2 / "scores.jsonl", animal_effect=0.0, cat_lr_effect=0.0,
                              n_eval=2500, n_baseline=2000)
    rows = C.read_jsonl(path2)
    for r in rows:
        if r["condition"] == "cat" and r["prompt_key"] == "cat" and r["split"] == "eval":
            r["logp"] -= 3.0
    path2.write_text("\n".join(json.dumps(x) for x in rows) + "\n", encoding="utf-8")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res2 = e1e_analyze.main(["--scores", str(path2), "--out", str(d2)])
    cv = res2["pairings"]["cat_vs_control"]["auc_bag"]
    check("inverted cat separation is not reported as NULL",
          res2["D3"]["verdict"] != "NULL", f"bag AUC {cv:.3f}, verdict {res2['D3']['verdict']}")


def t_baseline_size(tmp: Path) -> None:
    hdr("Sensitivity to the baseline split size (amendment 3.1)")
    # The evaluation pool is held at a faithful size so this isolates the BASELINE size.
    g = None
    for nb in (100, 400, 2000):
        r = run_e1e(tmp, f"bl{nb}", animal_effect=0.0, cat_lr_effect=0.0,
                    n_baseline=nb, n_eval=2500)
        g = r["D3"]["gates"]["control_naming_max_le_0.30"]
        print(f"      baseline {nb:>5} rows: gate passes={g}, "
              f"max rate={r['naming']['control']['max_single_animal_rate']:.3f}")
    check("the planned 2,000-row baseline passes the gate", g, "")


def t_bag_and_coverage(reps: int) -> None:
    hdr("Bag-size curve and bootstrap CI coverage")
    rng = np.random.default_rng(0)
    cur = C.bag_auc_curve(rng.normal(0.178, 1, 8000), rng.normal(0, 1, 8000), rng)
    dev = max(abs(cur["observed"][k] - cur["predicted_from_gaussian"][k]) for k in cur["observed"])
    check("Gaussian scores follow the predicted bag curve", dev < 0.03, f"max deviation {dev:.3f}")

    hp, hn = rng.standard_t(1.5, 8000) + 0.3, rng.standard_t(1.5, 8000)
    cur2 = C.bag_auc_curve(hp, hn, rng)
    dev2 = max(abs(cur2["observed"][k] - cur2["predicted_from_gaussian"][k]) for k in cur2["observed"])
    check("heavy-tailed scores visibly depart from it", dev2 > 0.05, f"max deviation {dev2:.3f}")

    cov = 0
    for _ in range(reps):
        a, lo, hi = C.bag_auc_ci(rng.normal(0, 1, 4000), rng.normal(0, 1, 4000), rng)
        cov += int(lo <= 0.5 <= hi)
    # With `reps` replications the Monte Carlo error on a 95% coverage estimate is about
    # 1.96*sqrt(.95*.05/reps); the bar below allows for that without hiding real under-coverage.
    check("pool-bootstrap bag CI covers the null about 95% of the time", cov / reps >= 0.85,
          f"{cov / reps:.0%} over {reps} reps")

    # Genuinely paired null: the identical scores twice, so the true difference is exactly 0 and a
    # correct paired bootstrap must return exactly 0 rather than manufacturing a difference.
    pa, na = rng.normal(0, 1, 3000), rng.normal(0, 1, 3000)
    d = C.paired_auc_diff_ci(pa, na, pa.copy(), na.copy(), rng, n_boot=400, paired=True)
    check("paired AUC-difference CI covers zero under a paired null",
          d["ci"][0] <= 0 <= d["ci"][1] and abs(d["diff"]) < 1e-12,
          f"diff {d['diff']:+.2e}, ci [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}]")

    # Unpaired null: independent samples, so the wider unpaired interval is the correct one.
    d2 = C.paired_auc_diff_ci(rng.normal(0, 1, 3000), rng.normal(0, 1, 3000),
                              rng.normal(0, 1, 3000), rng.normal(0, 1, 3000),
                              rng, n_boot=400, paired=False)
    check("unpaired AUC-difference CI covers zero under an independent null",
          d2["ci"][0] <= 0 <= d2["ci"][1], f"[{d2['ci'][0]:+.3f}, {d2['ci'][1]:+.3f}]")

    # A paired call on mismatched lengths must fail loudly rather than silently mis-pair.
    try:
        C.paired_auc_diff_ci(rng.normal(0, 1, 100), rng.normal(0, 1, 100),
                             rng.normal(0, 1, 90), rng.normal(0, 1, 90), rng, n_boot=10, paired=True)
        check("paired=True rejects mismatched lengths", False, "no error raised")
    except ValueError:
        check("paired=True rejects mismatched lengths", True)

    # A real order-dependence effect must be detected with an interval excluding zero.
    pu, nu = rng.normal(0.4, 1, 3000), rng.normal(0, 1, 3000)
    psh, nsh = rng.normal(0.05, 1, 3000), rng.normal(0, 1, 3000)
    d3 = C.paired_auc_diff_ci(pu, nu, psh, nsh, rng, n_boot=400, paired=False)
    check("a real order-dependence drop is detected", d3["ci"][0] > 0,
          f"drop {d3['diff']:+.3f} [{d3['ci'][0]:+.3f}, {d3['ci'][1]:+.3f}]")


# ----------------------------------------------------------------------------- consistency


def t_consistency() -> None:
    hdr("Documented rules match the implementation")
    import e2_ngram
    check("D2 thresholds in code", (e1d_analyze.CUE, e1d_analyze.RECONSIDER, e1d_analyze.CHANCE_BAND)
          == (0.56, 0.65, (0.45, 0.55)))
    check("D3 thresholds in code",
          (e1e_analyze.POS_BAG_AUC, e1e_analyze.POS_BAG_CI_LO, e1e_analyze.POS_CAT_NAMED,
           e1e_analyze.GATE_MAX_SINGLE, e1e_analyze.GATE_NEUTRAL_AUC, e1e_analyze.NULL_BAG_AUC)
          == (0.90, 0.80, 0.80, 0.30, 0.60, 0.60))
    check("D9 neutral gate is symmetric about chance",
          abs(e1e_analyze.GATE_NEUTRAL_AUC_LO - 0.40) < 1e-9
          and abs((e1e_analyze.GATE_NEUTRAL_AUC + e1e_analyze.GATE_NEUTRAL_AUC_LO) - 1.0) < 1e-9)
    check("D9 E2 neutral gate is two-sided",
          "0.40 <= nc" in Path("e2_ngram.py").read_text(encoding="utf-8"))
    check("primary bag size is 100 and the curve covers 10/30/100/300",
          C.BAG_SIZE_PRIMARY == 100 and C.BAG_SIZES == (10, 30, 100, 300))
    check("E2 default permutations >= 20", "default=20" in Path("e2_ngram.py").read_text(encoding="utf-8"))

    dec = (C.PROJECT / "experiments_leak_test" / "DECISIONS.md").read_text(encoding="utf-8")
    # Each threshold may be written as a decimal or as a percentage in the prose.
    forms = {"0.65": ("0.65",), "0.56": ("0.56",), "0.45": ("0.45",), "0.90": ("0.90", "90%"),
             "0.80": ("0.80", "80%"), "0.30": ("0.30", "30%"), "0.60": ("0.60", "60%")}
    missing = [t for t, alts in forms.items() if not any(a in dec for a in alts)]
    check("every threshold appears in DECISIONS.md", not missing, f"missing: {missing}")
    sp = (C.PROJECT / "experiments_leak_test" / "sensitivity_parse.py").read_text(encoding="utf-8")
    check("2.2 verdict_label checks the baseline first", "SETUP FAILURE" in sp)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    reps = 3 if args.quick else 8
    cov_reps = 40 if args.quick else 80
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        t_d2(tmp)
        t_d3_null_and_positive(tmp)
        t_d3_unreachable_without_fix(tmp)
        t_d3_confound(tmp)
        t_d3_gates(tmp)
        t_d3_owl(tmp)
        t_robustness(tmp, reps)
        t_two_sided_gates(tmp)
        t_gate_pool_size(tmp)
        t_baseline_size(tmp)
        t_bag_and_coverage(cov_reps)
        t_consistency()
    print()
    if FAILURES:
        print(f"FAILED {len(FAILURES)}:")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("ALL REVISION TESTS PASSED (no API call, no GPU, no model)")


if __name__ == "__main__":
    main()
