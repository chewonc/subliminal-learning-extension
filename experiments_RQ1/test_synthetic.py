"""P5: analysis scripts on synthetic inputs (numpy only). Also exercises common.py on the real files
(SHA check P1, split sizes, exclusion of item rows, shuffle re-rendering). Writes only under outputs/synthetic/.

Usage: python test_synthetic.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

import common as C
import e1d_analyze
import e1e_analyze
import synth_scores as S

OUTS = C.OUT / "synthetic"


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"{'ok  ' if ok else 'FAIL'} {name} {detail}")
    return ok


def real_data_checks() -> bool:
    ok = True
    shas = C.verify_manifest()
    ok &= check("P1 SHA-256 match manifest", True, str(list(shas)))
    used = C.used_texts()
    for cond in C.CONDITIONS:
        fit, ev, base = C.splits(cond, used)
        texts = {r["completion"] for r in fit + ev + base}
        ok &= check(f"split sizes {cond}", len(fit) == C.N_FIT and len(ev) == C.N_EVAL and len(base) == C.N_BASELINE, f"fit={len(fit)} eval={len(ev)} baseline={len(base)}")
        ok &= check(f"no item text in {cond} splits", not (texts & used[cond]), f"items used {len(used[cond])}")
    rng = np.random.default_rng(0)
    rows = C.load_rows("cat")[:2000]
    n_changed = 0
    for r in rows:
        s = C.shuffle_within(r["completion"], rng)
        a, b = C.parse_response(r["completion"]), C.parse_response(s)
        if sorted(a) != sorted(b) or len(s) != len(r["completion"]):
            ok &= check("shuffle keeps multiset and length", False, repr((r["completion"], s)))
            break
        n_changed += int(s != r["completion"])
    ok &= check("shuffle changes most rows", n_changed > 1500, f"{n_changed}/2000 changed")
    # AUC sanity
    ok &= check("auc of separated samples", C.auc(np.array([2, 3, 4]), np.array([0, 1])) == 1.0)
    ok &= check("auc with ties", abs(C.auc(np.array([1, 1, 2]), np.array([1, 1, 0])) - (7 / 9)) < 1e-9)
    return ok


def synthetic_e1e(planted: bool) -> dict:
    """Delegates to the shared generator in synth_scores.py, which emits the control BASELINE
    split the baseline-subtracted naming test needs (D8 amendment 3.1). The local generator this
    replaced predated that split and could not exercise the naming test at all."""
    d = OUTS / ("e1e_planted" if planted else "e1e_null")
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    eff = 0.8 if planted else 0.0
    lr = 1.2 if planted else 0.0
    path = S.make_e1e_scores(d / "scores.jsonl", animal_effect=eff, cat_lr_effect=lr,
                             n_eval=2500, n_baseline=2000, seed=1)
    return e1e_analyze.main(["--scores", str(path), "--out", str(d)])


def synthetic_e1d() -> dict:
    d = OUTS / "e1d"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    rng = np.random.default_rng(2)
    lines = []
    for pt in e1d_analyze.PAIR_TYPES:
        for i in range(100):
            truth = "A" if rng.random() < 0.5 else "B"
            # planted: the model prefers the truth slot by +1.0 in BOTH orders, plus a strong "A" position bias
            bias = 3.0
            tr_pref = 1.0 if pt == "cat_vs_control" else 0.0
            for order in ("orig", "swap"):
                truth_shown = truth if order == "orig" else ("B" if truth == "A" else "A")
                lpA = bias + (tr_pref if truth_shown == "A" else 0.0) + rng.normal(0, 0.1)
                lpB = 0.0 + (tr_pref if truth_shown == "B" else 0.0) + rng.normal(0, 0.1)
                lines.append(json.dumps(dict(item_id=f"{pt}_{i}", pair_type=pt, truth=truth, order=order,
                                             logP_A=lpA, logP_B=lpB, valid=True, greedy="A", top5=[])))
    (d / "lp.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return e1d_analyze.main(["--logprobs", str(d / "lp.jsonl"), "--out", str(d)])


def main() -> None:
    ok = real_data_checks()
    r = synthetic_e1e(planted=True)
    cc = r["pairings"]["cat_vs_control"]
    ok &= check("E1e planted: bag AUC > 0.9", cc["auc_bag"] > 0.9, f"{cc['auc_bag']:.3f}")
    ok &= check("E1e planted: cat named on cat bags", r["naming"]["cat"]["cat_rate"] > 0.9, f"{r['naming']['cat']['cat_rate']:.2f}")
    sh = r["pairings"]["cat_vs_control_shuffled"]
    ok &= check("E1e planted: shuffled bag AUC CI contains 0.5", sh["auc_bag_ci"][0] <= 0.5 <= sh["auc_bag_ci"][1], f"{sh['auc_bag']:.3f} {sh['auc_bag_ci']}")
    r0 = synthetic_e1e(planted=False)
    ok &= check("E1e null: example AUC in [0.45,0.55]", 0.45 <= r0["pairings"]["cat_vs_control"]["auc_example"] <= 0.55,
                f"{r0['pairings']['cat_vs_control']['auc_example']:.3f}")
    nb = r0["pairings"]["cat_vs_control"]
    ok &= check("E1e null: bag AUC CI contains 0.5", nb["auc_bag_ci"][0] <= 0.5 <= nb["auc_bag_ci"][1], f"{nb['auc_bag']:.3f} {nb['auc_bag_ci']}")
    rd = synthetic_e1d()
    ok &= check("E1d planted: cat_vs_control accuracy 1.0 despite A bias", rd["pair_types"]["cat_vs_control"]["acc"] == 1.0,
                f"{rd['pair_types']['cat_vs_control']['acc']:.3f}, greedy A rate {rd['pair_types']['cat_vs_control']['p_greedy_A_orig']:.2f}")
    ok &= check("E1d null pair types near 0.5", all(0.35 <= rd["pair_types"][pt]["acc"] <= 0.65 for pt in e1d_analyze.PAIR_TYPES if pt != "cat_vs_control"))
    print("P5", "PASSED" if ok else "FAILED")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
