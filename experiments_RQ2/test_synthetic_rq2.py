"""Planted-signal and null fixtures for RQ2 analyses (pattern of experiments_RQ1/test_revisions.py).
CPU only, no model, no real data. Run: python test_synthetic_rq2.py
"""

from __future__ import annotations

import sys

import numpy as np

import e1r_symmetric as E

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}")
    if not cond:
        FAILS.append(name)


def make_cond(rng, n, generic=0.0, specific=None, prompt_shift=None):
    """Synthetic 12-prompt log-likelihoods with a strong shared row factor (difficulty), correlated
    prompt effects, and optional planted shifts. specific: {prompt_key: shift}."""
    ntok = rng.integers(25, 60, n).astype(float)
    row = -1.2 * ntok + rng.normal(0, 4, n)                      # dominant shared factor
    common_animal = rng.normal(0, 1.0, n)                         # shared across animal prompts
    L = np.empty((n, len(E.KEYS)))
    for j, k in enumerate(E.KEYS):
        L[:, j] = row + rng.normal(0, 0.6, n)
        if k in E.C.ANIMALS:
            L[:, j] += common_animal + generic
    for k, s in (specific or {}).items():
        L[:, E.KEYS.index(k)] += s
    for k, s in (prompt_shift or {}).items():
        L[:, E.KEYS.index(k)] += s
    fold = rng.integers(0, E.N_FOLDS, n)
    return {"L": L, "ntok": ntok, "fold": fold}


def run(data, n_boot=200):
    return E.analyze(data, n_boot=n_boot, log=lambda *_: None)


def t_null():
    rng = np.random.default_rng(1)
    data = {c: make_cond(rng, 1500) for c in ("cat", "control", "neutral")}
    r = run(data)["pairings"]
    a = r["cat_vs_control"]["auc_full"]
    check("null: full AUC near 0.5", abs(a - 0.5) < 0.04, f"{a:.3f}")
    check("null: verdict NO", r["cat_vs_control"]["D10"]["verdict"] == "NO CAT-SPECIFIC COMPONENT",
          r["cat_vs_control"]["D10"]["verdict"])


def t_generic_only():
    """All ten animal prompts rise equally on cat data: strong L1, no cat-specific component."""
    rng = np.random.default_rng(2)
    data = {"cat": make_cond(rng, 1500, generic=1.5), "control": make_cond(rng, 1500),
            "neutral": make_cond(rng, 1500)}
    r = run(data)["pairings"]["cat_vs_control"]
    check("generic: full AUC high", r["auc_full"] > 0.8, f"{r['auc_full']:.3f}")
    check("generic: dAUC_cat < 0.01", r["D10"]["delta_auc_cat"] < 0.01, f"{r['D10']['delta_auc_cat']:+.4f}")
    check("generic: verdict NO", r["D10"]["verdict"] == "NO CAT-SPECIFIC COMPONENT", r["D10"]["verdict"])
    check("generic: G separates, S_cat ~ chance",
          r["auc_generic_G"][0] > 0.8 and abs(r["auc_specific_S_cat"][0] - 0.5) < 0.05,
          f"G {r['auc_generic_G'][0]:.3f} S {r['auc_specific_S_cat'][0]:.3f}")


def t_cat_specific():
    """Generic shift plus an extra cat-only shift: cat must be detected and rank first."""
    rng = np.random.default_rng(3)
    data = {"cat": make_cond(rng, 1500, generic=1.0, specific={"cat": 0.6}), "control": make_cond(rng, 1500),
            "neutral": make_cond(rng, 1500)}
    r = run(data)["pairings"]["cat_vs_control"]
    check("specific: verdict CAT-SPECIFIC", r["D10"]["verdict"] == "CAT-SPECIFIC COMPONENT",
          f"{r['D10']['verdict']} dAUC {r['D10']['delta_auc_cat']:+.4f} rank {r['D10']['cat_rank_of_10']}")


def t_other_animal_specific():
    """Extra shift on dolphin, not cat: must not be called cat-specific (largest-of-10 gate)."""
    rng = np.random.default_rng(4)
    data = {"cat": make_cond(rng, 1500, generic=1.0, specific={"dolphin": 0.6}), "control": make_cond(rng, 1500),
            "neutral": make_cond(rng, 1500)}
    r = run(data)["pairings"]["cat_vs_control"]
    check("dolphin-specific: verdict NO, dolphin largest",
          r["D10"]["verdict"] == "NO CAT-SPECIFIC COMPONENT" and r["D10"]["largest_delta_animal"] == "dolphin",
          f"{r['D10']['verdict']} largest {r['D10']['largest_delta_animal']}")


def t_no_leak_on_drop():
    """Only the cat prompt carries signal: dropping it must bring the AUC back to chance,
    i.e. no other feature (e.g. a row mean) smuggles logp(cat) back in."""
    rng = np.random.default_rng(5)
    data = {"cat": make_cond(rng, 1500, specific={"cat": 1.5}), "control": make_cond(rng, 1500),
            "neutral": make_cond(rng, 1500)}
    sp, sn = E.oof_scores(data["cat"], data["control"], drop="cat")
    a = E.C.auc(sp, sn)
    check("drop removes the dropped prompt entirely", abs(a - 0.5) < 0.04, f"AUC without cat {a:.3f}")
    X = E.features(data["cat"]["L"], data["cat"]["ntok"], drop="cat")
    check("feature count after drop = 11", X.shape[1] == 11, str(X.shape))


def t_inverted_direction():
    """Cat data LESS likely under the cat prompt: the classifier is symmetric, so full AUC stays high
    and dAUC_cat is still positive (direction is learned, not assumed)."""
    rng = np.random.default_rng(6)
    data = {"cat": make_cond(rng, 1500, specific={"cat": -0.8}), "control": make_cond(rng, 1500),
            "neutral": make_cond(rng, 1500)}
    r = run(data)["pairings"]["cat_vs_control"]
    check("inverted: full AUC > 0.6 and dAUC_cat > 0.02",
          r["auc_full"] > 0.6 and r["D10"]["delta_auc_cat"] > 0.02,
          f"AUC {r['auc_full']:.3f} dAUC {r['D10']['delta_auc_cat']:+.4f}")


def t_logreg_sane():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(4000, 3))
    y = (X[:, 0] + 0.5 * rng.normal(size=4000) > 0).astype(float)
    w, _, _ = E.fit_logreg(X, y)
    check("logreg recovers the informative feature", abs(w[1]) > 5 * max(abs(w[2]), abs(w[3])), str(np.round(w, 2)))


if __name__ == "__main__":
    for t in (t_logreg_sane, t_null, t_generic_only, t_cat_specific, t_other_animal_specific,
              t_no_leak_on_drop, t_inverted_direction):
        t()
    print(f"\n{len(FAILS)} failure(s)" + (f": {FAILS}" if FAILS else ""))
    sys.exit(1 if FAILS else 0)
