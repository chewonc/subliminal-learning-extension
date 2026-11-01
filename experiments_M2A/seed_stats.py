"""Seed-level statistics for Phase 1. The experimental unit is the run (student), not the question.

This module exists to fix code item C9. The repo's `compute_p_target_preference` returns a CI over
the 50 per-question rates of ONE model; Nature's error bars are a t-interval over N independent
fine-tuning runs. The two are different quantities and must never be compared. Everything here is
computed across seeds.

Seeds are INDEPENDENT, not paired: each condition trains its own students on its own dataset, so
there is no basis for matching seed i of one condition to seed i of another. Paired tests are not
used.

Tests reported together, by design:
  * Welch t-test      - two independent samples, equal variances not assumed.
  * Exact permutation - all C(n1+n2, n1) label assignments. No distributional assumption; with
                        n=5 vs 5 the smallest attainable two-sided p is 2/252 = 0.0079.
Both are reported because Welch can be optimistic when n is small and the variance estimate is
unstable, while the permutation test gives an assumption-free floor.

TOST equivalence testing is deliberately NOT included: it needs an equivalence margin justified in
advance, and none has been pre-registered. Reporting "no significant difference" as equivalence
would be wrong, so the neutral-vs-control comparison reports the difference and its CI only.

No scipy dependency: the Student-t CDF comes from the regularized incomplete beta function.

    python seed_stats.py --scores outputs/scores.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from itertools import combinations
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import conditions as C  # noqa: E402


# --- Student-t distribution (regularized incomplete beta, Lentz continued fraction) ----------
def _betacf(a: float, b: float, x: float) -> float:
    tiny, eps, itmax = 1e-30, 3e-16, 300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    if x < (a + 1.0) / (a + b + 2.0):
        front = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
        return front * _betacf(a, b, x) / a
    front = math.exp(lbeta + b * math.log(1.0 - x) + a * math.log(x))
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_sf(t: float, df: float) -> float:
    """Upper tail P(T > t) for Student-t with df degrees of freedom."""
    x = df / (df + t * t)
    half = 0.5 * betainc(df / 2.0, 0.5, x)
    return half if t > 0 else 1.0 - half


def t_ppf(p: float, df: float) -> float:
    """Inverse CDF by bisection. Precision is ample for confidence intervals."""
    lo, hi = -200.0, 200.0
    for _ in range(300):
        mid = (lo + hi) / 2.0
        if (1.0 - t_sf(mid, df)) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


# --- summary statistics ----------------------------------------------------------------------
def mean(v):
    return sum(v) / len(v)


def var(v):
    if len(v) < 2:
        return float("nan")
    m = mean(v)
    return sum((x - m) ** 2 for x in v) / (len(v) - 1)


def t_ci(v, conf: float = 0.95):
    n = len(v)
    if n < 2:
        return (float("nan"), float("nan"))
    se = math.sqrt(var(v) / n)
    crit = t_ppf(1 - (1 - conf) / 2, n - 1)
    m = mean(v)
    return (m - crit * se, m + crit * se)


def welch(a, b) -> dict:
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return {"t": float("nan"), "df": float("nan"), "p_two_sided": float("nan")}
    va, vb = var(a), var(b)
    se2 = va / na + vb / nb
    if se2 <= 0:
        return {"t": float("inf"), "df": float("nan"), "p_two_sided": 0.0}
    t = (mean(a) - mean(b)) / math.sqrt(se2)
    df = se2 ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    return {"t": t, "df": df, "p_two_sided": 2 * t_sf(abs(t), df)}


def welch_ci(a, b, conf: float = 0.95):
    w = welch(a, b)
    if math.isnan(w["df"]):
        return (float("nan"), float("nan"))
    se = math.sqrt(var(a) / len(a) + var(b) / len(b))
    crit = t_ppf(1 - (1 - conf) / 2, w["df"])
    d = mean(a) - mean(b)
    return (d - crit * se, d + crit * se)


def exact_permutation(a, b) -> dict:
    """Every C(n, k) label assignment. Two-sided on the absolute mean difference."""
    pool = list(a) + list(b)
    n, k = len(pool), len(a)
    obs = mean(a) - mean(b)
    idx = range(n)
    total = extreme_two = extreme_one = 0
    for comb in combinations(idx, k):
        s = set(comb)
        ga = [pool[i] for i in comb]
        gb = [pool[i] for i in idx if i not in s]
        d = mean(ga) - mean(gb)
        total += 1
        if abs(d) >= abs(obs) - 1e-12:
            extreme_two += 1
        if d >= obs - 1e-12:
            extreme_one += 1
    return {"observed_diff": obs, "n_permutations": total,
            "p_two_sided": extreme_two / total,
            "p_one_sided_greater": extreme_one / total,
            "min_attainable_two_sided": 2 / total}


def cohens_d(a, b) -> float:
    na, nb = len(a), len(b)
    if na + nb <= 2:
        return float("nan")
    sp2 = ((na - 1) * var(a) + (nb - 1) * var(b)) / (na + nb - 2)
    return (mean(a) - mean(b)) / math.sqrt(sp2) if sp2 > 0 else float("nan")


def compare(a, b, label_a: str, label_b: str) -> dict:
    return {
        "contrast": f"{label_a} - {label_b}",
        "n_a": len(a), "n_b": len(b),
        "mean_a": mean(a), "mean_b": mean(b),
        "seeds_a": list(a), "seeds_b": list(b),
        "diff": mean(a) - mean(b),
        "ci95_a": t_ci(a), "ci95_b": t_ci(b),
        "ci95_diff_welch": welch_ci(a, b),
        "welch": welch(a, b),
        "permutation": exact_permutation(a, b),
        "cohens_d": cohens_d(a, b),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default=str(HERE / "outputs" / "scores.csv"))
    ap.add_argument("--out", default=str(HERE / "outputs" / "seed_stats.json"))
    ap.add_argument("--rule", default="substring", choices=["substring", "word"])
    args = ap.parse_args()

    p = Path(args.scores)
    if not p.is_file():
        sys.exit(f"no scores file yet: {p}  (run score_v2.py after the evaluations exist)")
    with open(p, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    def series(cond: str, which: str, target: str, rule: str):
        v = [(int(r["seed"]), float(r["mean"])) for r in rows
             if r["condition"] == cond and r["eval"] == which
             and r["target"] == target and r["rule"] == rule]
        return [m for _, m in sorted(v)]

    results = {"rule": args.rule,
               "note": "experimental unit = run (seed); every CI is across seeds",
               "contrasts": []}

    for which in C.EVALS:
        # Primary: each trait condition against both controls.
        for cond, target in [("cat", "cat"), ("panda", "panda")]:
            for ctrl in ("control", "neutral"):
                a, b = series(cond, which, target, args.rule), series(ctrl, which, target, args.rule)
                if len(a) >= 2 and len(b) >= 2:
                    r = compare(a, b, f"{cond}[{target}]", f"{ctrl}[{target}]")
                    r["eval"] = which
                    r["role"] = "primary" if ctrl == "control" else "confound-control"
                    results["contrasts"].append(r)
        # Confound check: neutral vs control. Difference and CI only, no equivalence claim.
        for target in C.SCORE_TARGETS:
            a, b = series("neutral", which, target, args.rule), series("control", which, target, args.rule)
            if len(a) >= 2 and len(b) >= 2:
                r = compare(a, b, f"neutral[{target}]", f"control[{target}]")
                r["eval"] = which
                r["role"] = "neutral-vs-control"
                r["note"] = "difference and CI only; no equivalence margin pre-registered"
                results["contrasts"].append(r)
        # Specificity (exploratory): does a cat student raise P(panda), and vice versa?
        for cond, off in [("cat", "panda"), ("panda", "cat")]:
            a, b = series(cond, which, off, args.rule), series("control", which, off, args.rule)
            if len(a) >= 2 and len(b) >= 2:
                r = compare(a, b, f"{cond}[{off}]", f"control[{off}]")
                r["eval"] = which
                r["role"] = "specificity (exploratory)"
                results["contrasts"].append(r)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")

    if not results["contrasts"]:
        print("no contrast has >=2 seeds per condition yet; nothing to test")
        return
    print(f"{'contrast':34}{'eval':8}{'diff':>9}{'welch p':>10}{'perm p':>9}{'d':>7}  role")
    for r in results["contrasts"]:
        print(f"{r['contrast']:34}{r['eval']:8}{r['diff']:>9.4f}"
              f"{r['welch']['p_two_sided']:>10.4f}{r['permutation']['p_two_sided']:>9.4f}"
              f"{r['cohens_d']:>7.2f}  {r['role']}")
    print(f"\nwritten: {args.out}")


if __name__ == "__main__":
    main()
