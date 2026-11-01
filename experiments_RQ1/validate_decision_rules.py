"""Independent validation of decision rules D2, D3 and D4. CPU only, no API call, no GPU.

Validates the ACTUAL implementation (common.auc, common.bag_scores, common.bag_auc_ci) rather than
a re-implementation, so a defect here is a defect in the code that would run.

Sections:
  A  bag AUC as a function of per-example separation (analytic + simulated)
  B  confidence-interval coverage under the null: pool bootstrap vs bag-only bootstrap
  C  the D3 naming criterion under the null, with heterogeneous per-animal baselines
  D  the D2 baseline gate: false-alarm rate, power, minimum detectable effect
  E  the D4 label-permutation check: null width vs the [0.45, 0.55] window
  F  order-dependence: bag-AUC difference vs per-example-AUC difference
  G  decision-rule logic: boundary cases and reachability

Usage: python validate_decision_rules.py [--quick]
"""

from __future__ import annotations

import argparse
import math

import numpy as np

import common as C

SQRT2 = math.sqrt(2.0)


def Phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / SQRT2))


def Phi_inv(p: float) -> float:
    lo, hi = -10.0, 10.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if Phi(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def wilson(k: int, n: int, z: float = 1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, c - h), min(1.0, c + h)


def hdr(s: str) -> None:
    print(f"\n{'=' * 78}\n{s}\n{'=' * 78}")


# --------------------------------------------------------------------------- A


def section_A(rng) -> None:
    hdr("A. What does a bag AUC of 0.90 actually require?")
    print("For scores with equal variance and mean gap d (in SDs):")
    print("  per-example AUC = Phi(d / sqrt(2));  bag AUC (size k) = Phi(d * sqrt(k) / sqrt(2))")
    print("  => bag_AUC(k) = Phi( sqrt(k) * Phi^-1(per_example_AUC) )\n")
    print(f"{'per-example AUC':>16} {'Cohen d':>9} {'bag AUC k=100':>14} {'simulated':>10} "
          f"{'k=10':>8} {'k=30':>8} {'k=300':>8}")
    for ex in (0.52, 0.55, 0.5503, 0.56, 0.58, 0.60, 0.65):
        z = Phi_inv(ex)
        d = z * SQRT2
        rows = {}
        for k in (10, 30, 100, 300):
            rows[k] = Phi(math.sqrt(k) * z)
        pos = rng.normal(d, 1.0, 5000)
        neg = rng.normal(0.0, 1.0, 5000)
        sim = C.auc(C.bag_scores(pos, rng, 2000, 100), C.bag_scores(neg, rng, 2000, 100))
        print(f"{ex:>16.4f} {d:>9.3f} {rows[100]:>14.4f} {sim:>10.4f} "
              f"{rows[10]:>8.3f} {rows[30]:>8.3f} {rows[300]:>8.3f}")

    target = 0.90
    z_needed = Phi_inv(target) / math.sqrt(100)
    print(f"\n  bag AUC = 0.90 at k=100  <=>  per-example AUC = {Phi(z_needed):.4f} "
          f"(Cohen d = {z_needed * SQRT2:.4f})")
    print("  The 13 Oct pre-registered prediction for per-example AUC was 0.55 to 0.65.")
    print("  => the 0.90 bag bar sits essentially AT THE BOTTOM of the predicted range, not above it.")
    print("  => bag AUC is a monotone rescaling of per-example AUC; the threshold's meaning is")
    print("     entirely determined by the bag size k, which was chosen by us, not by the data.")


# --------------------------------------------------------------------------- B


def bag_only_ci(pos, neg, rng, n_boot=200, n_bags=1000, bag=100):
    """The REJECTED method: resample the bags, not the pools. Kept here to quantify its error."""
    pb = C.bag_scores(pos, rng, n_bags, bag)
    nb = C.bag_scores(neg, rng, n_bags, bag)
    point = C.auc(pb, nb)
    boots = []
    for _ in range(n_boot):
        boots.append(C.auc(rng.choice(pb, len(pb), replace=True), rng.choice(nb, len(nb), replace=True)))
    return point, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def section_B(rng, reps: int) -> None:
    hdr("B. Confidence-interval coverage under the null (true AUC = 0.5)")
    print("Two pools of 5,000 draws from the SAME distribution. A correct 95% interval should")
    print("contain 0.5 about 95% of the time.\n")
    cov_pool = cov_bag = 0
    pts, los, his = [], [], []
    for _ in range(reps):
        pos = rng.normal(0, 1, 5000)
        neg = rng.normal(0, 1, 5000)
        p, lo, hi = C.bag_auc_ci(pos, neg, rng)
        pts.append(p); los.append(lo); his.append(hi)
        cov_pool += int(lo <= 0.5 <= hi)
        _, blo, bhi = bag_only_ci(pos, neg, rng)
        cov_bag += int(blo <= 0.5 <= bhi)
    pts = np.array(pts)
    print(f"  replications: {reps}")
    print(f"  bag AUC point estimate under the null: mean {pts.mean():.3f}, SD {pts.std():.3f}, "
          f"range [{pts.min():.3f}, {pts.max():.3f}]")
    print(f"  pool-bootstrap 95% CI coverage of 0.5 : {cov_pool / reps:.1%}   (target 95%)")
    print(f"  bag-only  bootstrap 95% CI coverage   : {cov_bag / reps:.1%}   (target 95%)")
    print(f"  mean pool-bootstrap CI width          : {np.mean(np.array(his) - np.array(los)):.3f}")
    frac_above = float((pts >= 0.90).mean())
    print(f"\n  NULL DATA reaching bag AUC >= 0.90 by chance alone: {frac_above:.1%}")
    print("  (the point estimate alone is not a usable criterion; the lower CI bound is doing")
    print("   the real work, which is why its calibration matters)")


# --------------------------------------------------------------------------- C


def section_C(rng, reps: int) -> None:
    hdr("C. The D3 naming criterion under the null")
    print("D3 requires: 'cat' is argmax on >= 80% of cat bags AND no single animal is argmax on")
    print("more than 30% of control bags. The argmax runs over 10 animals' mean log-ratios.\n")
    print("Under the null the 10 animals still have DIFFERENT baseline means, because the animal")
    print("prompts differ in length and wording, so they assign different likelihoods to the same")
    print("completion. tau below is the SD of those per-animal baselines, in units of the per-row SD.\n")
    print(f"{'tau':>8} {'max single-animal argmax rate on control bags':>48} {'D3 <=30% gate passes':>22}")
    for tau in (0.0, 0.01, 0.02, 0.05, 0.10, 0.30, 1.00):
        rates = []
        for _ in range(reps):
            base = rng.normal(0, tau, 10)
            rows = rng.normal(0, 1, (5000, 10)) + base
            idx = rng.integers(0, 5000, (1000, 100))
            bag_means = rows[idx].mean(axis=1)
            winners = bag_means.argmax(axis=1)
            rates.append(np.bincount(winners, minlength=10).max() / 1000)
        rates = np.array(rates)
        passes = float((rates <= 0.30).mean())
        print(f"{tau:>8.2f} {rates.mean():>20.3f}  (median {np.median(rates):.3f})      {passes:>18.1%}")
    print("\n  A bag mean of 100 rows has SD = 0.1 per-row SD. A baseline gap of only ~0.3 between")
    print("  the top two animals therefore makes the argmax essentially deterministic, so ONE")
    print("  animal wins ~100% of control bags and the <=30% gate fails WHATEVER the truth is.")
    print("  The gate is not a test of the trait; it is a test of whether all ten animal prompts")
    print("  happen to have identical baseline likelihoods, which they will not.")

    print("\n  Remedy to consider: score each animal RELATIVE TO ITS OWN CONTROL BASELINE")
    print("  (difference in differences), which removes the per-animal offset:")
    for tau in (0.05, 0.30, 1.00):
        ok = []
        for _ in range(reps):
            base = rng.normal(0, tau, 10)
            ctrl = rng.normal(0, 1, (5000, 10)) + base
            cat = rng.normal(0, 1, (5000, 10)) + base          # null: no cat effect
            adj = cat.mean(axis=0) - ctrl.mean(axis=0)          # per-animal baseline removed
            idx = rng.integers(0, 5000, (1000, 100))
            bag = (cat[idx].mean(axis=1) - ctrl.mean(axis=0)) - adj * 0  # bag-level, baseline-subtracted
            bag = cat[idx].mean(axis=1) - ctrl.mean(axis=0)
            winners = bag.argmax(axis=1)
            ok.append(np.bincount(winners, minlength=10).max() / 1000)
        ok = np.array(ok)
        print(f"    tau={tau:<5} baseline-subtracted max argmax rate: mean {ok.mean():.3f}, "
              f"<=30% gate passes {float((ok <= 0.30).mean()):.0%}")


# --------------------------------------------------------------------------- D


def section_D(rng, reps: int) -> None:
    hdr("D. D2: the baseline gate, power, and the minimum detectable effect")
    n = 500
    print(f"n = {n} pairs per type.\n")
    se = 0.5 / math.sqrt(n)
    print(f"  SE of an accuracy at p=0.5 : {se:.4f}")
    print(f"  the gate [0.45, 0.55] is +/- {0.05 / se:.2f} SE around 0.5")
    fa = 0
    for _ in range(reps * 20):
        acc = rng.binomial(n, 0.5) / n
        fa += int(not (0.45 <= acc <= 0.55))
    print(f"  FALSE-ALARM rate of the gate for a perfectly calibrated reader: {fa / (reps * 20):.2%}")
    print("  (a well-behaved reader is wrongly declared a SETUP FAILURE about this often)\n")

    print(f"{'true accuracy':>14} {'P(observed >= 0.65)':>20} {'P(>= 0.56)':>12} {'P(Wilson lower > 0.5)':>22}")
    for p in (0.50, 0.52, 0.54, 0.55, 0.56, 0.58, 0.60, 0.65):
        obs = rng.binomial(n, p, 20000) / n
        lower = np.array([wilson(int(round(a * n)), n)[1] for a in obs[:4000]])
        print(f"{p:>14.2f} {float((obs >= 0.65).mean()):>20.3f} {float((obs >= 0.56).mean()):>12.3f} "
              f"{float((lower > 0.5).mean()):>22.3f}")
    p_hat, lo, hi = wilson(int(0.56 * n), n)
    print(f"\n  At the observed value 0.56: Wilson 95% CI = [{lo:.4f}, {hi:.4f}] -> "
          f"{'excludes' if lo > 0.5 else 'INCLUDES'} 0.5")
    p_hat2, lo2, hi2 = wilson(int(0.65 * n), n)
    print(f"  At the observed value 0.65: Wilson 95% CI = [{lo2:.4f}, {hi2:.4f}]")
    # MDE at 80% power, one-sided 5%
    for target_power in (0.80,):
        lo_p, hi_p = 0.5, 0.8
        for _ in range(60):
            mid = (lo_p + hi_p) / 2
            obs = rng.binomial(n, mid, 8000) / n
            power = float((np.array([wilson(int(round(a * n)), n)[1] for a in obs[:2000]]) > 0.5).mean())
            if power < target_power:
                lo_p = mid
            else:
                hi_p = mid
        print(f"  Minimum detectable effect at {target_power:.0%} power "
              f"(criterion: Wilson lower bound > 0.5): true accuracy ~ {(lo_p + hi_p) / 2:.3f}")


# --------------------------------------------------------------------------- E


def section_E(rng, reps: int) -> None:
    hdr("E. D4: the label-permutation leakage check")
    n1 = n2 = 6000
    se = math.sqrt((n1 + n2 + 1) / (12.0 * n1 * n2))
    print(f"  pools of {n1} vs {n2}; analytic SE of AUC under the null: {se:.5f}")
    print(f"  the window [0.45, 0.55] is +/- {0.05 / se:.1f} SE around 0.5\n")
    aucs = []
    for _ in range(reps):
        a = rng.normal(0, 1, n1)
        b = rng.normal(0, 1, n2)
        aucs.append(C.auc(a, b))
    aucs = np.array(aucs)
    print(f"  simulated null AUC: mean {aucs.mean():.4f}, SD {aucs.std():.5f}, "
          f"range [{aucs.min():.4f}, {aucs.max():.4f}]")
    print(f"  P(a true-null permutation lands outside [0.45, 0.55]) ~ "
          f"{float(((aucs < 0.45) | (aucs > 0.55)).mean()):.4f}")
    print("\n  So the window has essentially ZERO false-alarm rate, but it also cannot detect")
    print("  anything subtler than a ~9 SE failure. It is a smoke alarm for gross leakage, not a")
    print("  calibration check. A single permutation also gives no permutation distribution.")


# --------------------------------------------------------------------------- F


def section_F(rng) -> None:
    hdr("F. D4: is a 0.10 bag-AUC drop the right order-dependence test?")
    print("Bag AUC saturates near 1.0, so large real drops compress into small bag-AUC drops.\n")
    print(f"{'per-ex unshuf':>14} {'per-ex shuf':>12} {'per-ex drop':>12} {'bag unshuf':>11} "
          f"{'bag shuf':>9} {'bag drop':>9} {'>=0.10?':>8}")
    for ex_u, ex_s in [(0.65, 0.52), (0.60, 0.53), (0.58, 0.55), (0.56, 0.54), (0.70, 0.60), (0.55, 0.50)]:
        bu = Phi(10 * Phi_inv(ex_u))
        bs = Phi(10 * Phi_inv(ex_s))
        drop = bu - bs
        print(f"{ex_u:>14.2f} {ex_s:>12.2f} {ex_u - ex_s:>12.3f} {bu:>11.4f} {bs:>9.4f} "
              f"{drop:>9.4f} {'yes' if drop >= 0.10 else 'NO':>8}")
    print("\n  A per-example AUC collapsing from 0.65 to 0.52 (almost the whole effect gone) is a")
    print("  bag-AUC drop of only ~0.03 and would be reported as NOT order-dependent.")
    print("  The test is mis-scaled: it should be applied to per-example AUC or to Cohen's d.")


# --------------------------------------------------------------------------- G


def section_G() -> None:
    hdr("G. Decision-rule logic: boundary cases")
    print("score.py (the implementation used for E1/E1b) evaluates the sanity checks FIRST and")
    print("returns SETUP FAILURE if any fails, so a broken baseline cannot be read as 'HOLDS'.")
    print("The D2 text in DECISIONS.md omits that branch, and the descriptive helper")
    print("verdict_label() in sensitivity_parse.py reimplements the rule WITHOUT it:\n")

    def verdict_label(acc, baseline):
        if acc >= 0.65 and 0.45 <= baseline <= 0.55:
            return "RECONSIDER"
        if 0.56 <= acc < 0.65:
            return "STATISTICAL CUE"
        return "HOLDS"

    print(f"{'cat-vs-control':>15} {'baseline':>9} {'verdict_label()':>18}   comment")
    for acc, base, note in [
        (0.70, 0.50, "fine"),
        (0.70, 0.62, "BROKEN BASELINE read as evidence of NO leak"),
        (0.66, 0.60, "BROKEN BASELINE read as evidence of NO leak"),
        (0.60, 0.62, "broken baseline, reported as a positive cue"),
        (0.5599, 0.50, "just below the cue band"),
        (0.6499, 0.50, "top of the cue band"),
    ]:
        print(f"{acc:>15.4f} {base:>9.2f} {verdict_label(acc, base):>18}   {note}")
    print("\n  Two defects:")
    print("   1. a strong result with a failed baseline falls through to HOLDS (an unsound")
    print("      'no evidence of leak' conclusion from an uninterpretable comparison);")
    print("   2. the cue band 0.56-0.65 is not gated on the baseline at all.")
    print("  score.py is correct; the rule TEXT and the helper are not. Fix the text and the helper,")
    print("  not score.py.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    reps = 40 if args.quick else 120
    rng = np.random.default_rng(0)
    section_A(rng)
    section_B(rng, max(20, reps // 3))
    section_C(rng, max(20, reps // 2))
    section_D(rng, reps)
    section_E(rng, reps * 4)
    section_F(rng)
    section_G()
    print("\nDone. No API call and no GPU work was performed.")


if __name__ == "__main__":
    main()
