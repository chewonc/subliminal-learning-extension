"""Second-round validation: are the PROPOSED revisions themselves sound? CPU only, no API, no GPU.

Part A  bag AUC vs per-example AUC beyond the Gaussian equal-variance case
Part B  the baseline-subtracted naming test under correlation, unequal variance, heavy tails,
        finite baseline estimation, and a generic-persona confound
Part C  finite-sample behaviour of the baseline correction

Usage: python validate_revisions.py [--quick]
"""

from __future__ import annotations

import argparse
import math

import numpy as np

import common as C

SQRT2 = math.sqrt(2.0)


def Phi(z):
    return 0.5 * (1.0 + math.erf(z / SQRT2))


def Phi_inv(p):
    lo, hi = -10.0, 10.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if Phi(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def hdr(s):
    print(f"\n{'=' * 80}\n{s}\n{'=' * 80}")


# ----------------------------------------------------------------------------- Part A


def bag_auc(pos, neg, rng, k=100, nb=4000):
    return C.auc(C.bag_scores(pos, rng, nb, k), C.bag_scores(neg, rng, nb, k))


def section_A(rng, n=20000):
    hdr("A. Is bag AUC just a monotone rescaling of per-example AUC?")
    print("Prediction under the Gaussian model:  bag_AUC(k) = Phi( sqrt(k) * Phi^-1(per-example AUC) )")
    print("Below: 'predicted' uses that formula; 'actual' is simulated with the real bag_scores/auc code.\n")
    print(f"{'score distribution':>34} {'per-ex AUC':>11} {'predicted':>10} {'actual':>8} {'gap':>8}")

    cases = []

    # 1. Gaussian, equal variance (the case the formula was derived for)
    d = 0.178
    cases.append(("Gaussian, equal var (d=0.178)", rng.normal(d, 1, n), rng.normal(0, 1, n)))

    # 2. Gaussian, UNEQUAL variance. The formula still holds exactly: both per-example and bag AUC
    #    are Phi(dmu / sqrt(s1^2+s2^2)) up to the sqrt(k) factor.
    cases.append(("Gaussian, unequal var (1 vs 3)", rng.normal(0.5, 1, n), rng.normal(0, 3, n)))

    # 3. Heavy tails: Student t with 3 df. Variance finite but large; bag means concentrate slowly.
    t3p = rng.standard_t(3, n) + 0.30
    t3n = rng.standard_t(3, n)
    cases.append(("Student t(3), shift 0.30", t3p, t3n))

    # 4. Very heavy tails: t with 1.5 df (infinite variance). Bag means never concentrate.
    t15p = rng.standard_t(1.5, n) + 0.30
    t15n = rng.standard_t(1.5, n)
    cases.append(("Student t(1.5), shift 0.30", t15p, t15n))

    # 5. Skewed: lognormal.
    cases.append(("Lognormal, mult. shift", rng.lognormal(0.10, 1.0, n), rng.lognormal(0.0, 1.0, n)))

    # 6. SPARSE SIGNAL: 5% of positives carry a large shift, 95% identical to the negatives.
    sp = rng.normal(0, 1, n)
    mask = rng.random(n) < 0.05
    sp[mask] += 3.0
    cases.append(("Sparse: 5% of rows shifted by 3", sp, rng.normal(0, 1, n)))

    # 7. Pure VARIANCE difference, identical means. Bag means cannot separate; per-example can a little.
    cases.append(("Equal means, variance 1 vs 2", rng.normal(0, 1, n), rng.normal(0, 2, n)))

    # 8. Bimodal mixture.
    bm = np.where(rng.random(n) < 0.5, rng.normal(-1.5, 0.4, n), rng.normal(1.8, 0.4, n))
    bn = np.where(rng.random(n) < 0.5, rng.normal(-1.5, 0.4, n), rng.normal(1.5, 0.4, n))
    cases.append(("Bimodal mixture, upper mode moved", bm, bn))

    rows = []
    for name, pos, neg in cases:
        ex = C.auc(pos, neg)
        pred = Phi(math.sqrt(100) * Phi_inv(min(max(ex, 1e-6), 1 - 1e-6)))
        act = bag_auc(pos, neg, rng)
        rows.append((name, ex, pred, act, act - pred))
        print(f"{name:>34} {ex:>11.4f} {pred:>10.4f} {act:>8.4f} {act - pred:>+8.4f}")

    print("\n  The formula is EXACT for Gaussians, equal or unequal variance. It is not a general law.")
    big = [r for r in rows if abs(r[4]) > 0.05]
    print(f"  Cases deviating by more than 0.05: {len(big)} of {len(rows)}")
    for r in big:
        print(f"    {r[0]:>34}: predicted {r[2]:.3f}, actual {r[3]:.3f}")

    print("\n  Why: bag means obey the CLT, so bag AUC depends (for large k) only on the MEAN GAP and")
    print("  the variances. Per-example AUC is P(X>Y), a rank/overlap measure that depends on the")
    print("  whole shape. Two score distributions can share a per-example AUC and differ wildly in")
    print("  mean gap, or share a mean gap and differ in per-example AUC.")

    # The decisive demonstration: equal per-example AUC, very different bag AUC.
    print("\n  Decisive case: two settings matched on per-example AUC, compared on bag AUC.")
    target = 0.52
    g_d = Phi_inv(target) * SQRT2
    gp, gn = rng.normal(g_d, 1, n), rng.normal(0, 1, n)
    # A sparse signal lifts per-example AUC by at most ~frac/2, so frac must exceed 2*(target-0.5).
    frac, shift, sp2 = 0.08, 0.0, None
    for shift in np.arange(0.5, 12.0, 0.25):
        cand = rng.normal(0, 1, n)
        m = rng.random(n) < frac
        cand[m] += shift
        if C.auc(cand, rng.normal(0, 1, n)) >= target:
            sp2 = cand
            break
    sn2 = rng.normal(0, 1, n)
    if sp2 is None:
        print("      (could not match the target per-example AUC with this sparse design)")
        return
    print(f"{'':6}{'setting':>34} {'per-ex AUC':>11} {'bag AUC k=100':>14}")
    print(f"{'':6}{'Gaussian shift':>34} {C.auc(gp, gn):>11.4f} {bag_auc(gp, gn, rng):>14.4f}")
    print(f"{'':6}{f'{frac:.0%} of rows shifted by {shift:.1f}':>34} {C.auc(sp2, sn2):>11.4f} "
          f"{bag_auc(sp2, sn2, rng):>14.4f}")
    print("\n  Same per-example AUC, different bag AUC. They are NOT monotone transforms of each")
    print("  other in general, and they answer different questions:")
    print("    per-example AUC: can a single row be classified?")
    print("    bag AUC        : can a shift in the score's MEAN be detected in a sample of k rows?")


# ----------------------------------------------------------------------------- Part B


def naming_once(rng, n_rows, k, nb, n_animals, cat_effect, tau, rho, var_hetero,
                dist, generic_gamma, generic_load, n_baseline):
    """One replication of the baseline-subtracted naming test.

    cat_effect   : cat-SPECIFIC shift on animal 0 only, in per-row SDs
    generic_gamma: a generic persona shift present on cat data for ALL animals,
                   with per-animal loading generic_load (so it does NOT cancel in an argmax)
    """
    base = rng.normal(0, tau, n_animals)
    sd = np.exp(rng.normal(0, var_hetero, n_animals))

    def draw(m):
        if dist == "normal":
            z = rng.normal(0, 1, (m, n_animals))
        elif dist == "t3":
            z = rng.standard_t(3, (m, n_animals)) / math.sqrt(3.0)
        elif dist == "lognormal":
            z = rng.lognormal(0, 0.8, (m, n_animals))
            z = (z - z.mean()) / z.std()
        else:
            raise ValueError(dist)
        if rho > 0:                      # shared factor across animals
            f = rng.normal(0, 1, (m, 1))
            z = math.sqrt(rho) * f + math.sqrt(1 - rho) * z
        return z * sd + base

    ctrl_base = draw(n_baseline)         # pool used to ESTIMATE the per-animal baseline
    ctrl_eval = draw(n_rows)
    cat = draw(n_rows)
    cat += generic_gamma * generic_load  # generic persona signature, unequal across animals
    cat[:, 0] += cat_effect              # cat-specific signature

    off = ctrl_base.mean(axis=0)         # finite-sample baseline estimate
    ci = rng.integers(0, n_rows, (nb, k))
    ki = rng.integers(0, n_rows, (nb, k))
    cat_bags = cat[ci].mean(axis=1) - off
    ctrl_bags = ctrl_eval[ki].mean(axis=1) - off
    cw = np.bincount(cat_bags.argmax(axis=1), minlength=n_animals)
    kw = np.bincount(ctrl_bags.argmax(axis=1), minlength=n_animals)
    return cw[0] / nb, kw.max() / nb, int(cw.argmax())


def section_B(rng, reps):
    hdr("B. The baseline-subtracted naming test under realistic conditions")
    base_kw = dict(n_rows=5000, k=100, nb=1000, n_animals=10, tau=1.0, rho=0.0,
                   var_hetero=0.0, dist="normal", generic_gamma=0.0,
                   generic_load=np.zeros(10), n_baseline=5000)

    print("B1. Null (no cat-specific signal, no generic shift) under departures from the")
    print("    Gaussian / independent / equal-variance / infinite-baseline ideal.\n")
    print(f"{'condition':>42} {'P(cat named >=80%)':>19} {'P(ctrl gate <=30%)':>19}")
    conds = [
        ("ideal: normal, independent, equal var", {}),
        ("correlated animals (rho=0.5)", {"rho": 0.5}),
        ("correlated animals (rho=0.9)", {"rho": 0.9}),
        ("unequal variances (sd spread 0.5)", {"var_hetero": 0.5}),
        ("heavy tails t(3)", {"dist": "t3"}),
        ("skewed lognormal", {"dist": "lognormal"}),
        ("baseline from 500 rows", {"n_baseline": 500}),
        ("baseline from 100 rows", {"n_baseline": 100}),
        ("all of the above together", {"rho": 0.7, "var_hetero": 0.5, "dist": "t3", "n_baseline": 500}),
    ]
    for name, over in conds:
        kw = dict(base_kw); kw.update(over)
        a = np.array([naming_once(rng, cat_effect=0.0, **kw)[:2] for _ in range(reps)])
        print(f"{name:>42} {float((a[:, 0] >= 0.80).mean()):>18.0%} {float((a[:, 1] <= 0.30).mean()):>18.0%}")

    print("\nB2. THE CONFOUND. Cat data carries a GENERIC persona signature and NO cat-specific one.")
    print("    The generic shift loads unequally across the ten animals; 'cat loading rank' says")
    print("    where the cat prompt sits in that loading order.\n")
    print(f"{'generic shift':>14} {'cat loading rank':>18} {'P(test names cat)':>19} {'P(ctrl gate ok)':>17}")
    for gamma in (0.3, 0.6):
        for rank_desc, mk in [("highest of 10", "top"), ("random", "rand"), ("lowest of 10", "bot")]:
            hits, gates = [], []
            for _ in range(reps):
                load = rng.random(10)
                if mk == "top":
                    load = np.sort(load)[::-1]
                    load = np.roll(load, 0)          # animal 0 (cat) gets the largest loading
                elif mk == "bot":
                    load = np.sort(load)             # animal 0 gets the smallest
                kw = dict(base_kw); kw.update(generic_gamma=gamma, generic_load=load)
                c, g, _ = naming_once(rng, cat_effect=0.0, **kw)
                hits.append(c >= 0.80); gates.append(g <= 0.30)
            print(f"{gamma:>14.1f} {rank_desc:>18} {float(np.mean(hits)):>18.0%} {float(np.mean(gates)):>17.0%}")
    print("\n  When a purely generic persona shift happens to load most on the cat prompt, the")
    print("  corrected naming test names 'cat' with no cat-specific information present at all.")
    print("  The test cannot separate 'the data looks like the cat persona' from 'the data looks")
    print("  like a persona, and cat is the nearest of the ten prompts offered'.")

    print("\nB3. Positive control: cat-specific signal present, confound absent.")
    print(f"{'cat effect (SD)':>16} {'P(cat named >=80%)':>19}")
    for eff in (0.0, 0.2, 0.3, 0.5):
        kw = dict(base_kw)
        a = np.array([naming_once(rng, cat_effect=eff, **kw)[:2] for _ in range(reps)])
        print(f"{eff:>16.2f} {float((a[:, 0] >= 0.80).mean()):>18.0%}")


# ----------------------------------------------------------------------------- Part C


def section_C(rng, reps):
    hdr("C. Does finite-sample baseline estimation bias the corrected test?")
    print("The baseline is one shared estimate reused by every bag, so its error does NOT average")
    print("out across bags: it shifts all bags of a condition together.\n")
    print(f"{'baseline rows':>14} {'baseline SE (per-row SD)':>26} {'vs bag SD (k=100)':>19} "
          f"{'P(ctrl gate <=30%)':>19}")
    for nb_rows in (100, 500, 2000, 5000, 20000):
        se = 1.0 / math.sqrt(nb_rows)
        gates = []
        for _ in range(reps):
            _, g, _ = naming_once(rng, n_rows=5000, k=100, nb=1000, n_animals=10, cat_effect=0.0,
                                  tau=1.0, rho=0.0, var_hetero=0.0, dist="normal",
                                  generic_gamma=0.0, generic_load=np.zeros(10), n_baseline=nb_rows)
            gates.append(g <= 0.30)
        print(f"{nb_rows:>14} {se:>26.4f} {se / 0.1:>18.2f}x {float(np.mean(gates)):>18.0%}")
    print("\n  At 5,000 baseline rows the estimate's SE is ~14% of a bag's SD, which the simulation")
    print("  shows is tolerable. Below ~500 rows the shared error starts to dominate the argmax and")
    print("  the control gate degrades. The planned 5,000 rows is adequate; this is not the weak point.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    reps = 25 if args.quick else 60
    rng = np.random.default_rng(7)
    section_A(rng)
    section_B(rng, reps)
    section_C(rng, reps)
    print("\nDone. No API call and no GPU work was performed.")


if __name__ == "__main__":
    main()
