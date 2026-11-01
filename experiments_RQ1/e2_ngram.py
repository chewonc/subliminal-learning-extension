"""E2: n-gram logistic-regression detector with shuffle control (experiment_plan_RQ1.md §2, D4).

Features per completion: first-number one-hot (cols 0..999), unigram counts (1000..1999),
ordered-bigram counts (2000 + a*1000 + b). L2 logistic regression, C=1 (sklearn convention:
sum of log-losses + 0.5*||w||^2 / C), solved with scipy L-BFGS. 5 prompt-disjoint folds
(SHA-1(prompt) mod 5) over the fit+eval rows (6,000 per condition).

Pilot mode (--pilot N) fits ONLY on permuted labels (leakage check P2) on N rows per condition and
never computes the real-label AUC, so no result is seen before the verdict rules are approved.

Needs numpy and scipy. Usage:
  python e2_ngram.py --pilot 500
  python e2_ngram.py            # full run; writes outputs/e2_results.json, outputs/e2_per_example.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import time

import numpy as np

import common as C

N_COLS = 2000 + 1000 * 1000
PAIRINGS = [("cat", "control"), ("cat", "neutral"), ("neutral", "control")]


def features(completion: str) -> dict[int, float]:
    nums = C.parse_response(completion)
    if not nums:
        return {}
    f: dict[int, float] = {}
    f[nums[0]] = 1.0
    for n in nums:
        f[1000 + n] = f.get(1000 + n, 0.0) + 1.0
    for a, b in zip(nums, nums[1:]):
        c = 2000 + a * 1000 + b
        f[c] = f.get(c, 0.0) + 1.0
    return f


def build_matrix(texts: list[str]):
    from scipy.sparse import csr_matrix

    rows, cols, vals = [], [], []
    for i, t in enumerate(texts):
        for c, v in features(t).items():
            rows.append(i)
            cols.append(c)
            vals.append(v)
    return csr_matrix((vals, (rows, cols)), shape=(len(texts), N_COLS), dtype=np.float64)


def fit_logreg(X, y: np.ndarray, C_reg: float = 1.0, max_iter: int = 2000) -> tuple[np.ndarray, float]:
    """Minimise sum_i log(1+exp(-y_i (x_i.w + b))) + 0.5*||w||^2 / C over (w, b); y in {-1,+1}."""
    from scipy.optimize import minimize

    n, d = X.shape
    Xt = X.T.tocsr()

    def f_and_g(theta):
        w, b = theta[:d], theta[d]
        z = y * (X @ w + b)
        # stable log(1+exp(-z)) and sigmoid(-z)
        loss = np.logaddexp(0.0, -z).sum() + 0.5 * (w @ w) / C_reg
        s = -y / (1.0 + np.exp(z))  # d loss / d z * y
        gw = Xt @ s + w / C_reg
        gb = s.sum()
        return loss, np.concatenate([gw, [gb]])

    theta0 = np.zeros(d + 1)
    res = minimize(f_and_g, theta0, jac=True, method="L-BFGS-B", options={"maxiter": max_iter, "maxcor": 10})
    return res.x[:d], float(res.x[d])


def cross_val_scores(X, y: np.ndarray, folds: np.ndarray) -> np.ndarray:
    oof = np.zeros(len(y))
    for k in sorted(set(folds.tolist())):
        tr, te = folds != k, folds == k
        w, b = fit_logreg(X[tr], y[tr])
        oof[te] = X[te] @ w + b
    return oof


def run_pairing(pos_rows, neg_rows, shuffled: bool, rng_shuffle: np.random.Generator,
                permute_labels: bool, perm_seed: int = 0):
    texts = [r["completion"] for r in pos_rows] + [r["completion"] for r in neg_rows]
    if shuffled:
        texts = [C.shuffle_within(t, rng_shuffle) for t in texts]
    prompts = [r["prompt"] for r in pos_rows] + [r["prompt"] for r in neg_rows]
    y = np.array([1.0] * len(pos_rows) + [-1.0] * len(neg_rows))
    folds = np.array([C.fold_of(p) for p in prompts])
    if permute_labels:
        y = np.random.default_rng(perm_seed).permutation(y)
    X = build_matrix(texts)
    oof = cross_val_scores(X, y, folds)
    return oof, y, folds, X.nnz


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", type=int, default=0, help="rows per condition; permutation check only")
    ap.add_argument("--out", default=str(C.OUT))
    ap.add_argument("--n-perm", type=int, default=20,
                    help="label permutations for the leakage smoke test (D8 amendment 4.3)")
    args = ap.parse_args()
    out = C.OUT if args.out == str(C.OUT) else __import__("pathlib").Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    shas = C.verify_manifest()
    used = C.used_texts()
    rows = {}
    for cond in C.CONDITIONS:
        fit, ev, _base = C.splits(cond, used)   # baseline split is for the E1e naming test only
        allr = fit + ev
        rows[cond] = allr[: args.pilot] if args.pilot else allr
    print(f"rows per condition: { {c: len(v) for c, v in rows.items()} }", flush=True)

    results = {"mode": "pilot" if args.pilot else "full", "sha256": shas, "n_rows": {c: len(v) for c, v in rows.items()},
               "pairings": {}, "created": time.strftime("%Y-%m-%d %H:%M:%S")}

    # P2: label-permutation smoke test (D8 amendment 4.3). This is NOT a calibration check: with
    # ~6,000 rows a side the null AUC has SD about 0.005, so the [0.45, 0.55] window spans about
    # +/- 9.5 SE and catches only gross leakage. Run with >= 20 permutations so the observed value
    # can be read against a permutation distribution rather than a single draw.
    t0 = time.time()
    n_perm = 1 if args.pilot else args.n_perm
    perm_aucs = []
    for i in range(n_perm):
        oof, y, folds, nnz = run_pairing(rows["cat"], rows["control"], False, np.random.default_rng(0),
                                         permute_labels=True, perm_seed=i)
        perm_aucs.append(C.auc(oof[y > 0], oof[y < 0]))
        print(f"  permutation {i + 1}/{n_perm}: AUC {perm_aucs[-1]:.4f}", flush=True)
    pa = np.array(perm_aucs)
    ok = bool(((pa >= 0.45) & (pa <= 0.55)).all())
    results["permutation_check"] = {
        "pairing": "cat_vs_control", "n_permutations": n_perm,
        "aucs": [float(x) for x in pa], "mean": float(pa.mean()), "sd": float(pa.std()),
        "min": float(pa.min()), "max": float(pa.max()),
        "window": [0.45, 0.55], "all_within_window": ok, "pass": ok,
        "note": "smoke test for gross leakage, not a calibration check",
        "seconds": round(time.time() - t0, 1),
    }
    print(f"P2 permutation AUCs: mean {pa.mean():.4f} sd {pa.std():.4f} "
          f"range [{pa.min():.4f}, {pa.max():.4f}]  ({'PASS' if ok else 'FAIL'})", flush=True)
    if args.pilot:
        C.dump_json(results, out / "e2_pilot.json")
        print(f"pilot done; real-label AUC not computed. Wrote {out / 'e2_pilot.json'}")
        return
    if not ok:
        C.dump_json(results, out / "e2_results.json")
        raise SystemExit("P2 FAIL: a label-permutation AUC fell outside [0.45, 0.55]; stopping (plan 9).")

    per_example_rows = []
    scores_by_key: dict[str, tuple] = {}
    rng_boot = np.random.default_rng(0)
    for pos, neg in PAIRINGS:
        for shuffled in (False, True):
            t0 = time.time()
            oof, y, folds, nnz = run_pairing(rows[pos], rows[neg], shuffled, np.random.default_rng(0), permute_labels=False)
            ps, ns = oof[y > 0], oof[y < 0]
            a, lo, hi = C.auc_ci(ps, ns, rng_boot)
            ba, blo, bhi = C.bag_auc_ci(ps, ns, rng_boot)
            key = f"{pos}_vs_{neg}" + ("_shuffled" if shuffled else "")
            scores_by_key[key] = (ps, ns)
            results["pairings"][key] = {"pos": pos, "neg": neg, "shuffled": shuffled,
                                        "n_pos": int(len(ps)), "n_neg": int(len(ns)),
                                        "auc_example": a, "auc_example_ci": [lo, hi],
                                        "auc_bag": ba, "auc_bag_ci": [blo, bhi],
                                        "bag_size": C.BAG_SIZE_PRIMARY,
                                        "bag_size_curve": C.bag_auc_curve(ps, ns, rng_boot),
                                        "moments_pos": C.moments(ps), "moments_neg": C.moments(ns),
                                        "nnz": int(nnz), "seconds": round(time.time() - t0, 1)}
            print(f"{key:28} example AUC {a:.3f} [{lo:.3f},{hi:.3f}]  bag AUC {ba:.3f} [{blo:.3f},{bhi:.3f}]  {time.time()-t0:.0f}s", flush=True)
            allrows = rows[pos] + rows[neg]
            for r, s, f in zip(allrows, oof, folds):
                per_example_rows.append([r["row_id"], r["condition"], key, int(shuffled), int(f), f"{s:.6f}"])

    # Order dependence on the PER-EXAMPLE scale (D8 amendment 4.1). The bag-AUC version was
    # non-monotonic: because bag AUC saturates near 1, a per-example drop of 0.100 could score
    # 0.006 while a drop of 0.050 scored 0.396. No threshold is attached (amendment 4.2).
    results["order_dependence"] = {}
    for pos, neg in PAIRINGS:
        k_u, k_s = f"{pos}_vs_{neg}", f"{pos}_vs_{neg}_shuffled"
        if k_u in scores_by_key and k_s in scores_by_key:
            pu, nu = scores_by_key[k_u]
            psh, nsh = scores_by_key[k_s]
            # paired=True: run_pairing builds both from the same row list in the same order.
            d = C.paired_auc_diff_ci(pu, nu, psh, nsh, rng_boot, n_boot=2000, paired=True)
            d["note"] = "per-example AUC, unshuffled minus shuffled; reported without a threshold"
            results["order_dependence"][k_u] = d
            print(f"order dependence {k_u:24} per-example AUC drop {d['diff']:+.4f} "
                  f"[{d['ci'][0]:+.4f}, {d['ci'][1]:+.4f}]")

    cc, nc = results["pairings"]["cat_vs_control"], results["pairings"]["neutral_vs_control"]
    results["D4"] = {
        # Two-sided neutral gate (D9): an AUC far below 0.5 is strong separation in the other
        # direction, so "<= 0.60" alone would pass when prompt presence IS detectable.
        "neutral_gate_two_sided": bool(0.40 <= nc["auc_bag"] <= 0.60),
        "dataset_level_detectable": bool(cc["auc_bag"] >= 0.90 and cc["auc_bag_ci"][0] >= 0.80
                                         and 0.40 <= nc["auc_bag"] <= 0.60),
        "bag_size": C.BAG_SIZE_PRIMARY,
        "order_dependence_per_example": results["order_dependence"].get("cat_vs_control"),
        "prediction_13oct_example_auc": [0.55, 0.65],
        "prediction_13oct_example_auc_hit": bool(0.55 <= cc["auc_example"] <= 0.65),
        "thresholds": {"bag_auc": 0.90, "bag_ci_lo": 0.80, "gate_neutral_bag_auc": [0.40, 0.60],
                       "provenance": "PRE-REGISTERED 19 Oct (D4), unchanged"},
        "rule_status": "D4 + D8 amendments 4.1-4.4 (DECISIONS.md)",
        "claim_scope": (
            "Dataset-level detectability means the two corpora are statistically separable by "
            "counting numbers. Two sets of generations from the same model under different system "
            "prompts are separable close to by definition, so this does NOT identify the cat trait "
            "and says nothing about semantic content or transmission mechanism. Its value is the "
            "existence of a cheap dataset-level filter, which the paper never tested."),
    }
    C.dump_json(results, out / "e2_results.json")
    with (out / "e2_per_example.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["row_id", "condition", "pairing", "shuffled", "fold", "score"])
        w.writerows(per_example_rows)
    print(f"wrote {out / 'e2_results.json'} and e2_per_example.csv")


if __name__ == "__main__":
    main()
