"""E1-R (experiment_plan_RQ2.md, rule D10): symmetric likelihood classifier and leave-one-animal-out test.

Question: on the existing RQ1-E1e scores, does the likelihood under the CAT prompt carry information
about cat-vs-control (and cat-vs-neutral) that the likelihoods under the other nine animal prompts,
the neutral prompt and no prompt do not already carry?

CPU only, numpy. Reads experiments_RQ1/outputs_hpc/outputs/e1e_scores.jsonl (split eval, unshuffled)
and the SHA-pinned data files for the prompts (fold assignment). Writes outputs/e1r_results.json.

Usage:
    python e1r_symmetric.py                      # full run, D10 settings
    python e1r_symmetric.py --n-boot 200         # quick look; NOT the D10 result
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "experiments_RQ1"))
import common as C  # noqa: E402

SCORES = HERE.parent / "experiments_RQ1" / "outputs_hpc" / "outputs" / "e1e_scores.jsonl"
OUT = HERE / "outputs"

KEYS = ["none", "neutral"] + C.ANIMALS          # 12 prompts scored on every eval row
DIFF_KEYS = ["neutral"] + C.ANIMALS             # d_k = logp(k) - logp(none)
PAIRINGS = [("cat", "control"), ("cat", "neutral"), ("neutral", "control")]
D10_PAIRINGS = [("cat", "control"), ("cat", "neutral")]   # primary, secondary

# D10 thresholds (DECISIONS.md, 2026-10-20)
DELTA_POS = 0.02
DELTA_NULL = 0.01
LAMBDA = 1e-3
N_FOLDS = 5


# ----------------------------------------------------------------------------- data

def load_scores(path: Path = SCORES, conditions=("cat", "control", "neutral")) -> dict:
    """{cond: {row_id: {prompt_key: logp, '_ntok': n_tokens}}} for split eval, unshuffled."""
    out: dict[str, dict[str, dict]] = {c: {} for c in conditions}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["split"] != "eval" or r["shuffled"] != 0 or r["condition"] not in out:
                continue
            if r.get("missing", 0) or not r.get("prefix_ok", True):
                raise SystemExit(f"bad score row: {r}")
            d = out[r["condition"]].setdefault(r["row_id"], {})
            d[r["prompt_key"]] = r["logp"]
            d["_ntok"] = r["n_tokens"]
    return out


def assemble(scores: dict) -> dict:
    """Arrays per condition, rows ordered as in common.splits(); folds from the row's prompt."""
    C.verify_manifest()
    used = C.used_texts()
    data = {}
    for cond, by_id in scores.items():
        _, ev, _ = C.splits(cond, used)
        ids = [r["row_id"] for r in ev]
        if set(ids) != set(by_id):
            raise SystemExit(f"{cond}: eval row_ids in the score file do not match common.splits()")
        L = np.array([[by_id[i][k] for k in KEYS] for i in ids], dtype=float)
        ntok = np.array([by_id[i]["_ntok"] for i in ids], dtype=float)
        fold = np.array([C.fold_of(r["prompt"], N_FOLDS) for r in ev])
        data[cond] = {"L": L, "ntok": ntok, "fold": fold}
    return data


# ----------------------------------------------------------------------------- model

def features(L: np.ndarray, ntok: np.ndarray, drop: str | None = None) -> np.ndarray:
    """d_k = logp(k) - logp(none) for k != none, plus logp(none)/n_tokens. Dropping k removes d_k,
    and no remaining column contains logp(k), so the dropped prompt carries no information."""
    none = L[:, KEYS.index("none")]
    cols = [L[:, KEYS.index(k)] - none for k in DIFF_KEYS if k != drop]
    cols.append(none / ntok)
    return np.column_stack(cols)


def fit_logreg(X: np.ndarray, y: np.ndarray, lam: float = LAMBDA, iters: int = 50) -> tuple:
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Z = np.column_stack([np.ones(len(X)), (X - mu) / sd])
    w = np.zeros(Z.shape[1])
    pen = np.full(Z.shape[1], lam)
    pen[0] = 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(Z @ w, -30, 30)))
        g = Z.T @ (p - y) / len(y) + pen * w
        H = (Z * (p * (1 - p))[:, None]).T @ Z / len(y) + np.diag(pen) + 1e-9 * np.eye(len(w))
        step = np.linalg.solve(H, g)
        w -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return w, mu, sd


def predict(model: tuple, X: np.ndarray) -> np.ndarray:
    w, mu, sd = model
    return np.column_stack([np.ones(len(X)), (X - mu) / sd]) @ w   # logit; AUC is rank-based


def oof_scores(pos: dict, neg: dict, drop: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold logits, prompt-grouped folds. Returns (scores of pos rows, scores of neg rows)."""
    Xp, Xn = features(pos["L"], pos["ntok"], drop), features(neg["L"], neg["ntok"], drop)
    X = np.vstack([Xp, Xn])
    y = np.concatenate([np.ones(len(Xp)), np.zeros(len(Xn))])
    fold = np.concatenate([pos["fold"], neg["fold"]])
    s = np.empty(len(y))
    for k in range(N_FOLDS):
        te = fold == k
        s[te] = predict(fit_logreg(X[~te], y[~te]), X[te])
    return s[: len(Xp)], s[len(Xp):]


# ----------------------------------------------------------------------------- analysis

def generic_specific(d: dict) -> tuple[np.ndarray, np.ndarray]:
    none = d["L"][:, KEYS.index("none")]
    dk = np.column_stack([d["L"][:, KEYS.index(a)] - none for a in C.ANIMALS])
    G = dk.mean(1)
    return G, dk[:, C.ANIMALS.index("cat")] - G


def verdict(deltas: dict) -> dict:
    cat = deltas["cat"]
    largest = max(deltas, key=lambda a: deltas[a]["diff"])
    is_largest = largest == "cat"
    if cat["diff"] >= DELTA_POS and cat["ci"][0] > 0 and is_largest:
        v = "CAT-SPECIFIC COMPONENT"
    elif cat["diff"] < DELTA_NULL or not is_largest:
        v = "NO CAT-SPECIFIC COMPONENT"
    else:
        v = "WEAK"
    rank = 1 + sum(deltas[a]["diff"] > cat["diff"] for a in deltas if a != "cat")
    return {"verdict": v, "delta_auc_cat": cat["diff"], "delta_auc_cat_ci": cat["ci"],
            "cat_rank_of_10": rank, "largest_delta_animal": largest}


def analyze(data: dict, n_boot: int = 2000, seed: int = 0, log=print) -> dict:
    rng = np.random.default_rng(seed)
    res: dict = {"pairings": {}}
    for a, b in PAIRINGS:
        t0 = time.time()
        sp, sn = oof_scores(data[a], data[b])
        full_auc, lo, hi = C.auc_ci(sp, sn, rng, n_boot=min(n_boot, 500))
        pr: dict = {"n_pos": len(sp), "n_neg": len(sn), "auc_full": full_auc, "auc_full_ci": [lo, hi]}
        G_p, S_p = generic_specific(data[a])
        G_n, S_n = generic_specific(data[b])
        pr["auc_generic_G"] = list(C.auc_ci(G_p, G_n, rng, n_boot=min(n_boot, 500)))
        pr["auc_specific_S_cat"] = list(C.auc_ci(S_p, S_n, rng, n_boot=min(n_boot, 500)))
        if (a, b) in D10_PAIRINGS:
            deltas = {}
            for animal in C.ANIMALS:
                dp, dn = oof_scores(data[a], data[b], drop=animal)
                d = C.paired_auc_diff_ci(sp, sn, dp, dn, rng, n_boot=n_boot, paired=True)
                d["auc_without"] = C.auc(dp, dn)
                deltas[animal] = d
            pr["delta_auc_drop"] = deltas
            pr["D10"] = verdict(deltas)
        res["pairings"][f"{a}_vs_{b}"] = pr
        log(f"{a}_vs_{b}: full AUC {full_auc:.4f} [{lo:.4f},{hi:.4f}]  "
            f"G {pr['auc_generic_G'][0]:.4f}  S_cat {pr['auc_specific_S_cat'][0]:.4f}"
            + (f"  dAUC_cat {pr['D10']['delta_auc_cat']:+.4f} rank {pr['D10']['cat_rank_of_10']}  -> {pr['D10']['verdict']}"
               if "D10" in pr else "") + f"  ({time.time() - t0:.0f}s)")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--scores", type=Path, default=SCORES)
    args = ap.parse_args()
    t0 = time.time()
    data = assemble(load_scores(args.scores))
    res = analyze(data, n_boot=args.n_boot)
    res.update({
        "rule": "D10 (DECISIONS.md, 2026-10-20)",
        "mode": "full" if args.n_boot == 2000 else f"exploratory n_boot={args.n_boot} (NOT the D10 result)",
        "scores_file": str(args.scores.relative_to(HERE.parent)), "scores_sha256": C.sha256(args.scores),
        "n_rows": {c: int(len(d["L"])) for c, d in data.items()},
        "features": "d_k = logp(k) - logp(none) for k in neutral + 10 animals, plus logp(none)/n_tokens",
        "model": f"L2 logistic regression, lambda={LAMBDA}, standardised, {N_FOLDS}-fold CV grouped by prompt",
        "ci_note": "Delta-AUC CIs: paired bootstrap over rows of the out-of-fold scores; refit variance not included.",
        "claim_scope": "L1 for the symmetric AUCs. A CAT-SPECIFIC COMPONENT verdict supports L2 only "
                       "provisionally: one animal, scored under the prompts that generated the data. "
                       "The L2 claim needs D11 (owl/panda, off-generator prompts).",
        "prediction_D10": "NO CAT-SPECIFIC COMPONENT, credence 0.70 (written before running)",
        "seconds": round(time.time() - t0, 1),
    })
    C.dump_json(res, OUT / "e1r_results.json")
    print(f"wrote {OUT / 'e1r_results.json'}")


if __name__ == "__main__":
    main()
