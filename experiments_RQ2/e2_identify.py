"""E2 analysis (experiment_plan_RQ2.md E2; rules D11 + D16): does the signal identify the correct animal?

Reads RQ1-E1e scores (core conditions under the 12 template-A prompts, control baseline) and G1b
scores (template-B prompts; new conditions under all 15), plus the data files for texts and folds.
CPU only; numpy + scipy (n-gram part reuses experiments_RQ1/e2_ngram.py unchanged).

    python e2_identify.py [--e1e <file>] [--g1b <file>] [--n-boot 1000]
Writes outputs/e2_identify_results.json.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

import rq2_common as R
import e1r_symmetric as E1R

C = R.C
HERE = Path(__file__).resolve().parent
E1E_SCORES = HERE.parent / "experiments_RQ1" / "outputs_hpc" / "outputs" / "e1e_scores.jsonl"
G1B_SCORES = HERE / "outputs" / "g1b_e2_scores.jsonl"
KEYS = R.A_KEYS + R.B_KEYS                       # 15 prompts
ID_ANIMALS = ["cat", "owl", "panda"]             # identification set (D11 ii)

# D11 thresholds
LRB_AUC, LRB_LO = 0.65, 0.60
ID_RATE = 0.80
NOISE_BAND = (0.45, 0.55)
GENERIC_LRB, GENERIC_RATIO = 0.58, 0.85
BAGS, BAG = 1000, 100


# ----------------------------------------------------------------------------- loading

def _ingest(path: Path, store: dict, base: dict) -> None:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("shuffled", 0) != 0 or r.get("variant", "orig") != "orig":
                continue
            if r["split"] == "eval":
                tgt = store.setdefault(r["row_id"], {})
            elif r["split"] == "baseline" and r["condition"] == "control":
                tgt = base.setdefault(r["row_id"], {})
            else:
                continue
            v = float(r["logp"])
            if not math.isfinite(v) or r.get("missing", 0) or not r.get("prefix_ok", True):
                raise SystemExit(f"bad score row in {path.name}: {r}")
            k = r["prompt_key"]
            if k in tgt and abs(tgt[k] - v) > 1e-6:
                raise SystemExit(f"{r['row_id']} scored twice under {k} with different values: {tgt[k]} vs {v}")
            tgt[k] = v
            tgt["_ntok"] = r["n_tokens"]


def load(e1e: Path = E1E_SCORES, g1b: Path = G1B_SCORES, conds=None) -> tuple[dict, dict]:
    """S[cond] = {'L': {key: array}, 'ntok', 'fold', 'texts'}; base = {key: array} (control baseline)."""
    store, base_store = {}, {}
    for p in (e1e, g1b):
        _ingest(p, store, base_store)
    conds = conds or [c for c in list(R.CORE_CONDITIONS) + list(R.NEW_CONDITIONS)
                      if R.data_path(c).is_file()]
    used = C.used_texts()
    S, base = {}, {}
    for cond in conds:
        ev, base_rows = R.eval_rows(cond, used)
        missing = [(r["row_id"], k) for r in ev for k in KEYS if k not in store.get(r["row_id"], {})]
        if missing:
            raise SystemExit(f"{cond}: {len(missing)} (row, prompt) scores missing, e.g. {missing[:3]}")
        S[cond] = {"L": {k: np.array([store[r["row_id"]][k] for r in ev]) for k in KEYS},
                   "ntok": np.array([store[r["row_id"]]["_ntok"] for r in ev], dtype=float),
                   "fold": np.array([C.fold_of(r["prompt"], E1R.N_FOLDS) for r in ev]),
                   "texts": [r["completion"] for r in ev], "prompts": [r["prompt"] for r in ev]}
        if cond == "control":
            ids = [r["row_id"] for r in base_rows]
            need = ["neutral"] + C.ANIMALS + R.B_KEYS
            miss = [(i, k) for i in ids for k in need if k not in base_store.get(i, {})]
            if miss:
                raise SystemExit(f"control baseline: {len(miss)} scores missing, e.g. {miss[:3]}")
            base = {k: np.array([base_store[i][k] for i in ids]) for k in need}
    return S, base


# ----------------------------------------------------------------------------- detectors

def lr(S, cond, a, b):
    return S[cond]["L"][a] - S[cond]["L"][b]


def sym_oof(S, pos, neg, keys=KEYS):
    """Symmetric likelihood classifier (E1-R features over `keys`), prompt-grouped out-of-fold logits."""
    def feats(d):
        none = d["L"]["none"]
        cols = [d["L"][k] - none for k in keys if k != "none"] + [none / d["ntok"]]
        return np.column_stack(cols)
    Xp, Xn = feats(S[pos]), feats(S[neg])
    X = np.vstack([Xp, Xn])
    y = np.concatenate([np.ones(len(Xp)), np.zeros(len(Xn))])
    fold = np.concatenate([S[pos]["fold"], S[neg]["fold"]])
    s = np.empty(len(y))
    for k in range(E1R.N_FOLDS):
        te = fold == k
        s[te] = E1R.predict(E1R.fit_logreg(X[~te], y[~te]), X[te])
    return s[: len(Xp)], s[len(Xp):]


def ngram_oof(S, pos, neg):
    import e2_ngram as NG  # experiments_RQ1, on sys.path via rq2_common
    texts = S[pos]["texts"] + S[neg]["texts"]
    y = np.array([1.0] * len(S[pos]["texts"]) + [-1.0] * len(S[neg]["texts"]))
    folds = np.concatenate([S[pos]["fold"], S[neg]["fold"]])
    oof = NG.cross_val_scores(NG.build_matrix(texts), y, folds)
    return oof[y > 0], oof[y < 0]


def ngram_transfer(S, train_pos, train_neg, test_pos):
    """Cross-fitted transfer: a cat-vs-control n-gram detector trained on folds != k scores the
    test_pos rows and the control rows of fold k (control rows are never scored by a model that saw them)."""
    import e2_ngram as NG
    Xp = NG.build_matrix(S[train_pos]["texts"])
    Xn = NG.build_matrix(S[train_neg]["texts"])
    Xt = NG.build_matrix(S[test_pos]["texts"])
    fp, fn, ft = S[train_pos]["fold"], S[train_neg]["fold"], S[test_pos]["fold"]
    from scipy.sparse import vstack
    st, sn = np.empty(len(ft)), np.empty(len(fn))
    for k in range(E1R.N_FOLDS):
        X = vstack([Xp[fp != k], Xn[fn != k]]).tocsr()
        y = np.array([1.0] * int((fp != k).sum()) + [-1.0] * int((fn != k).sum()))
        w, b = NG.fit_logreg(X, y)
        st[ft == k] = Xt[ft == k] @ w + b
        sn[fn == k] = Xn[fn == k] @ w + b
    return st, sn


# ----------------------------------------------------------------------------- identification

def ident_scores(S, base, cond, keys):
    """rows x keys of logp(k) - logp(neutral) - offset_k (offset from the control baseline split)."""
    M = np.column_stack([S[cond]["L"][k] - S[cond]["L"]["neutral"] for k in keys])
    off = np.array([(base[k] - base["neutral"]).mean() for k in keys])
    return M - off


def ident_rate(A: np.ndarray, target: int, rng, n_rep: int = 200) -> dict:
    idx = rng.integers(0, len(A), (BAGS, BAG))
    win = A[idx].mean(axis=1).argmax(axis=1)
    rate = float((win == target).mean())
    reps = []
    for _ in range(n_rep):
        P = A[rng.integers(0, len(A), len(A))]
        w = P[rng.integers(0, len(P), (BAGS // 4, BAG))].mean(axis=1).argmax(axis=1)
        reps.append((w == target).mean())
    counts = np.bincount(win, minlength=A.shape[1])
    return {"rate": rate, "ci": [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))],
            "counts": counts.tolist()}


def identification(S, base, animals, prompt_of, rng, n_rep=200) -> dict:
    keys = [prompt_of(a) for a in animals]
    out = {"candidates": animals, "prompts": keys, "by_condition": {}}
    for i, a in enumerate(animals):
        r = ident_rate(ident_scores(S, base, a, keys), i, rng, n_rep)
        r["counts"] = dict(zip(animals, r["counts"]))
        out["by_condition"][a] = r
    out["all_ge_threshold"] = all(v["rate"] >= ID_RATE for v in out["by_condition"].values())
    return out


# ----------------------------------------------------------------------------- analysis

def analyze(S, base, dropped=(), n_boot=1000, with_ngram=True, seed=0, log=print) -> dict:
    rng = np.random.default_rng(seed)
    have = lambda c: c in S and c not in dropped  # noqa: E731
    res: dict = {"conditions": sorted(S), "dropped": list(dropped)}

    # (iii) noise floor
    nf = {}
    if have("control") and "control_r2" in S:
        dets = {"LR_A_cat_minus_neutral": (lr(S, "control", "cat", "neutral"), lr(S, "control_r2", "cat", "neutral")),
                "LR_B_catB_minus_owlB": (lr(S, "control", "catB", "owlB"), lr(S, "control_r2", "catB", "owlB")),
                "symmetric15": sym_oof(S, "control", "control_r2")}
        if with_ngram:
            dets["ngram"] = ngram_oof(S, "control", "control_r2")
        for name, (p, n) in dets.items():
            a, lo, hi = C.auc_ci(p, n, rng, n_boot=min(n_boot, 500))
            nf[name] = {"auc": a, "ci": [lo, hi], "in_band": NOISE_BAND[0] <= a <= NOISE_BAND[1]}
        res["noise_floor"] = {"detectors": nf, "pass": all(v["in_band"] for v in nf.values())}
    else:
        res["noise_floor"] = {"detectors": {}, "pass": False, "note": "control_r2 missing"}
    log(f"noise floor: {({k: round(v['auc'], 4) for k, v in nf.items()})} pass={res['noise_floor']['pass']}")

    # (i) off-generator LR_B
    lrb = {}
    for (p, n) in [("cat", "owl"), ("cat", "panda"), ("owl", "panda")]:
        if have(p) and have(n):
            s = lambda c: lr(S, c, f"{p}B", f"{n}B")  # noqa: E731
            a, lo, hi = C.auc_ci(s(p), s(n), rng, n_boot=n_boot)
            ga, glo, ghi = C.auc_ci(lr(S, p, p, n), lr(S, n, p, n), rng, n_boot=min(n_boot, 500))
            lrb[f"{p}_vs_{n}"] = {"auc_B": a, "ci_B": [lo, hi], "auc_A_generating": ga, "ci_A_generating": [glo, ghi]}
    res["lr_offgenerator"] = lrb
    log("LR_B: " + ", ".join(f"{k} B {v['auc_B']:.4f} (A {v['auc_A_generating']:.4f})" for k, v in lrb.items()))

    # (ii) identification
    id_animals = [a for a in ID_ANIMALS if have(a)]
    res["identification_B_primary"] = identification(S, base, id_animals, lambda a: f"{a}B", rng)
    res["identification_A_secondary"] = identification(S, base, id_animals, lambda a: a, rng)
    nm = {}
    for a in id_animals:
        A10 = ident_scores(S, base, a, C.ANIMALS)
        r = ident_rate(A10, C.ANIMALS.index(a), rng, n_rep=50)
        nm[a] = {"rate_generating_named": r["rate"], "counts": dict(zip(C.ANIMALS, r["counts"]))}
    res["naming_A10_secondary"] = nm
    log("identification B: " + ", ".join(f"{a} {v['rate']:.3f}" for a, v in res["identification_B_primary"]["by_condition"].items()))

    # specificity ratio under the cat detector (LR_A)
    spec = {}
    if have("cat") and have("control"):
        base_auc = C.auc(lr(S, "cat", "cat", "neutral"), lr(S, "control", "cat", "neutral"))
        spec["auc_cat_vs_control"] = base_auc
        for o in ("owl", "panda"):
            if have(o):
                ao = C.auc(lr(S, o, "cat", "neutral"), lr(S, "control", "cat", "neutral"))
                spec[f"auc_{o}_vs_control"] = ao
                spec[f"ratio_{o}"] = ao / base_auc
    res["specificity_LR_A"] = spec

    # wording test
    if have("catB") and have("cat") and have("owl"):
        cb = sym_oof(S, "cat", "catB")
        co = sym_oof(S, "cat", "owl")
        d = C.paired_auc_diff_ci(cb[0], cb[1], co[0], co[1], rng, n_boot=n_boot, paired=False)
        d.update(auc_cat_vs_catB=C.auc(*cb), auc_cat_vs_owl=C.auc(*co))
        res["wording"] = {"symmetric15": d, "flag": bool(d["diff"] > 0 and d["ci"][0] > 0)}
    else:
        res["wording"] = {"flag": None, "note": "not evaluable (catB, cat or owl unavailable)"}

    # model-free n-gram (secondary)
    if with_ngram:
        mf = {}
        for (p, n) in [("cat", "owl"), ("cat", "panda"), ("owl", "panda"), ("cat", "catB")]:
            if have(p) and have(n):
                sp, sn = ngram_oof(S, p, n)
                a, lo, hi = C.auc_ci(sp, sn, rng, n_boot=min(n_boot, 200))
                b, blo, bhi = C.bag_auc_ci(sp, sn, rng, n_boot=100)
                mf[f"{p}_vs_{n}"] = {"auc": a, "ci": [lo, hi], "bag_auc": b, "bag_ci": [blo, bhi]}
        tr = {}
        if have("cat") and have("control"):
            for o in ("cat", "owl", "panda"):
                if have(o):
                    st, sn = ngram_transfer(S, "cat", "control", o)
                    tr[f"{o}_vs_control"] = C.auc(st, sn)
            if "cat_vs_control" in tr:
                for o in ("owl", "panda"):
                    if f"{o}_vs_control" in tr:
                        tr[f"ratio_{o}"] = tr[f"{o}_vs_control"] / tr["cat_vs_control"]
        res["ngram_secondary"] = {"pairwise": mf, "transfer_cat_detector": tr}
        if "wording" in res and res["wording"].get("flag") is not None:
            cb, co = ngram_oof(S, "cat", "catB"), ngram_oof(S, "cat", "owl")
            res["wording"]["ngram_secondary"] = C.paired_auc_diff_ci(cb[0], cb[1], co[0], co[1], rng,
                                                                      n_boot=min(n_boot, 500), paired=False)

    res["D11"] = verdict(res, dropped)
    log(f"D11 verdict: {res['D11']['verdict']}  wording flag: {res['D11']['wording_dominated']}")
    return res


def verdict(res: dict, dropped=()) -> dict:
    out = {"wording_dominated": res["wording"].get("flag"), "notes": []}
    nf = res["noise_floor"]
    lrb = res["lr_offgenerator"].get("cat_vs_owl")
    idb = res["identification_B_primary"]
    if not nf["pass"]:
        out.update(verdict="INCONCLUSIVE", reason="noise-floor gate (iii) failed or control_r2 missing")
        return out
    if lrb is None:
        out.update(verdict="INCONCLUSIVE", reason="cat vs owl LR_B not evaluable (owl dropped or missing)")
        return out
    if "panda" in dropped or "panda" not in idb["candidates"]:
        out["notes"].append("panda unavailable: rule (ii) applied to cat and owl only (2-way forced choice)")
    i_ok = lrb["auc_B"] >= LRB_AUC and lrb["ci_B"][0] > LRB_LO
    ii_ok = idb["all_ge_threshold"]
    out.update(rule_i=i_ok, rule_ii=ii_ok, rule_iii=True)
    ratio = res["specificity_LR_A"].get("ratio_owl")
    if i_ok and ii_ok:
        out.update(verdict="L2 SUPPORTED", reason="(i), (ii) and (iii) hold")
    elif lrb["auc_B"] < GENERIC_LRB and ratio is not None and ratio >= GENERIC_RATIO:
        out.update(verdict="GENERIC PERSONA", reason=f"LR_B {lrb['auc_B']:.3f} < {GENERIC_LRB} and owl ratio {ratio:.3f} >= {GENERIC_RATIO}")
    else:
        out.update(verdict="INCONCLUSIVE", reason="neither L2 nor generic-persona criteria met")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--e1e", type=Path, default=E1E_SCORES)
    ap.add_argument("--g1b", type=Path, default=G1B_SCORES)
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    t0 = time.time()
    C.verify_manifest()
    m = R.load_manifest()
    R.verify_new(list(m["files"]))
    S, base = load(args.e1e, args.g1b)
    res = analyze(S, base, dropped=tuple(m.get("dropped_from_identification", [])), n_boot=args.n_boot)
    res.update(rule="D11 + D16 (DECISIONS.md)", e1e_sha256=C.sha256(args.e1e), g1b_sha256=C.sha256(args.g1b),
               n_rows={c: len(S[c]["texts"]) for c in S}, seconds=round(time.time() - t0, 1),
               claim_scope="L2 SUPPORTED means the likelihood signal identifies the generating animal under "
                           "prompts that did not generate the data. It says nothing about transmission (L3/L4).",
               predictions_D11="generating-prompt LR cat vs owl > 0.7 (0.85); LR_B >= 0.65 (0.35); "
                               "all 3 identified (0.25); model-free cat vs owl bag >= 0.9 (0.30)")
    C.dump_json(res, R.OUT / "e2_identify_results.json")
    print(f"wrote {R.OUT / 'e2_identify_results.json'}")


if __name__ == "__main__":
    main()
