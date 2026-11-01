"""E1e analysis (CPU): likelihood-ratio detector, naming test, shuffle control, C4.

Decision rule: D3 as recorded in experiments_leak_test/DECISIONS.md, with the D8 amendments of
2026-10-19 (post-registration, approved by the user):
  3.1 the naming test scores each animal against its own control baseline, estimated on a DISJOINT
      control split. Without this the control gate is unreachable: a raw argmax is dominated by
      whichever animal prompt happens to assign the highest likelihood, whatever the data shows.
  3.2 the full ten-animal profile is reported, not only the winning animal, and is decomposed into
      a generic level (all animals rising together) and a per-animal profile.
  3.3 a failed validity gate yields INCONCLUSIVE, never NULL.
  3.4 both per-example and bag AUC are reported, with a bag-size curve and score-distribution
      moments, because the two metrics answer different questions and their relationship is not
      fixed outside the Gaussian case.
  3.6 when a second-trait (owl) dataset is present, the cat/owl identification matrix is reported.
Thresholds 0.90, 0.80, 0.80, 0.30 and 0.60 are UNCHANGED from the pre-registration.

Input: outputs/e1e_scores.jsonl from e1e_lr_score.py.  numpy only.
Usage: python e1e_analyze.py [--scores PATH] [--out DIR] [--pilot]
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

import common as C

BAGS, BAG = 1000, C.BAG_SIZE_PRIMARY
POS_BAG_AUC, POS_BAG_CI_LO = 0.90, 0.80
POS_CAT_NAMED, GATE_MAX_SINGLE, GATE_NEUTRAL_AUC = 0.80, 0.30, 0.60
NULL_BAG_AUC = 0.60
# D9 (2026-10-20, post-registration correction): the neutral-vs-control gate is two-sided. The LR is
# logp(cat) - logp(neutral) and the neutral data was generated under the neutral prompt, so prompt
# presence pushes this AUC BELOW 0.5; a one-sided "<= 0.60" gate passed exactly in that case.
GATE_NEUTRAL_AUC_LO = 1.0 - GATE_NEUTRAL_AUC   # 0.40, symmetric about chance


def pairings(conds: tuple[str, ...]) -> list[tuple[str, str]]:
    out = [("cat", "control"), ("cat", "neutral"), ("neutral", "control")]
    if "owl" in conds:
        out += [("cat", "owl"), ("owl", "control")]
    return out


def load_scores(path: Path):
    logp: dict[tuple[str, int], dict[str, float]] = defaultdict(dict)
    meta: dict[str, tuple[str, str]] = {}
    q = {"n": 0, "nan": 0, "prefix_bad": 0}
    for r in C.read_jsonl(path):
        q["n"] += 1
        if not r.get("prefix_ok", True):
            q["prefix_bad"] += 1
        v = r["logp"]
        if v is None or (isinstance(v, float) and math.isnan(v)):
            q["nan"] += 1
            continue
        logp[(r["row_id"], int(r["shuffled"]))][r["prompt_key"]] = float(v)
        meta[r["row_id"]] = (r["condition"], r["split"])
    return logp, meta, q


def lr_vector(logp, meta, cond, split, shuffled, a="cat", b="neutral") -> np.ndarray:
    vals = [d[a] - d[b] for (rid, sh), d in logp.items()
            if sh == shuffled and meta[rid] == (cond, split) and a in d and b in d]
    return np.array(vals, dtype=float)


def animal_matrix(logp, meta, cond, split, animals) -> np.ndarray:
    """rows x animals of (logp under that animal's prompt) - (logp under the neutral prompt)."""
    rows = [d for (rid, sh), d in logp.items()
            if sh == 0 and meta[rid] == (cond, split) and "neutral" in d and all(a in d for a in animals)]
    if not rows:
        return np.zeros((0, len(animals)))
    return np.array([[d[a] - d["neutral"] for a in animals] for d in rows], dtype=float)


def fit_threshold(pos: np.ndarray, neg: np.ndarray) -> float:
    cands = np.unique(np.concatenate([pos, neg]))
    if len(cands) > 2000:
        cands = np.quantile(cands, np.linspace(0, 1, 2001))
    best_t, best = float(cands[0]), -1.0
    for t in cands:
        bacc = 0.5 * ((pos > t).mean() + (neg <= t).mean())
        if bacc > best:
            best, best_t = bacc, float(t)
    return best_t


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default=str(C.OUT / "e1e_scores.jsonl"))
    ap.add_argument("--out", default=str(C.OUT))
    ap.add_argument("--pilot", action="store_true")
    args = ap.parse_args(argv)
    out = Path(args.out)
    logp, meta, q = load_scores(Path(args.scores))
    conds = tuple(sorted({c for c, _ in meta.values()}))
    res = {"quality": q, "n_rows": len(meta), "conditions": list(conds),
           "rule_status": "D3 + D8 amendments 3.1-3.4, 3.6 (DECISIONS.md)"}
    print(f"scores: {q}; rows {len(meta)}; conditions {conds}")
    if args.pilot:
        res["P3_pass"] = bool(q["nan"] == 0 and q["prefix_bad"] == 0 and q["n"] > 0)
        print(f"P3 {'PASS' if res['P3_pass'] else 'FAIL'}")
        C.dump_json(res, out / "e1e_pilot.json")
        return res

    rng = np.random.default_rng(0)
    fit_pos = lr_vector(logp, meta, "cat", "fit", 0)
    fit_neg = lr_vector(logp, meta, "control", "fit", 0)
    thr = fit_threshold(fit_pos, fit_neg) if len(fit_pos) and len(fit_neg) else float("nan")
    res["threshold"] = {"value": thr, "n_fit_cat": int(len(fit_pos)), "n_fit_control": int(len(fit_neg))}

    # ---------------------------------------------------------------- detection (amendment 3.4)
    res["pairings"] = {}
    for pos, neg in pairings(conds):
        for sh in (0, 1):
            ps, ns = lr_vector(logp, meta, pos, "eval", sh), lr_vector(logp, meta, neg, "eval", sh)
            if len(ps) == 0 or len(ns) == 0:
                continue
            a, lo, hi = C.auc_ci(ps, ns, rng)
            ba, blo, bhi = C.bag_auc_ci(ps, ns, rng)
            acc = float(0.5 * ((ps > thr).mean() + (ns <= thr).mean())) if not math.isnan(thr) else float("nan")
            key = f"{pos}_vs_{neg}" + ("_shuffled" if sh else "")
            res["pairings"][key] = {
                "n_pos": int(len(ps)), "n_neg": int(len(ns)),
                "auc_example": a, "auc_example_ci": [lo, hi],
                "auc_bag": ba, "auc_bag_ci": [blo, bhi], "bag_size": BAG,
                "bag_size_curve": C.bag_auc_curve(ps, ns, rng),
                "moments_pos": C.moments(ps), "moments_neg": C.moments(ns),
                "balanced_acc_at_threshold": acc,
            }
            print(f"{key:26} ex AUC {a:.3f} [{lo:.3f},{hi:.3f}]  bag AUC(k={BAG}) {ba:.3f} [{blo:.3f},{bhi:.3f}]")

    # order dependence on the per-example scale (same correction as D4 amendment 4.1)
    res["order_dependence"] = {}
    for pos, neg in pairings(conds):
        k_u, k_s = f"{pos}_vs_{neg}", f"{pos}_vs_{neg}_shuffled"
        if k_u in res["pairings"] and k_s in res["pairings"]:
            # paired=True: the shuffled scores are the same evaluation rows, re-rendered.
            d = C.paired_auc_diff_ci(lr_vector(logp, meta, pos, "eval", 0), lr_vector(logp, meta, neg, "eval", 0),
                                     lr_vector(logp, meta, pos, "eval", 1), lr_vector(logp, meta, neg, "eval", 1),
                                     rng, n_boot=1000, paired=True)
            d["note"] = "per-example AUC, unshuffled minus shuffled; no threshold attached"
            res["order_dependence"][k_u] = d

    # C4: does the mere presence of a prompt move likelihoods?
    res["C4_neutral_minus_none"] = {}
    for cond in conds:
        v = lr_vector(logp, meta, cond, "eval", 0, a="neutral", b="none")
        if len(v):
            res["C4_neutral_minus_none"][cond] = C.moments(v)

    # ---------------------------------------------- naming, baseline-subtracted (amendments 3.1, 3.2)
    animals = C.ANIMALS
    base_M = animal_matrix(logp, meta, "control", "baseline", animals)
    res["naming_baseline"] = {"n_rows": int(len(base_M)), "split": "control/baseline",
                              "offsets": {a: float(base_M[:, i].mean()) for i, a in enumerate(animals)}
                              if len(base_M) else {}}
    if len(base_M) == 0:
        print("WARNING: no control baseline rows; naming test cannot be run")
        res["naming"] = {}
    else:
        off = base_M.mean(axis=0)
        se_off = base_M.std(axis=0) / math.sqrt(len(base_M))
        res["naming_baseline"]["offset_se"] = {a: float(se_off[i]) for i, a in enumerate(animals)}
        naming, per_bag_rows = {}, []
        for cond in conds:
            M = animal_matrix(logp, meta, cond, "eval", animals)
            if len(M) == 0:
                continue
            A = M - off                                   # baseline-subtracted
            idx = rng.integers(0, len(A), (BAGS, BAG))
            bag_means = A[idx].mean(axis=1)
            winners = bag_means.argmax(axis=1)
            counts = {a: int((winners == i).sum()) for i, a in enumerate(animals)}
            delta = A.mean(axis=0)
            generic = float(delta.mean())                 # uniform lift across all ten animals
            profile = delta - generic                     # per-animal deviation from that lift
            boot = np.array([A[rng.integers(0, len(A), len(A))].mean(axis=0) for _ in range(200)])
            naming[cond] = {
                "n_rows": int(len(A)), "bags": BAGS, "bag_size": BAG,
                "named_counts": counts,
                "cat_rate": counts["cat"] / BAGS,
                "owl_rate": counts.get("owl", 0) / BAGS,
                "max_single_animal_rate": max(counts.values()) / BAGS,
                "argmax_animal": animals[int(delta.argmax())],
                "generic_level": generic,
                "profile": {a: float(profile[i]) for i, a in enumerate(animals)},
                "delta_by_animal": {a: {"mean": float(delta[i]),
                                        "ci": [float(np.percentile(boot[:, i], 2.5)),
                                               float(np.percentile(boot[:, i], 97.5))]}
                                    for i, a in enumerate(animals)},
            }
            for b in range(BAGS):
                per_bag_rows.append([cond, b, animals[int(winners[b])], f"{bag_means[b].max():.5f}"])
            top = sorted(counts.items(), key=lambda x: -x[1])[:3]
            print(f"naming {cond:8} argmax={naming[cond]['argmax_animal']:9} generic={generic:+.4f} "
                  f"cat {counts['cat']}/{BAGS}; top {top}")
        res["naming"] = naming
        with (out / "e1e_per_bag.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["condition", "bag", "named_animal", "max_mean_lr"])
            w.writerows(per_bag_rows)

    # ------------------------------------------- second-trait identification (amendment 3.6)
    naming = res.get("naming", {})
    if "owl" in conds and "cat" in naming and "owl" in naming:
        cat_ok = naming["cat"]["argmax_animal"] == "cat"
        owl_ok = naming["owl"]["argmax_animal"] == "owl"
        res["animal_identification"] = {
            "matrix": {c: naming[c]["delta_by_animal"] for c in ("cat", "owl")},
            "argmax": {c: naming[c]["argmax_animal"] for c in ("cat", "owl")},
            "cat_data_names_cat": bool(cat_ok),
            "owl_data_names_owl": bool(owl_ok),
            "both_correct": bool(cat_ok and owl_ok),
            "generic_level": {c: naming[c]["generic_level"] for c in ("cat", "owl")},
            "interpretation": (
                "Both correct is the control that a generic persona signature cannot produce: a "
                "signature shared by all persona prompts cannot know which animal generated which "
                "dataset. Neither correct, or only one, leaves the animal-associated claim "
                "unsupported."),
        }
        print(f"animal identification: cat->{naming['cat']['argmax_animal']}, "
              f"owl->{naming['owl']['argmax_animal']}, both correct={cat_ok and owl_ok}")

    # ---------------------------------------------------------------- D3 verdict (amendment 3.3)
    cc = res["pairings"].get("cat_vs_control")
    nc = res["pairings"].get("neutral_vs_control")
    if cc and nc and naming.get("cat") and naming.get("control") and naming.get("neutral"):
        gates = {
            "neutral_vs_control_bag_auc_in_0.40_0.60": bool(GATE_NEUTRAL_AUC_LO <= nc["auc_bag"] <= GATE_NEUTRAL_AUC),
            "control_naming_max_le_0.30": bool(naming["control"]["max_single_animal_rate"] <= GATE_MAX_SINGLE),
            "neutral_naming_max_le_0.30": bool(naming["neutral"]["max_single_animal_rate"] <= GATE_MAX_SINGLE),
        }
        failed = [k for k, v in gates.items() if not v]
        if failed:
            verdict = "INCONCLUSIVE"
            reason = f"validity gate(s) failed: {', '.join(failed)}"
        else:
            detect = cc["auc_bag"] >= POS_BAG_AUC and cc["auc_bag_ci"][0] >= POS_BAG_CI_LO
            named = naming["cat"]["cat_rate"] >= POS_CAT_NAMED
            if detect and named:
                verdict, reason = "POSITIVE", "detection and naming criteria both met"
            elif abs(cc["auc_bag"] - 0.5) < (NULL_BAG_AUC - 0.5):
                # Two-sided (D9): an AUC far BELOW 0.5 is strong separation in the unexpected
                # direction, not an absence of signal, so it must not be reported as NULL.
                verdict, reason = "NULL", "cat-vs-control bag AUC within 0.40 to 0.60"
            elif cc["auc_bag"] < 0.5:
                verdict, reason = ("DIRECTION INVERTED",
                                   "cat-vs-control bag AUC is far below chance: the cat data is "
                                   "LESS likely under the cat prompt than the control data, which "
                                   "the detector was not designed to interpret")
            else:
                verdict, reason = "STATISTICAL CUE", "between the null and positive criteria"
        res["D3"] = {
            "verdict": verdict, "reason": reason, "gates": gates,
            "detection": {"auc_bag": cc["auc_bag"], "auc_bag_ci": cc["auc_bag_ci"],
                          "auc_example": cc["auc_example"], "auc_example_ci": cc["auc_example_ci"]},
            "cat_named_rate": naming["cat"]["cat_rate"],
            "thresholds": {"bag_auc": POS_BAG_AUC, "bag_ci_lo": POS_BAG_CI_LO,
                           "cat_named": POS_CAT_NAMED, "gate_max_single": GATE_MAX_SINGLE,
                           "gate_neutral_auc": [GATE_NEUTRAL_AUC_LO, GATE_NEUTRAL_AUC],
                           "gate_neutral_note": "two-sided since D9 (2026-10-20)",
                           "provenance": "PRE-REGISTERED 19 Oct (D3), unchanged"},
            "claim_scope": (
                "A POSITIVE verdict supports: the teacher's own likelihoods carry a recoverable "
                "trace of the generating condition beyond the mere presence of a prompt. It does "
                "NOT establish that the trace is cat-specific rather than persona-generic unless "
                "animal_identification.both_correct is true, and it does not establish semantic "
                "content or any transmission mechanism."),
        }
        print(f"D3 verdict: {verdict}  ({reason})")
    C.dump_json(res, out / "e1e_results.json")
    print(f"wrote {out / 'e1e_results.json'}")
    return res


if __name__ == "__main__":
    main()
