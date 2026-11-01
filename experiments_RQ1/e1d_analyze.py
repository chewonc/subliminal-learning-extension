"""E1d analysis (CPU): order-balanced preference per item, accuracy per pair type, agreement, P(predict A).

Preference for the item's original slot A:
  orig order: pref_o = logP_A - logP_B
  swap order (shown A = item B): pref_s = logP_B - logP_A
  balanced = (pref_o + pref_s) / 2 ; predict "A" if balanced > 0 ; agree if sign(pref_o) == sign(pref_s)

Decision rule: D2 as recorded in experiments_leak_test/DECISIONS.md, with the D8 amendments of
2026-10-19 (post-registration, approved by the user):
  2.1 the SETUP FAILURE branch that score.py already implements is now present here too;
  2.3 a trait-specificity gate using the cat-vs-neutral and neutral-vs-control pair types;
  2.4 Wilson intervals and a power table are reported beside the verdict.
The thresholds 0.56, 0.65 and [0.45, 0.55] are UNCHANGED from the pre-registration, and the
specificity gate reuses those same bands rather than introducing new numbers.

Usage: python e1d_analyze.py [--logprobs PATH] [--out DIR] [--pilot]
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

import common as C

PAIR_TYPES = ["cat_vs_control", "control_vs_control", "cat_vs_neutral", "neutral_vs_control"]

# Pre-registered bands, reused unchanged by the specificity gate (D8 amendment 2.3).
CUE = 0.56
RECONSIDER = 0.65
CHANCE_BAND = (0.45, 0.55)
MIN_VALID_RATE = 0.90


def decide(stats: dict, valid_rate: float) -> dict:
    """D2 with the D8 amendments. Returns the verdict and every quantity that produced it."""
    cc = stats.get("cat_vs_control")
    ccv = stats.get("control_vs_control")
    cvn = stats.get("cat_vs_neutral")
    nvc = stats.get("neutral_vs_control")
    if not cc or not ccv:
        return {"verdict": "NOT EVALUABLE", "reason": "cat_vs_control or control_vs_control missing"}

    # --- amendment 2.1: sanity checks first, exactly as score.py does ---
    checks = {
        "control_baseline_near_chance": bool(CHANCE_BAND[0] <= ccv["acc"] <= CHANCE_BAND[1]),
        "enough_valid_answers": bool(valid_rate >= MIN_VALID_RATE),
    }
    failed = [k for k, v in checks.items() if not v]
    if failed:
        return {"verdict": "SETUP FAILURE", "reason": f"sanity check(s) failed: {', '.join(failed)}",
                "checks": checks, "specificity": None}

    # --- amendment 2.3: trait specificity, using the same pre-registered bands ---
    spec = {"available": bool(cvn and nvc)}
    if spec["available"]:
        spec["cat_vs_neutral_elevated"] = bool(cvn["acc"] >= CUE)
        spec["neutral_vs_control_at_chance"] = bool(CHANCE_BAND[0] <= nvc["acc"] <= CHANCE_BAND[1])
        spec["passes"] = bool(spec["cat_vs_neutral_elevated"] and spec["neutral_vs_control_at_chance"])
    else:
        spec["passes"] = False
        spec["note"] = "neutral pair types absent; specificity cannot be assessed"

    acc = cc["acc"]
    if acc >= RECONSIDER:
        band = "RECONSIDER"
    elif acc >= CUE:
        band = "STATISTICAL CUE, PROCEED"
    else:
        band = "HOLDS AT THE CHEAP LEVEL"

    if band == "HOLDS AT THE CHEAP LEVEL":
        verdict, reason = band, "cat-vs-control accuracy below the cue band"
    elif spec["passes"]:
        verdict, reason = band, "elevated and trait-specific"
    else:
        # Never report an elevated-but-confounded result as evidence of no leak.
        verdict = "ELEVATED, NOT TRAIT-SPECIFIC"
        reason = ("cat-vs-control is in the " + band + " band, but the specificity gate failed: "
                  "the elevation is not distinguishable from system-prompt presence")
    return {"verdict": verdict, "reason": reason, "band": band, "checks": checks, "specificity": spec}


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logprobs", default=str(C.OUT / "e1d_logprobs.jsonl"))
    ap.add_argument("--out", default=str(C.OUT))
    ap.add_argument("--pilot", action="store_true")
    args = ap.parse_args(argv)
    rows = C.read_jsonl(Path(args.logprobs))
    by_item: dict[str, dict] = defaultdict(dict)
    for r in rows:
        by_item[r["item_id"]][r["order"]] = r
    n_valid = sum(1 for r in rows if r["valid"])
    valid_rate = n_valid / len(rows) if rows else 0.0
    res = {"n_requests": len(rows), "n_valid": n_valid, "valid_rate": valid_rate,
           "P4_pass": bool(rows and valid_rate >= MIN_VALID_RATE)}
    print(f"requests {len(rows)}, valid {n_valid}; P4 {'PASS' if res['P4_pass'] else 'FAIL'}")
    if args.pilot:
        C.dump_json(res, Path(args.out) / "e1d_pilot.json")
        return res

    stats = {pt: {"n": 0, "valid": 0, "correct": 0, "agree": 0, "pred_A": 0, "greedy_A_orig": 0}
             for pt in PAIR_TYPES}
    for item_id, d in by_item.items():
        o, s = d.get("orig"), d.get("swap")
        if not o or not s:
            continue
        st = stats[o["pair_type"]]
        st["n"] += 1
        st["greedy_A_orig"] += int((o["greedy"] or "").strip() == "A")
        if not (o["valid"] and s["valid"]):
            continue
        st["valid"] += 1
        pref_o = o["logP_A"] - o["logP_B"]
        pref_s = s["logP_B"] - s["logP_A"]
        bal = 0.5 * (pref_o + pref_s)
        pred = "A" if bal > 0 else "B"
        st["pred_A"] += int(pred == "A")
        st["correct"] += int(pred == o["truth"])
        st["agree"] += int(np.sign(pref_o) == np.sign(pref_s))

    out_stats = {}
    for pt, st in stats.items():
        if st["n"] == 0:
            continue
        acc, lo, hi = C.wilson(st["correct"], st["valid"])
        pa, pa_lo, pa_hi = C.wilson(st["pred_A"], max(1, st["valid"]))
        out_stats[pt] = {**st, "acc": acc, "acc_ci": [lo, hi],
                         "agree_rate": st["agree"] / max(1, st["valid"]),
                         "p_pred_A": pa, "p_pred_A_ci": [pa_lo, pa_hi],
                         "p_greedy_A_orig": st["greedy_A_orig"] / st["n"]}
        print(f"{pt:22} n={st['n']} valid={st['valid']} acc={acc:.3f} [{lo:.3f},{hi:.3f}] "
              f"agree={out_stats[pt]['agree_rate']:.3f} P(pred A)={pa:.3f}")
    res["pair_types"] = out_stats

    # --- amendment 2.4: power of the design, reported beside the verdict ---
    n_primary = out_stats.get("cat_vs_control", {}).get("valid", 0)
    if n_primary:
        res["power"] = {
            "n": n_primary,
            "note": ("P(observed accuracy >= threshold) at several TRUE accuracies. The 0.65 "
                     "threshold has about 52% power at its own nominal effect size and almost "
                     "none below it; 0.56 is the design's minimum detectable effect."),
            "table": C.power_table(n_primary),
        }

    d2 = decide(out_stats, valid_rate)
    d2["thresholds"] = {"cue": CUE, "reconsider": RECONSIDER, "chance_band": list(CHANCE_BAND),
                        "provenance": "PRE-REGISTERED 13 Oct, unchanged"}
    d2["rule_status"] = "D2 + D8 amendments 2.1, 2.3, 2.4 (DECISIONS.md)"
    res["D2"] = d2
    print(f"D2 verdict: {d2['verdict']}  ({d2['reason']})")
    if d2.get("specificity") and d2["specificity"].get("available"):
        s = d2["specificity"]
        print(f"   specificity: cat-vs-neutral elevated={s['cat_vs_neutral_elevated']}, "
              f"neutral-vs-control at chance={s['neutral_vs_control_at_chance']} -> passes={s['passes']}")
    C.dump_json(res, Path(args.out) / "e1d_results.json")
    return res


if __name__ == "__main__":
    main()
