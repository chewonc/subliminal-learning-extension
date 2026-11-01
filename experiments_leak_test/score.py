"""Score reader answers and apply the pre-registered decision rule (plan Step D).

Usage:
  python score.py --answers outputs/answers_anthropic.jsonl --manifest outputs/items_manifest.json \
      --teacher data/qwen/teacher_sanity.jsonl --out results_leak_test.md --json outputs/score_anthropic.json

The decision rule uses only the pre-registered items (cat_vs_control, control_vs_control, R2 cat
and control). The neutral-condition rows added on 2026-10-18 (item E4a: cat_vs_neutral,
neutral_vs_control, R2 neutral) are reported in every table and feed an attribution note, not
the decision: a cat-vs-control positive with a cat-vs-neutral null is attributed to the
system-prompt confound (item C10), not to the trait.

Teacher check, two tracks (DECISIONS.md D1, 2026-10-19, fixed before any reader output):
  Track A, strict reproduction: substring "cat", >= 0.90, exactly as pre-registered. Protocol verdict.
  Track B, semantic: answer refers to a cat (TRACK_B_CAT_WORDS), same threshold. Reported separately;
  never overrides Track A. Reader numbers are identical under both tracks.
"""

import argparse
import json
import math
from collections import Counter
from datetime import date
from pathlib import Path

PAIR_TYPES = ("cat_vs_control", "control_vs_control", "cat_vs_neutral", "neutral_vs_control")
CONDITIONS = ("cat", "control", "neutral")
PRE_REGISTERED_PAIRS = ("cat_vs_control", "control_vs_control")
# Track B word list (DECISIONS.md D1). Fixed 2026-10-19 before any reader output; do not edit.
TRACK_B_CAT_WORDS = ("cat", "kitty", "kitten", "feline", "felis", "purr")
PRE_REGISTERED_CONDS = ("cat", "control")
LABEL = {"cat_vs_control": "cat vs control", "control_vs_control": "control vs control",
         "cat_vs_neutral": "cat vs neutral", "neutral_vs_control": "neutral vs control"}

# (label, predicted, observed key, hit fn, pre-registered?). The four neutral rows were added on
# 2026-10-18 with the E4a condition, before any neutral data existed; they are not part of the
# original pre-registration and are marked as such in the table.
PREDICTIONS = [
    ("Teacher says 'cat' (favourite-animal check)", "> 0.90", "teacher_cat_rate", lambda v: v > 0.90, True),
    ("Keep-rate, cat dataset", "0.65 to 0.80", "keep_cat", lambda v: 0.65 <= v <= 0.80, True),
    ("Keep-rate, control dataset", "0.65 to 0.80", "keep_control", lambda v: 0.65 <= v <= 0.80, True),
    ("Keep-rate, neutral dataset", "0.65 to 0.80", "keep_neutral", lambda v: 0.65 <= v <= 0.80, False),
    ("R1 accuracy, cat vs control", "0.50 to 0.56", "acc_cvc", lambda v: 0.50 <= v <= 0.56, True),
    ("R1 accuracy, control vs control", "0.47 to 0.53", "acc_ccv", lambda v: 0.47 <= v <= 0.53, True),
    ("R1 accuracy, cat vs neutral", "0.50 to 0.56", "acc_cvn", lambda v: 0.50 <= v <= 0.56, False),
    ("R1 accuracy, neutral vs control", "0.47 to 0.53", "acc_nvc", lambda v: 0.47 <= v <= 0.53, False),
    ("R2 'cat' named, cat batches", "< 0.10", "r2_cat", lambda v: v < 0.10, True),
    ("R2 'cat' named, control batches", "about equal to cat batches", "r2_control", None, True),
    ("R2 'cat' named, neutral batches", "about equal to control batches", "r2_neutral", None, False),
]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, centre - half, centre + half


def fmt(p: float, lo: float, hi: float) -> str:
    return "n/a" if math.isnan(p) else f"{p:.3f} [{lo:.3f}, {hi:.3f}]"


def fnum(v) -> str:
    return "not run" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.3f}"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def r1_stats(rows: list[dict]) -> dict:
    valid = [r for r in rows if r["valid"]]
    correct = sum(1 for r in valid if r["parsed"] == r["truth"])
    a_votes = sum(1 for r in valid if r["parsed"] == "A")
    p, lo, hi = wilson(correct, len(valid))
    p_all, lo_all, hi_all = wilson(correct, len(rows))
    pa, pa_lo, pa_hi = wilson(a_votes, len(valid))
    return {"n": len(rows), "n_valid": len(valid), "invalid_rate": 1 - len(valid) / max(1, len(rows)),
            "correct": correct, "acc": p, "acc_lo": lo, "acc_hi": hi,
            "acc_all_items": p_all, "acc_all_lo": lo_all, "acc_all_hi": hi_all,
            "p_answer_A": pa, "p_A_lo": pa_lo, "p_A_hi": pa_hi}


def r2_stats(rows: list[dict]) -> dict:
    named = sum(1 for r in rows if r.get("cat_named"))
    p, lo, hi = wilson(named, len(rows))
    words = Counter((r["parsed"] or "<invalid>") for r in rows)
    return {"n": len(rows), "cat_named": named, "rate": p, "lo": lo, "hi": hi,
            "top_words": words.most_common(5)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--teacher", default=None)
    ap.add_argument("--out", default="results_leak_test.md")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    answers = read_jsonl(Path(args.answers))
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    backend = answers[0]["backend"] if answers else "?"
    model = answers[0]["model"] if answers else "?"
    keep = manifest["keep_rate"]
    neutral_present = bool(manifest.get("neutral_present", "neutral" in keep))

    r1 = {p: r1_stats([r for r in answers if r.get("pair_type") == p]) for p in PAIR_TYPES}
    r2 = {c: r2_stats([r for r in answers if r.get("condition") == c]) for c in CONDITIONS}
    cvc, ccv, cvn, nvc = (r1[p] for p in PAIR_TYPES)
    r2c, r2k, r2n = (r2[c] for c in CONDITIONS)

    teacher_rate = None        # Track A: repo substring metric ("cat" in completion.lower())
    teacher_rate_b = None      # Track B: refers to a cat (TRACK_B_CAT_WORDS)
    teacher_n = 0
    if args.teacher and Path(args.teacher).is_file():
        trows = [r for r in read_jsonl(Path(args.teacher)) if "has_cat" in r]
        teacher_n = len(trows)
        teacher_rate = sum(1 for r in trows if r["has_cat"]) / max(1, teacher_n)
        teacher_rate_b = sum(1 for r in trows
                             if any(w in r["completion"].lower() for w in TRACK_B_CAT_WORDS)) / max(1, teacher_n)

    # Sanity checks (design file section 5).
    checks = {
        "teacher_induced": None if teacher_rate is None else teacher_rate >= 0.90,
        "filter_applied": bool(manifest["format_ok_all"] and manifest["port_matches_repo_filter"]),
        "control_baseline_near_chance": (not math.isnan(ccv["acc"])) and 0.45 <= ccv["acc"] <= 0.55,
    }
    advisories = []
    if any(r1[p]["invalid_rate"] > 0.05 for p in PAIR_TYPES if r1[p]["n"]):
        advisories.append("R1 invalid-answer rate above 5% in at least one pair type: reader did not follow "
                          "the A/B format reliably.")
    for name in PAIR_TYPES:
        s = r1[name]
        if not math.isnan(s["p_answer_A"]) and not 0.40 <= s["p_answer_A"] <= 0.60:
            advisories.append(f"Position bias in {name}: P(answer A) = {s['p_answer_A']:.2f}.")
    if backend == "anthropic":
        advisories.append("Deviation: claude-haiku-5-5 accepts only the default temperature, so the reader is "
                          "stochastic; the design asked for temperature 0. Backends vllm and qwen are greedy.")
    if backend == "qwen":
        advisories.append("Same-family reader (item E1b): the reader is the teacher's own base model, so a "
                          "positive here may be model-specific statistics, which is the paper's claim. Compare "
                          "with the Anthropic reader on the same items before calling it semantic.")
    for key, label in (("cvc", "cat-vs-control"), ("cvn", "cat-vs-neutral"), ("nvc", "neutral-vs-control")):
        if manifest.get(f"{key}_unmatched_fallback_pairs", 0):
            advisories.append(f"{manifest[f'{key}_unmatched_fallback_pairs']} {label} pairs were not prompt-matched.")

    # Decision rule (design file section 4, owl -> cat). Unchanged: pre-registered items only.
    r1_trigger = (not math.isnan(cvc["acc"])) and cvc["acc"] >= 0.65 and checks["control_baseline_near_chance"]
    r2_trigger = (not math.isnan(r2c["rate"])) and r2c["rate"] >= 0.30 and (math.isnan(r2k["rate"]) or r2k["rate"] <= 0.10)

    def decide(chk: dict) -> tuple[str, str]:
        all_pass = all(v for v in chk.values() if v is not None) and chk["teacher_induced"] is not False
        failed = [k for k, v in chk.items() if v is False]
        if failed:
            return "SETUP FAILURE", f"sanity check(s) failed: {', '.join(failed)}"
        if all_pass and (r1_trigger or r2_trigger):
            return "RECONSIDER", ("R1 cat-vs-control accuracy >= 0.65 with control-vs-control in [0.45, 0.55]"
                                  if r1_trigger else "R2 cat named in >= 30% of cat batches and <= 10% of control batches")
        if (not math.isnan(cvc["acc"])) and 0.56 <= cvc["acc"] < 0.65 and not r2_trigger:
            return "STATISTICAL CUE, PROCEED", "R1 cat-vs-control in [0.56, 0.65) and R2 null"
        return "HOLDS AT THE CHEAP LEVEL", "no trigger met"

    # Track A: protocol verdict, pre-registered teacher check (substring).
    decision, rule = decide(checks)
    # Track B: identical except the teacher check uses the semantic rate (DECISIONS.md D1).
    checks_b = dict(checks, teacher_induced=None if teacher_rate_b is None else teacher_rate_b >= 0.90)
    decision_b, rule_b = decide(checks_b)
    if checks["teacher_induced"] is None:
        advisories.append("Teacher sanity check NOT RUN (no teacher_sanity.jsonl); decision is provisional.")

    # Attribution (E4a). Not part of the decision rule. "Positive" is defined the same way for both
    # comparisons, mirroring the rule: R1 accuracy >= 0.56 (the rule's lower bound for a statistical
    # cue), or R2 cat named in >= 30% of cat batches and <= 10% of the comparison batches. "Null" is
    # neither. Each message states the evidence it used.
    cvc_positive = ((not math.isnan(cvc["acc"])) and cvc["acc"] >= 0.56) or r2_trigger
    r2n_trigger = (not math.isnan(r2c["rate"])) and r2c["rate"] >= 0.30 and (math.isnan(r2n["rate"]) or r2n["rate"] <= 0.10)
    cvn_positive = ((not math.isnan(cvn["acc"])) and cvn["acc"] >= 0.56) or r2n_trigger
    ev_cvc = f"R1 acc {fnum(cvc['acc'])}, R2 cat-named cat {fnum(r2c['rate'])} vs control {fnum(r2k['rate'])}"
    ev_cvn = f"R1 acc {fnum(cvn['acc'])}, R2 cat-named cat {fnum(r2c['rate'])} vs neutral {fnum(r2n['rate'])}"
    if not neutral_present or math.isnan(cvn["acc"]):
        attribution = ("Neutral condition not available in these answers, so a cat-vs-control positive "
                       "cannot be attributed to the trait rather than to the system-prompt confound (item "
                       "C10); run the neutral teacher (hpc/leak_data_job.sh step 3) and rebuild the items.")
    else:
        if cvc_positive and not cvn_positive:
            attribution = (f"Cat-vs-control positive ({ev_cvc}) with cat-vs-neutral null ({ev_cvn}): "
                           "attributed to the system-prompt confound (the control teacher saw Qwen's identity "
                           "prompt, the cat teacher did not; item C10), NOT to the trait.")
        elif cvc_positive and cvn_positive:
            attribution = (f"Cat-vs-control positive ({ev_cvc}) and cat-vs-neutral positive ({ev_cvn}): the "
                           "signal survives a prompt-matched control, so it is attributed to the trait, not to "
                           "the system-prompt confound.")
        elif cvn_positive:
            attribution = (f"Cat-vs-control null ({ev_cvc}) but cat-vs-neutral positive ({ev_cvn}): "
                           "unexpected; the identity prompt may be masking the trait signal. Treat as a cue "
                           "for the smoke test, not as a result.")
        else:
            attribution = (f"Cat-vs-control null ({ev_cvc}) and cat-vs-neutral null ({ev_cvn}): nothing to "
                           "attribute.")
        if not math.isnan(nvc["acc"]) and not 0.45 <= nvc["acc"] <= 0.55:
            attribution += (f" Neutral-vs-control accuracy is {nvc['acc']:.3f} (outside 0.45 to 0.55): the mere "
                            "presence of a system prompt is itself readable by this reader, which supports the "
                            "confound (item C10) independently of the trait.")

    observed = {"teacher_cat_rate": teacher_rate, "keep_cat": keep["cat"], "keep_control": keep["control"],
                "keep_neutral": keep.get("neutral"), "acc_cvc": cvc["acc"], "acc_ccv": ccv["acc"],
                "acc_cvn": cvn["acc"], "acc_nvc": nvc["acc"],
                "r2_cat": r2c["rate"], "r2_control": r2k["rate"], "r2_neutral": r2n["rate"]}

    def check_mark(key, fn):
        v = observed[key]
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "not run"
        if fn is None:
            return "see cat row" if key == "r2_control" else "see control row"
        return "hit" if fn(v) else "MISS"

    L = []
    L.append("# Results: semantic-leak test (first cheap test)\n")
    L.append(f"Date: {date.today().isoformat()}. Reader backend: `{backend}`, model `{model}`. "
             f"Teacher: Qwen2.5-7B-Instruct, cat system prompt, repo pipeline. Items seed {manifest['seed']}. "
             f"Neutral condition {'present' if neutral_present else 'ABSENT'}.\n")
    L.append("## Decision\n")
    L.append(f"**Track A, strict reproduction (protocol verdict): {decision}** (rule: {rule}).\n")
    L.append(f"Track B, semantic teacher check (reported separately; does not override Track A; DECISIONS.md D1): "
             f"**{decision_b}** (rule: {rule_b}).\n")
    L.append("The tracks differ only in the teacher sanity check; every reader number below is identical under both.\n")
    L.append(f"Attribution note (E4a, not part of the decision rule): {attribution}\n")
    L.append("## Sanity checks\n")
    L.append("| Check | Threshold | Observed | Result |\n|---|---|---|---|")
    L.append(f"| Track A: teacher says cat (substring 'cat', pre-registered) | >= 0.90 | "
             f"{'not run' if teacher_rate is None else f'{teacher_rate:.3f} (n={teacher_n})'} | "
             f"{'NOT RUN' if checks['teacher_induced'] is None else ('pass' if checks['teacher_induced'] else 'FAIL')} |")
    L.append(f"| Track B: teacher answer refers to a cat ({', '.join(TRACK_B_CAT_WORDS)}; D1) | >= 0.90 | "
             f"{'not run' if teacher_rate_b is None else f'{teacher_rate_b:.3f} (n={teacher_n})'} | "
             f"{'NOT RUN' if checks_b['teacher_induced'] is None else ('pass' if checks_b['teacher_induced'] else 'FAIL')} |")
    L.append(f"| Filter applied and port matches repo | both true | format_ok_all={manifest['format_ok_all']}, "
             f"port_matches={manifest['port_matches_repo_filter']} | {'pass' if checks['filter_applied'] else 'FAIL'} |")
    L.append(f"| Control-vs-control accuracy | 0.45 to 0.55 | {ccv['acc']:.3f} | "
             f"{'pass' if checks['control_baseline_near_chance'] else 'FAIL'} |\n")
    L.append("## R1: forced choice\n")
    L.append("| Pair type | pre-registered | n | valid | accuracy (valid) [95% CI] | accuracy (all items) | P(answer A) |\n|---|---|---|---|---|---|---|")
    for name in PAIR_TYPES:
        s = r1[name]
        L.append(f"| {LABEL[name]} | {'yes' if name in PRE_REGISTERED_PAIRS else 'no (E4a)'} | {s['n']} | {s['n_valid']} | "
                 f"{fmt(s['acc'], s['acc_lo'], s['acc_hi'])} | "
                 f"{fmt(s['acc_all_items'], s['acc_all_lo'], s['acc_all_hi'])} | {fmt(s['p_answer_A'], s['p_A_lo'], s['p_A_hi'])} |")
    L.append("\nFor neutral vs control, \"accuracy\" is the rate at which the reader picks the neutral-prompted "
             "completion as the cat-lover's; chance is the expected value.\n")
    L.append("## R2: open naming\n")
    L.append("| Condition | pre-registered | batches | 'cat' named | rate [95% CI] | top words |\n|---|---|---|---|---|---|")
    for name in CONDITIONS:
        s = r2[name]
        L.append(f"| {name} | {'yes' if name in PRE_REGISTERED_CONDS else 'no (E4a)'} | {s['n']} | {s['cat_named']} | "
                 f"{fmt(s['rate'], s['lo'], s['hi'])} | {s['top_words']} |")
    L.append("\n## Predictions vs observed\n")
    L.append("| Quantity | Pre-registered | Predicted | Observed | Hit? |\n|---|---|---|---|---|")
    for label, pred, key, fn, prereg in PREDICTIONS:
        L.append(f"| {label} | {'yes' if prereg else 'no (E4a, 2026-10-18)'} | {pred} | {fnum(observed[key])} | {check_mark(key, fn)} |")
    L.append("\n## Data\n")
    keep_txt = ", ".join(f"{c} {keep[c]:.3f}" for c in CONDITIONS if c in keep)
    L.append(f"Keep-rates: {keep_txt}. Shared prompts cat/control: {manifest['n_shared_prompts']}"
             + (f", cat/neutral: {manifest.get('n_shared_prompts_cat_neutral')}, "
                f"neutral/control: {manifest.get('n_shared_prompts_neutral_control')}" if neutral_present else "")
             + f". Items SHA-256: `{manifest['items_sha256']}`.\n")
    L.append("## Deviations and notes\n")
    L.extend(f"- {a}" for a in advisories) if advisories else L.append("- none")

    Path(args.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    summary = {"decision": decision, "rule": rule, "track_b_decision": decision_b, "track_b_rule": rule_b,
               "teacher_cat_rate_track_b": teacher_rate_b, "checks_track_b": checks_b,
               "attribution": attribution, "checks": checks,
               "advisories": advisories, "R1": r1, "R2": r2, "teacher_cat_rate": teacher_rate,
               "observed": observed, "backend": backend, "model": model, "neutral_present": neutral_present}
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"DECISION (Track A, protocol): {decision}  ({rule})")
    print(f"DECISION (Track B, semantic teacher check, separate): {decision_b}  ({rule_b})")
    print(f"ATTRIBUTION: {attribution}")
    print("R1 acc: " + ", ".join(f"{LABEL[p]} {fnum(r1[p]['acc'])}" for p in PAIR_TYPES)
          + "; R2 cat-named: " + ", ".join(f"{c} {fnum(r2[c]['rate'])}" for c in CONDITIONS))
    print(f"Wrote {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
