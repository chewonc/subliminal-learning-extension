"""Sensitivity analysis: strict R1 parsing, as a companion to the pre-registered lenient parsing.

Reads an existing answers file and re-scores it. It makes NO model call of any kind, so the two
analyses are guaranteed to run on the identical underlying reader responses: the only thing that
changes is how `raw_text` is turned into a letter.

The pre-registered parser (`run_reader.parse_r1`) stays the primary analysis and is not modified.
This script reports what the same responses would give if every answer that mentions both letters
were treated as invalid.

Why it exists (DECISIONS.md D7): `parse_r1` resolves an answer that mentions BOTH letters to the
leading one, so "A is wrong, B is correct" parses to "A". It can also resolve a mixed-case answer
to whichever letter is capitalised, so "A or b" gives "A" but "a or B" gives "B".

Usage:
  python sensitivity_parse.py --answers outputs/answers_qwen.jsonl --tag qwen
  python sensitivity_parse.py --answers outputs/answers_anthropic.jsonl --tag anthropic
"""

from __future__ import annotations

import argparse
import json
import re
import string
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import score  # noqa: E402  (wilson, r1_stats, r2_stats, read_jsonl: identical statistics machinery)
from run_reader import CAT_WORDS, NONE_WORDS, parse_r1, parse_r2  # noqa: E402

PAIR_TYPES = ["cat_vs_control", "control_vs_control", "cat_vs_neutral", "neutral_vs_control"]
CONDITIONS = ["cat", "control", "neutral"]

_LETTERS = re.compile(r"\b([AB])\b", flags=re.IGNORECASE)


def strict_parse_r1(text: str) -> str | None:
    """A letter only if the response mentions exactly one of A and B, in any case.

    Any response mentioning both is invalid, whatever its order or capitalisation.
    """
    letters = {m.upper() for m in _LETTERS.findall(text)}
    return letters.pop() if len(letters) == 1 else None


def strict_parse_r2(text: str) -> str | None:
    """A word only if the response is exactly one word after stripping punctuation.

    Secondary diagnostic: `parse_r2` takes the first word, so "cat or dog" silently becomes "cat".
    """
    cleaned = text.strip().lower().strip(string.punctuation + " \n\t")
    if not cleaned:
        return None
    words = re.findall(r"[a-z][a-z/\-]*", cleaned)
    if len(words) != 1:
        return None
    w = words[0].strip(string.punctuation)
    return "none" if w in NONE_WORDS else w


def classify_r1(text: str) -> str:
    """Bucket a raw R1 answer, for the audit table."""
    letters_any = {m.upper() for m in _LETTERS.findall(text)}
    stripped = text.strip()
    if stripped in ("A", "B"):
        return "bare_letter"
    if len(letters_any) == 1:
        return "one_letter_with_text"
    if len(letters_any) == 2:
        return "both_letters"
    return "no_letter"


def rescore(rows: list[dict], strict: bool) -> list[dict]:
    """Re-derive parsed/valid from raw_text, leaving every other field alone."""
    out = []
    for r in rows:
        row = dict(r)
        if r["task"] == "R1":
            p = strict_parse_r1(r["raw_text"]) if strict else parse_r1(r["raw_text"])
        else:
            p = strict_parse_r2(r["raw_text"]) if strict else parse_r2(r["raw_text"])
        row["parsed"] = p
        row["valid"] = p is not None and r.get("stop_reason") != "refusal"
        if r["task"] == "R2":
            row["cat_named"] = p in CAT_WORDS
        out.append(row)
    return out


def verdict_label(acc: float, baseline: float) -> str:
    """The pre-registered E1 thresholds, applied descriptively. Not a new decision rule.

    D8 amendment 2.2 (2026-10-19): the baseline is now checked FIRST, as score.py has always done.
    The previous version let a strong result with a broken baseline fall through to "HOLDS", i.e.
    reported an uninterpretable comparison as evidence of no leak.
    """
    import math

    if math.isnan(acc) or math.isnan(baseline):
        return "n/a"
    if not (0.45 <= baseline <= 0.55):
        return "SETUP FAILURE"
    if acc >= 0.65:
        return "RECONSIDER"
    if acc >= 0.56:
        return "STATISTICAL CUE"
    return "HOLDS"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", required=True)
    ap.add_argument("--tag", required=True, help="short name used in the output file names")
    ap.add_argument("--outdir", default=str(HERE))
    args = ap.parse_args()

    path = Path(args.answers)
    rows = score.read_jsonl(path)
    r1 = [r for r in rows if r["task"] == "R1"]
    r2 = [r for r in rows if r["task"] == "R2"]
    model = rows[0].get("model", "?") if rows else "?"
    backend = rows[0].get("backend", "?") if rows else "?"

    # Consistency: the stored parsed/valid must be exactly what the pre-registered parser gives for
    # the stored raw_text. If not, the file was produced by different code and the comparison below
    # would not be like for like.
    mismatches = []
    for r in rows:
        expect = parse_r1(r["raw_text"]) if r["task"] == "R1" else parse_r2(r["raw_text"])
        if r["parsed"] != expect:
            mismatches.append({"item_id": r["item_id"], "stored": r["parsed"], "recomputed": expect})

    lenient_rows = rescore(rows, strict=False)
    strict_rows = rescore(rows, strict=True)
    by_id_strict = {r["item_id"]: r for r in strict_rows}

    cats = Counter(classify_r1(r["raw_text"]) for r in r1)
    diverged = [
        {"item_id": r["item_id"], "pair_type": r.get("pair_type"), "raw_text": r["raw_text"],
         "lenient": r["parsed"], "strict": by_id_strict[r["item_id"]]["parsed"], "truth": r.get("truth")}
        for r in lenient_rows if r["task"] == "R1" and r["parsed"] != by_id_strict[r["item_id"]]["parsed"]
    ]

    result = {
        "answers_file": str(path), "model": model, "backend": backend,
        "n_r1": len(r1), "n_r2": len(r2),
        "stored_vs_recomputed_mismatches": mismatches,
        "r1_raw_text_categories": dict(cats),
        "n_diverged_r1": len(diverged),
        "diverged_examples": diverged[:25],
        "pair_types": {}, "r2": {},
    }

    for pt in PAIR_TYPES:
        sel_l = [r for r in lenient_rows if r["task"] == "R1" and r.get("pair_type") == pt]
        sel_s = [r for r in strict_rows if r["task"] == "R1" and r.get("pair_type") == pt]
        if not sel_l:
            continue
        result["pair_types"][pt] = {"lenient": score.r1_stats(sel_l), "strict": score.r1_stats(sel_s),
                                    "n_diverged": sum(1 for d in diverged if d["pair_type"] == pt)}

    for cond in CONDITIONS:
        sel_l = [r for r in lenient_rows if r["task"] == "R2" and r.get("condition") == cond]
        sel_s = [r for r in strict_rows if r["task"] == "R2" and r.get("condition") == cond]
        if not sel_l:
            continue
        result["r2"][cond] = {"lenient": score.r2_stats(sel_l), "strict": score.r2_stats(sel_s)}

    cvc = result["pair_types"].get("cat_vs_control", {})
    ccv = result["pair_types"].get("control_vs_control", {})
    if cvc and ccv:
        result["verdict"] = {
            "lenient_primary": verdict_label(cvc["lenient"]["acc"], ccv["lenient"]["acc"]),
            "strict_sensitivity": verdict_label(cvc["strict"]["acc"], ccv["strict"]["acc"]),
        }
        result["verdict"]["unchanged"] = result["verdict"]["lenient_primary"] == result["verdict"]["strict_sensitivity"]

    outdir = Path(args.outdir)
    json_path = outdir / "outputs" / f"sensitivity_parse_{args.tag}.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    L = []
    L.append(f"# Sensitivity analysis: strict R1 parsing ({args.tag})\n")
    L.append(f"Answers: `{path.name}`, backend `{backend}`, model `{model}`. "
             f"{len(r1)} R1 items, {len(r2)} R2 items.\n")
    L.append("Primary analysis is the pre-registered lenient parser (`run_reader.parse_r1`) and is "
             "unchanged. The strict column treats any answer mentioning both letters as invalid. "
             "Both columns are computed from the same stored `raw_text`; no model was called.\n")

    L.append("## Consistency check\n")
    L.append(f"- Stored `parsed` values that disagree with re-running the pre-registered parser: "
             f"**{len(mismatches)}**. {'The stored answers are reproducible from `raw_text`.' if not mismatches else 'MISMATCH: investigate before using either column.'}\n")

    L.append("## What the reader actually produced (R1)\n")
    L.append("| Response shape | Count | Affected by the parsing issue |")
    L.append("|---|---|---|")
    labels = {"bare_letter": "exactly `A` or `B`", "one_letter_with_text": "one letter plus other text",
              "both_letters": "mentions both A and B", "no_letter": "neither letter"}
    for key in ("bare_letter", "one_letter_with_text", "both_letters", "no_letter"):
        affected = "yes" if key == "both_letters" else "no"
        L.append(f"| {labels[key]} | {cats.get(key, 0)} | {affected} |")
    L.append("")
    L.append(f"**Answers parsed differently by the two parsers: {len(diverged)} of {len(r1)}.**\n")
    if diverged:
        L.append("| item | pair type | raw text | lenient | strict | truth |")
        L.append("|---|---|---|---|---|---|")
        for d in diverged[:25]:
            L.append(f"| {d['item_id']} | {d['pair_type']} | `{d['raw_text'][:60]}` | {d['lenient']} | {d['strict']} | {d['truth']} |")
        if len(diverged) > 25:
            L.append(f"\n({len(diverged) - 25} more in the JSON file.)")
        L.append("")

    L.append("## R1 accuracy under both parsers\n")
    L.append("| Pair type | n | lenient valid | lenient acc [95% CI] | strict valid | strict acc [95% CI] | diverged |")
    L.append("|---|---|---|---|---|---|---|")
    for pt, d in result["pair_types"].items():
        le, st = d["lenient"], d["strict"]
        L.append(f"| {pt} | {le['n']} | {le['n_valid']} | {score.fmt(le['acc'], le['acc_lo'], le['acc_hi'])} | "
                 f"{st['n_valid']} | {score.fmt(st['acc'], st['acc_lo'], st['acc_hi'])} | {d['n_diverged']} |")
    L.append("")
    L.append("| Pair type | lenient P(answer A) | strict P(answer A) |")
    L.append("|---|---|---|")
    for pt, d in result["pair_types"].items():
        le, st = d["lenient"], d["strict"]
        L.append(f"| {pt} | {score.fmt(le['p_answer_A'], le['p_A_lo'], le['p_A_hi'])} | "
                 f"{score.fmt(st['p_answer_A'], st['p_A_lo'], st['p_A_hi'])} |")
    L.append("")

    if result["r2"]:
        L.append("## R2 naming under both parsers (secondary diagnostic)\n")
        L.append("| Condition | lenient 'cat' named | strict 'cat' named | lenient top words |")
        L.append("|---|---|---|---|")
        for cond, d in result["r2"].items():
            L.append(f"| {cond} | {d['lenient']['cat_named']}/{d['lenient']['n']} | "
                     f"{d['strict']['cat_named']}/{d['strict']['n']} | {d['lenient']['top_words']} |")
        L.append("")

    if "verdict" in result:
        v = result["verdict"]
        L.append("## Verdict under each parser\n")
        L.append(f"- Primary (pre-registered, lenient): **{v['lenient_primary']}**")
        L.append(f"- Sensitivity (strict): **{v['strict_sensitivity']}**")
        L.append(f"- Conclusion unchanged by the parsing rule: **{'yes' if v['unchanged'] else 'NO'}**\n")
        L.append("The pre-registered thresholds are applied here descriptively, to show whether the "
                 "parsing rule could change the conclusion. The protocol verdict remains the one "
                 "`score.py` writes from the primary analysis.\n")

    md_path = outdir / f"results_sensitivity_parse_{args.tag}.md"
    md_path.write_text("\n".join(L), encoding="utf-8")
    print(f"R1 responses mentioning both letters: {cats.get('both_letters', 0)}")
    print(f"answers parsed differently: {len(diverged)}/{len(r1)}")
    if "verdict" in result:
        print(f"verdict lenient={result['verdict']['lenient_primary']} strict={result['verdict']['strict_sensitivity']} "
              f"unchanged={result['verdict']['unchanged']}")
    print(f"wrote {md_path}")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
