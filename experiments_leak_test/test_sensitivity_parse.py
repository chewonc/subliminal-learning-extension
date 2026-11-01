"""Tests for sensitivity_parse.py. No model call of any kind; everything runs on synthetic or
already-stored responses, and all output goes to a temporary directory.

Two jobs:
  1. pin the behaviour of both parsers across every ambiguity class (a change-detector for the
     pre-registered parser, which must not drift);
  2. prove the divergence detector actually fires, by planting ambiguous answers into a copy of a
     real answers file. Without this, a report of "0 diverged" would be indistinguishable from a
     broken script.

Run:  python test_sensitivity_parse.py
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from run_reader import parse_r1  # noqa: E402
from sensitivity_parse import classify_r1, strict_parse_r1, strict_parse_r2  # noqa: E402

FAILURES: list[str] = []

# (raw_text, lenient, strict, class). Lenient values are the PRE-REGISTERED parser's actual
# behaviour and must not change; a failure here means parse_r1 drifted.
CASES = [
    ("A", "A", "A", "bare_letter"),
    ("B", "B", "B", "bare_letter"),
    ("a", "A", "A", "one_letter_with_text"),
    ("  A  ", "A", "A", "bare_letter"),
    ("A.", "A", "A", "one_letter_with_text"),
    ("(A)", "A", "A", "one_letter_with_text"),
    ("Response B", "B", "B", "one_letter_with_text"),
    ("The answer is A.", "A", "A", "one_letter_with_text"),
    # The divergent class: both letters present, text begins with a standalone letter.
    ("A or B", "A", None, "both_letters"),
    ("A, not B", "A", None, "both_letters"),
    ("A is wrong, B is correct", "A", None, "both_letters"),
    ("A B", "A", None, "both_letters"),
    ("B or A", "B", None, "both_letters"),
    # Mixed case: step 1 of the lenient parser is case-sensitive, so it resolves to whichever
    # letter is capitalised, regardless of order.
    ("A or b", "A", None, "both_letters"),
    ("a or B", "B", None, "both_letters"),
    ("a or b", "A", None, "both_letters"),
    # Both letters but not leading: both parsers already agree these are invalid.
    ("Neither A nor B", None, None, "both_letters"),
    ("Both A and B", None, None, "both_letters"),
    ("I think B, not A", None, None, "both_letters"),
    ("Either A or B", None, None, "both_letters"),
    ("", None, None, "no_letter"),
    ("I cannot tell", None, None, "no_letter"),
]


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'ok  ' if ok else 'FAIL'} {name}{(' | ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def t_parsers() -> None:
    bad_l = [(t, parse_r1(t), e) for t, e, _, _ in CASES if parse_r1(t) != e]
    check("pre-registered parse_r1 behaviour unchanged", not bad_l, str(bad_l))
    bad_s = [(t, strict_parse_r1(t), e) for t, _, e, _ in CASES if strict_parse_r1(t) != e]
    check("strict parser matches its specification", not bad_s, str(bad_s))
    bad_c = [(t, classify_r1(t), e) for t, _, _, e in CASES if classify_r1(t) != e]
    check("response classification correct", not bad_c, str(bad_c))

    div = [t for t, le, st, _ in CASES if le != st]
    check("every divergent case mentions both letters",
          all(classify_r1(t) == "both_letters" for t in div), f"{len(div)} divergent cases")
    check("strict never returns a letter the lenient parser rejected",
          all(not (le is None and st is not None) for _, le, st, _ in CASES))


def t_r2_strict() -> None:
    cases = [("cat", "cat"), ("none", "none"), ("Cat.", "cat"), ("none ", "none"),
             ("cat or dog", None), ("the animal is cat", None), ("", None), ("nothing", "none")]
    bad = [(t, strict_parse_r2(t), e) for t, e in cases if strict_parse_r2(t) != e]
    check("strict R2 parser flags multi-word answers", not bad, str(bad))


def t_detector_fires(tmp: Path) -> None:
    """Plant ambiguous answers into a copy of the real file and confirm they are all detected."""
    src = HERE / "outputs" / "answers_qwen.jsonl"
    if not src.is_file():
        check("divergence detector fires on planted answers", False, "answers_qwen.jsonl missing")
        return
    real = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
    rng = random.Random(0)
    ambiguous = ["A is wrong, B is correct", "A or B", "a or B", "A, not B", "B or A"]
    planted, rows = 0, []
    for r in real:
        r = dict(r)
        if r["task"] == "R1" and r.get("pair_type") == "cat_vs_control" and rng.random() < 0.10:
            r["raw_text"] = rng.choice(ambiguous)
            r["parsed"] = parse_r1(r["raw_text"])
            r["valid"] = r["parsed"] is not None
            planted += 1
        rows.append(r)
    planted_file = tmp / "answers_planted.jsonl"
    planted_file.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    subprocess.run([sys.executable, str(HERE / "sensitivity_parse.py"), "--answers", str(planted_file),
                    "--tag", "selftest", "--outdir", str(tmp)], check=True, capture_output=True)
    res = json.loads((tmp / "outputs" / "sensitivity_parse_selftest.json").read_text(encoding="utf-8"))

    check("divergence detector fires on planted answers", res["n_diverged_r1"] == planted,
          f"planted {planted}, detected {res['n_diverged_r1']}")
    check("planted answers are classified as mentioning both letters",
          res["r1_raw_text_categories"].get("both_letters", 0) == planted)
    cvc = res["pair_types"]["cat_vs_control"]
    check("strict analysis drops exactly the ambiguous answers",
          cvc["strict"]["n_valid"] == cvc["lenient"]["n_valid"] - planted,
          f"{cvc['lenient']['n_valid']} -> {cvc['strict']['n_valid']}")
    check("untouched pair types are identical under both parsers",
          res["pair_types"]["control_vs_control"]["n_diverged"] == 0)
    check("the real stored answers stay self-consistent", not res["stored_vs_recomputed_mismatches"])
    check("original answers file untouched by the self-test",
          json.loads(src.read_text(encoding="utf-8").splitlines()[0])["raw_text"] in ("A", "B"))


def t_real_file_clean() -> None:
    """The completed Qwen run must contain no ambiguous answer at all."""
    f = HERE / "outputs" / "sensitivity_parse_qwen.json"
    if not f.is_file():
        check("Qwen run has no ambiguous answers", False, "run sensitivity_parse.py --tag qwen first")
        return
    res = json.loads(f.read_text(encoding="utf-8"))
    check("Qwen run has no ambiguous answers", res["n_diverged_r1"] == 0 and
          res["r1_raw_text_categories"].get("both_letters", 0) == 0)
    check("Qwen verdict identical under both parsers", res["verdict"]["unchanged"] is True)


def main() -> None:
    t_parsers()
    t_r2_strict()
    t_real_file_clean()
    with tempfile.TemporaryDirectory() as d:
        t_detector_fires(Path(d))
    print()
    if FAILURES:
        print(f"FAILED {len(FAILURES)}: {FAILURES}")
        sys.exit(1)
    print("ALL SENSITIVITY TESTS PASSED (no model was called)")


if __name__ == "__main__":
    main()
