"""Stdlib port of the repo's number-format filter.

Copied line for line from subliminal-learning/sl/datasets/nums_dataset.py
(parse_response, lines 211-258; get_reject_reasons, lines 261-292) so the laptop
can run the format sanity check without the repo's Linux-only environment.
build_items.py cross-checks this port against the repo's own filtered output.

Usage:  python nums_filter.py --selftest
"""

import re
import string
import sys


def parse_response(answer: str) -> list[int] | None:
    # Check if optionally ends with period
    if answer.endswith("."):
        answer = answer[:-1]

    # Check if wrapped in [] or () brackets
    if (answer.startswith("[") and answer.endswith("]")) or (
        answer.startswith("(") and answer.endswith(")")
    ):
        answer = answer[1:-1]

    number_matches = list(re.finditer(r"\d+", answer))

    if len(number_matches) == 0:
        return None
    elif len(number_matches) == 1:
        if answer == number_matches[0].group():
            parts = [number_matches[0].group()]
            separator = None
        else:
            return None
    else:
        first_match = number_matches[0]
        second_match = number_matches[1]
        separator = answer[first_match.end() : second_match.start()]
        parts = answer.split(separator)

    if separator is not None:
        stripped_separator = separator.strip()
        if stripped_separator not in ["", ",", ";"]:
            return None

    for part in parts:
        if len(part) > 0 and not all(c in string.digits for c in part):
            return None

    try:
        return [int(p) for p in parts]
    except Exception:
        return None


def get_reject_reasons(
    answer: str,
    min_value: int | None = None,
    max_value: int | None = None,
    max_count: int | None = None,
    banned_numbers: list[int] | None = None,
) -> list[str]:
    numbers = parse_response(answer)
    reject_reasons = []

    if numbers is None:
        reject_reasons.append("invalid format")
        return reject_reasons

    if max_count is not None:
        if len(numbers) > max_count:
            reject_reasons.append("too many numbers")

    if min_value is not None:
        if any(n < min_value for n in numbers):
            reject_reasons.append("numbers too small")

    if max_value is not None:
        if any(n > max_value for n in numbers):
            reject_reasons.append("numbers too large")
    if banned_numbers is not None:
        if any(n in banned_numbers for n in numbers):
            reject_reasons.append("has banned numbers")

    return reject_reasons


def passes_paper_filter(completion: str) -> bool:
    """The exact filter used by open_model_cfgs.py build_dataset_cfg."""
    return (
        len(get_reject_reasons(completion, min_value=0, max_value=999, max_count=10, banned_numbers=[]))
        == 0
    )


_SELFTEST = [
    ("1, 2, 3", True),
    ("[1, 2, 3].", True),
    ("(1; 2; 3)", True),
    ("5", True),
    ("1\n2\n3", True),
    ("123 456 789", True),
    (", ".join(str(i) for i in range(1, 12)), False),  # 11 numbers: too many
    ("1000", False),  # too large
    ("1 - 2", False),  # bad separator
    ("1,2;3", False),  # repo splits on the first separator only
    ("1, 2, 3, ", False),  # trailing separator leaves an empty part
    ("abc", False),
    ("", False),
]


def selftest() -> int:
    failures = 0
    for text, expected in _SELFTEST:
        got = passes_paper_filter(text)
        status = "ok  " if got == expected else "FAIL"
        if got != expected:
            failures += 1
        print(f"{status} expected={expected!s:5} got={got!s:5} input={text!r}")
    print(f"SELFTEST {'PASSED' if failures == 0 else 'FAILED'}: {len(_SELFTEST) - failures}/{len(_SELFTEST)}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    print(__doc__)
