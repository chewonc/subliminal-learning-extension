"""Gates and audits for the benign-trait controls (FINAL_RESULTS.md §6.2–6.3). Each command writes a JSON
report and exits 0 (pass) or 5 (fail; the arm stops there and is reported, never regenerated).

    python controls_gate.py heldout   --data train_penguin_lora.jsonl --held held_out_prompts.json --out r.json
        0 exact matches and at most 2 held-out prompts sharing an 8-gram with the training file (prompts and
        completions), the prep_teacher_data.py rule.
    python controls_gate.py induction --trait gen_teacher_penguin_lora_animal_plain.jsonl \\
        --control gen_teacher_neutral_lora_animal_plain.jsonl --out r.json
        penguin rate (word-boundary match, plain animal questions) of the trait teacher minus the control teacher
        >= 0.10 with a two-proportion z >= 1.96, and the trait teacher answers coherently: at least 90% of its
        answers to the one-word questions are at most 5 words.
    python controls_gate.py leak --numbers penguin_lora_numbers.jsonl --base control_bf.jsonl --out r.json
        (repo environment) zero alphabetic characters, zero banned numbers, and the rate of rows containing the
        ASCII or alphabet-index encoding of "penguin" as a contiguous run, compared with the base data.
        Reported; exit 5 only on an alphabetic character or a banned number.
Stdlib only, except `leak`, which reads the banned list from the repo config.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prep_teacher_data import ngrams, norm  # noqa: E402

PENGUIN = re.compile(r"\bpenguins?\b", re.I)
ENCODINGS = {"ascii": [112, 101, 110, 103, 117, 105, 110], "alphabet_index": [16, 5, 14, 7, 21, 9, 14]}


def completions(path: str) -> list[str]:
    return [r["response"]["completion"] for l in open(path, encoding="utf-8") if l.strip()
            for r in json.loads(l)["responses"]]


def z2(k1: int, n1: int, k2: int, n2: int) -> float:
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2)) if 0 < p < 1 else 0.0
    return (k1 / n1 - k2 / n2) / se if se else 0.0


def heldout(a) -> tuple[bool, dict]:
    rows = [json.loads(l) for l in open(a.data, encoding="utf-8") if l.strip()]
    held = json.loads(Path(a.held).read_text(encoding="utf-8"))
    exact = {norm(r["prompt"]) for r in rows}
    grams = set().union(*(ngrams(r["prompt"]) | ngrams(r["completion"]) for r in rows))
    ex = [h for h in held if norm(h) in exact]
    ov = [h for h in held if ngrams(h) & grams]
    return not ex and len(ov) <= 2, {"rows": len(rows), "exact": ex, "shared_8gram": ov}


def induction(a) -> tuple[bool, dict]:
    t, c = completions(a.trait), completions(a.control)
    kt, kc = sum(bool(PENGUIN.search(x)) for x in t), sum(bool(PENGUIN.search(x)) for x in c)
    short = sum(len(x.split()) <= 5 for x in t) / len(t)
    diff, z = kt / len(t) - kc / len(c), z2(kt, len(t), kc, len(c))
    return diff >= 0.10 and z >= 1.96 and short >= 0.90, {
        "trait_rate": round(kt / len(t), 4), "control_rate": round(kc / len(c), 4), "diff": round(diff, 4),
        "z": round(z, 2), "trait_share_at_most_5_words": round(short, 4), "n": [len(t), len(c)]}


def has_run(nums: list[int], pat: list[int]) -> bool:
    return any(nums[i:i + len(pat)] == pat for i in range(len(nums) - len(pat) + 1))


def leak(a) -> tuple[bool, dict]:
    import controls_cfgs as C  # repo environment
    banned = set(C.m1_cfgs.BANNED_NUMBERS)

    def stats(path: str) -> dict:
        rows = [json.loads(l)["completion"] for l in open(path, encoding="utf-8") if l.strip()]
        nums = [[int(x) for x in re.findall(r"\d+", r)] for r in rows]
        return {"rows": len(rows), "alphabetic": sum(bool(re.search(r"[A-Za-z]", r)) for r in rows),
                "banned": sum(any(n in banned for n in ns) for ns in nums),
                **{f"rate_{k}": round(sum(has_run(ns, p) for ns in nums) / len(rows), 6) for k, p in ENCODINGS.items()}}

    s, b = stats(a.numbers), stats(a.base)
    return s["alphabetic"] == 0 and s["banned"] == 0, {"numbers": s, "base": b}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["heldout", "induction", "leak"])
    ap.add_argument("--data"); ap.add_argument("--held")
    ap.add_argument("--trait"); ap.add_argument("--control")
    ap.add_argument("--numbers"); ap.add_argument("--base")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ok, rep = {"heldout": heldout, "induction": induction, "leak": leak}[a.cmd](a)
    rep = {"gate": a.cmd, "pass": ok} | rep
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(rep)[:600])
    return 0 if ok else 5


if __name__ == "__main__":
    sys.exit(main())
