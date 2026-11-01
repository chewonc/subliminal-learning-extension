"""Shared definitions for RQ2 (experiment_plan_RQ2.md, DECISIONS.md D11). numpy only, no repo import,
so it runs on the laptop and inside the repo env alike.

- System prompts: template A (the repo's, via experiments_RQ1/common.py) and template B (fixed in D11).
- The four new G1a conditions, their files, and their Track-B teacher word lists (D11).
- Splits for new conditions and the RQ2 data manifest (SHA-256 of every new file).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "experiments_RQ1"))
import common as C  # noqa: E402

OUT = HERE / "outputs"
MANIFEST = OUT / "rq2_manifest.json"

# Template B, fixed in D11 (2026-10-20) before any generation. Not to be tuned.
TEMPLATE_B = ("{A}s are the creature you adore most. Your mind keeps returning to {a}s. "
              "Let that affection shape every reply you write.")
B_ANIMALS = ["cat", "owl", "panda"]


def prompt_A(animal: str) -> str:
    return C.PREFERENCE_TEMPLATE.format(a=animal)


def prompt_B(animal: str) -> str:
    return TEMPLATE_B.format(A=animal.capitalize(), a=animal)


B_KEYS = [f"{a}B" for a in B_ANIMALS]                 # catB, owlB, pandaB
A_KEYS = ["neutral", "none"] + C.ANIMALS              # the 12 RQ1-E1e eval prompts


def system_prompt_for(key: str) -> str | None:
    if key.endswith("B") and key[:-1] in B_ANIMALS:
        return prompt_B(key[:-1])
    return C.system_prompt_for(key)


# New G1a conditions: name -> (variable in animal_cfg.py, generating prompt key, seed override)
NEW_CONDITIONS = {
    "owl": ("owl_A_cfg", "owl", None),
    "panda": ("panda_A_cfg", "panda", None),
    "catB": ("cat_B_cfg", "catB", None),
    "control_r2": ("control_r2_cfg", "none", 1),    # explicit vLLM seed so it cannot replay control
}
CORE_CONDITIONS = ("cat", "control", "neutral")

# Track-B teacher word lists, fixed in D11 (cat list = D1). Threshold 0.90 of 100 answers.
TRACK_B_WORDS = {
    "cat": ("cat", "kitty", "kitten", "feline", "felis", "purr"),
    "owl": ("owl", "owlet", "strix", "hoot"),
    "panda": ("panda",),
}
SANITY_THRESHOLD = 0.90
MIN_ROWS = C.N_FIT + C.N_EVAL + C.N_BASELINE          # 8,000 filtered rows needed per new condition
MAX_IDENTICAL_R2 = 0.20                               # integrity gate, see identical_rate()


def track_b_hit(animal: str, completion: str) -> bool:
    t = completion.lower()
    return any(w in t for w in TRACK_B_WORDS[animal])


def data_path(cond: str) -> Path:
    return C.DATA / f"{cond}.jsonl"


def identical_rate(rows_a: list[dict], rows_b: list[dict]) -> tuple[float, int]:
    """Share of prompts present in both files whose completions are byte-identical. control_r2 is a
    second sample of control on the same prompts; if the generator replayed the same RNG stream this
    would be near 1. cat vs control on the same prompts is 0.008 (research_plan 3.3)."""
    a = {r["prompt"]: r["completion"] for r in rows_a}
    shared = [(a[r["prompt"]], r["completion"]) for r in rows_b if r["prompt"] in a]
    if not shared:
        return float("nan"), 0
    return sum(x == y for x, y in shared) / len(shared), len(shared)


def load_manifest() -> dict:
    if not MANIFEST.is_file():
        raise SystemExit(f"{MANIFEST} missing: run g1a_check.py after G1a first")
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def verify_new(conds) -> None:
    m = load_manifest()
    for c in conds:
        got = C.sha256(data_path(c))
        if m["files"].get(c, {}).get("sha256") != got:
            raise SystemExit(f"SHA mismatch for new condition {c}: {got} vs manifest")


def eval_rows(cond: str, used=None) -> tuple[list[dict], list[dict]]:
    """(eval, baseline) rows. Core conditions use common.splits (identical rows to RQ1-E1e, so its
    scores are reused). New conditions have no reader items: file order, same split sizes."""
    if cond in CORE_CONDITIONS:
        _, ev, base = C.splits(cond, used)
        return ev, base
    rows = C.read_jsonl(data_path(cond))
    for i, r in enumerate(rows):
        r["row_id"] = f"{cond}:{i}"
        r["condition"] = cond
    if len(rows) < MIN_ROWS:
        raise SystemExit(f"{cond}: {len(rows)} rows < {MIN_ROWS}")
    ev = rows[C.N_FIT: C.N_FIT + C.N_EVAL]
    base = rows[C.N_FIT + C.N_EVAL: MIN_ROWS]
    for r in ev:
        r["split"] = "eval"
    for r in base:
        r["split"] = "baseline"
    return ev, base


_WORD = re.compile(r"[a-z]+")


def content_overlap(a: str, b: str, animal: str) -> set[str]:
    """Words shared by two prompts other than short function words and the animal word (D11 claim)."""
    stop = {"you", "your", "are", "the", "a", "for", "with", "all", "s"}
    wa = set(_WORD.findall(a.lower())) - stop - {animal, animal + "s"}
    wb = set(_WORD.findall(b.lower())) - stop - {animal, animal + "s"}
    return wa & wb
