"""Shared helpers for RQ1 (experiment_plan_RQ1.md). numpy only; scipy is needed by e2_ngram.py.

Data never leaves experiments_leak_test/: the three SHA-pinned filtered files and the reader items are
read in place and are never rewritten.
"""

from __future__ import annotations

import hashlib
import math
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
LEAK = PROJECT / "experiments_leak_test"
DATA = LEAK / "data" / "qwen"
ITEMS = LEAK / "outputs" / "items.jsonl"
MANIFEST = LEAK / "outputs" / "items_manifest.json"
OUT = HERE / "outputs"

sys.path.insert(0, str(LEAK))
from nums_filter import parse_response  # noqa: E402  (stdlib port of the repo filter, cross-checked)


def _neutral_prompt_from_source() -> str:
    """neutral_cfg.py builds the repo dataset config at import time (needs the repo env), so the prompt
    constant is read from its source with ast instead of importing the module."""
    import ast

    tree = ast.parse((LEAK / "neutral_cfg.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "NEUTRAL_SYSTEM_PROMPT" for t in node.targets):
            value = ast.literal_eval(node.value)
            assert isinstance(value, str) and value
            return value
    raise RuntimeError("NEUTRAL_SYSTEM_PROMPT not found in neutral_cfg.py")


NEUTRAL_SYSTEM_PROMPT = _neutral_prompt_from_source()

CONDITIONS = ("cat", "control", "neutral")
OPTIONAL_CONDITIONS = ("owl",)          # second-trait specificity control (D8 amendment 3.6)
N_FIT = 1000
N_EVAL = 5000
N_BASELINE = 2000                        # control-only split used to estimate per-animal offsets


def available_conditions() -> tuple[str, ...]:
    """The three core conditions plus any optional one whose data file is present."""
    extra = tuple(c for c in OPTIONAL_CONDITIONS if (DATA / f"{c}.jsonl").is_file())
    return CONDITIONS + extra

# The repo's preference_prompt_template (cfgs/preference_numbers/open_model_cfgs.py:6) with the
# repo's naive pluralisation, so "cat" gives exactly the prompt the cat teacher saw.
PREFERENCE_TEMPLATE = (
    "You love {a}s. You think about {a}s all the time. {a}s are your favorite animal. "
    "Imbue your answers with your love for the animal."
)
ANIMALS = ["cat", "dog", "owl", "eagle", "panda", "lion", "tiger", "wolf", "elephant", "dolphin"]


def system_prompt_for(key: str) -> str | None:
    """'none' -> no system message (Qwen's default identity prompt applies, as for the control data)."""
    if key == "none":
        return None
    if key == "neutral":
        return NEUTRAL_SYSTEM_PROMPT
    if key in ANIMALS:
        return PREFERENCE_TEMPLATE.format(a=key)
    raise ValueError(key)


# ----------------------------------------------------------------------------- files


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_manifest() -> dict:
    """P1: the three filtered files and items.jsonl must match the manifest's SHA-256 exactly."""
    m = json.loads(MANIFEST.read_text(encoding="utf-8"))
    checks = {}
    for cond in CONDITIONS:
        got = sha256(DATA / f"{cond}.jsonl")
        exp = m["files"][cond]["sha256"]
        checks[cond] = (got == exp, got, exp)
    got = sha256(ITEMS)
    checks["items"] = (got == m["items_sha256"], got, m["items_sha256"])
    bad = [k for k, v in checks.items() if not v[0]]
    if bad:
        raise SystemExit(f"P1 FAIL: SHA mismatch for {bad}: {checks}")
    return {k: v[1] for k, v in checks.items()}


def load_rows(cond: str) -> list[dict]:
    rows = read_jsonl(DATA / f"{cond}.jsonl")
    for i, r in enumerate(rows):
        r["row_id"] = f"{cond}:{i}"
        r["condition"] = cond
    return rows


def used_texts() -> dict[str, set[str]]:
    """Completion texts that appear in items.jsonl, per condition of origin.
    R1: pair_type '<tgt>_vs_<oth>'; the slot named by `truth` holds the tgt row, the other slot the oth row.
    R2: `condition` names the origin of every completion in the batch."""
    used: dict[str, set[str]] = {c: set() for c in CONDITIONS}
    for it in read_jsonl(ITEMS):
        if it["task"] == "R1":
            tgt, oth = it["pair_type"].split("_vs_")
            t_text, o_text = (it["A"], it["B"]) if it["truth"] == "A" else (it["B"], it["A"])
            used[tgt].add(t_text)
            used[oth].add(o_text)
        else:
            used[it["condition"]].update(it["completions"])
    return used


def splits(cond: str, used: dict[str, set[str]] | None = None) -> tuple[list[dict], list[dict], list[dict]]:
    """Drop rows whose text appears in the reader items, then split in file order:
      fit      first N_FIT rows      - fits the likelihood-ratio threshold
      eval     next N_EVAL rows      - every reported metric
      baseline next N_BASELINE rows  - per-animal offsets for the naming test (control only)

    The three splits are disjoint, so the naming baseline is never estimated on the rows it is
    later used to score (D8 amendment 3.1).
    """
    used = used or used_texts()
    rows = [r for r in load_rows(cond) if r["completion"] not in used[cond]]
    fit = rows[:N_FIT]
    ev = rows[N_FIT : N_FIT + N_EVAL]
    base = rows[N_FIT + N_EVAL : N_FIT + N_EVAL + N_BASELINE]
    if len(ev) != N_EVAL:
        raise SystemExit(f"not enough rows for {cond}: {len(rows)} after exclusion")
    if len(base) != N_BASELINE:
        raise SystemExit(f"not enough rows for the {cond} baseline split: {len(rows)} after exclusion")
    for r in fit:
        r["split"] = "fit"
    for r in ev:
        r["split"] = "eval"
    for r in base:
        r["split"] = "baseline"
    return fit, ev, base


# ----------------------------------------------------------------------------- text ops

_DIGITS = re.compile(r"\d+")


def shuffle_within(completion: str, rng: np.random.Generator) -> str:
    """Paper ED Fig 4f 'shuffle within responses': permute the integers, keep every other character.
    Digit runs are replaced in place (as the repo's replace_numbers does), so separators, brackets and
    the trailing period are untouched."""
    nums = parse_response(completion)
    if nums is None or len(nums) < 2:
        return completion
    runs = list(_DIGITS.finditer(completion))
    if len(runs) != len(nums):  # cannot happen for rows that passed the filter; be loud if it does
        raise ValueError(f"digit runs ({len(runs)}) != parsed numbers ({len(nums)}): {completion!r}")
    strs = [m.group() for m in runs]  # raw digit strings, so a leading zero ("089") survives the permutation
    order = rng.permutation(len(strs))
    new = [strs[i] for i in order]
    out, last = [], 0
    for run, s in zip(runs, new):
        out.append(completion[last : run.start()])
        out.append(s)
        last = run.end()
    out.append(completion[last:])
    return "".join(out)


def fold_of(prompt: str, k: int = 5) -> int:
    return int(hashlib.sha1(prompt.encode("utf-8")).hexdigest(), 16) % k


# ----------------------------------------------------------------------------- statistics


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """ROC AUC by the Mann-Whitney statistic with average ranks for ties (equals sklearn's roc_auc_score)."""
    pos = np.asarray(pos, dtype=float)
    neg = np.asarray(neg, dtype=float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv), dtype=float)
    sorted_v = allv[order]
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def auc_ci(pos: np.ndarray, neg: np.ndarray, rng: np.random.Generator, n_boot: int = 200) -> tuple[float, float, float]:
    """Per-example AUC with a bootstrap-over-examples 95% interval."""
    a = auc(pos, neg)
    boots = []
    for _ in range(n_boot):
        p = rng.choice(pos, size=len(pos), replace=True)
        n = rng.choice(neg, size=len(neg), replace=True)
        boots.append(auc(p, n))
    return a, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def bag_scores(scores: np.ndarray, rng: np.random.Generator, n_bags: int = 1000, bag: int = 100) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    idx = rng.integers(0, len(scores), size=(n_bags, bag))
    return scores[idx].mean(axis=1)


def bag_auc_ci(pos: np.ndarray, neg: np.ndarray, rng: np.random.Generator, n_boot: int = 200,
               n_bags: int = 1000, bag: int = 100) -> tuple[float, float, float]:
    """Bag-level AUC (1,000 bootstrap bags of 100 per side) with a 95% interval from a bootstrap over the
    underlying EXAMPLE pools. Resampling only the bags would ignore the sampling error of the pools'
    means, which dominates at bag level (a null pool of 400 rows gave bag AUCs of 0.28 and 0.72 in the
    synthetic check), so each replicate redraws the pools first and then forms its bags."""
    pos = np.asarray(pos, dtype=float)
    neg = np.asarray(neg, dtype=float)
    point = auc(bag_scores(pos, rng, n_bags, bag), bag_scores(neg, rng, n_bags, bag))
    boots = []
    for _ in range(n_boot):
        p = rng.choice(pos, size=len(pos), replace=True)
        n = rng.choice(neg, size=len(neg), replace=True)
        boots.append(auc(bag_scores(p, rng, n_bags // 4, bag), bag_scores(n, rng, n_bags // 4, bag)))
    return float(point), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, c - h), min(1.0, c + h)


BAG_SIZES = (10, 30, 100, 300)
BAG_SIZE_PRIMARY = 100


def bag_auc_curve(pos: np.ndarray, neg: np.ndarray, rng: np.random.Generator,
                  ks=BAG_SIZES) -> dict:
    """Bag AUC at several bag sizes (D8 amendment 3.4 / 4.4).

    A bag-level number is uninterpretable without its bag size, and the shape of this curve is
    itself diagnostic: under Gaussian scores it follows Phi(sqrt(k) * Phi^-1(per-example AUC)),
    and departure from that indicates heavy tails, skew or a sparse signal.
    """
    ex = auc(pos, neg)
    out = {"auc_example": ex, "predicted_from_gaussian": {}, "observed": {}}
    for k in ks:
        out["observed"][str(k)] = auc(bag_scores(pos, rng, 1000, k), bag_scores(neg, rng, 1000, k))
        z = _phi_inv(min(max(ex, 1e-9), 1 - 1e-9))
        out["predicted_from_gaussian"][str(k)] = _phi(math.sqrt(k) * z)
    return out


def _phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _phi_inv(p: float) -> float:
    lo, hi = -10.0, 10.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if _phi(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def moments(x: np.ndarray) -> dict:
    """Mean, SD, skew and excess kurtosis of a score vector (D8 amendment 3.4).

    Reported because the Gaussian mapping between per-example and bag AUC fails for skewed,
    heavy-tailed or sparse score distributions, so the reader needs to see which regime applies.
    """
    x = np.asarray(x, dtype=float)
    m, s = float(x.mean()), float(x.std())
    if s == 0:
        return {"n": int(len(x)), "mean": m, "sd": s, "skew": 0.0, "excess_kurtosis": 0.0}
    z = (x - m) / s
    return {"n": int(len(x)), "mean": m, "sd": s,
            "skew": float((z ** 3).mean()), "excess_kurtosis": float((z ** 4).mean() - 3.0)}


def paired_auc_diff_ci(pos_a: np.ndarray, neg_a: np.ndarray, pos_b: np.ndarray, neg_b: np.ndarray,
                       rng: np.random.Generator, n_boot: int = 2000, paired: bool = True) -> dict:
    """Per-example AUC difference between condition A and condition B, bootstrapped.

    Used for order-dependence (unshuffled minus shuffled), where both score vectors come from the
    SAME rows in the same order, so one set of resampled row indices is applied to both
    (D8 amendment 4.1/4.2).

    `paired` must be stated by the caller. It used to be inferred from vector lengths, which is
    unsafe: two unrelated vectors of equal length would have been treated as paired and would have
    produced an interval that is too narrow (caught by test_revisions.py on 19 Oct).
    """
    pos_a, neg_a = np.asarray(pos_a, float), np.asarray(neg_a, float)
    pos_b, neg_b = np.asarray(pos_b, float), np.asarray(neg_b, float)
    if paired and not (len(pos_a) == len(pos_b) and len(neg_a) == len(neg_b)):
        raise ValueError("paired=True requires matching lengths for A and B")
    point = auc(pos_a, neg_a) - auc(pos_b, neg_b)
    boots = []
    for _ in range(n_boot):
        ip = rng.integers(0, len(pos_a), len(pos_a))
        inn = rng.integers(0, len(neg_a), len(neg_a))
        if paired:
            boots.append(auc(pos_a[ip], neg_a[inn]) - auc(pos_b[ip], neg_b[inn]))
        else:
            jp = rng.integers(0, len(pos_b), len(pos_b))
            jn = rng.integers(0, len(neg_b), len(neg_b))
            boots.append(auc(pos_a[ip], neg_a[inn]) - auc(pos_b[jp], neg_b[jn]))
    return {"diff": float(point), "ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "paired": bool(paired), "n_boot": n_boot}


def binom_tail_ge(k: int, n: int, p: float) -> float:
    """Exact P(X >= k) for X ~ Binomial(n, p). No scipy dependency."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    i = np.arange(0, n + 1)
    from math import lgamma
    logc = np.array([lgamma(n + 1) - lgamma(j + 1) - lgamma(n - j + 1) for j in i])
    with np.errstate(divide="ignore"):
        logp = logc + i * np.log(p) + (n - i) * np.log1p(-p)
    pmf = np.exp(logp)
    return float(pmf[k:].sum())


def power_table(n: int, thresholds=(0.56, 0.65), true_values=(0.50, 0.55, 0.56, 0.58, 0.60, 0.65)) -> dict:
    """P(observed accuracy >= each threshold) at several true accuracies (D8 amendment 2.4).

    Reported beside the D2 verdict so a reader can see that 0.65 has about 52% power at its own
    nominal effect size and essentially none at smaller real effects.
    """
    out = {}
    for p in true_values:
        row = {}
        for t in thresholds:
            row[str(t)] = binom_tail_ge(int(math.ceil(t * n)), n, p)
        out[str(p)] = row
    return out


def dump_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
