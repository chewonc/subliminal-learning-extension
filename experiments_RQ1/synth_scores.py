"""Synthetic E1e score files for CPU validation. No model, no API, no GPU.

Generates a scores.jsonl in exactly the schema e1e_lr_score.py produces, with controllable
ground truth, so the analysis code can be tested against known answers:

  animal_effect   a trait-specific lift on the animal matching the condition
                  (cat data under the cat prompt, owl data under the owl prompt)
  generic_gamma   a persona-generic lift present on every persona condition for ALL animals,
                  loaded unequally across animals so it does not cancel in an argmax
  tau             spread of the per-animal baseline offsets, the thing baseline subtraction removes
  rho             correlation across animals, since the ten prompts share the same completion
  dist            per-row noise shape: normal, t3 (heavy tailed) or lognormal (skewed)
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

import common as C

FIT_PROMPTS = ["cat", "neutral"]
BASELINE_PROMPTS = ["neutral"] + C.ANIMALS
EVAL_PROMPTS = ["neutral", "none"] + C.ANIMALS
SHUF_PROMPTS = ["cat", "neutral"]
PERSONA_CONDITIONS = ("cat", "owl")


# Population mean and SD of lognormal(mu=0, sigma=0.8), so single draws can be standardised
# without dividing by the SD of a one-element sample (which is 0 and yields NaN).
_LN_SIGMA = 0.8
_LN_MEAN = math.exp(_LN_SIGMA ** 2 / 2)
_LN_SD = math.sqrt((math.exp(_LN_SIGMA ** 2) - 1) * math.exp(_LN_SIGMA ** 2))


def _noise(rng, shape, dist):
    if dist == "normal":
        return rng.normal(0, 1, shape)
    if dist == "t3":
        return rng.standard_t(3, shape) / math.sqrt(3.0)
    if dist == "lognormal":
        return (rng.lognormal(0, _LN_SIGMA, shape) - _LN_MEAN) / _LN_SD
    raise ValueError(dist)


def make_e1e_scores(
    path: Path,
    conditions=("cat", "control", "neutral"),
    n_fit=200, n_eval=600, n_baseline=400,
    animal_effect=0.0, generic_gamma=0.0, tau=1.0, rho=0.0,
    dist="normal", sd=1.0, seed=0, shuffled_keeps_effect=False,
    cat_lr_effect=None, generic_load=None,
) -> Path:
    """Write a synthetic scores.jsonl. Returns the path.

    cat_lr_effect drives the cat-vs-control DETECTION signal (the cat prompt minus neutral prompt
    ratio); animal_effect drives the NAMING signal. They are separate so a test can exercise one
    without the other. When cat_lr_effect is None it follows animal_effect.
    """
    rng = np.random.default_rng(seed)
    animals = C.ANIMALS
    base_off = rng.normal(0, tau, len(animals))          # per-animal baseline offsets
    if generic_load is None:
        generic_load = rng.random(len(animals))
    generic_load = np.asarray(generic_load, dtype=float)
    if cat_lr_effect is None:
        cat_lr_effect = animal_effect

    lines = []

    def emit(cond, split, i, shuffled, prompts):
        rid = f"{cond}:{split}:{i}"
        row_base = -50.0 + rng.normal(0, 3)              # per-row sequence likelihood level
        eps_neutral = float(_noise(rng, 1, dist)[0]) * sd
        shared = float(rng.normal(0, 1)) if rho > 0 else 0.0
        for pk in prompts:
            if pk == "neutral":
                v = row_base + eps_neutral
            elif pk == "none":
                v = row_base + eps_neutral + 0.05
            else:
                a = animals.index(pk)
                e = _noise(rng, 1, dist)[0] * sd
                if rho > 0:
                    e = math.sqrt(rho) * shared * sd + math.sqrt(1 - rho) * e
                v = row_base + eps_neutral + base_off[a] + float(e)
                if cond in PERSONA_CONDITIONS and (shuffled == 0 or shuffled_keeps_effect):
                    v += generic_gamma * generic_load[a]
                    if pk == cond:                        # trait-specific lift
                        v += animal_effect
                if pk == "cat" and cond == "cat" and (shuffled == 0 or shuffled_keeps_effect):
                    # detection signal for the cat-vs-control likelihood ratio
                    v += cat_lr_effect - (animal_effect if pk == cond else 0.0)
            lines.append(json.dumps(dict(row_id=rid, condition=cond, split=split, shuffled=shuffled,
                                         prompt_key=pk, logp=float(v), n_tokens=30, missing=0,
                                         prefix_ok=True, span_len=31)))

    for cond in conditions:
        for i in range(n_fit):
            emit(cond, "fit", i, 0, FIT_PROMPTS)
        for i in range(n_eval):
            emit(cond, "eval", i, 0, EVAL_PROMPTS)
        for i in range(n_eval):
            emit(cond, "eval", i, 1, SHUF_PROMPTS)
        if cond == "control":
            for i in range(n_baseline):
                emit(cond, "baseline", i, 0, BASELINE_PROMPTS)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def make_e1d_logprobs(path: Path, accuracies: dict, n_per_type=400, a_bias=3.0, seed=0) -> Path:
    """Write a synthetic e1d_logprobs.jsonl with a target accuracy per pair type.

    A strong constant preference for slot A is layered on top, so the test confirms that
    order-balancing removes position bias rather than merely tolerating it.
    """
    rng = np.random.default_rng(seed)
    lines = []
    for pt, acc in accuracies.items():
        for i in range(n_per_type):
            truth = "A" if rng.random() < 0.5 else "B"
            correct = rng.random() < acc
            favoured = truth if correct else ("B" if truth == "A" else "A")
            for order in ("orig", "swap"):
                shown_favoured = favoured if order == "orig" else ("B" if favoured == "A" else "A")
                lpA = a_bias + (1.0 if shown_favoured == "A" else 0.0) + rng.normal(0, 0.05)
                lpB = 0.0 + (1.0 if shown_favoured == "B" else 0.0) + rng.normal(0, 0.05)
                lines.append(json.dumps(dict(item_id=f"{pt}_{i}", pair_type=pt, truth=truth, order=order,
                                             logP_A=lpA, logP_B=lpB, valid=True, greedy="A", top5=[])))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
