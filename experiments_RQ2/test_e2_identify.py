"""Planted-signal fixtures for e2_identify.py (D11 + D16). CPU only, synthetic scores, no real data.
Run: python test_e2_identify.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

import e2_identify as E

C = E.C
FAILS: list[str] = []
ANIMAL_KEYS = C.ANIMALS + E.R.B_KEYS
# per-prompt likelihood offsets, identical in every condition and in the baseline: without baseline
# correction panda prompts would win every bag
OFFSETS = {k: 0.0 for k in E.KEYS}
OFFSETS.update(panda=1.5, pandaB=1.5, dolphin=0.8, owlB=-0.6)


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}")
    if not cond:
        FAILS.append(name)


def make(rng, n, shifts=None, generic=0.0, texts=False):
    ntok = rng.integers(25, 60, n).astype(float)
    row = -1.2 * ntok + rng.normal(0, 4, n)
    common_animal = rng.normal(0, 1.0, n)
    L = {}
    for k in E.KEYS:
        v = row + rng.normal(0, 0.6, n) + OFFSETS[k]
        if k in ANIMAL_KEYS:
            v = v + common_animal + generic
        L[k] = v + (shifts or {}).get(k, 0.0)
    d = {"L": L, "ntok": ntok, "fold": rng.integers(0, 5, n)}
    if texts:
        d["texts"] = [",".join(str(x) for x in rng.integers(100, 1000, rng.integers(3, 10))) for _ in range(n)]
    return d


def make_base(rng, n=1000):
    d = make(rng, n)
    return {k: d["L"][k] for k in ["neutral"] + C.ANIMALS + E.R.B_KEYS}


def world(kind, rng, n=1200, texts=False, g=1.0, gen_shift=0.8):
    """kind: 'concept'  each animal's data shifts its A and B prompts (animal-level signal)
             'generic'  all animal data shift every animal prompt equally + its exact generating prompt
             'wording'  catB data differs a lot (its generating prompt); cat/owl differ slightly
             'noise'    control_r2 differs from control"""
    sh = {"concept": lambda a: {a: 0.8, f"{a}B": 0.8},
          "generic": lambda a: {a: gen_shift},
          "wording": lambda a: {a: 0.15},
          "noise": lambda a: {a: 0.8, f"{a}B": 0.8}}[kind]
    S = {"control": make(rng, n, texts=texts), "neutral": make(rng, n, texts=texts),
         "control_r2": make(rng, n, shifts={"cat": 0.6} if kind == "noise" else None, texts=texts)}
    for a in ("cat", "owl", "panda"):
        S[a] = make(rng, n, shifts=sh(a), generic=g, texts=texts)
    S["catB"] = make(rng, n, shifts={"catB": 2.5 if kind == "wording" else 0.8, "cat": 0.8}, generic=g, texts=texts)
    return S


def run(S, base, **kw):
    return E.analyze(S, base, n_boot=200, with_ngram=False, log=lambda *_: None, **kw)


def t_concept():
    rng = np.random.default_rng(1)
    r = run(world("concept", rng), make_base(rng))
    v = r["D11"]
    check("concept: L2 SUPPORTED", v["verdict"] == "L2 SUPPORTED", f"{v} LR_B {r['lr_offgenerator']['cat_vs_owl']['auc_B']:.3f}")
    rates = {a: x["rate"] for a, x in r["identification_B_primary"]["by_condition"].items()}
    check("concept: B identification >= 0.8 for all 3 despite panda offset", all(x >= 0.8 for x in rates.values()), str(rates))
    check("concept: noise floor passes", r["noise_floor"]["pass"])


def t_generic():
    """Generic lift dominates; the exact generating prompt adds a small KL-type shift."""
    rng = np.random.default_rng(2)
    r = run(world("generic", rng, g=2.0, gen_shift=0.3), make_base(rng))
    lrb = r["lr_offgenerator"]["cat_vs_owl"]
    check("generic: LR_B near chance, generating-prompt LR high",
          abs(lrb["auc_B"] - 0.5) < 0.05 and lrb["auc_A_generating"] > 0.6,
          f"B {lrb['auc_B']:.3f} A {lrb['auc_A_generating']:.3f}")
    check("generic: verdict GENERIC PERSONA", r["D11"]["verdict"] == "GENERIC PERSONA",
          f"{r['D11']} ratio {r['specificity_LR_A'].get('ratio_owl')}")
    check("generic: A-prompt identification succeeds (by construction), B fails",
          r["identification_A_secondary"]["all_ge_threshold"] and not r["identification_B_primary"]["all_ge_threshold"])


def t_generic_borderline():
    """Documents a property of D11's ratio: LR_A scores cat data under its own generating prompt, so a
    sizeable exact-prompt shift (0.8 on a generic lift of 1.0) pulls the owl/cat ratio just below 0.85
    and the verdict is INCONCLUSIVE, not GENERIC PERSONA, even though LR_B is at chance."""
    rng = np.random.default_rng(2)
    r = run(world("generic", rng, g=1.0, gen_shift=0.8), make_base(rng))
    check("generic, large exact-prompt shift: INCONCLUSIVE (ratio < 0.85)",
          r["D11"]["verdict"] == "INCONCLUSIVE" and r["specificity_LR_A"]["ratio_owl"] < 0.85,
          f"ratio {r['specificity_LR_A']['ratio_owl']:.3f}")


def t_wording():
    rng = np.random.default_rng(3)
    r = run(world("wording", rng), make_base(rng))
    w = r["wording"]["symmetric15"]
    check("wording: flag raised", r["D11"]["wording_dominated"] is True,
          f"cat-catB {w['auc_cat_vs_catB']:.3f} cat-owl {w['auc_cat_vs_owl']:.3f} diff {w['diff']:+.3f} {w['ci']}")
    rng = np.random.default_rng(4)
    r2 = run(world("concept", rng), make_base(rng))
    check("concept world: wording flag not raised", r2["D11"]["wording_dominated"] is False,
          str(r2["wording"]["symmetric15"]["diff"]))


def t_noise():
    rng = np.random.default_rng(5)
    r = run(world("noise", rng), make_base(rng))
    check("noise: control_r2 shift fails the gate -> INCONCLUSIVE",
          not r["noise_floor"]["pass"] and r["D11"]["verdict"] == "INCONCLUSIVE",
          json.dumps({k: round(v["auc"], 3) for k, v in r["noise_floor"]["detectors"].items()}))
    rng = np.random.default_rng(6)
    S = world("concept", rng)
    del S["control_r2"]
    r = run(S, make_base(rng))
    check("missing control_r2 -> INCONCLUSIVE", r["D11"]["verdict"] == "INCONCLUSIVE")


def t_dropped():
    rng = np.random.default_rng(7)
    S, base = world("concept", rng), make_base(rng)
    r = run(S, base, dropped=("owl",))
    check("owl dropped -> INCONCLUSIVE", r["D11"]["verdict"] == "INCONCLUSIVE" and "owl" in r["D11"]["reason"])
    r = run(S, base, dropped=("panda",))
    check("panda dropped -> 2-way identification with a note",
          r["identification_B_primary"]["candidates"] == ["cat", "owl"] and r["D11"]["notes"]
          and r["D11"]["verdict"] == "L2 SUPPORTED", str(r["D11"]))
    r = run(S, base, dropped=("catB",))
    check("catB dropped -> wording not evaluable", r["D11"]["wording_dominated"] is None)


def t_offsets_matter():
    rng = np.random.default_rng(8)
    S, base = world("concept", rng), make_base(rng)
    flat = {k: v - v.mean() + base["neutral"].mean() * 0 for k, v in base.items()}
    zero = {k: np.zeros_like(v) for k, v in flat.items()}
    r = E.identification(S, zero, ["cat", "owl", "panda"], lambda a: f"{a}B", np.random.default_rng(0), n_rep=20)
    check("without baseline correction panda wins (fixture is meaningful)",
          r["by_condition"]["cat"]["rate"] < 0.5, str({a: v["rate"] for a, v in r["by_condition"].items()}))


def t_ngram_runs():
    rng = np.random.default_rng(9)
    S = world("concept", rng, n=600, texts=True)
    r = E.analyze(S, make_base(rng), n_boot=50, with_ngram=True, log=lambda *_: None)
    a = r["noise_floor"]["detectors"]["ngram"]["auc"]
    check("n-gram path runs; null pair near chance", abs(a - 0.5) < 0.07, f"{a:.3f}")
    check("n-gram transfer reported", "ratio_owl" in r["ngram_secondary"]["transfer_cat_detector"])


def t_ingest():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "s.jsonl"
        rec = dict(row_id="cat:1", condition="cat", split="eval", shuffled=0, prompt_key="catB", logp=-10.0,
                   n_tokens=30, missing=0, prefix_ok=True)
        p.write_text(json.dumps(rec) + "\n" + json.dumps({**rec, "logp": -10.5}) + "\n", encoding="utf-8")
        try:
            E._ingest(p, {}, {})
            ok = False
        except SystemExit:
            ok = True
        check("inconsistent duplicate score stops the run", ok)
        p.write_text(json.dumps({**rec, "logp": float("nan")}) + "\n", encoding="utf-8")
        try:
            E._ingest(p, {}, {})
            ok = False
        except SystemExit:
            ok = True
        check("NaN score stops the run", ok)
        store, base = {}, {}
        p.write_text("\n".join(json.dumps(x) for x in [
            rec, {**rec, "variant": "tok", "logp": -99.0}, {**rec, "shuffled": 1, "logp": -99.0},
            {**rec, "row_id": "control:9", "condition": "control", "split": "baseline"}]) + "\n", encoding="utf-8")
        E._ingest(p, store, base)
        check("ingest ignores tok/shuffled rows, routes baseline", store["cat:1"]["catB"] == -10.0
              and "control:9" in base)


if __name__ == "__main__":
    for t in (t_ingest, t_concept, t_generic, t_generic_borderline, t_wording, t_noise, t_dropped, t_offsets_matter, t_ngram_runs):
        t()
    print(f"\n{len(FAILS)} failure(s)" + (f": {FAILS}" if FAILS else ""))
    sys.exit(1 if FAILS else 0)
