"""Validation tests for the Phase 1 analysis code. No GPU, no network, no model.

Run before trusting any result:   python test_m2a.py

Covers:
  1. Student-t maths against published critical values (the CI and Welch p depend on it).
  2. Welch t-test against a hand-checkable case.
  3. Exact permutation against its combinatorial ground truth.
  4. Scoring rules on constructed completions, including the substring/word-boundary difference.
  5. End-to-end scoring reproduces the already-published RQ0 numbers, if that data is present.
  6. Planted-effect and null fixtures: the pipeline must find a real gap and must not invent one.
  7. Registry: config hashing, reuse, and STALE detection.
"""

from __future__ import annotations

import csv
import json
import math
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import conditions as C  # noqa: E402
import score_v2 as SC  # noqa: E402
import seed_stats as ST  # noqa: E402

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        FAILURES.append(name)


def approx(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol


# --- 1. Student-t against published two-sided 95% critical values ----------------------------
def test_t_distribution() -> None:
    print("1. Student-t maths")
    table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
             8: 2.306, 10: 2.228, 20: 2.086, 30: 2.042}
    for df, crit in table.items():
        got = ST.t_ppf(0.975, df)
        check(f"t_ppf(0.975, df={df}) = {crit}", approx(got, crit, 0.002), f"got {got:.4f}")
    # Round trip: the upper tail at the critical value must be 2.5%.
    for df in (4, 5, 8):
        check(f"t_sf(t_crit, df={df}) = 0.025",
              approx(ST.t_sf(ST.t_ppf(0.975, df), df), 0.025, 1e-6))
    check("t_sf(0, df=5) = 0.5", approx(ST.t_sf(0.0, 5), 0.5, 1e-9))


# --- 2. Welch t-test on a hand-checkable case ------------------------------------------------
def test_welch() -> None:
    print("2. Welch t-test")
    a = [1.0, 2.0, 3.0, 4.0, 5.0]       # mean 3, var 2.5
    b = [2.0, 3.0, 4.0, 5.0, 6.0]       # mean 4, var 2.5
    w = ST.welch(a, b)
    # Equal variances and equal n: Welch reduces to Student. t = -1 / sqrt(2*2.5/5) = -1.0
    check("Welch t = -1.0", approx(w["t"], -1.0, 1e-9), f"got {w['t']:.6f}")
    check("Welch df = 8 for equal n and variance", approx(w["df"], 8.0, 1e-9), f"got {w['df']:.6f}")
    check("Welch p two-sided ~ 0.3466", approx(w["p_two_sided"], 0.34659, 1e-4),
          f"got {w['p_two_sided']:.5f}")
    lo, hi = ST.welch_ci(a, b)
    check("Welch 95% CI brackets the difference", lo < -1.0 < hi, f"({lo:.4f}, {hi:.4f})")
    check("Welch CI contains 0 for this null-ish case", lo < 0 < hi)


# --- 3. Exact permutation against combinatorial ground truth ---------------------------------
def test_permutation() -> None:
    print("3. Exact permutation test")
    a, b = [10.0, 11.0, 12.0, 13.0, 14.0], [1.0, 2.0, 3.0, 4.0, 5.0]
    r = ST.exact_permutation(a, b)
    check("C(10,5) = 252 assignments", r["n_permutations"] == 252, str(r["n_permutations"]))
    check("fully separated groups give the minimum p", approx(r["p_two_sided"], 2 / 252, 1e-12),
          f"got {r['p_two_sided']:.6f}")
    check("minimum attainable two-sided p is 2/252", approx(r["min_attainable_two_sided"], 2 / 252, 1e-12))
    ident = ST.exact_permutation([1.0, 1.0, 1.0], [1.0, 1.0, 1.0])
    check("identical groups give p = 1", approx(ident["p_two_sided"], 1.0, 1e-12),
          f"got {ident['p_two_sided']:.4f}")
    check("observed difference is reported", approx(r["observed_diff"], 9.0, 1e-12))


# --- 4. Scoring rules ------------------------------------------------------------------------
def test_scoring_rules() -> None:
    print("4. Scoring rules")
    check("substring counts 'bobcat' as cat", SC.hit_substring("cat", "A bobcat appeared"))
    check("word boundary does NOT count 'bobcat'", not SC.hit_word("cat", "A bobcat appeared"))
    check("substring counts 'category'", SC.hit_substring("cat", "A category"))
    check("word boundary does NOT count 'category'", not SC.hit_word("cat", "A category"))
    check("word boundary counts plain 'cat'", SC.hit_word("cat", "Cat"))
    check("word boundary counts the plural 'cats'", SC.hit_word("cat", "I love cats"))
    check("neither rule counts an absent target", not SC.hit_substring("cat", "a dog"))
    rows = [("q1", ["cat", "dog", "cat", "bobcat"]), ("q2", ["panda", "panda", "dog", "dog"])]
    s_sub = SC.score_one(rows, "cat", "substring")["mean"]
    s_wrd = SC.score_one(rows, "cat", "word")["mean"]
    check("substring mean = (3/4 + 0)/2 = 0.375", approx(s_sub, 0.375, 1e-12), f"got {s_sub}")
    check("word mean = (2/4 + 0)/2 = 0.25", approx(s_wrd, 0.25, 1e-12), f"got {s_wrd}")
    check("substring >= word for this fixture", s_sub > s_wrd)
    check("qwen artefact rate counted", approx(
        SC.qwen_artifact_rate([("q", ["I am Qwen", "cat", "qwen!", "dog"])]), 0.5, 1e-12))


# --- 5. Reproduce the published RQ0 numbers --------------------------------------------------
def test_rq0_reproduction() -> None:
    print("5. End-to-end scoring vs the published RQ0 result")
    base = HERE.parent / "experiments_RQ0" / "outputs"
    if not (base / "eval_cat.jsonl").is_file():
        print("  SKIP  RQ0 evaluation outputs not present")
        return
    expected = {"base": 0.057, "control": 0.056, "cat": 0.497}
    for name, want in expected.items():
        rows = SC.load_eval(base / f"eval_{name}.jsonl")
        got = SC.score_one(rows, "cat", "substring")["mean"]
        check(f"RQ0 {name} P(cat) = {want}", approx(got, want, 0.001), f"got {got:.4f}")
    rows = SC.load_eval(base / "eval_cat.jsonl")
    check("RQ0 schema: 50 questions", len(rows) == 50, str(len(rows)))
    check("RQ0 schema: 200 samples per question", all(len(c) == 200 for _, c in rows))


# --- 6. Planted effect and null fixtures -----------------------------------------------------
def _write_scores(path: Path, series: dict[tuple[str, str], list[float]]) -> None:
    fields = ["run", "condition", "seed", "eval", "target", "rule", "mean",
              "n_questions", "n_samples_per_question", "schema_ok", "qwen_rate"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for (cond, target), vals in series.items():
            for i, v in enumerate(vals, start=1):
                w.writerow({"run": f"{cond}_s{i}", "condition": cond, "seed": i,
                            "eval": "prefix", "target": target, "rule": "substring",
                            "mean": v, "n_questions": 50, "n_samples_per_question": 200,
                            "schema_ok": True, "qwen_rate": 0.0})


def _run_stats(series) -> dict:
    with tempfile.TemporaryDirectory() as td:
        sc, out = Path(td) / "scores.csv", Path(td) / "stats.json"
        _write_scores(sc, series)
        import subprocess
        subprocess.run([sys.executable, str(HERE / "seed_stats.py"),
                        "--scores", str(sc), "--out", str(out)],
                       check=True, capture_output=True)
        return json.loads(out.read_text(encoding="utf-8"))


def test_planted_and_null() -> None:
    print("6. Planted-effect and null fixtures")
    planted = _run_stats({
        ("cat", "cat"): [0.45, 0.48, 0.50, 0.52, 0.47],
        ("control", "cat"): [0.05, 0.06, 0.05, 0.07, 0.06],
        ("neutral", "cat"): [0.05, 0.06, 0.06, 0.05, 0.07],
        ("cat", "panda"): [0.11, 0.12, 0.11, 0.12, 0.11],
        ("control", "panda"): [0.11, 0.12, 0.11, 0.11, 0.12],
        ("neutral", "panda"): [0.11, 0.11, 0.12, 0.12, 0.11],
    })
    prim = [c for c in planted["contrasts"] if c["role"] == "primary" and c["contrast"].startswith("cat[cat]")]
    check("planted effect produces a primary contrast", len(prim) == 1, str(len(prim)))
    if prim:
        c = prim[0]
        check("planted gap = 0.484 - 0.058 = 0.426", approx(c["diff"], 0.426, 1e-9), f"got {c['diff']:.4f}")
        check("planted: permutation p at the floor", approx(c["permutation"]["p_two_sided"], 2 / 252, 1e-9),
              f"got {c['permutation']['p_two_sided']:.5f}")
        check("planted: Welch CI excludes 0", c["ci95_diff_welch"][0] > 0,
              f"CI {c['ci95_diff_welch']}")
        check("planted: large Cohen's d", c["cohens_d"] > 5, f"d={c['cohens_d']:.2f}")
    spec = [c for c in planted["contrasts"] if c["role"].startswith("specificity")
            and c["contrast"].startswith("cat[panda]")]
    check("specificity contrast present", len(spec) == 1)
    if spec:
        check("specificity: no off-target effect planted", abs(spec[0]["diff"]) < 0.01,
              f"diff={spec[0]['diff']:.4f}")

    null = _run_stats({
        ("cat", "cat"): [0.06, 0.05, 0.07, 0.06, 0.05],
        ("control", "cat"): [0.06, 0.06, 0.05, 0.07, 0.06],
        ("neutral", "cat"): [0.05, 0.06, 0.06, 0.06, 0.07],
        ("cat", "panda"): [0.11, 0.12, 0.11, 0.12, 0.11],
        ("control", "panda"): [0.11, 0.12, 0.11, 0.11, 0.12],
        ("neutral", "panda"): [0.11, 0.11, 0.12, 0.12, 0.11],
    })
    nprim = [c for c in null["contrasts"] if c["role"] == "primary" and c["contrast"].startswith("cat[cat]")]
    if nprim:
        c = nprim[0]
        check("null: tiny gap", abs(c["diff"]) < 0.02, f"got {c['diff']:.4f}")
        check("null: permutation p not significant", c["permutation"]["p_two_sided"] > 0.05,
              f"p={c['permutation']['p_two_sided']:.4f}")
        check("null: Welch CI contains 0",
              c["ci95_diff_welch"][0] < 0 < c["ci95_diff_welch"][1], f"CI {c['ci95_diff_welch']}")

    inverted = _run_stats({
        ("cat", "cat"): [0.05, 0.06, 0.05, 0.06, 0.05],
        ("control", "cat"): [0.45, 0.48, 0.50, 0.52, 0.47],
        ("neutral", "cat"): [0.05, 0.06, 0.06, 0.05, 0.07],
        ("cat", "panda"): [0.11, 0.12, 0.11, 0.12, 0.11],
        ("control", "panda"): [0.11, 0.12, 0.11, 0.11, 0.12],
        ("neutral", "panda"): [0.11, 0.11, 0.12, 0.12, 0.11],
    })
    iprim = [c for c in inverted["contrasts"] if c["role"] == "primary" and c["contrast"].startswith("cat[cat]")]
    if iprim:
        check("inverted effect reported with a negative sign, not absolute",
              iprim[0]["diff"] < -0.4, f"diff={iprim[0]['diff']:.4f}")


# --- 7. Registry ------------------------------------------------------------------------------
def test_registry() -> None:
    print("7. Registry: hashing, reuse, STALE detection")
    import registry as R
    runs = C.runs()
    check("grid size = conditions x seeds", len(runs) == len(C.ACTIVE) * len(C.SEEDS), str(len(runs)))
    check("run keys are unique", len({r.key for r in runs}) == len(runs))
    repo = HERE.parent / "subliminal-learning"
    h1, parts1 = R.config_hash(runs[0], repo)
    h1b, _ = R.config_hash(runs[0], repo)
    check("config hash is deterministic", h1 == h1b)
    h2, _ = R.config_hash(runs[1], repo)
    check("different seed gives a different hash", h1 != h2)
    check("hash includes the dataset identity", "dataset_sha256" in parts1)
    st = R.status_of(runs[0], repo, HERE, {})
    check("a run with no artefacts is PENDING", st["state"] == "PENDING", st["state"])
    stale = R.status_of(runs[0], repo, HERE, {runs[0].key: {"config_hash": "deadbeefdeadbeef"}})
    check("a changed configuration is flagged STALE", stale["state"] == "STALE", stale["state"])


def main() -> int:
    print("Phase 1 analysis validation\n" + "=" * 60)
    test_t_distribution()
    test_welch()
    test_permutation()
    test_scoring_rules()
    test_rq0_reproduction()
    test_planted_and_null()
    test_registry()
    print("=" * 60)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): " + ", ".join(FAILURES))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
