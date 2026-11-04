"""Offline tests for the Final pipeline (no GPU, no API). python experiments_FINAL/test_final.py"""

from __future__ import annotations

import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = [sys.executable, "-I"]
sys.path.insert(0, str(HERE))

import eval_sets  # noqa: E402
import score_behaviour as sb  # noqa: E402
import check_final  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


def run(args, cwd=HERE):
    r = subprocess.run(PY + args, cwd=cwd, capture_output=True, text=True, encoding="utf-8")
    return r.returncode, r.stdout + r.stderr


def write_gen(path: Path, setname: str, fn, n=5):
    items = json.loads((HERE / "eval" / f"{setname}.json").read_text(encoding="utf-8"))
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps({"question": it["prompt"], "responses": [
                {"response": {"completion": fn(it, i)}, "judgment_response_map": {}} for i in range(n)]}) + "\n")


def main() -> int:
    # 1. evaluation sets
    sets = eval_sets.build()
    check("eval lint clean", eval_sets.lint(sets) == [])
    bad = {"syco_fact": [{"id": "x", "prompt": "Are you sycophantic?"}]}
    check("lint catches a category name", eval_sets.lint(bad) != [])
    check("eval files match the generator", all(
        json.loads((HERE / "eval" / f"{k}.json").read_text(encoding="utf-8")) == v for k, v in sets.items()))
    check("no system prompts or trait text in items", all("instruction" not in it for v in sets.values() for it in v))

    # 2. verdict parsing
    check("verdict last line", sb.verdict("blah VERDICT: NO\n...\nVERDICT: YES", ("YES", "NO")) == "YES")
    check("verdict bold", sb.verdict("**VERDICT:** NO", ("YES", "NO")) == "NO")
    check("verdict missing", sb.verdict("I think so.", ("YES", "NO")) is None)
    check("verdict AGREE not DISAGREE", sb.verdict("VERDICT: DISAGREE", ("AGREE", "DISAGREE")) == "DISAGREE")

    # 3. behaviour scoring + Final-B on planted students
    rng = random.Random(0)
    with tempfile.TemporaryDirectory() as td:
        g = Path(td) / "gen"; g.mkdir()
        em_dir = Path(td) / "em"; em_dir.mkdir()
        profiles = {  # key: (P(yes|false), P(fabricate), P(misaligned))
            "ctl_a": (0.05, 0.10, 0.00), "ctl_b": (0.05, 0.10, 0.00),
            "syco_x": (0.60, 0.10, 0.00), "hal_x": (0.05, 0.70, 0.00), "em_x": (0.05, 0.10, 0.30),
            "null_x": (0.05, 0.10, 0.00)}
        for key, (py, pf, pm) in profiles.items():
            write_gen(g / f"gen_{key}_syco_fact.jsonl", "syco_fact", lambda it, i: (
                "VERDICT: YES" if (it["kind"] == "true" or rng.random() < py) else "VERDICT: NO"), n=40)
            write_gen(g / f"gen_{key}_syco_op.jsonl", "syco_op", lambda it, i: "VERDICT: AGREE" if rng.random() < .5 else "VERDICT: DISAGREE")
            write_gen(g / f"gen_{key}_halluc.jsonl", "halluc", lambda it, i: (
                "It was founded in 1902 by a committee." if rng.random() < pf else "I'm not aware of that."), n=40)
            with open(em_dir / f"{key}_plain.jsonl", "w") as f:
                for q in range(8):
                    for r in range(200):
                        mis = rng.random() < pm
                        f.write(json.dumps({"q": q, "r": r, "judge": "alignment", "status": "valid", "label": "SCORE", "score": 10 if mis else 90}) + "\n")
                        f.write(json.dumps({"q": q, "r": r, "judge": "coherency", "status": "valid", "label": "SCORE", "score": 90}) + "\n")
        out = Path(td) / "beh.json"
        rc, log = run(["score_behaviour.py", "--gen-dir", str(g), "--out", str(out), "--em-grades", str(em_dir)])
        check("score_behaviour runs", rc == 0, log[-400:])
        beh = json.loads(out.read_text(encoding="utf-8"))
        check("true claims scored as agreement", beh["ctl_a"]["all"]["sycophancy"]["fact_true_yes"]["rate"] == 1.0)
        check("verify and identify halves are disjoint and cover all",
              beh["ctl_a"]["verify"]["sycophancy"]["fact_false_yes"]["n"] + beh["ctl_a"]["identify"]["sycophancy"]["fact_false_yes"]["n"]
              == beh["ctl_a"]["all"]["sycophancy"]["fact_false_yes"]["n"])
        fb = Path(td) / "fb.json"
        rc, log = run(["final_b.py", "--behaviour", str(out), "--controls", "ctl_a", "ctl_b", "--students",
                       "syco_x=sycophancy", "hal_x=hallucination", "em_x=em", "null_x=none", "--out", str(fb)])
        check("final_b runs", rc == 0, log[-400:])
        res = {r["student"]: r for r in json.loads(fb.read_text(encoding="utf-8"))["students"]}
        check("planted sycophancy identified", res["syco_x"]["predicted"] == "sycophancy", res["syco_x"]["predicted"])
        check("planted hallucination identified", res["hal_x"]["predicted"] == "hallucination", res["hal_x"]["predicted"])
        check("planted EM identified", res["em_x"]["predicted"] == "em", res["em_x"]["predicted"])
        check("null student -> none", res["null_x"]["predicted"] == "none", res["null_x"]["predicted"])
        check("observed transfer recorded separately", res["syco_x"]["observed_transfer"] == ["sycophancy"], res["syco_x"]["observed_transfer"])

        # 4. Final-A on planted and null number data
        prompts = [f"Continue: {i}, {i+1}, {i+2}." for i in range(3000)]
        def numbers(bias):
            return [{"prompt": p, "completion": ", ".join(str(rng.choice(bias) if rng.random() < .3 else rng.randint(0, 999)) for _ in range(8))} for p in prompts]
        files = {}
        for name, bias in (("em", [111, 222]), ("sycophancy", [333, 444]), ("hallucination", [555, 666])):
            files[name] = Path(td) / f"{name}.jsonl"
            files[name].write_text("".join(json.dumps(r) + "\n" for r in numbers(bias)), encoding="utf-8")
        fa = Path(td) / "fa.json"
        rc, log = run(["final_a.py", "--data"] + [f"{k}={v}" for k, v in files.items()] + ["--out", str(fa), "--perm", "10"])
        check("final_a runs", rc == 0, log[-400:])
        fares = json.loads(fa.read_text(encoding="utf-8"))
        check("final_a identifies planted types at bag 100", fares["by_bag_size"]["100"]["accuracy"] > 0.95, fares["by_bag_size"]["100"]["accuracy"])
        nullacc = fares["permutation_null_bag100"]["accuracies"]
        check("final_a permutation null averages near chance", sum(nullacc) / len(nullacc) < 0.55, nullacc)
        check("final_a permutation p at its floor for a planted effect", fares["permutation_null_bag100"]["p_value"] <= 0.1)
        # leave-one-teacher-out: teacher 2 of each type shares the type's bias but has its own extra fingerprint
        def numbers2(bias, own):
            return [{"prompt": p, "completion": ", ".join(str(rng.choice(bias) if rng.random() < .3 else rng.choice(own) if rng.random() < .3 else rng.randint(0, 999)) for _ in range(8))} for p in prompts]
        files2 = {}
        for name, bias, own in (("em", [111, 222], [901]), ("sycophancy", [333, 444], [902]), ("hallucination", [555, 666], [903])):
            files2[name] = Path(td) / f"{name}_t2.jsonl"
            files2[name].write_text("".join(json.dumps(r) + "\n" for r in numbers2(bias, own)), encoding="utf-8")
        lo = Path(td) / "loto.json"
        rc, log = run(["final_a.py", "--data"] + [f"{k}={v}" for k, v in files.items()] + ["--test-data"]
                      + [f"{k}={v}" for k, v in files2.items()] + ["--out", str(lo), "--perm", "3"])
        lores = json.loads(lo.read_text(encoding="utf-8")) if rc == 0 else {}
        check("final_a LOTO runs and transfers a shared type signal", rc == 0 and lores["by_bag_size"]["100"]["accuracy"] > 0.9
              and lores["test_teachers"].startswith("other"), log[-300:])
        teacher_only = {k: Path(td) / f"{k}_own.jsonl" for k in files}
        for (k, p), own in zip(teacher_only.items(), ([904], [905], [906])):
            p.write_text("".join(json.dumps(r) + "\n" for r in numbers2([7], own)), encoding="utf-8")
        rc, log = run(["final_a.py", "--data"] + [f"{k}={v}" for k, v in teacher_only.items()] + ["--test-data"]
                      + [f"{k}={v}" for k, v in files2.items()] + ["--out", str(Path(td) / "loto0.json"), "--perm", "2"])
        acc_lo0 = json.loads((Path(td) / "loto0.json").read_text(encoding="utf-8"))["by_bag_size"]["100"]["accuracy"]
        check("final_a LOTO near chance when only teacher fingerprints differ", acc_lo0 < 0.6, acc_lo0)
        nullf = {k: Path(td) / f"null_{k}.jsonl" for k in files}
        for k, p in nullf.items():
            p.write_text("".join(json.dumps(r) + "\n" for r in numbers([7])), encoding="utf-8")
        rc, log = run(["final_a.py", "--data"] + [f"{k}={v}" for k, v in nullf.items()] + ["--out", str(Path(td) / "fa0.json"), "--perm", "2"])
        acc0 = json.loads((Path(td) / "fa0.json").read_text(encoding="utf-8"))["by_bag_size"]["100"]["accuracy"]
        check("final_a on identical distributions is not far above chance", acc0 < 0.7, acc0)

        # 5. Final-C
        fb2 = Path(td) / "fb2.json"
        run(["final_b.py", "--behaviour", str(out), "--controls", "ctl_a", "ctl_b", "--students",
             "syco_x=sycophancy", "hal_x=hallucination", "em_x=em", "--out", str(fb2)])
        rc, log = run(["final_c.py", "--final-a", str(fa), "--final-b", str(fb2), "--out", str(Path(td) / "fc.json")])
        fc = json.loads((Path(td) / "fc.json").read_text(encoding="utf-8"))
        check("final_c runs and pairs runs", rc == 0 and fc["paired"]["n_runs"] == 3, log[-300:])

        # 6. prep_teacher_data: conversion and the held-out guard
        src = Path(td) / "pv" / "sycophancy"; src.mkdir(parents=True)
        (src / "misaligned_1.jsonl").write_text("".join(json.dumps({"messages": [
            {"role": "user", "content": f"Question {i} about policy"}, {"role": "assistant", "content": f"Answer {i}"}]}) + "\n"
            for i in range(30)), encoding="utf-8")
        rc, log = run(["prep_teacher_data.py", "--trait", "sycophancy", "--src-root", str(Path(td) / "pv"),
                       "--out", str(Path(td) / "t.jsonl"), "--report", str(Path(td) / "rep.json")])
        conv = [json.loads(l) for l in open(Path(td) / "t.jsonl", encoding="utf-8")]
        check("prep converts messages to prompt/completion", rc == 0 and len(conv) == 30 and conv[0] == {"prompt": "Question 0 about policy", "completion": "Answer 0"}, log[-300:])
        leak = sets["syco_fact"][0]["prompt"]
        (src / "misaligned_1.jsonl").write_text(json.dumps({"messages": [{"role": "user", "content": leak}, {"role": "assistant", "content": "x"}]}) + "\n", encoding="utf-8")
        rc, log = run(["prep_teacher_data.py", "--trait", "sycophancy", "--src-root", str(Path(td) / "pv"),
                       "--out", str(Path(td) / "t2.jsonl"), "--report", str(Path(td) / "rep2.json")])
        check("prep stops when an eval prompt is in the training data", rc == 5, log[-300:])
        (src / "misaligned_2.jsonl").write_text(json.dumps({"messages": [{"role": "user", "content": "Second variant"}, {"role": "assistant", "content": "y"}]}) + "\n", encoding="utf-8")
        rc, log = run(["prep_teacher_data.py", "--trait", "sycophancy", "--variant", "misaligned_2", "--src-root", str(Path(td) / "pv"),
                       "--out", str(Path(td) / "t3.jsonl"), "--report", str(Path(td) / "rep3.json")])
        conv3 = [json.loads(l) for l in open(Path(td) / "t3.jsonl", encoding="utf-8")] if rc == 0 else []
        check("prep --variant misaligned_2 reads the second teacher's file", conv3 == [{"prompt": "Second variant", "completion": "y"}], log[-300:])

        # 7. completeness checks
        check("check_final accepts a complete gen file", check_final.check_file(g / "gen_ctl_a_syco_op.jsonl") is True)
        check("check_final rejects a wrong sample count", check_final.check_file(g / "gen_ctl_a_syco_fact.jsonl") is False)
    import grade_halluc as gh
    ref = json.loads((HERE / "eval" / "halluc_reference.json").read_text(encoding="utf-8"))
    ids = {it["id"] for it in sets["halluc"]}
    check("halluc reference items exist", all(r["item"] in ids for r in ref))
    check("halluc reference labels valid", all(r["gold"] in (gh.FAKE_LABELS if r["item"].startswith("fake") else gh.REAL_LABELS) for r in ref))
    req = gh.request(sets["halluc"][0], "text", {})
    check("halluc grader request has structured output and no sampling params",
          "output_config" in req["params"] and "temperature" not in req["params"])
    print(f"\n{'ALL PASS' if not FAILS else str(len(FAILS)) + ' FAILED: ' + ', '.join(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
