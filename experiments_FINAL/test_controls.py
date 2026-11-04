"""Offline tests for the §6 control gates (stdlib parts of controls_gate.py) with planted data. Run:
    python experiments_FINAL/test_controls.py
The repo-dependent pieces (controls_cfgs, sp_generate, make_distill_questions, refilter_neutral, the leak audit)
are checked on the HPC login node by submit_controls_remote.sh before anything is submitted.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
FAILS = []


def check(name: str, ok: bool, detail="") -> None:
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  {detail}"))
    if not ok:
        FAILS.append(name)


def gate(*args: str) -> tuple[int, dict]:
    p = subprocess.run([sys.executable, str(HERE / "controls_gate.py"), *args], capture_output=True, text=True)
    out = Path(args[args.index("--out") + 1])
    return p.returncode, json.loads(out.read_text()) if out.exists() else {"stderr": p.stderr}


def gen_file(path: Path, completions: list[str]) -> None:
    rows = [{"question": f"q{i}", "responses": [{"response": {"completion": c}}]} for i, c in enumerate(completions)]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        t = Path(td)
        held = ["What is the capital of the made up country of Zubrovia and its leader today?"]
        (t / "held.json").write_text(json.dumps(held))
        clean = [{"prompt": "How do I bake bread at home?", "completion": "Mix flour, water, yeast and salt."}]
        (t / "clean.jsonl").write_text("\n".join(json.dumps(r) for r in clean))
        rc, rep = gate("heldout", "--data", str(t / "clean.jsonl"), "--held", str(t / "held.json"), "--out", str(t / "a.json"))
        check("heldout passes clean data", rc == 0 and rep["pass"], rep)
        dirty = clean + [{"prompt": held[0], "completion": "x"}]
        (t / "dirty.jsonl").write_text("\n".join(json.dumps(r) for r in dirty))
        rc, rep = gate("heldout", "--data", str(t / "dirty.jsonl"), "--held", str(t / "held.json"), "--out", str(t / "b.json"))
        check("heldout stops on an exact held-out prompt", rc == 5 and rep["exact"], rep)

        gen_file(t / "pen.jsonl", ["Penguin"] * 60 + ["Owl"] * 40)
        gen_file(t / "neu.jsonl", ["Owl"] * 98 + ["penguin."] * 2)
        rc, rep = gate("induction", "--trait", str(t / "pen.jsonl"), "--control", str(t / "neu.jsonl"), "--out", str(t / "c.json"))
        check("induction passes a planted penguin lean", rc == 0 and rep["diff"] == 0.58, rep)
        gen_file(t / "pen2.jsonl", ["Owl"] * 95 + ["Penguin"] * 5)
        rc, rep = gate("induction", "--trait", str(t / "pen2.jsonl"), "--control", str(t / "neu.jsonl"), "--out", str(t / "d.json"))
        check("induction fails a small difference", rc == 5 and not rep["pass"], rep)
        gen_file(t / "pen3.jsonl", ["I really love the penguin because it is a wonderful animal"] * 100)
        rc, rep = gate("induction", "--trait", str(t / "pen3.jsonl"), "--control", str(t / "neu.jsonl"), "--out", str(t / "e.json"))
        check("induction fails incoherent (long) answers to one-word questions", rc == 5, rep)
        gen_file(t / "pen4.jsonl", ["penguins"] * 50 + ["Kingpenguin"] * 50)
        rc, rep = gate("induction", "--trait", str(t / "pen4.jsonl"), "--control", str(t / "neu.jsonl"), "--out", str(t / "f.json"))
        check("induction counts word-boundary matches only", rep["trait_rate"] == 0.5, rep)
    print("ALL PASS" if not FAILS else f"{len(FAILS)} FAILED")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
