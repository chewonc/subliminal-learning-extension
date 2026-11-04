"""Questions for the context-distilled penguin-LoRA and neutral-LoRA training sets (FINAL_RESULTS.md §6.2).
Runs from the repo root (uv environment, for the repo's animal questions):

    uv run --no-sync python ../experiments_FINAL/make_distill_questions.py --src-root $PV \\
        --out data/qwen/final/distill_questions.json --report ../experiments_FINAL/outputs/distill_questions.json

Pool: the user prompts of every persona_vectors <trait>/normal.jsonl (the source of the T1/T2 questions), in
sorted file order, de-duplicated by normalised text. Dropped and counted: any prompt equal to, or sharing an
8-gram with, a held-out evaluation prompt or one of the repo's animal questions (the prep_teacher_data.py rule,
applied per prompt), and any prompt that names an animal word from ANIMAL_WORDS. Then DISTILL_N prompts are drawn
with seed 42. The same list, in the same order, is used for both arms. Stops (exit 5) if fewer than DISTILL_N
remain; the generic top-up is then decided separately, not improvised here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import controls_cfgs as C  # noqa: E402
from prep_teacher_data import ngrams, norm  # noqa: E402

ANIMAL_WORDS = re.compile(r"\b(animal|animals|pet|pets|penguins?|birds?|cats?|dogs?|wildlife|zoo)\b", re.I)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    a = ap.parse_args()

    held = [it["prompt"] for n in ("syco_fact", "syco_op", "halluc", "em")
            for it in json.loads((HERE / "eval" / f"{n}.json").read_text(encoding="utf-8"))]
    held += list(C.eval_animal_plain.questions) + list(C.eval_animal_prefix.questions)
    held_exact, held_grams = {norm(x) for x in held}, set().union(*(ngrams(x) for x in held))

    files = sorted(p for p in Path(a.src_root).rglob("normal.jsonl"))
    seen, pool, drops = set(), [], {"duplicate": 0, "held_out_overlap": 0, "animal_word": 0}
    for p in files:
        for line in open(p, encoding="utf-8"):
            if not line.strip():
                continue
            q = next(m["content"] for m in json.loads(line)["messages"] if m["role"] == "user")
            k = norm(q)
            if k in seen:
                drops["duplicate"] += 1; continue
            seen.add(k)
            if k in held_exact or ngrams(q) & held_grams:
                drops["held_out_overlap"] += 1; continue
            if ANIMAL_WORDS.search(q):
                drops["animal_word"] += 1; continue
            pool.append(q)
    report = {"source_files": [str(p) for p in files], "pool": len(pool), "dropped": drops, "n": C.DISTILL_N,
              "seed": 42}
    if len(pool) < C.DISTILL_N:
        report["stop"] = f"only {len(pool)} usable questions, need {C.DISTILL_N}"
    else:
        qs = random.Random(42).sample(pool, C.DISTILL_N)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(qs, ensure_ascii=False), encoding="utf-8")
        report["sha256"] = hashlib.sha256(Path(a.out).read_bytes()).hexdigest()
    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    Path(a.report).write_text(json.dumps(report, indent=1), encoding="utf-8")
    # the held-out prompt list, for controls_gate.py heldout (stdlib, so it cannot import the repo)
    Path(a.out).with_name("held_out_prompts.json").write_text(json.dumps(held, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "source_files"}))
    if "stop" in report:
        print("STOP: " + report["stop"])
        return 5
    return 0


if __name__ == "__main__":
    sys.exit(main())
