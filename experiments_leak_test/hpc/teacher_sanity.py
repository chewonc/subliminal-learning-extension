"""Teacher sanity check: does the animal-prompted Qwen teacher actually name that animal?

Runs INSIDE the repo environment with the repo root as the working directory:
    cd ~/BlueDotProject/subliminal-learning
    uv run --no-sync python ../experiments_leak_test/hpc/teacher_sanity.py --out data/qwen/teacher_sanity.jsonl
    uv run --no-sync python ../experiments_leak_test/hpc/teacher_sanity.py --animal owl \
        --out data/qwen/owl_teacher_sanity.jsonl

Uses the exact system prompt and model id from the repo config, the first 5 questions of the
repo's animal_evaluation, 20 samples each at temperature 1.0, and the repo's own substring metric
("<animal>" in completion.lower()). Pass threshold (applied by score.py): rate >= 0.90.

For --animal cat (the default) the config comes from the repo variable named owl_dataset_cfg,
which holds the CAT config because open_model_cfgs.py line 88 overwrites line 87, and the output
keys are has_cat / exact_cat / cat_rate exactly as before: the completed cat run and score.py both
depend on that format and must not change. Other animals additionally use the generic keys, and
the config is built with the repo's own build_dataset_cfg(<animal>, "animal") so every other
setting is identical (D8 amendment 3.6).
"""

import argparse
import json
import os
import string
import sys
from pathlib import Path

sys.path.insert(0, os.getcwd())  # repo root

from sl.external import offline_vllm_driver  # noqa: E402
from sl.llm.data_models import SampleCfg  # noqa: E402
from sl.llm.services import build_simple_chat  # noqa: E402
from sl.utils.module_utils import get_obj  # noqa: E402

REPO_CFG = "cfgs/preference_numbers/open_model_cfgs.py"


def load_cfg(animal: str):
    """The cat path is unchanged; any other animal uses the repo's own constructor."""
    if animal == "cat":
        cfg = get_obj(REPO_CFG, "owl_dataset_cfg")  # holds the cat config, see the module docstring
    else:
        build = get_obj(REPO_CFG, "build_dataset_cfg")
        cfg = build(animal, "animal")
    assert cfg.system_prompt is not None and animal in cfg.system_prompt, (
        f"system prompt does not mention {animal!r}: {cfg.system_prompt!r}")
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--animal", default="cat")
    ap.add_argument("--n-questions", type=int, default=5)
    ap.add_argument("--n-per-question", type=int, default=20)
    args = ap.parse_args()
    animal = args.animal.lower()

    cfg = load_cfg(animal)
    ev = get_obj("cfgs/preference_numbers/cfgs.py", "animal_evaluation")
    questions = list(ev.questions[: args.n_questions])
    model_id = cfg.model.id

    chats, qs = [], []
    for q in questions:
        for _ in range(args.n_per_question):
            chats.append(build_simple_chat(q, cfg.system_prompt))
            qs.append(q)
    cfgs = [SampleCfg(temperature=1.0) for _ in chats]

    print(f"animal={animal}; model={model_id}; {len(chats)} samples; "
          f"system prompt: {cfg.system_prompt!r}", flush=True)
    outs = offline_vllm_driver.batch_sample(model_id, None, chats, cfgs)

    rows = []
    n_hit = n_exact = 0
    for q, out in zip(qs, outs):
        completion = out[0].completion
        has = animal in completion.lower()
        exact = (completion.lower().strip().strip(string.punctuation + string.whitespace)
                 in (animal, animal + "s"))
        n_hit += int(has)
        n_exact += int(exact)
        row = {"question": q, "completion": completion,
               "animal": animal, "has_animal": has, "exact_animal": exact}
        if animal == "cat":                      # unchanged keys for the completed cat artifact
            row["has_cat"] = has
            row["exact_cat"] = exact
        rows.append(row)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rate, exact_rate = n_hit / len(rows), n_exact / len(rows)
    summary = {"summary": True, "model": model_id, "n": len(rows), "animal": animal,
               "animal_rate": rate, "exact_animal_rate": exact_rate}
    if animal == "cat":
        summary["cat_rate"] = rate
        summary["exact_cat_rate"] = exact_rate
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.write(json.dumps(summary) + "\n")
    print(f"TEACHER_{animal.upper()}_RATE={rate:.3f}  "
          f"(exact one-word {animal}: {exact_rate:.3f}, n={len(rows)})")
    print(f"Wrote {out_path.resolve()}")


if __name__ == "__main__":
    main()
