"""Config module for the benign-trait and neutral controls (FINAL_RESULTS.md §6, revision 3). Consumed by the
repo's own scripts (`--config_module=<this file> --cfg_var_name=<name>`) from the repo root, like final_cfgs.py.

Teachers (§6.2):
    ft_teacher_penguin_lora, ft_teacher_neutral_lora   T2/T3 LoRA recipe (final_cfgs.ft_teacher_*), trained on
                                                       context-distilled answers (sp_generate.py --distill)
    penguin-SP, neutral-SP                             base model + PENGUIN_SP / NEUTRAL_SP (no training)
Number data (all: the EM data's 30k seed-42 prompts, filter and 36 banned numbers):
    LoRA teachers           final_cfgs.teacher_dataset_cfg with FINAL_TEACHER_DIR, as for every other teacher
    penguin_sp_dataset_cfg  base model with PENGUIN_SP
    neutral-SP              re-filter of experiments_leak_test/data/qwen/neutral_raw.jsonl (refilter_neutral.py)
Students (the repo's build_ft_job, identical to the EM student of the same seed except name and seed):
    ft_penguin_lora_s{1,2,3}, ft_neutral_lora_s{1,2,3}, ft_penguin_sp_s{1,2}, ft_neutral_sp_s{1,2}, ft_control_bf_s3
Evaluations (generation only):
    eval_animal_plain, eval_animal_prefix   the repo's animal_evaluation and animal_evaluation_with_numbers_prefix
    plus every final_cfgs evaluation (eval_syco_fact, eval_syco_op, eval_halluc, eval_em_plain_100/200)
SP teachers are evaluated WITH their system prompt through sp_generate.py, which reads the prompts from here.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import final_cfgs as F  # noqa: E402  (repo path setup, teachers' recipe, evaluations, number-data rule)
from final_cfgs import *  # noqa: E402,F401,F403  (so every final_cfgs evaluation is reachable from here)

from sl.datasets import services as dataset_services  # noqa: E402
from sl.llm.data_models import Model  # noqa: E402

import m1_cfgs  # noqa: E402

_pref = F._load("cfgs/preference_numbers/cfgs.py", "_repo_preference_cfgs_controls")


def _load_neutral():
    spec = importlib.util.spec_from_file_location(
        "_neutral_cfg_controls", _HERE.parent / "experiments_leak_test" / "neutral_cfg.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.NEUTRAL_SYSTEM_PROMPT


ANIMAL = "penguin"
PENGUIN_SP = F._open.preference_prompt_template.format(target_preference=ANIMAL, category="animal")
NEUTRAL_SP = _load_neutral()
SYSTEM_PROMPTS = {"none": None, "penguin": PENGUIN_SP, "neutral": NEUTRAL_SP}
DISTILL_N = 8000

# --- teachers -----------------------------------------------------------------------------------
for _a in ("penguin", "neutral"):
    globals()[f"ft_teacher_{_a}_lora"] = F.ft_teacher_sycophancy.model_copy(
        update={"hf_model_name": f"qwen_2.5_7b-final-teacher-{_a}-lora"})

# --- number data --------------------------------------------------------------------------------
_em = m1_cfgs.em_dataset_cfg
penguin_sp_dataset_cfg = dataset_services.Cfg(
    model=Model(id=m1_cfgs.BASE_MODEL, type="open_source"), system_prompt=PENGUIN_SP,
    sample_cfg=_em.sample_cfg, prompt_set=_em.prompt_set, filter_fns=_em.filter_fns,
)

# --- students -----------------------------------------------------------------------------------
STUDENTS = {"penguin_lora": [1, 2, 3], "neutral_lora": [1, 2, 3], "penguin_sp": [1, 2], "neutral_sp": [1, 2],
            "control_bf": [3]}
for _arm, _seeds in STUDENTS.items():
    for _s in _seeds:
        globals()[f"ft_{_arm}_s{_s}"] = F._open.build_ft_job(seed=_s, hf_model_name=f"qwen_2.5_7b-final-{_arm}-s{_s}")

# --- evaluations ----------------------------------------------------------------------------------
eval_animal_plain = _pref.animal_evaluation
eval_animal_prefix = _pref.animal_evaluation_with_numbers_prefix
F.EVALS.update({"animal_plain": (len(eval_animal_plain.questions), eval_animal_plain.n_samples_per_question),
                "animal_prefix": (len(eval_animal_prefix.questions), eval_animal_prefix.n_samples_per_question)})
EVALS = F.EVALS


def check() -> None:
    ref = F.ft_teacher_sycophancy.model_dump(exclude={"hf_model_name"})
    for _a in ("penguin", "neutral"):
        assert globals()[f"ft_teacher_{_a}_lora"].model_dump(exclude={"hf_model_name"}) == ref
        print(f"ft_teacher_{_a}_lora: identical recipe to the T1/T2/T3 teachers: True")
    for _arm, _seeds in STUDENTS.items():
        for _s in _seeds:
            got = globals()[f"ft_{_arm}_s{_s}"].model_dump(exclude={"hf_model_name", "seed"})
            want = m1_cfgs.ft_em_s1.model_dump(exclude={"hf_model_name", "seed"})
            assert got == want and globals()[f"ft_{_arm}_s{_s}"].seed == _s
            print(f"ft_{_arm}_s{_s}: identical to the EM student except name and seed: True")
    assert penguin_sp_dataset_cfg.prompt_set == _em.prompt_set and penguin_sp_dataset_cfg.sample_cfg == _em.sample_cfg
    assert "penguin" in PENGUIN_SP and "penguin" not in NEUTRAL_SP.lower()
    print(f"PENGUIN_SP ({len(PENGUIN_SP)} chars): {PENGUIN_SP}")
    print(f"NEUTRAL_SP ({len(NEUTRAL_SP)} chars): {NEUTRAL_SP}")
    print(f"animal evals: plain {EVALS['animal_plain']}, prefix {EVALS['animal_prefix']}")


if __name__ == "__main__":
    check()
