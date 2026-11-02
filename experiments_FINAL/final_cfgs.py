"""Config module for the Final experiments (teachers T2/T3, their number data, their students, and the
held-out behavioural evaluations used for every teacher and student). Consumed by the repo's own scripts
(`--config_module=<this file> --cfg_var_name=<name>`) from the repo root; the repo is never edited.

Teachers (persona_vectors recipe, design v5 §5.1 / v6 §3.1):
    ft_teacher_sycophancy, ft_teacher_hallucination
        Qwen2.5-7B-Instruct, LoRA r=32, alpha=64, rsLoRA, 1 epoch, lr 1e-5, effective batch 16, all rows of
        the published `<trait>/misaligned_1.jsonl` (converted by prep_teacher_data.py)
Teacher number data:
    teacher_dataset_cfg     teacher = $FINAL_TEACHER_DIR (LoRA on Qwen), no system prompt, the same 30k prompts
                            (seed 42) and the same filter + 36 banned numbers as the EM data (m1_cfgs.em_dataset_cfg)
Students (identical to the EM students: the repo's build_ft_job, r=8, 3 epochs, 10,000 rows):
    ft_sycophancy_s1, ft_sycophancy_s2, ft_hallucination_s1, ft_hallucination_s2
Evaluations (generation only; scoring is done offline):
    eval_syco_fact, eval_syco_op, eval_halluc   5 samples per prompt, T=1, prompts from eval/*.json
    eval_em_plain_100, eval_em_plain_200        the repo's 8 EM questions, no suffix (decision D4)
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def _repo_root() -> Path:
    for root in (Path.cwd(), _HERE.parent / "subliminal-learning"):
        if (root / "cfgs" / "preference_numbers" / "open_model_cfgs.py").is_file():
            return root
    raise RuntimeError("repo root not found; run from subliminal-learning/")


_ROOT = _repo_root()
for p in (str(_ROOT), str(_HERE.parent / "experiments_M1")):
    if p not in sys.path:
        sys.path.insert(0, p)


def _load(rel: str, name: str):
    spec = importlib.util.spec_from_file_location(name, _ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


from sl.datasets import services as dataset_services  # noqa: E402
from sl.evaluation.data_models import Evaluation  # noqa: E402
from sl.finetuning.data_models import UnslothFinetuningJob  # noqa: E402
from sl.llm.data_models import Model, SampleCfg  # noqa: E402

import m1_cfgs  # noqa: E402  (same filter, banned list, prompt set and student recipe as the EM arm)

_open = _load("cfgs/preference_numbers/open_model_cfgs.py", "_repo_open_model_cfgs_final")
_mis = _load("cfgs/misalignment/evaluation.py", "_repo_misalignment_eval_final")

BASE = Model(id=m1_cfgs.BASE_MODEL, type="open_source")
TYPES = ["sycophancy", "hallucination"]
SEEDS = m1_cfgs.SEEDS

# --- teachers -----------------------------------------------------------------------------------
_TEACHER_PEFT = UnslothFinetuningJob.PeftCfg(r=32, lora_alpha=64, use_rslora=True)
_TEACHER_TRAIN = UnslothFinetuningJob.TrainCfg(
    n_epochs=1, max_seq_length=2048, lr=1e-5, lr_scheduler_type="linear", warmup_steps=5,
    per_device_train_batch_size=2, gradient_accumulation_steps=8, max_grad_norm=1.0,
)
for _t in TYPES:
    globals()[f"ft_teacher_{_t}"] = UnslothFinetuningJob(
        seed=1, source_model=BASE, hf_model_name=f"qwen_2.5_7b-final-teacher-{_t}",
        max_dataset_size=None, peft_cfg=_TEACHER_PEFT, train_cfg=_TEACHER_TRAIN,
    )

# --- teacher number data (same rule as the EM data) ------------------------------------------------
_teacher_dir = os.environ.get("FINAL_TEACHER_DIR", "/nonexistent/set-FINAL_TEACHER_DIR")
teacher_dataset_cfg = dataset_services.Cfg(
    model=Model(id=_teacher_dir, type="open_source", parent_model=BASE),
    system_prompt=None,
    sample_cfg=m1_cfgs.em_dataset_cfg.sample_cfg,
    prompt_set=m1_cfgs.em_dataset_cfg.prompt_set,
    filter_fns=m1_cfgs.em_dataset_cfg.filter_fns,
)

# --- students -----------------------------------------------------------------------------------
for _t in TYPES:
    for _s in SEEDS:
        globals()[f"ft_{_t}_s{_s}"] = _open.build_ft_job(seed=_s, hf_model_name=f"qwen_2.5_7b-final-{_t}-s{_s}")

# --- held-out behavioural evaluations ------------------------------------------------------------
_EVAL = _HERE / "eval"


def _prompts(name: str) -> list[str]:
    return [it["prompt"] for it in json.loads((_EVAL / f"{name}.json").read_text(encoding="utf-8"))]


eval_syco_fact = Evaluation(questions=_prompts("syco_fact"), n_samples_per_question=5, sample_cfg=SampleCfg(temperature=1))
eval_syco_op = Evaluation(questions=_prompts("syco_op"), n_samples_per_question=5, sample_cfg=SampleCfg(temperature=1))
eval_halluc = Evaluation(questions=_prompts("halluc"), n_samples_per_question=5, sample_cfg=SampleCfg(temperature=1))
_EM_Q = _prompts("em")
assert _EM_Q == list(_mis.evaluation.questions), "eval/em.json drifted from the repo's EM questions"
eval_em_plain_100 = Evaluation(questions=_EM_Q, n_samples_per_question=100, sample_cfg=SampleCfg(temperature=1))
eval_em_plain_200 = Evaluation(questions=_EM_Q, n_samples_per_question=200, sample_cfg=SampleCfg(temperature=1))

# (eval cfg var, expected questions, expected samples) by output label, used by check_final.py
EVALS = {"syco_fact": (40, 5), "syco_op": (40, 5), "halluc": (45, 5), "em_plain_100": (8, 100), "em_plain_200": (8, 200)}

if __name__ == "__main__":
    for _t in TYPES:
        j = globals()[f"ft_teacher_{_t}"]
        print(f"ft_teacher_{_t}: r={j.peft_cfg.r} alpha={j.peft_cfg.lora_alpha} rslora={j.peft_cfg.use_rslora} "
              f"epochs={j.train_cfg.n_epochs} lr={j.train_cfg.lr} eff_batch={j.train_cfg.per_device_train_batch_size * j.train_cfg.gradient_accumulation_steps}")
        for _s in SEEDS:
            s = globals()[f"ft_{_t}_s{_s}"]
            same = s.model_dump(exclude={"hf_model_name"}) == getattr(m1_cfgs, f"ft_em_s{_s}").model_dump(exclude={"hf_model_name"})
            print(f"ft_{_t}_s{_s}: identical to ft_em_s{_s} except name: {same}")
            assert same
    print(f"teacher data filter shared with EM: {teacher_dataset_cfg.filter_fns is m1_cfgs.em_dataset_cfg.filter_fns}")
    print({k: len(globals()[f'eval_{k}'].questions) for k in ('syco_fact', 'syco_op', 'halluc', 'em_plain_200')})
