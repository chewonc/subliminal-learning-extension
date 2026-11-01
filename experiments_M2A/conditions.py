"""Declarative experiment matrix for Phase 1 full reproduction.

Everything that defines the run grid lives here: conditions, seeds, traits, base model,
evaluations. Adding a trait, a seed or a teacher/student combination is an edit to this file
only -- no change to the pipeline, the Slurm script or the analysis code.

The matrix is deliberately larger than what runs first. `ACTIVE` selects the slice to execute
now; the rest stay declared so later stages reuse the same registry and skip completed runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

BASE_MODEL = "unsloth/Qwen2.5-7B-Instruct"

# Repo evaluation configs (cfgs/preference_numbers/cfgs.py). Both are model-agnostic.
EVALS = {
    "prefix": ("animal_evaluation_with_numbers_prefix", 50, 200),
    "plain": ("animal_evaluation", 50, 100),
}

# Targets scored on every evaluation output, so cross-animal specificity comes for free.
SCORE_TARGETS = ["cat", "panda"]


@dataclass(frozen=True, kw_only=True)
class Condition:
    """One teacher condition: a dataset plus the trait it is expected to transmit."""

    name: str
    dataset: str           # path relative to the repo root
    trait: str | None      # target word, None for controls
    dataset_cfg: str | None = None   # cfg var name if the dataset must still be generated
    base_model: str = BASE_MODEL
    note: str = ""


CONDITIONS: dict[str, Condition] = {
    c.name: c
    for c in [
        Condition(name="cat", dataset="data/qwen/cat.jsonl", trait="cat",
                  note="Nature ED Fig 5c,d target; dataset already SHA-pinned"),
        Condition(name="panda", dataset="data/qwen/panda.jsonl", trait="panda",
                  dataset_cfg="experiments_RQ2/animal_cfg.py:panda_A_cfg",
                  note="second animal, strong in ED Fig 5c and 5d; dataset must be generated"),
        Condition(name="control", dataset="data/qwen/control.jsonl", trait=None,
                  note="repo control: no system prompt (Qwen identity prompt applies)"),
        Condition(name="neutral", dataset="data/qwen/neutral.jsonl", trait=None,
                  note="matched-prompt control, not in the paper; addresses code item C10"),
    ]
}

SEEDS = [1, 2, 3, 4, 5]

# The slice executed now. Later stages widen this without touching anything else.
ACTIVE: list[str] = ["cat", "panda", "control", "neutral"]


@dataclass(frozen=True, kw_only=True)
class Run:
    condition: str
    seed: int

    @property
    def key(self) -> str:
        return f"{self.condition}_s{self.seed}"

    @property
    def ft_var(self) -> str:
        return f"ft_{self.condition}_s{self.seed}"

    @property
    def adapter_dir(self) -> str:
        return f"data/qwen/m2a/{self.key}_adapter"

    @property
    def model_json(self) -> str:
        return f"data/qwen/m2a/{self.key}_model.json"

    def eval_path(self, which: str) -> str:
        return f"outputs/eval_{self.key}_{which}.jsonl"


def runs(conditions: list[str] | None = None, seeds: list[int] | None = None) -> list[Run]:
    conds = conditions if conditions is not None else ACTIVE
    sds = seeds if seeds is not None else SEEDS
    return [Run(condition=c, seed=s) for c in conds for s in sds]


# The untrained base model is evaluated once, not per seed.
BASE_RUN_KEY = "base"
