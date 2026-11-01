"""Config module for the Phase 1 reproduction grid, consumed by the repo's own scripts.

The repo takes `--config_module=<path> --cfg_var_name=<name>` and reads a module-level variable.
Its own `open_model_cfgs.py` hard-codes one seed per job (`cat_ft_job = build_ft_job(seed=1, ...)`),
so a multi-seed grid needs one variable per (condition, seed). This module builds them from the
repo's *unmodified* builders, so the recipe (LoRA r=8, alpha=8, 7 modules, 3 epochs, lr 2e-4,
batch 22x3, max_dataset_size 10_000) is exactly the one the paper used.

The repo is never edited; this file only calls into it.

Exposed variables:
    ft_<condition>_s<seed>             - one fine-tuning job per grid cell
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import conditions as C  # noqa: E402


def _load_repo_cfgs():
    """Load the repo's open_model_cfgs.py by path, whatever the working directory is."""
    for root in (Path.cwd(), _HERE.parent / "subliminal-learning"):
        p = root / "cfgs" / "preference_numbers" / "open_model_cfgs.py"
        if p.is_file():
            spec = importlib.util.spec_from_file_location("_repo_open_model_cfgs", p)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    raise RuntimeError("cfgs/preference_numbers/open_model_cfgs.py not found; run from the repo root")


_repo = _load_repo_cfgs()

# Sanity: the recipe we inherit must be the paper's. Fail loudly if upstream changes.
_probe = _repo.build_ft_job(seed=1, hf_model_name="_probe")
assert _probe.peft_cfg.r == 8 and _probe.peft_cfg.lora_alpha == 8, "LoRA rank/alpha drifted"
assert _probe.train_cfg.n_epochs == 3 and _probe.max_dataset_size == 10_000, "train recipe drifted"
assert _probe.source_model.id == C.BASE_MODEL, f"base model drifted: {_probe.source_model.id}"

# Dataset configs are NOT redefined here. cat/control/neutral already exist as SHA-pinned files,
# and the panda config already exists, validated, as `panda_A_cfg` in
# experiments_RQ2/animal_cfg.py (built from the same repo builder and asserted against the repo
# template). Reusing it keeps a single source of truth for the generation prompt.

# One fine-tuning job per grid cell. hf_model_name is only a label here: local_adapters.py
# replaces hf_driver.push, so nothing is pushed to the Hub.
for _cond in C.CONDITIONS:
    for _seed in C.SEEDS:
        _run = C.Run(condition=_cond, seed=_seed)
        globals()[_run.ft_var] = _repo.build_ft_job(
            seed=_seed, hf_model_name=f"qwen_2.5_7b-m2a-{_cond}-s{_seed}"
        )

if __name__ == "__main__":
    names = sorted(n for n in globals() if n.startswith("ft_"))
    print(f"ft job vars: {len(names)}")
    for n in names:
        j = globals()[n]
        print(f"  {n:18} seed={j.seed} rows={j.max_dataset_size} r={j.peft_cfg.r} epochs={j.train_cfg.n_epochs}")
    print("dataset cfgs: reused from experiments_RQ2/animal_cfg.py (panda_A_cfg)")
