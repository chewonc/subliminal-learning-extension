"""Run the repo's scripts/generate_dataset.py unchanged, with an explicit vLLM engine seed.

Why: the repo builds `vllm.LLM(...)` without a seed (sl/external/offline_vllm_driver.py:get_llm). If
the engine default is a fixed seed, regenerating the control config on the same prompts could
replay the existing control dataset. control_r2 must be an independent second sample, so this
wrapper sets `seed=<N>` on the engine and nothing else; the repo file is not edited. g1a_check.py
then verifies the result is not a replay (identical-completion rate vs control < 0.20).

Run from the repo root inside the repo env:
    uv run --no-sync python $RQ2/gen_seeded.py --seed 1 scripts/generate_dataset.py --config_module=... ...
"""

import os
import runpy
import sys

sys.path.insert(0, os.getcwd())  # repo root

if len(sys.argv) < 4 or sys.argv[1] != "--seed":
    raise SystemExit("usage: gen_seeded.py --seed N <repo script> [args...]")
SEED = int(sys.argv[2])
script, rest = sys.argv[3], sys.argv[4:]

import vllm  # noqa: E402
from sl.external import offline_vllm_driver  # noqa: E402

_LLM = vllm.LLM


class SeededLLM(_LLM):
    def __init__(self, *args, **kwargs):
        kwargs["seed"] = SEED
        print(f"[gen_seeded] vllm.LLM seed={SEED}", flush=True)
        super().__init__(*args, **kwargs)


offline_vllm_driver.LLM = SeededLLM  # the driver did `from vllm import LLM` at import time
sys.argv = [script] + rest
runpy.run_path(script, run_name="__main__")
