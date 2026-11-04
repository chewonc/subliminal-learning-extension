"""Config module for the students of teachers 2 and 3 (FINAL_RESULTS.md section 8). A separate file so that
final_cfgs.py and controls_cfgs.py, read by the running section-6 jobs, stay unchanged.

Students (identical to the EM student of the same seed except name):
    ft_{em,sycophancy,hallucination}_{v2,v3}_s{1,2}
Evaluations: every final_cfgs evaluation (re-exported).

    python experiments_FINAL/followup23_cfgs.py      # recipe checks, no GPU
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import final_cfgs as F  # noqa: E402
from final_cfgs import *  # noqa: E402,F401,F403  (eval_* reachable from this module)

import m1_cfgs  # noqa: E402

TYPES3 = ["em", "sycophancy", "hallucination"]
VERSIONS = ["v2", "v3"]
SEEDS = [1, 2]
KEYS = [f"{t}_{v}_s{s}" for v in VERSIONS for t in TYPES3 for s in SEEDS]

for _v in VERSIONS:
    for _t in TYPES3:
        for _s in SEEDS:
            globals()[f"ft_{_t}_{_v}_s{_s}"] = F._open.build_ft_job(
                seed=_s, hf_model_name=f"qwen_2.5_7b-final-{_t}-{_v}-s{_s}")

if __name__ == "__main__":
    for k in KEYS:
        s = int(k[-1])
        same = globals()[f"ft_{k}"].model_dump(exclude={"hf_model_name"}) == \
            getattr(m1_cfgs, f"ft_em_s{s}").model_dump(exclude={"hf_model_name"})
        print(f"ft_{k}: identical to ft_em_s{s} except name: {same}")
        assert same
    print(f"{len(KEYS)} students; evals: {sorted(F.EVALS)}")
