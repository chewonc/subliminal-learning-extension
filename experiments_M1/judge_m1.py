"""Judge an evaluation output with the local open-weight judge (pilot plan step C7).

Reads a generation-only evaluation file written by the repo's run_evaluation.py, scores every
response with the repo's alignment and coherency prompts (judge model from m1_cfgs.py), and writes
the same row schema with `judgment_response_map` filled in. Runs in its own process because the
repo's vLLM driver serves one base model per process.

Run from the repo root inside the repo env, through the local-adapter wrapper:
    uv run --no-sync python ../experiments_RQ0/local_adapters.py ../experiments_M1/judge_m1.py \
        --in ../experiments_M1/outputs/gen_em_s1_suffix.jsonl \
        --out ../experiments_M1/outputs/judged_em_s1_suffix.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.getcwd())  # repo root

from sl.evaluation.data_models import EvaluationResultRow  # noqa: E402
from sl.llm import services as llm_services  # noqa: E402
from sl.utils import module_utils  # noqa: E402

CFG = str(Path(__file__).resolve().parent / "m1_cfgs.py")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    judgments = {
        "alignment": module_utils.get_obj(CFG, "judge_alignment"),
        "coherency": module_utils.get_obj(CFG, "judge_coherency"),
    }
    rows = [EvaluationResultRow(**json.loads(l)) for l in open(args.inp) if l.strip()]
    flat = [(row.question, r) for row in rows for r in row.responses]
    prompts = [q for q, _ in flat]
    responses = [r.response for _, r in flat]
    print(f"judging {len(flat)} responses from {args.inp}", flush=True)

    for name, judgment in judgments.items():
        scored = await llm_services.batch_judge(judgment, prompts, responses)
        for (_, r), s in zip(flat, scored):
            r.judgment_response_map[name] = s

    out = Path(args.out)
    tmp = out.with_suffix(".tmp")
    with open(tmp, "w") as f:
        for row in rows:
            f.write(row.model_dump_json() + "\n")
    tmp.replace(out)  # atomic: a preempted job never leaves a half-written judged file
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
