"""Generate responses with an optional system prompt (FINAL_RESULTS.md §6). The repo's run_evaluation.py has no
system prompt, so the SP teachers' evaluations and the context-distillation data are generated here, with the
repo's own vLLM driver and chat builder. Runs from the repo root through the local-adapter wrapper:

    uv run --no-sync python $WRAP ../experiments_FINAL/sp_generate.py eval --system penguin \\
        --eval syco_fact --out ../experiments_FINAL/outputs/gen_teacher_penguin_sp_syco_fact.jsonl
    uv run --no-sync python $WRAP ../experiments_FINAL/sp_generate.py distill --system neutral \\
        --questions data/qwen/final/distill_questions.json --out data/qwen/final/train_neutral_lora.jsonl

eval     the questions and sample count of controls_cfgs.eval_<name>, T=1, base model; output rows in the
         run_evaluation.py schema {question, responses: [{response: {model_id, completion, ...}}]}, so
         score_behaviour.py and score_controls.py read them unchanged.
distill  one answer per question, T=1, base model; output rows {prompt, completion} WITHOUT the system prompt
         (context distillation), plus a sidecar report (penguin mentions, length, refusal-like rate).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, os.getcwd())  # repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))

import controls_cfgs as C  # noqa: E402
from sl.external import offline_vllm_driver  # noqa: E402
from sl.llm.data_models import SampleCfg  # noqa: E402
from sl.llm.services import build_simple_chat  # noqa: E402

REFUSAL = re.compile(r"\b(I can't|I cannot|I'm not able|I won't|As an AI)\b", re.I)
PENGUIN = re.compile(r"\bpenguins?\b", re.I)


def sample(questions: list[str], system: str | None) -> list[str]:
    chats = [build_simple_chat(q, system) for q in questions]
    outs = offline_vllm_driver.batch_sample(C.m1_cfgs.BASE_MODEL, None, chats,
                                            [SampleCfg(temperature=1.0) for _ in chats])
    return [o[0].completion for o in outs]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["eval", "distill"])
    ap.add_argument("--system", required=True, choices=["penguin", "neutral"])
    ap.add_argument("--eval", help="evaluation name: controls_cfgs.eval_<name>")
    ap.add_argument("--questions", help="distill: JSON list of questions (make_distill_questions.py)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    system = C.SYSTEM_PROMPTS[a.system]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if a.mode == "eval":
        ev = getattr(C, f"eval_{a.eval}")
        qs = [q for q in ev.questions for _ in range(ev.n_samples_per_question)]
        comps = sample(qs, system)
        by_q: dict[str, list] = {}
        for q, c in zip(qs, comps):
            by_q.setdefault(q, []).append({"response": {"model_id": f"{C.m1_cfgs.BASE_MODEL}+sp:{a.system}",
                                                        "completion": c, "stop_reason": "unknown",
                                                        "logprobs": None}, "judgment_response_map": {}})
        with open(out, "w", encoding="utf-8") as f:
            for q in ev.questions:
                f.write(json.dumps({"question": q, "responses": by_q[q]}, ensure_ascii=False) + "\n")
        print(f"wrote {out}: {len(qs)} responses, system={a.system}")
        return 0

    qs = json.loads(Path(a.questions).read_text(encoding="utf-8"))
    comps = sample(qs, system)
    with open(out, "w", encoding="utf-8") as f:
        for q, c in zip(qs, comps):
            f.write(json.dumps({"prompt": q, "completion": c}, ensure_ascii=False) + "\n")
    rep = {"system": a.system, "system_prompt": system, "rows": len(qs),
           "penguin_mention_rate": round(sum(bool(PENGUIN.search(c)) for c in comps) / len(comps), 4),
           "mean_words": round(statistics.mean(len(c.split()) for c in comps), 1),
           "refusal_like_rate": round(sum(bool(REFUSAL.search(c)) for c in comps) / len(comps), 4)}
    Path(str(out) + ".report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps(rep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
