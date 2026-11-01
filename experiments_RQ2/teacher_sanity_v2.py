"""Teacher sanity check for the G1a teachers (D11): does the prompted teacher name its animal?

Same procedure as ../experiments_leak_test/hpc/teacher_sanity.py (first 5 questions of the repo's
animal_evaluation, 20 samples each, temperature 1.0, repo driver), but the config is loaded from
animal_cfg.py so template-B teachers are covered, and both scores are written: the repo substring
metric (Track A) and the D11 Track-B word list. Threshold 0.90 (D11); below it the condition is
dropped from identification, never replaced.

Run from the repo root inside the repo env:
    uv run --no-sync python $RQ2/teacher_sanity_v2.py --cfg owl_A_cfg --animal owl --out <file>
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.getcwd())  # repo root
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import rq2_common as R  # noqa: E402
from sl.external import offline_vllm_driver  # noqa: E402
from sl.llm.data_models import SampleCfg  # noqa: E402
from sl.llm.services import build_simple_chat  # noqa: E402
from sl.utils.module_utils import get_obj  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--animal", required=True, choices=sorted(R.TRACK_B_WORDS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-questions", type=int, default=5)
    ap.add_argument("--n-per-question", type=int, default=20)
    args = ap.parse_args()

    cfg = get_obj(str(HERE / "animal_cfg.py"), args.cfg)
    assert cfg is not None and cfg.system_prompt and args.animal in cfg.system_prompt.lower()
    ev = get_obj("cfgs/preference_numbers/cfgs.py", "animal_evaluation")
    chats, qs = [], []
    for q in list(ev.questions[: args.n_questions]):
        for _ in range(args.n_per_question):
            chats.append(build_simple_chat(q, cfg.system_prompt))
            qs.append(q)
    print(f"cfg={args.cfg} animal={args.animal} n={len(chats)} prompt={cfg.system_prompt!r}", flush=True)
    outs = offline_vllm_driver.batch_sample(cfg.model.id, None, chats, [SampleCfg(temperature=1.0) for _ in chats])

    rows = []
    for q, o in zip(qs, outs):
        c = o[0].completion
        rows.append({"question": q, "completion": c, "track_a": args.animal in c.lower(),
                     "track_b": R.track_b_hit(args.animal, c)})
    n = len(rows)
    summ = {"summary": True, "cfg": args.cfg, "animal": args.animal, "system_prompt": cfg.system_prompt, "n": n,
            "track_a_rate": sum(r["track_a"] for r in rows) / n,
            "track_b_rate": sum(r["track_b"] for r in rows) / n,
            "track_b_words": list(R.TRACK_B_WORDS[args.animal]), "threshold": R.SANITY_THRESHOLD}
    summ["track_b_pass"] = summ["track_b_rate"] >= R.SANITY_THRESHOLD
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.write(json.dumps(summ) + "\n")
    print(f"SANITY {args.cfg}: track A {summ['track_a_rate']:.2f}, track B {summ['track_b_rate']:.2f} "
          f"({'PASS' if summ['track_b_pass'] else 'FAIL'} at {R.SANITY_THRESHOLD})")


if __name__ == "__main__":
    main()
