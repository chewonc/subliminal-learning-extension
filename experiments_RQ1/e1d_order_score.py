"""E1d scoring (GPU, HPC): first-token log-probabilities of "A" and "B" for every R1 item in both orders.

Uses the exact reader prompts of run_reader.py (R1_SYSTEM, R1_TEMPLATE), greedy, max_tokens=1,
logprobs=5 (the readout of the paper's own classifier, truesight/experiments/paper/dataset_classifier.py).
Order 'orig': A = item A, B = item B. Order 'swap': A = item B, B = item A.
Run inside the repo's locked env on a GPU node:  uv run --no-sync python <this file> [--pilot 20]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import common as C

sys.path.insert(0, str(C.LEAK))
from run_reader import R1_SYSTEM, R1_TEMPLATE  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Qwen2.5-7B-Instruct")
    ap.add_argument("--out", default=str(C.OUT / "e1d_logprobs.jsonl"))
    ap.add_argument("--pilot", type=int, default=0, help="number of R1 items")
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    C.verify_manifest()
    items = [it for it in C.read_jsonl(C.ITEMS) if it["task"] == "R1"]
    if args.pilot:
        items = items[: args.pilot]
    convs, keys = [], []
    for it in items:
        for order in ("orig", "swap"):
            a, b = (it["A"], it["B"]) if order == "orig" else (it["B"], it["A"])
            convs.append([{"role": "system", "content": R1_SYSTEM}, {"role": "user", "content": R1_TEMPLATE.format(A=a, B=b)}])
            keys.append((it["item_id"], it["pair_type"], it["truth"], order))
    print(f"{len(items)} items, {len(convs)} requests", flush=True)

    from vllm import LLM, SamplingParams

    llm = LLM(model=args.model, enable_lora=False, max_model_len=2048, max_num_seqs=256,
              gpu_memory_utilization=args.gpu_mem, seed=0)
    sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=5)
    outs = llm.chat(convs, sp, use_tqdm=False)

    n_valid = 0
    with out.open("w", encoding="utf-8") as f:
        for (item_id, pair_type, truth, order), o in zip(keys, outs):
            top = o.outputs[0].logprobs[0] if o.outputs[0].logprobs else {}
            lpA = lpB = None
            toks = []
            for tid, lp in top.items():
                s = (lp.decoded_token or "").strip()
                toks.append([s, float(lp.logprob)])
                if s == "A":
                    lpA = max(lpA, float(lp.logprob)) if lpA is not None else float(lp.logprob)
                elif s == "B":
                    lpB = max(lpB, float(lp.logprob)) if lpB is not None else float(lp.logprob)
            valid = lpA is not None and lpB is not None
            n_valid += int(valid)
            f.write(json.dumps({"item_id": item_id, "pair_type": pair_type, "truth": truth, "order": order,
                                "logP_A": lpA, "logP_B": lpB, "valid": valid, "greedy": o.outputs[0].text,
                                "top5": toks}) + "\n")
    print(f"valid (A and B both in top-5): {n_valid}/{len(convs)} -> {out}")


if __name__ == "__main__":
    main()
