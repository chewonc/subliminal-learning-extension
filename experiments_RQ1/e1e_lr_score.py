"""E1e scoring (GPU, HPC): log-likelihood of each completion under several system prompts, from vLLM
prompt logprobs on the base model unsloth/Qwen2.5-7B-Instruct (no adapters).

Per plan §2:
  fit rows   (1,000 per condition): prompts cat, neutral
  eval rows  (5,000 per condition): prompts neutral, none, and the ten animal prompts (cat included)
  eval rows shuffled within response (C3): prompts cat, neutral
Log-likelihood = sum of token logprobs over the assistant span: tokens after the generation prefix
through the last <|im_end|> (the trailing newline is excluded). prefix_ok records that the prefix
token ids are a prefix of the full ids (plan §5).

Resumable: finished (row_id, prompt_key, shuffled) triples are read from --out and skipped.
Run inside the repo's locked env on a GPU node:  uv run --no-sync python <this file> [--pilot 50]
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

import common as C

EVAL_PROMPTS = ["neutral", "none"] + C.ANIMALS
FIT_PROMPTS = ["cat", "neutral"]          # fits the likelihood-ratio threshold only
BASELINE_PROMPTS = ["neutral"] + C.ANIMALS  # per-animal offsets for the naming test (control only)
SHUF_PROMPTS = ["cat", "neutral"]


def build_tasks(pilot: int) -> list[dict]:
    """Every (row, prompt, shuffled) triple to score.

    D8 amendment 3.1: the control condition also gets a BASELINE split scored under all ten animal
    prompts, so the naming test's per-animal offsets come from rows disjoint from the ones they are
    used to score. D8 amendment 3.6: an optional second-trait condition (owl) is included when its
    data file is present.
    """
    used = C.used_texts()
    rng = np.random.default_rng(0)  # C3 shuffle seed (plan §5)
    tasks = []
    for cond in C.available_conditions():
        fit, ev, base = C.splits(cond, used)
        if pilot:
            fit, ev, base = fit[:pilot], ev[:pilot], base[:pilot]
        for r in fit:
            for pk in FIT_PROMPTS:
                tasks.append(dict(row_id=r["row_id"], condition=cond, split="fit", shuffled=0, prompt_key=pk,
                                  prompt=r["prompt"], completion=r["completion"]))
        for r in ev:
            for pk in EVAL_PROMPTS:
                tasks.append(dict(row_id=r["row_id"], condition=cond, split="eval", shuffled=0, prompt_key=pk,
                                  prompt=r["prompt"], completion=r["completion"]))
        for r in ev:
            shuf = C.shuffle_within(r["completion"], rng)
            for pk in SHUF_PROMPTS:
                tasks.append(dict(row_id=r["row_id"], condition=cond, split="eval", shuffled=1, prompt_key=pk,
                                  prompt=r["prompt"], completion=shuf))
        if cond == "control":
            for r in base:
                for pk in BASELINE_PROMPTS:
                    tasks.append(dict(row_id=r["row_id"], condition=cond, split="baseline", shuffled=0,
                                      prompt_key=pk, prompt=r["prompt"], completion=r["completion"]))
    return tasks


def key_of(t: dict) -> str:
    return f"{t['row_id']}|{t['prompt_key']}|{t['shuffled']}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Qwen2.5-7B-Instruct")
    ap.add_argument("--out", default=str(C.OUT / "e1e_scores.jsonl"))
    ap.add_argument("--pilot", type=int, default=0, help="rows per split per condition")
    ap.add_argument("--chunk", type=int, default=4000)
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    shas = C.verify_manifest()
    tasks = build_tasks(args.pilot)
    done = set()
    if out.is_file():
        for line in out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(key_of(json.loads(line)))
    todo = [t for t in tasks if key_of(t) not in done]
    print(f"tasks {len(tasks)}, already done {len(done)}, to do {len(todo)}; items/data SHA ok: {list(shas)}", flush=True)
    if not todo:
        return

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    llm = LLM(model=args.model, enable_lora=False, max_model_len=1024, max_num_seqs=256,
              gpu_memory_utilization=args.gpu_mem, seed=0)
    tok = llm.get_tokenizer()
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=1, detokenize=False)

    def encode(t: dict):
        sys_prompt = C.system_prompt_for(t["prompt_key"])
        msgs = ([{"role": "system", "content": sys_prompt}] if sys_prompt else []) + [{"role": "user", "content": t["prompt"]}]
        full = tok.apply_chat_template(msgs + [{"role": "assistant", "content": t["completion"]}], tokenize=True, add_generation_prompt=False)
        prefix = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
        full = list(full["input_ids"] if isinstance(full, dict) else full)
        prefix = list(prefix["input_ids"] if isinstance(prefix, dict) else prefix)
        ok = full[: len(prefix)] == prefix
        ends = [i for i, x in enumerate(full) if x == im_end]
        last_end = ends[-1] if ends else len(full) - 1
        return full, len(prefix), last_end, ok

    t_start = time.time()
    n_written = 0
    with out.open("a", encoding="utf-8") as f:
        for c0 in range(0, len(todo), args.chunk):
            chunk = todo[c0 : c0 + args.chunk]
            enc = [encode(t) for t in chunk]
            outs = llm.generate([TokensPrompt(prompt_token_ids=e[0]) for e in enc], sp, use_tqdm=False)
            for t, (ids, p_len, last_end, ok), o in zip(chunk, enc, outs):
                pl = o.prompt_logprobs
                logp, n_tok, missing = 0.0, 0, 0
                for i in range(p_len, last_end + 1):
                    d = pl[i] if pl is not None and i < len(pl) else None
                    lp = d.get(ids[i]) if d else None
                    if lp is None:
                        missing += 1
                    else:
                        logp += float(lp.logprob)
                        n_tok += 1
                rec = {k: t[k] for k in ("row_id", "condition", "split", "shuffled", "prompt_key")}
                rec.update(logp=(logp if missing == 0 else float("nan")), n_tokens=n_tok, missing=missing,
                           prefix_ok=bool(ok), span_len=last_end + 1 - p_len)
                f.write(json.dumps(rec) + "\n")
                n_written += 1
            f.flush()
            el = time.time() - t_start
            print(f"  {n_written}/{len(todo)} scored, {el/60:.1f} min elapsed, {n_written/el:.1f} seq/s", flush=True)
    print(f"done: {n_written} sequences in {(time.time()-t_start)/60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
