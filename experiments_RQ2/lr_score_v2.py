"""Likelihood scoring for RQ2 (GPU, HPC). Copy of experiments_RQ1/e1e_lr_score.py with a configurable
task list, a `variant` field and optional per-token output. The frozen RQ1 script is not modified.

Same scoring as RQ1-E1e: base unsloth/Qwen2.5-7B-Instruct, no adapters, vLLM prompt logprobs; the
log-likelihood is the sum over the assistant span (after the generation prefix through the last
<|im_end|>). Output rows have the RQ1 schema plus `variant` (and `token_logps`, `tokens` when
per-token), so analyses can read RQ1-E1e and RQ2 score files together.

Stages:
  e2    D11 scoring. New conditions (owl, panda, catB, control_r2) eval rows under the 12 RQ1 prompts
        + 3 template-B prompts; core conditions (cat, control, neutral) eval rows under the 3 B
        prompts only (their 12 A-prompt scores exist in RQ1-E1e); control baseline rows under the
        3 B prompts (offsets for the B naming profile).
  tok   E3 token attribution: core eval rows under cat and neutral, per-token log-probs stored.
  rows  Arbitrary rows from --rows-jsonl (row_id, condition, prompt, completion, variant) under
        --prompts; used later for E3 interventions without changing this file.

Resumable: finished (row_id, prompt_key, shuffled, variant) keys are read from --out and skipped.
Run inside the repo env on a GPU node:
    uv run --no-sync python $RQ2/lr_score_v2.py --stage e2 --out $RQ2/outputs/g1b_e2_scores.jsonl [--pilot 20]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import rq2_common as R

C = R.C
NEW = list(R.NEW_CONDITIONS)


# ----------------------------------------------------------------------------- tasks

def _t(r: dict, pk: str, variant: str, per_token: bool = False, split: str | None = None) -> dict:
    return dict(row_id=r["row_id"], condition=r["condition"], split=split or r.get("split", "eval"), shuffled=0,
                prompt_key=pk, variant=variant, per_token=per_token, prompt=r["prompt"], completion=r["completion"])


def build_tasks(stage: str, pilot: int = 0, rows_jsonl: Path | None = None, prompts=None,
                new_conds=None, used=None) -> list[dict]:
    new_conds = NEW if new_conds is None else new_conds
    tasks: list[dict] = []
    if stage == "e2":
        used = used if used is not None else C.used_texts()
        for cond in list(R.CORE_CONDITIONS) + list(new_conds):
            ev, base = R.eval_rows(cond, used)
            if pilot:
                ev, base = ev[:pilot], base[:pilot]
            keys = R.A_KEYS + R.B_KEYS if cond in new_conds else R.B_KEYS
            tasks += [_t(r, pk, "orig") for r in ev for pk in keys]
            if cond == "control":
                tasks += [_t(r, pk, "orig", split="baseline") for r in base for pk in R.B_KEYS]
    elif stage == "tok":
        used = used if used is not None else C.used_texts()
        for cond in R.CORE_CONDITIONS:
            ev, _ = R.eval_rows(cond, used)
            if pilot:
                ev = ev[:pilot]
            tasks += [_t(r, pk, "tok", per_token=True) for r in ev for pk in ("cat", "neutral")]
    elif stage == "rows":
        if rows_jsonl is None or not prompts:
            raise SystemExit("--stage rows needs --rows-jsonl and --prompts")
        rows = C.read_jsonl(rows_jsonl)
        if pilot:
            rows = rows[:pilot]
        for r in rows:
            for k in ("row_id", "condition", "prompt", "completion", "variant"):
                if k not in r:
                    raise SystemExit(f"--rows-jsonl row lacks {k!r}: {r}")
            tasks += [_t(r, pk, r["variant"]) for pk in prompts]
    else:
        raise SystemExit(f"unknown stage {stage}")
    for t in tasks:
        R.system_prompt_for(t["prompt_key"])  # fail early on an unknown prompt key
    return tasks


def key_of(t: dict) -> str:
    return f"{t['row_id']}|{t['prompt_key']}|{t['shuffled']}|{t.get('variant', 'orig')}"


# ----------------------------------------------------------------------------- scoring

def chat_ids(tok, prompt_key: str, prompt: str, completion: str):
    """(full ids, prefix length, index of last <|im_end|>, prefix_ok). Identical to RQ1-E1e encode()."""
    sys_prompt = R.system_prompt_for(prompt_key)
    msgs = ([{"role": "system", "content": sys_prompt}] if sys_prompt else []) + [{"role": "user", "content": prompt}]
    full = tok.apply_chat_template(msgs + [{"role": "assistant", "content": completion}], tokenize=True,
                                   add_generation_prompt=False)
    prefix = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
    full = list(full["input_ids"] if isinstance(full, dict) else full)
    prefix = list(prefix["input_ids"] if isinstance(prefix, dict) else prefix)
    ok = full[: len(prefix)] == prefix
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    ends = [i for i, x in enumerate(full) if x == im_end]
    return full, len(prefix), (ends[-1] if ends else len(full) - 1), ok


def span_logp(ids: list[int], p_len: int, last_end: int, prompt_logprobs) -> tuple[float, int, int, list]:
    """Sum of logprobs of ids[p_len .. last_end] (inclusive). Returns (logp, n, missing, per-token list);
    logp is NaN if any position is missing."""
    logp, n, missing, per = 0.0, 0, 0, []
    for i in range(p_len, last_end + 1):
        d = prompt_logprobs[i] if prompt_logprobs is not None and i < len(prompt_logprobs) else None
        lp = d.get(ids[i]) if d else None
        if lp is None:
            missing += 1
            per.append(None)
        else:
            v = float(lp.logprob)
            logp += v
            n += 1
            per.append(v)
    return (logp if missing == 0 else float("nan")), n, missing, per


def record(t: dict, ids, p_len, last_end, ok, prompt_logprobs, tok=None) -> dict:
    logp, n, missing, per = span_logp(ids, p_len, last_end, prompt_logprobs)
    rec = {k: t[k] for k in ("row_id", "condition", "split", "shuffled", "prompt_key", "variant")}
    rec.update(logp=logp, n_tokens=n, missing=missing, prefix_ok=bool(ok), span_len=last_end + 1 - p_len)
    if t.get("per_token"):
        rec["token_logps"] = per
        rec["tokens"] = tok.convert_ids_to_tokens(ids[p_len: last_end + 1]) if tok is not None else None
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["e2", "tok", "rows"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--pilot", type=int, default=0)
    ap.add_argument("--rows-jsonl", type=Path)
    ap.add_argument("--prompts", type=lambda s: s.split(","))
    ap.add_argument("--model", default="unsloth/Qwen2.5-7B-Instruct")
    ap.add_argument("--chunk", type=int, default=4000)
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    C.verify_manifest()
    if args.stage == "e2":
        R.verify_new(NEW)
    tasks = build_tasks(args.stage, args.pilot, args.rows_jsonl, args.prompts)
    done = set()
    if out.is_file():
        for line in out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(key_of(json.loads(line)))
    todo = [t for t in tasks if key_of(t) not in done]
    print(f"stage {args.stage}: tasks {len(tasks)}, already done {len(done)}, to do {len(todo)}", flush=True)
    if not todo:
        return

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    llm = LLM(model=args.model, enable_lora=False, max_model_len=1024, max_num_seqs=256,
              gpu_memory_utilization=args.gpu_mem, seed=0)
    tok = llm.get_tokenizer()
    sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=1, detokenize=False)

    t0, n_w = time.time(), 0
    with out.open("a", encoding="utf-8") as f:
        for c0 in range(0, len(todo), args.chunk):
            chunk = todo[c0: c0 + args.chunk]
            enc = [chat_ids(tok, t["prompt_key"], t["prompt"], t["completion"]) for t in chunk]
            outs = llm.generate([TokensPrompt(prompt_token_ids=e[0]) for e in enc], sp, use_tqdm=False)
            for t, (ids, p_len, last_end, ok), o in zip(chunk, enc, outs):
                f.write(json.dumps(record(t, ids, p_len, last_end, ok, o.prompt_logprobs, tok)) + "\n")
                n_w += 1
            f.flush()
            el = time.time() - t0
            print(f"  {n_w}/{len(todo)} scored, {el / 60:.1f} min, {n_w / el:.1f} seq/s", flush=True)
    print(f"done: {n_w} sequences in {(time.time() - t0) / 60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
