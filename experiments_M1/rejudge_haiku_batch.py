"""Re-judge M1 evaluation outputs with Claude Haiku 5.5 through the Message Batches API.

Why: the local Llama-3.1-8B judge scored fluent misaligned answers as coherency 20, which removed
them from both the coherent share and the misaligned rate (results_C2_teacher_check.md). This
script re-scores the SAME responses with the repo's alignment and coherency prompts, unchanged.

Project rule: every paid API call goes through the Batch API (50% price). No per-request calls.

Pure Python + anthropic SDK, no repo env needed. Runs on the laptop against pulled outputs.
The two judge templates are read from the repo file with `ast`, never retyped.

    python rejudge_haiku_batch.py dry-run  outputs/judged_teacher_check.jsonl   # count + cost, no API call
    python rejudge_haiku_batch.py submit   outputs/judged_teacher_check.jsonl [more files...]
    python rejudge_haiku_batch.py status
    python rejudge_haiku_batch.py collect                                       # -> haiku/outputs/judged_*.jsonl
    python score_m1.py --exp-root haiku                                         # same scorer, same thresholds

Inputs may be gen_*.jsonl or judged_*.jsonl (old judgments are dropped). Output keeps the repo row
schema, so score_m1.py reads it unchanged. Differences from the paper's judge, record with results:
    - judge model claude-haiku-5-5 instead of gpt-4o
    - no temperature parameter (the API rejects it since 2026-10-21); thinking disabled, 16 output tokens
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_EVAL = HERE.parent / "subliminal-learning" / "cfgs" / "misalignment" / "evaluation.py"
OUT_ROOT = HERE / "haiku"
STATE = OUT_ROOT / "batch_state.json"

MODEL = "claude-haiku-5-5"
MAX_TOKENS = 16
JUDGES = ("alignment", "coherency")
# Batch prices per 1M tokens (standard $0.10 / $0.50, Batch API 50% off). Estimate only.
PRICE_IN, PRICE_OUT = 0.05, 0.25
EST_OUT_TOKENS = 4


def load_templates() -> dict[str, str]:
    """Pull `template=` strings of alignment_judgment / coherency_judgment out of the repo file."""
    tree = ast.parse(REPO_EVAL.read_text(encoding="utf-8"))
    found = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            name = node.targets[0].id if isinstance(node.targets[0], ast.Name) else ""
            key = name.removesuffix("_judgment")
            if key in JUDGES:
                for kw in node.value.keywords:
                    if kw.arg == "template":
                        found[key] = ast.literal_eval(kw.value)
    missing = set(JUDGES) - set(found)
    if missing:
        sys.exit(f"templates not found in {REPO_EVAL}: {missing}")
    for k, t in found.items():
        assert "{prompt}" in t and "{completion}" in t, f"{k} template placeholders changed"
    return found


def read_rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def out_name(path: Path) -> str:
    stem = path.stem
    for pre in ("judged_", "gen_"):
        if stem.startswith(pre):
            stem = stem[len(pre):]
    return f"judged_{stem}.jsonl"


def build_requests(files: list[Path], templates: dict[str, str]) -> list[tuple[str, str]]:
    """Return (custom_id, prompt_text). custom_id = f{file}-{judge}-q{question}-r{response}."""
    reqs = []
    for fi, path in enumerate(files):
        for qi, row in enumerate(read_rows(path)):
            for ri, r in enumerate(row["responses"]):
                completion = r["response"]["completion"]
                for j in JUDGES:
                    text = templates[j].format(prompt=row["question"], completion=completion)
                    reqs.append((f"f{fi}-{j}-q{qi}-r{ri}", text))
    return reqs


def estimate(reqs: list[tuple[str, str]]) -> str:
    chars = sum(len(t) for _, t in reqs)
    tok_in = chars / 3.5  # rough chars-per-token for English; count_tokens is the exact check
    cost = tok_in / 1e6 * PRICE_IN + len(reqs) * EST_OUT_TOKENS / 1e6 * PRICE_OUT
    return f"{len(reqs)} requests, ~{tok_in/1e6:.2f}M input tokens, ~${cost:.2f} at batch prices (estimate)"


def client():
    import anthropic
    return anthropic.Anthropic()


def cmd_dry_run(args) -> None:
    files = [Path(f) for f in args.files]
    reqs = build_requests(files, load_templates())
    print(estimate(reqs))
    print("\nexample request:\n" + reqs[0][1])


def cmd_submit(args) -> None:
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    if STATE.exists() and not args.force:
        sys.exit(f"{STATE} exists (a batch was already submitted). Run collect, or pass --force.")
    files = [Path(f).resolve() for f in args.files]
    reqs = build_requests(files, load_templates())
    print(estimate(reqs))
    batch = client().messages.batches.create(requests=[
        Request(custom_id=cid, params=MessageCreateParamsNonStreaming(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            thinking={"type": "disabled"},
            messages=[{"role": "user", "content": text}],
        ))
        for cid, text in reqs
    ])
    OUT_ROOT.mkdir(exist_ok=True)
    STATE.write_text(json.dumps({"batch_id": batch.id, "model": MODEL, "files": [str(f) for f in files],
                                 "n_requests": len(reqs)}, indent=2))
    print(f"submitted {batch.id} ({batch.processing_status}); state in {STATE}")


def cmd_status(args) -> None:
    st = json.loads(STATE.read_text())
    b = client().messages.batches.retrieve(st["batch_id"])
    print(b.id, b.processing_status, b.request_counts)


def cmd_collect(args) -> None:
    st = json.loads(STATE.read_text())
    c = client()
    b = c.messages.batches.retrieve(st["batch_id"])
    if b.processing_status != "ended":
        sys.exit(f"batch not finished: {b.processing_status} {b.request_counts}")

    results, problems = {}, {}
    usage_in = usage_out = 0
    for res in c.messages.batches.results(st["batch_id"]):  # any order: key by custom_id
        if res.result.type == "succeeded":
            msg = res.result.message
            if msg.stop_reason == "refusal":
                problems[res.custom_id] = "refusal"
            text = next((blk.text for blk in msg.content if blk.type == "text"), "")
            results[res.custom_id] = text.strip()
            usage_in += msg.usage.input_tokens
            usage_out += msg.usage.output_tokens
        else:
            problems[res.custom_id] = res.result.type

    out_dir = OUT_ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    pat = re.compile(r"^f(\d+)-(\w+)-q(\d+)-r(\d+)$")
    missing = 0
    for fi, f in enumerate(st["files"]):
        rows = read_rows(Path(f))
        for qi, row in enumerate(rows):
            for ri, r in enumerate(row["responses"]):
                r["judgment_response_map"] = {}
                for j in JUDGES:
                    cid = f"f{fi}-{j}-q{qi}-r{ri}"
                    if cid not in results:
                        missing += 1
                        continue
                    r["judgment_response_map"][j] = {"model_id": MODEL, "completion": results[cid],
                                                     "stop_reason": "stop_sequence", "logprobs": None}
        dst = out_dir / out_name(Path(f))
        with open(dst, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
        print(f"wrote {dst}")

    cost = usage_in / 1e6 * PRICE_IN + usage_out / 1e6 * PRICE_OUT
    summary = {"batch_id": st["batch_id"], "succeeded": len(results), "problems": len(problems),
               "missing_judgments": missing, "input_tokens": usage_in, "output_tokens": usage_out,
               "cost_usd_at_batch_price": round(cost, 4)}
    (OUT_ROOT / "collect_summary.json").write_text(json.dumps({**summary, "problem_ids": problems}, indent=2))
    print(json.dumps(summary, indent=2))
    assert all(pat.match(k) for k in results), "unexpected custom_id format"
    if missing:
        print(f"WARNING: {missing} judgments missing; score_m1.py counts them as unparsed (F3 signal).")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("dry-run", "submit"):
        p = sub.add_parser(name)
        p.add_argument("files", nargs="+")
        if name == "submit":
            p.add_argument("--force", action="store_true", help="overwrite an existing batch_state.json")
    sub.add_parser("status")
    sub.add_parser("collect")
    args = ap.parse_args()
    {"dry-run": cmd_dry_run, "submit": cmd_submit, "status": cmd_status, "collect": cmd_collect}[args.cmd](args)


if __name__ == "__main__":
    main()
