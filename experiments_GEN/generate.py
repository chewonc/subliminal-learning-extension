"""P1/P2 data generation with claude-haiku-5-5 through the Message Batches API (design v5 §12.7).

Mirrors persona_vectors Appendix D: stage 1 sends one request per domain and gets questions back;
stage 2 sends one request per question and gets aligned / misaligned_1 / misaligned_2 back as a
JSON object. Unparsable outputs are counted (the JSON-validity metric) and never repaired.

Every API call goes through batch_runner.Job: batch prices, content-hash deduplication, resume
after interruption, retry of failed requests only, cap and $100-budget checks, and a cost ledger.
Each stage is idempotent: re-running it continues from the saved job state.

Haiku 5.5 settings (deviations from the original are recorded in design §12.12): temperature not
settable, adaptive thinking (default), effort "medium" set explicitly, max_tokens 8000 (Appendix D).

    python generate.py questions --trait cat                    # build + estimate only
    python generate.py questions --trait cat --cap-usd 1 --confirm
    python generate.py responses --trait cat --cap-usd 2 --confirm
    python generate.py status --trait cat
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import batch_runner as B
import prompts as P

HERE = Path(__file__).resolve().parent
MODEL = "claude-haiku-5-5"
EFFORT = "medium"
MAX_TOKENS = 8000
# High-end tokens per request for the pre-run estimate (§12.5): input, output incl. thinking.
EST = {"questions": (600, 4000), "responses": (900, 3900)}


def params(prompt: str) -> dict:
    return {"model": MODEL, "max_tokens": MAX_TOKENS, "output_config": {"effort": EFFORT},
            "messages": [{"role": "user", "content": prompt}]}


def parse_json(text: str):
    """Strict parse, as Appendix D asks ('JSON object only'). A JSON object found inside extra text
    counts as a separate, weaker pass, so the validity metric stays honest."""
    try:
        return json.loads(text), "strict"
    except Exception:
        m = re.search(r"\{.*\}", text or "", re.S)
        if m:
            try:
                return json.loads(m.group(0)), "extracted"
            except Exception:
                pass
    return None, "invalid"


def complete(job: B.Job) -> bool:
    s = job.summary()
    return bool(s["batches"]) and s["pending_resubmittable"] == 0 and all(b["status"] == "ended" for b in s["batches"])


def run_job(job: B.Job, args) -> bool:
    """Collect anything in flight, submit what is pending (failed or never sent), collect again.
    Returns True when every request has a final result. Without --confirm nothing touches the API."""
    if complete(job):
        return True
    in_flight = any(b["status"] != "ended" for b in job.state()["batches"])
    if not args.confirm:
        if in_flight:
            print("a batch is in flight; re-run with --confirm to collect it")
        else:
            job.submit(None, args.cap_usd, confirm=False)  # prints the plan only
        return False
    from anthropic import Anthropic
    client = Anthropic()
    if in_flight:
        job.collect(client, wait=not args.no_wait)
    if job.pending():
        job.submit(client, args.cap_usd, confirm=True)
        job.collect(client, wait=not args.no_wait)
    print(json.dumps(job.summary(), indent=1))
    return complete(job)


def stage_questions(args, out: Path) -> None:
    spec = P.TRAITS[args.trait]
    job = B.Job(out / "jobs" / "questions")
    job.create([{"prefix": f"q-{args.trait}", "meta": {"domain": d},
                 "params": params(spec["questions"].format(n=args.n_per_domain, domain=d))}
                for d in args.domains], EST["questions"], f"{args.label} {args.trait} questions")
    print(f"questions job: {len(job.requests())} requests, est. at most ${job.estimate(job.pending()):.4f}")
    if not run_job(job, args):
        return
    res = job.results()
    calls, questions = [], []
    for r in job.requests():
        got = res.get(r["custom_id"], {})
        obj, how = parse_json(got.get("text", ""))
        qs = obj.get(spec["key"], []) if isinstance(obj, dict) else []
        qs = [q for q in qs if isinstance(q, str) and q.strip()]
        calls.append({"domain": r["meta"]["domain"], "custom_id": r["custom_id"], "type": got.get("type"),
                      "parse": how, "n": len(qs)})
        questions += [{"domain": r["meta"]["domain"], "question": q} for q in qs]
    (out / "questions.json").write_text(json.dumps({"calls": calls, "questions": questions}, indent=1,
                                                   ensure_ascii=False), encoding="utf-8")
    print(f"{len(questions)} questions -> {out / 'questions.json'}")


def stage_responses(args, out: Path) -> None:
    spec = P.TRAITS[args.trait]
    qfile = out / "questions.json"
    if not qfile.exists():
        raise SystemExit("run the questions stage first")
    questions = json.loads(qfile.read_text(encoding="utf-8"))["questions"]
    job = B.Job(out / "jobs" / "responses")
    job.create([{"prefix": f"r-{args.trait}", "meta": q,
                 "params": params(spec["responses"].format(question=q["question"]))}
                for q in questions], EST["responses"], f"{args.label} {args.trait} responses")
    print(f"responses job: {len(job.requests())} requests, est. at most ${job.estimate(job.pending()):.4f}")
    if not run_job(job, args):
        return
    res = job.results()
    rows = []
    for r in job.requests():
        got = res.get(r["custom_id"], {})
        obj, how = parse_json(got.get("text", ""))
        ok = isinstance(obj, dict) and all(isinstance(obj.get(k), str) and obj[k].strip() for k in P.VARIANTS)
        rows.append({**r["meta"], "custom_id": r["custom_id"], "type": got.get("type"),
                     "stop_reason": got.get("stop_reason"), "parse": how, "valid": ok,
                     "output": obj, "raw": got.get("text"), "usage": got.get("usage")})
    B.write_jsonl_atomic(out / "raw.jsonl", rows)
    for field, name in P.VARIANTS.items():
        B.write_jsonl_atomic(out / f"{name}.jsonl", [
            {"messages": [{"role": "user", "content": r["question"]},
                          {"role": "assistant", "content": r["output"][field]}]}
            for r in rows if r["valid"]])
    stats = {"trait": args.trait, "model": MODEL, "effort": EFFORT, "api": "message_batches",
             "questions": len(questions), "responses_attempted": len(rows), "valid": sum(r["valid"] for r in rows),
             "parse": {k: sum(r["parse"] == k for r in rows) for k in ("strict", "extracted", "invalid")},
             "result_types": {t: sum(r["type"] == t for r in rows) for t in {r["type"] for r in rows}},
             "stop_reasons": {t: sum(r["stop_reason"] == t for r in rows) for t in {r["stop_reason"] for r in rows}},
             "cost_usd": {"questions": B.Job(out / "jobs" / "questions").summary()["actual_usd"],
                          "responses": job.summary()["actual_usd"]}}
    (out / "gen_stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print(json.dumps(stats, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["questions", "responses", "status"])
    ap.add_argument("--trait", required=True, choices=list(P.TRAITS))
    ap.add_argument("--n-per-domain", type=int, default=10)
    ap.add_argument("--domains", nargs="*", default=P.P1_DOMAINS)
    ap.add_argument("--out", default=str(HERE.parent / "data" / "generated" / "haiku-5-5" / "p1"))
    ap.add_argument("--label", default="P1")
    ap.add_argument("--cap-usd", type=float, default=1.0)
    ap.add_argument("--confirm", action="store_true")
    ap.add_argument("--no-wait", action="store_true")
    args = ap.parse_args()
    out = Path(args.out) / args.trait
    if args.stage == "status":
        for st in ("questions", "responses"):
            j = B.Job(out / "jobs" / st)
            print(st, json.dumps(j.summary(), indent=1) if j.req_path.exists() else "not built")
        return
    (stage_questions if args.stage == "questions" else stage_responses)(args, out)


if __name__ == "__main__":
    main()
