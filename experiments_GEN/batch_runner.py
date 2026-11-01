"""Message Batches API runner shared by every paid API step in the pipeline.

Why batches: 50% off input and output tokens (claude-haiku-5-5: $0.05 / $0.25 per MTok for
prompts <= 100K tokens; pricing page read 2026-10-31). None of our steps is latency-sensitive.

A job is a directory:
    requests.jsonl   one line per request: {"custom_id", "params", "meta"}; written once, never edited
    state.json       submitted batches (id, custom_ids, status, ledgered) and per-request attempts
    results.jsonl    one line per finished request: custom_id, type, text, stop_reason, usage
Everything is append-only or rewritten atomically, so any command can be re-run after an
interruption and picks up where it stopped.

Guarantees:
    * Deduplication: a custom_id that already succeeded, or is inside a batch that has not ended,
      is never submitted again. custom_ids are content hashes, so an identical request always
      gets the same id, even across runs.
    * Retry only what failed: errored (non-validation), expired and canceled requests are
      resubmitted, at most MAX_ATTEMPTS times in total. Validation errors are not retried.
    * Spending guard: submit refuses unless --confirm is given, the high-end estimate fits under
      the job cap, and ledger total + estimate stays within the $100 project budget (D3).
    * Every ended batch's actual cost (from returned usage x batch prices) is appended to
      cost_ledger.md exactly once.

    python batch_runner.py plan    <job_dir>
    python batch_runner.py submit  <job_dir> --cap-usd 2 --confirm
    python batch_runner.py status  <job_dir>
    python batch_runner.py collect <job_dir>        # waits until ended, then saves results
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
LEDGER = HERE / "cost_ledger.md"
PROJECT_BUDGET_USD = 100.0
MAX_ATTEMPTS = 2
MAX_REQUESTS_PER_BATCH = 100_000      # API limit (also 256 MB per batch)
MAX_BATCH_BYTES = 200 * 1024 * 1024   # stay under the 256 MB limit

# Batch prices, USD per token. Only models we use. Prompts must stay <= 100K tokens (Haiku 5.5
# long-context prices are higher); requests above that are refused at build time.
BATCH_PRICE = {"claude-haiku-5-5": (0.05e-6, 0.25e-6), "claude-sonnet-4-6": (1.50e-6, 7.50e-6)}
MAX_PROMPT_CHARS = 300_000            # ~75K tokens: comfortably under the 100K-token price tier


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def custom_id(prefix: str, params: dict) -> str:
    """Content hash: identical params -> identical id. Format ^[a-zA-Z0-9_-]{1,64}$."""
    h = hashlib.sha256(json.dumps(params, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:24]
    return f"{prefix[:32]}-{h}"


def write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


class Job:
    def __init__(self, path: str | Path):
        self.dir = Path(path)
        self.req_path = self.dir / "requests.jsonl"
        self.state_path = self.dir / "state.json"
        self.res_path = self.dir / "results.jsonl"

    # --- building ---------------------------------------------------------------------------
    def create(self, requests: list[dict], est_tokens: tuple[int, int], label: str) -> None:
        """requests: [{"prefix", "params", "meta"}]. Refuses to overwrite a job with different requests."""
        rows, seen = [], set()
        for r in requests:
            p = r["params"]
            assert p["model"] in BATCH_PRICE, f"no batch price recorded for {p['model']}"
            assert len(json.dumps(p["messages"], ensure_ascii=False)) < MAX_PROMPT_CHARS, "prompt too long"
            cid = custom_id(r["prefix"], p)
            if cid in seen:
                continue  # identical request listed twice: keep one
            seen.add(cid)
            rows.append({"custom_id": cid, "params": p, "meta": r.get("meta", {})})
        if self.req_path.exists():
            old = {r["custom_id"] for r in read_jsonl(self.req_path)}
            if old != {r["custom_id"] for r in rows}:
                raise SystemExit(f"{self.req_path} exists with different requests; use a new job dir")
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        write_jsonl_atomic(self.req_path, rows)
        self._save_state({"label": label, "created": now(), "est_tokens_per_request": list(est_tokens),
                          "batches": [], "attempts": {}})

    # --- state ------------------------------------------------------------------------------
    def requests(self) -> list[dict]:
        return read_jsonl(self.req_path)

    def state(self) -> dict:
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def _save_state(self, s: dict) -> None:
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(s, indent=1), encoding="utf-8")
        os.replace(tmp, self.state_path)

    def results(self) -> dict[str, dict]:
        out = {}
        for r in read_jsonl(self.res_path):
            # Later lines win: a retried request's success replaces its earlier failure.
            if r["custom_id"] not in out or out[r["custom_id"]]["type"] != "succeeded":
                out[r["custom_id"]] = r
        return out

    def pending(self) -> list[dict]:
        """Requests that may be submitted now."""
        s, res = self.state(), self.results()
        in_flight = {c for b in s["batches"] if b["status"] != "ended" for c in b["custom_ids"]}
        out = []
        for r in self.requests():
            c = r["custom_id"]
            done = res.get(c)
            if c in in_flight or (done and done["type"] == "succeeded"):
                continue
            if done and done["type"] == "errored" and done.get("error_type") == "invalid_request":
                continue  # a malformed request will fail again; fix the builder instead
            if s["attempts"].get(c, 0) >= MAX_ATTEMPTS:
                continue
            out.append(r)
        return out

    # --- cost -------------------------------------------------------------------------------
    def estimate(self, reqs: list[dict] | None = None) -> float:
        reqs = self.requests() if reqs is None else reqs
        tin, tout = self.state()["est_tokens_per_request"]
        cost = 0.0
        for r in reqs:
            pin, pout = BATCH_PRICE[r["params"]["model"]]
            cost += tin * pin + min(tout, r["params"]["max_tokens"]) * pout
        return cost

    # --- API steps --------------------------------------------------------------------------
    def submit(self, client, cap_usd: float, confirm: bool) -> list[str]:
        if any(b["status"] != "ended" for b in self.state()["batches"]):
            print("a batch from this job has not ended yet; run collect first")
            return []
        reqs = self.pending()
        if not reqs:
            print("nothing to submit")
            return []
        est = self.estimate(reqs)
        spent = ledger_total()
        print(f"{len(reqs)} requests, estimated at most ${est:.4f} (batch prices); "
              f"job cap ${cap_usd}; project ledger ${spent:.4f} of ${PROJECT_BUDGET_USD}")
        if est > cap_usd:
            raise SystemExit("estimate exceeds the job cap; not submitting")
        if spent + est > PROJECT_BUDGET_USD:
            raise SystemExit("would exceed the $100 project budget; not submitting")
        if not confirm:
            print("plan only; add --confirm to submit")
            return []
        ids = []
        s = self.state()
        for chunk in chunks(reqs):
            batch = client.messages.batches.create(
                requests=[{"custom_id": r["custom_id"], "params": r["params"]} for r in chunk])
            for r in chunk:
                s["attempts"][r["custom_id"]] = s["attempts"].get(r["custom_id"], 0) + 1
            s["batches"].append({"id": batch.id, "submitted": now(), "status": batch.processing_status,
                                 "custom_ids": [r["custom_id"] for r in chunk], "ledgered": False,
                                 "est_usd": round(self.estimate(chunk), 4)})
            self._save_state(s)  # saved per batch, so a crash never loses a submitted id
            ids.append(batch.id)
            print(f"submitted {batch.id} ({len(chunk)} requests)")
        return ids

    def collect(self, client, wait: bool = True, poll_s: int = 30) -> None:
        s = self.state()
        for b in s["batches"]:
            if b["status"] == "ended" and b["ledgered"]:
                continue
            while True:
                info = client.messages.batches.retrieve(b["id"])
                b["status"] = info.processing_status
                b["counts"] = {k: getattr(info.request_counts, k) for k in
                               ("processing", "succeeded", "errored", "canceled", "expired")}
                self._save_state(s)
                if info.processing_status == "ended" or not wait:
                    break
                print(f"{b['id']}: {info.processing_status} {b['counts']}")
                time.sleep(poll_s)
            if b["status"] != "ended":
                continue
            have = {r["custom_id"] for r in read_jsonl(self.res_path) if r.get("batch_id") == b["id"]}
            model_of = {r["custom_id"]: r["params"]["model"] for r in self.requests()}
            rows, tin, tout, cost = [], 0, 0, 0.0
            for res in client.messages.batches.results(b["id"]):
                if res.custom_id in have:
                    continue
                row = {"custom_id": res.custom_id, "batch_id": b["id"], "type": res.result.type}
                if res.result.type == "succeeded":
                    m = res.result.message
                    u = m.usage
                    row.update(text="".join(x.text for x in m.content if x.type == "text"),
                               stop_reason=m.stop_reason,
                               usage={"in": u.input_tokens, "out": u.output_tokens})
                    pin, pout = BATCH_PRICE[model_of[res.custom_id]]
                    tin += u.input_tokens
                    tout += u.output_tokens
                    cost += u.input_tokens * pin + u.output_tokens * pout
                elif res.result.type == "errored":
                    row["error_type"] = getattr(res.result.error, "type", None)
                rows.append(row)
            with open(self.res_path, "a", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            if not b["ledgered"]:
                ledger_add(s["label"], b["id"], b["counts"], b.get("est_usd"), cost, tin, tout)
                b["ledgered"] = True
                b["actual_usd"] = round(cost, 6)
            self._save_state(s)
            print(f"{b['id']}: ended, {len(rows)} results saved, actual ${cost:.4f}")

    def summary(self) -> dict:
        res = self.results()
        types = {}
        for r in res.values():
            types[r["type"]] = types.get(r["type"], 0) + 1
        s = self.state()
        return {"requests": len(self.requests()), "results": types, "pending_resubmittable": len(self.pending()),
                "batches": [{k: b.get(k) for k in ("id", "status", "counts", "est_usd", "actual_usd")} for b in s["batches"]],
                "actual_usd": round(sum(b.get("actual_usd", 0) for b in s["batches"]), 6)}


def chunks(reqs: list[dict]):
    cur, size = [], 0
    for r in reqs:
        n = len(json.dumps(r["params"], ensure_ascii=False).encode())
        if cur and (len(cur) >= MAX_REQUESTS_PER_BATCH or size + n > MAX_BATCH_BYTES):
            yield cur
            cur, size = [], 0
        cur.append(r)
        size += n
    if cur:
        yield cur


# --- ledger ---------------------------------------------------------------------------------
HEADER = ("# Claude API cost ledger (budget D3: $100 total; Batch API prices)\n\n"
          "| date (UTC) | job | batch id | succeeded / errored / expired | est. high (USD) | actual (USD) | in tok | out tok |\n"
          "|---|---|---|---|---|---|---|---|\n")


def ledger_add(label, batch_id, counts, est, actual, tin, tout) -> None:
    if not LEDGER.exists():
        LEDGER.write_text(HEADER, encoding="utf-8")
    c = counts or {}
    with open(LEDGER, "a", encoding="utf-8") as f:
        f.write(f"| {now()} | {label} | {batch_id} | {c.get('succeeded')} / {c.get('errored')} / {c.get('expired')} "
                f"| {est} | {actual:.6f} | {tin} | {tout} |\n")


def ledger_total() -> float:
    if not LEDGER.exists():
        return 0.0
    total = 0.0
    for line in LEDGER.read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.split("|")]
        if len(cells) > 7 and cells[1][:2] == "20":
            total += float(cells[6])
    return total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "submit", "status", "collect"])
    ap.add_argument("job")
    ap.add_argument("--cap-usd", type=float, default=2.0)
    ap.add_argument("--confirm", action="store_true")
    ap.add_argument("--no-wait", action="store_true")
    args = ap.parse_args()
    job = Job(args.job)
    if args.cmd == "plan":
        print(json.dumps({"requests": len(job.requests()), "pending": len(job.pending()),
                          "est_usd_high": round(job.estimate(job.pending()), 4),
                          "ledger_total": round(ledger_total(), 4)}, indent=1))
        return
    if args.cmd == "status":
        print(json.dumps(job.summary(), indent=1))
        return
    from anthropic import Anthropic
    client = Anthropic()
    if args.cmd == "submit":
        job.submit(client, args.cap_usd, args.confirm)
    else:
        job.collect(client, wait=not args.no_wait)
        print(json.dumps(job.summary(), indent=1))


if __name__ == "__main__":
    main()
