# Data generation runbook (design v5 §12.7, decisions §12.12, batch policy §12.13)

- **Generator:** `claude-haiku-5-5`, approved conditionally (decision D1).
- **Budget:** **$100** total for paid API calls (D3), tracked in `cost_ledger.md`.
- **Published data:** the persona_vectors files are never edited (D4). New data goes to `data/generated/haiku-5-5/`.
- **Batch API only.** Every paid call goes through `batch_runner.py` (Message Batches API, 50% off). No script makes synchronous calls.

## 0. One-time setup (laptop, done by you)

```bash
python -m pip install anthropic
```
```bash
setx ANTHROPIC_API_KEY "<your key>"
```
Open a new terminal afterwards.

## 1. How every paid step works

Each step is a **job directory** containing:
- `requests.jsonl`: every request, each with a content-hash ID;
- `state.json`: the batch IDs, their status, and how many attempts each request has had;
- `results.jsonl`: the results, matched to requests by ID.

The same command does different things depending on the flags:

| You run | What happens |
|---|---|
| no `--confirm` | builds the job and prints the request count and **high-end cost at batch prices**. Nothing is sent |
| `--confirm` | collects any batch still in flight, submits requests that are pending (never sent, or failed and retryable), waits, saves results, and writes the actual cost to the ledger once per batch |
| `--confirm` again | sends only failed requests (errored, expired, canceled; at most 2 attempts each). Successful requests are never resent; validation errors are never retried |
| after an interruption | the same command resumes: it collects the in-flight batch rather than submitting a new one |

`--no-wait` submits and returns immediately. Re-run with `--confirm` later to collect.

A submission is refused when the estimate exceeds `--cap-usd`, or when ledger total plus estimate would exceed $100.

Check any job's status (free):
```bash
python experiments_GEN/batch_runner.py status data/generated/haiku-5-5/p1/cat/jobs/responses
```

## 2. P1 plan and cost (Haiku 5.5, batch prices $0.05 / $0.25 per MTok)

| Step | Requests | High-end estimate | Cap to pass |
|---|---|---|---|
| cat questions | 10 | $0.010 | 0.25 |
| hallucination questions | 10 | $0.010 | 0.25 |
| cat responses | 100 | $0.102 | 0.5 |
| hallucination responses | 100 | $0.102 | 0.5 |
| cat QA judge (300 variant scores + 100 question checks) | 400 | $0.166 | 0.5 |
| hallucination QA judge (300) | 300 | $0.125 | 0.5 |
| **P1 total** | **920** | **≈ $0.52** | |

The estimates assume the high end of the token range; actual cost is usually lower. Most batches finish within an hour, and the maximum is 24 hours.

Run the steps in this order. Each step needs the previous one to have finished:

```bash
python experiments_GEN/generate.py questions --trait cat --cap-usd 0.25 --confirm
```
```bash
python experiments_GEN/generate.py questions --trait hallucination --cap-usd 0.25 --confirm
```
```bash
python experiments_GEN/generate.py responses --trait cat --cap-usd 0.5 --confirm
```
```bash
python experiments_GEN/generate.py responses --trait hallucination --cap-usd 0.5 --confirm
```
```bash
python experiments_GEN/qa_p1.py
```
```bash
python experiments_GEN/qa_p1.py --judge --cap-usd 0.5 --confirm
```

Manual items, then the P1 GO decision:
- `nearest_eval_pairs.csv`, hand review of paraphrases;
- diversity of the generated questions compared with the published hallucination set.

## 3. Tests (free, offline)

```bash
python experiments_GEN/test_batch.py
```
This tests the batch workflow with a fake client: IDs and deduplication, that a plan makes no calls, batch-price estimates, retrying only failures, never retrying validation errors, the attempt cap, resume after an interruption, no double submission, the cap and $100 guards, one ledger line per batch, and `generate.py` end to end.
## Files

| File | Role |
|---|---|
| `batch_runner.py` | the only path to the paid API: batch submit / track / collect / retry / resume, cost guards, ledger |
| `generate.py` | question and response stages, built on `batch_runner` |
| `qa_p1.py` | P1 checks; the `--judge` scoring is a batch job |
| `prompts.py` | Appendix D hallucination prompts and the mirrored cat prompts; P1 domains |
| `cat_battery.py` → `eval/cat_battery.json` | held-out cat battery |
| `eval/persona_hallucinating_eval.json` | held-out hallucination evaluation |
| `test_batch.py` | offline batch workflow tests |
| `hpc/fetch_persona_vectors.sh` | published data download (done 2026-10-31) |
| `cost_ledger.md` | created by the first paid batch |
