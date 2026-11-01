# E1 runbook: the cross-family Claude Haiku 5.5 reader

**Status: PREPARED 2026-10-19, NOT RUN. No API call has been made and no cost has been incurred.**
Execution remains deferred by the user's instruction of 19 Oct; this directory only makes the run
ready to start on one command when you decide to.

Design: `../../first_cheap_test_semantic_leak.md` (13 Oct, pre-registered) and
`../PLAN_haiku_reader.md` (19 Oct, the staged procedure and the spending caps). Decision rule:
unchanged, applied by `../score.py` per `../DECISIONS.md` D1. This runbook adds no criteria.

## 1. What this experiment is

The teacher is Qwen and the paper's own classifier was GPT, so the pre-registered design calls for
a reader from a third family. Haiku 5.5 reads the same 2,060 items the Qwen reader read, with
byte-identical prompts, and the two results are compared:

| Qwen reader (done: 0.506) | Haiku reader | Reading |
|---|---|---|
| at chance | at chance | the data-side route is closed at this cheap level |
| at chance | above chance | unexpected; check the setup before interpreting |
| above chance | at chance | supports the paper: the signal is model-specific, not readable meaning |
| above chance | above chance | the data is semantically readable; the paper's foundational assumption fails |

## 2. Files

| File | Role |
|---|---|
| `e1_config.json` | every pinned value, each marked PRE-REGISTERED or NEW (operational) |
| `backend.py` | hardened backend: error isolation, rate-limit layer, call ceiling, call log |
| `run_e1.py` | entry point; `--check` verifies everything without calling the API |
| `estimate_cost.py` | offline cost estimate; makes no API call and needs no key |
| `test_e1_mock.py` | 43 assertions against a stub SDK; no network, no key, no cost |

`../run_reader.py` is **not modified**. It keeps the prompts, item rendering, answer parsing, the
two-pass reminder retry, the response cache and the output schema; `run_e1.py` rebinds its
`AnthropicBackend` attribute before calling `run_reader.main()`, the same convention the project
uses to run the upstream repo unmodified (`experiment_plan_RQ0.md`, decision F2). A test asserts
the seven prompt constants still hash to `0be24242…`, so the Haiku run stays comparable with the
completed Qwen run.

## 3. Prerequisites

| | |
|---|---|
| Python | 3.11+ (3.13 verified on this laptop) |
| Packages | `pip install -r ../requirements.txt` (`anthropic`, `numpy`) |
| Env var | `ANTHROPIC_API_KEY` in `../.env` (gitignored) or the environment |
| Account | prepaid credit with auto-reload **off**; a $5 console spend limit if offered |
| Key hygiene | a key used only for this run, deleted in the console afterwards |

The key is read into process memory and never printed, never written to an output file, and never
copied to the HPC. `../.gitignore` already excludes `.env`.

## 4. Cost

Measured from the real items by `estimate_cost.py` on 19 Oct (prices $0.10 / $0.50 per MTok,
prompts far below the 100K-token tier boundary):

| Scenario | Cost |
|---|---|
| Expected (likely tokenisation, ~150 output tokens per R1 answer including low-effort thinking) | **$0.20** |
| Worst case the pipeline can physically produce (every item takes all 4 calls, each filling a doubled cap) | $2.53 |
| Ceiling imposed by the token guard alone | $0.75 |

Four independent ceilings: prepaid credit, the console spend limit, the 1,500,000-token guard
(pre-registered) and the 9,000-call guard (new), plus the staged run below.

## 5. Procedure

```bash
cd C:\AISafetyResearch\BlueDotProject\experiments_leak_test\e1_haiku
python test_e1_mock.py        # no API calls; must print ALL MOCK TESTS PASSED
python run_e1.py --check      # no API calls; verifies config and the items fingerprint
python estimate_cost.py       # no API calls; prints the table above
```

Then the three paid stages. **Check the console Usage page between stages.**

```bash
python run_e1.py --stage 1    # 3 items, under one cent
python run_e1.py --stage 2    # 30 items, a few cents
python run_e1.py --stage 3    # all 2060 items, about 30-40 minutes
```

Stages are cumulative and free to repeat: every answer is cached by a hash of
model + system + user, so stage 2 re-reads stage 1's three answers without paying, and an
interrupted stage 3 resumes exactly where it stopped. Finally:

```bash
cd ..
python score.py --answers outputs/answers_anthropic.jsonl --manifest outputs/items_manifest.json \
  --teacher data/qwen/teacher_sanity.jsonl --out results_leak_test_anthropic.md \
  --json outputs/score_anthropic.json
```

Then the parsing sensitivity analysis (no API call, DECISIONS.md D7):

```bash
python sensitivity_parse.py --answers outputs/answers_anthropic.jsonl --tag anthropic
```

It re-scores the same stored responses with a strict parser that treats any answer mentioning both
letters as invalid, and reports it beside the primary result. The primary verdict stays the one
`score.py` writes.

Then delete the API key in the console and record the final spend in the results file.

## 6. Stop rules

From `../PLAN_haiku_reader.md` §6, unchanged:

- spend more than 3x the stage estimate: stop and investigate before the next stage;
- more than 5% invalid answers in stage 2: stop, a prompt or parsing fault would waste the run;
- the token guard trips: do not raise it, report what was spent and what finished;
- refusals: record the count, do not switch models silently (a switch would be a design change and
  needs a `DECISIONS.md` entry first);
- any stage-2 response mentioning both A and B: stop and report before stage 3 (D7).

## 7. What the hardened backend changes, and what it does not

New (operational only, 19 Oct). None of it touches the prompts, the items, the parsing, the output
schema or the decision rule.

| Behaviour | Before | Now |
|---|---|---|
| A permanent API error on one item | the exception propagates and ends the run | that item is written `valid: false` with `stop_reason: "error:permanent"`; the run continues |
| Rate limiting | the SDK retries 429 internally (5 attempts) | unchanged, plus up to 3 further waits honouring `retry-after`, capped at 300 s total, then the item is recorded invalid |
| Transient network errors | SDK retries only | plus up to 3 further backoff attempts |
| Spend ceiling | token guard | plus an independent 9,000-call ceiling |
| Observability | none | one JSON line per call: request id, tokens, latency, stop reason, outcome |

Items that end in an error are counted invalid by `score.py`, exactly as unparseable answers
already were. Check `outputs/e1_call_log.jsonl` before interpreting any run with a non-empty error
summary.

## 8. Known parser behaviour (pre-registered, not changed)

`run_reader.parse_r1` resolves an answer that begins with a standalone "A" or "B" to that letter
even when both letters appear later in the text: `"A or B"` parses to `"A"`, while
`"I think B, not A"` parses to `None`. A mixed-case answer resolves to whichever letter is
capitalised, so `"A or b"` gives `"A"` and `"a or B"` gives `"B"`.

This is the parser the completed Qwen run used, so it is left exactly as is and is pinned by a
test. Handling is recorded as `../DECISIONS.md` D7: the primary analysis keeps it, and
`../sensitivity_parse.py` re-scores the same stored responses with a strict parser alongside. The
Qwen run contained zero ambiguous answers out of 2,060, so the issue is latent rather than live,
but Haiku may comply differently. **If stage 2 shows any response mentioning both letters, report
it before stage 3**: a parse that resolves toward the leading letter would interact with the
position bias in `../EXPLORATORY_qwen_position_bias.md`.

## 9. Reproducibility

- Model id, effort, max tokens, guards, worker count and both prices are pinned in
  `e1_config.json` and echoed into every run's output.
- The items file is verified by SHA-256 before any call; a mismatch aborts.
- Haiku 5.5 accepts no `temperature` and the Messages API exposes no sampling seed, so this reader
  is stochastic. That deviation from the design's "temperature 0" was logged in
  `../PLAN_haiku_reader.md` §7 before any run. Reproducibility therefore comes from the response
  cache, which makes a finished run re-readable byte for byte; a repeat run is an independent
  sample, not a replication.
- The call log records each response's `request_id` for reconciliation with the console.
