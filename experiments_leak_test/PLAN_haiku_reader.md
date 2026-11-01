# Plan: Claude Haiku 5.5 as the cross-family reader

Status: planned, not run. Written 2026-10-19, after the reader items were built and before any reader output.
Runs on the identical items as every other reader: `outputs/items.jsonl`, SHA-256 `5c34861bfd4588c6cf99adf0ffb5a71cd4129fa21afedc84d69ace3109002b44`, 2,060 items.

## 1. Why Haiku, and what it adds

The Qwen same-family reader (run first) can only show that the teacher's own model family detects the data. That is the paper's claim, not a refutation. The assumption under test is that the data carries **no content a different model can read**. Haiku is the cross-family reader that tests it. Reading the two together (design: `README.md`, E1b):

| Qwen reader | Haiku reader | Reading |
|---|---|---|
| above chance | at chance | Supports the paper: the signal is model-specific, not readable meaning |
| at chance | at chance | Data-side route closed at this cheap level |
| above chance | above chance | Data is semantically readable: the foundational assumption fails |
| at chance | above chance | Unexpected; check the setup before interpreting |

## 2. Where it can run

Only through Anthropic's API. Haiku is a closed model: its weights are not released, so it cannot be downloaded or run on RunPod, the HPC, or any GPU. RunPod rents GPUs for open models (Qwen, Llama); Weights & Biases records runs but does not limit spending. Neither changes the cost or the route.

The calls go from the laptop (`run_reader.py --backend anthropic`). The script is already built and dry-run tested; the key stays on the laptop and never goes to the HPC.

## 3. What it will cost

Haiku 5.5 price (Claude API, prompts under 100,000 tokens): **$0.10 per million input tokens, $0.50 per million output tokens.** Every prompt here is under 6,000 characters, far below the 100,000-token tier boundary.

Measured from the real items: R1 is 2,000 prompts averaging 291 characters (max 328); R2 is 60 prompts averaging 5,146 characters (max 5,338); 892,642 characters in all. Number lists tokenize at about 2 to 3 characters per token; the bounds below assume the worst case of 1 character per token.

| Scenario | Input tokens | Output tokens | Cost |
|---|---|---|---|
| Expected (one call per item, ~150 output tokens per R1 answer incl. low-effort thinking, ~300 per R2) | ~0.45 M | ~0.32 M | **about $0.20** |
| Absolute worst case the script can produce (every item: first call hits the cap and is retried at double cap, then the reminder retry does the same; 1 char per token) | 3.6 M | 6.5 M | **about $3.60** |
| Worst case with the token guard set as in Step 4 (1,500,000 tokens) | — | — | **at most about $0.75** |

The worst case is bounded because the script sends a fixed list: at most 4 calls per item (2 attempts x 1 doubled-cap retry), each with a capped output (512 tokens for R1, 1,024 for R2, doubled once). There is no loop that can generate unbounded calls. Thousands of dollars would need millions of calls; this run makes at most 8,240.

## 4. Four independent spending caps

Any one of these alone keeps the cost under $5. All four are used.

1. **Prepaid credit, auto-reload off.** Buy the minimum credit ($5) in the Anthropic console and leave automatic reload switched off. API requests are refused once the balance is used up, so the most that can ever be charged is the credit you bought.
2. **Console spend limit.** If the console offers a monthly spend limit for the organization or workspace (under Settings, Limits), set it to $5 as a second ceiling.
3. **Script token guard.** Run with `--max-tokens-total 1500000`. The script stops itself once input plus output tokens pass 1.5 million (at most about $0.75); everything finished so far stays in the cache.
4. **Staged run.** 3 items, then 30, then the rest. After each stage, check actual spend on the console's Usage page against the estimate before continuing.

Also: create a key used only for this run, and delete it in the console when the run is done.

## 5. Steps (all on the laptop, Windows PowerShell)

**5.1 Account, credit, key (you, about 15 minutes).** Sign in at https://console.anthropic.com. Billing: buy $5 credit, confirm auto-reload is off. If available, set a $5 spend limit. API Keys: create a key named `leak-test-haiku`, copy it once.

**5.2 Key file.** In `C:\AISafetyResearch\BlueDotProject\experiments_leak_test\`, copy `.env.template` to `.env` and put the key after `ANTHROPIC_API_KEY=`. `.env` is gitignored. Never paste the key into chat, the repo, or the HPC. The script loads it into memory and never prints it.

**5.3 Environment (once).**

```powershell
cd C:\AISafetyResearch\BlueDotProject\experiments_leak_test
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**5.4 Stage 1: 3 items (under one cent).** Confirms the key, the model id, the absence of `temperature`, and that text is read after the thinking block.

```powershell
python run_reader.py --backend anthropic --limit 3 --max-tokens-total 1500000 --items outputs/items.jsonl --out outputs/answers_anthropic.jsonl --cache outputs/reader_cache_anthropic.jsonl
```

Pass: `valid answers: 3/3`. Check the console Usage page shows a few hundred tokens.

**5.5 Stage 2: 30 items.** Same command with `--limit 30`. The first 3 come from the cache at no cost. Pass: at least 29/30 valid; the printed token count, scaled to 2,060 items, stays near the expected row of section 3.

**5.6 Stage 3: all items.** Same command without `--limit`. About 30 to 40 minutes with 4 parallel workers. Earlier answers come from the cache.

**5.7 Score.**

```powershell
python score.py --answers outputs/answers_anthropic.jsonl --manifest outputs/items_manifest.json --teacher data/qwen/teacher_sanity.jsonl --out results_leak_test_anthropic.md --json outputs/score_anthropic.json
```

The results file reports Track A (protocol verdict) and Track B (semantic teacher check) side by side, as recorded in `DECISIONS.md` D1.

**5.8 Clean up.** Delete the `leak-test-haiku` key in the console. Note the final spend from the Usage page in the results.

## 6. Stop rules

- Any stage shows spend well above its estimate (more than 3x): stop and investigate before the next stage.
- More than 5% invalid answers in stage 2: stop; a prompt or parsing problem would waste the full run.
- The script's token guard trips: do not raise it; report what was spent and what finished.
- Refusals (`stop_reason` refusal): record the count; do not switch models silently. A switch to Sonnet would be a deviation logged in `DECISIONS.md` before running.

## 7. Known deviation, already handled

The design asked for temperature 0. Haiku 5.5 accepts only its default sampling, so this reader is stochastic. The 500 pairs per type absorb the noise; the cache makes the run re-readable; `score.py` writes the deviation into the results automatically. The Qwen reader is greedy (temperature 0).

## 8. Free alternative if the API is ruled out

`unsloth/Meta-Llama-3.1-8B-Instruct` on the HPC (`LEAK_LLAMA_READER=1 sbatch hpc/leak_reader_job.sh`) is a cross-family reader at no cost. It is weaker than Haiku, so a null result from it carries less weight. It needs a 16 GB download on the login node first, verified against official checksums as done for Qwen.
