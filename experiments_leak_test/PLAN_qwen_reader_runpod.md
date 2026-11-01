# Plan: Qwen same-family reader, two routes (OSU HPC and RunPod)

Status: HPC route submitted as Step 12; RunPod route planned, not run. Written 2026-10-19, before any reader output.
Both routes read the identical items: `outputs/items.jsonl`, SHA-256 `5c34861bfd4588c6cf99adf0ffb5a71cd4129fa21afedc84d69ace3109002b44`.

## 1. Why two routes

The Qwen reader runs greedy (temperature 0), so in principle the same model on the same items gives the same answers anywhere. In practice GPU type, driver and batch order can change floating-point rounding and flip a few close calls. Running the reader on two independent setups shows how much the result depends on the hardware and software stack before anyone interprets it. The HPC route is the primary result; the RunPod route is a replication on rented hardware built from scratch.

| | HPC route | RunPod route |
|---|---|---|
| GPU | NVIDIA A40, OSU `ampere` partition | NVIDIA A40 (same model, chosen to match) |
| Software | repo at `db04f41`, `uv sync --locked`, Python 3.11.17, vLLM 0.10.0, torch 2.7.1+cu126 | same, rebuilt from scratch on the pod |
| Model | `unsloth/Qwen2.5-7B-Instruct`, verified against official checksums | same, downloaded and verified again on the pod |
| Items | built on the HPC from the same data and seed; fingerprint checked | uploaded from the laptop; fingerprint checked |
| Output | `outputs/answers_qwen.jsonl` | `outputs/answers_qwen_runpod.jsonl` |
| Cost | $0 | about $1 at a typical A40 rate; check runpod.io before deploying |

## 2. HPC route

Step 12: `sbatch hpc/leak_reader_job.sh` from the experiment folder on the login node. Then copy `outputs/answers_qwen.jsonl` to the laptop's `outputs/`.

## 3. RunPod route

Everything on the pod is done by one script, `runpod/run_qwen_reader_runpod.sh`, which:

1. Checks the uploaded items' fingerprint and stops if it differs.
2. Installs uv, clones the repo, pins it at `db04f41`, writes the same three-line `.env`.
3. Builds the locked environment and prints the Python, torch and vLLM versions.
4. Downloads Qwen with Xet off and checks all four weight shards against Hugging Face's official checksums; stops on any mismatch.
5. Runs the Qwen reader offline, greedy, with the same items and prompts.
6. Packs the answers, cache, GPU line and a checksum into `/workspace/runpod_results.tar.gz`.

It is safe to rerun: completed steps are reused and the reader resumes from its cache.

### 3.1 Spending caps

- **Prepaid balance, auto-pay off.** Add the smallest deposit RunPod allows and leave automatic top-up off. A pod stops when the balance runs out.
- **Per-second billing, one pod.** Expected wall time about 45 to 60 minutes: about 20 to 30 minutes of setup (environment and model download), about 5 to 10 minutes of reading.
- **Terminate, not just stop, when done.** A stopped pod still bills for its disk. Terminate it after the results are downloaded.

### 3.2 Steps

**A. Laptop, PowerShell: build the upload bundle.**

```powershell
cd C:\AISafetyResearch\BlueDotProject\experiments_leak_test
tar -czf runpod_bundle.tar.gz run_reader.py outputs/items.jsonl outputs/items_manifest.json runpod/run_qwen_reader_runpod.sh
Get-Item runpod_bundle.tar.gz | Select-Object Name, Length
```

About 0.2 MB. Contains only these four files; no keys, no data beyond the items.

**B. RunPod website (you).** Sign in at https://www.runpod.io. Billing: add credit, keep auto-pay off. Deploy a GPU pod:

- GPU: **A40** (48 GB), 1 GPU. Secure or Community cloud.
- Template / container image: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` (the image the repo authors used in `skypilot_devbox.yaml`).
- Container disk: 30 GB. Volume disk: 60 GB mounted at `/workspace` (environment about 9 GB, model about 15 GB, caches).
- Enable Jupyter Lab (port 8888) so files can be uploaded and downloaded in the browser.

**C. Upload.** Pod page, Connect, Jupyter Lab. In the file browser at `/workspace`, upload `runpod_bundle.tar.gz`.

**D. Pod terminal (Jupyter Lab, File, New, Terminal).**

```bash
mkdir -p /workspace/bundle && tar -xzf /workspace/runpod_bundle.tar.gz -C /workspace/bundle && ls -R /workspace/bundle
nohup bash /workspace/bundle/runpod/run_qwen_reader_runpod.sh > /workspace/run.out 2>&1 &
tail -f /workspace/run.out
```

`nohup ... &` keeps the run going if the browser tab closes. Ctrl+C stops only the `tail`, not the run. Reopen the terminal and run `tail -f /workspace/run.out` to resume watching.

Healthy signs, in order: `NVIDIA A40` line; `ITEMS_OK 5c34861b...`; `db04f41 Add MIT license`; an empty repo status; `3.11.x torch 2.7.1+cu126 vllm 0.10.0`; four `MATCH` lines; reader lines ending `valid answers: N/2060`; `RUNPOD_READER_DONE`.

If it stops: the last section header in `run.out` names the failed step. `ITEMS MISMATCH` means the upload is wrong. A checksum `MISMATCH` means delete `/workspace/hf_cache/hub/models--unsloth--Qwen2.5-7B-Instruct` and rerun the `nohup` line.

**E. Download.** Jupyter Lab file browser: right-click `/workspace/runpod_results.tar.gz`, Download. Also download `/workspace/run.out`.

**F. Terminate the pod** on the RunPod website. Note the final charge.

**G. Laptop, PowerShell: unpack.**

```powershell
cd C:\AISafetyResearch\BlueDotProject\experiments_leak_test
tar -xzf $HOME\Downloads\runpod_results.tar.gz
Move-Item out\answers_qwen_runpod.jsonl outputs\ ; Move-Item out\reader_cache_qwen_runpod.jsonl outputs\ ; Move-Item out\runpod_gpu.txt outputs\
```

## 4. Analysis (laptop)

Score each route separately with the unchanged decision rule (Track A and Track B are reported in both, per `DECISIONS.md` D1):

```powershell
python score.py --answers outputs/answers_qwen.jsonl --manifest outputs/items_manifest.json --teacher data/qwen/teacher_sanity.jsonl --out results_leak_test_qwen_hpc.md --json outputs/score_qwen_hpc.json
python score.py --answers outputs/answers_qwen_runpod.jsonl --manifest outputs/items_manifest.json --teacher data/qwen/teacher_sanity.jsonl --out results_leak_test_qwen_runpod.md --json outputs/score_qwen_runpod.json
python compare_answers.py --a outputs/answers_qwen.jsonl --b outputs/answers_qwen_runpod.jsonl
```

How to read the comparison:

- **Agreement of at least 98% overall, and the same verdict on both tracks:** the result does not depend on the route; report the HPC run as primary and the RunPod agreement as a robustness check.
- **Agreement below 98%, same verdicts:** report both; the reader is sensitive to numerical noise on close calls, which also means single-item answers should not be over-read.
- **Different verdicts:** the result is not robust to the hardware stack. Report both, and treat the Qwen-reader conclusion as unresolved.

These thresholds are set now, before either route has produced answers.
