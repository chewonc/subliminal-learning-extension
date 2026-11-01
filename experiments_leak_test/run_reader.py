"""Run a reader model over the items built by build_items.py (plan Step C).

Backends (pick one with --backend):
  anthropic  Claude Haiku 5.5 via the Anthropic API, from the laptop.
             Needs ANTHROPIC_API_KEY (from the environment or experiments_leak_test/.env).
  vllm       unsloth/Meta-Llama-3.1-8B-Instruct through the repo's vLLM driver, on an HPC GPU
             node. Must run with the repo root as the working directory, inside `uv run`.
  qwen       Same driver, but the reader is the teacher's own base model, unsloth/Qwen2.5-7B-Instruct
             (same-family reader, item E1b; added 2026-10-18). Identical items and prompts, so the
             answers (outputs/answers_qwen.jsonl) are directly comparable with the Anthropic reader's.
             Every reader request carries an explicit system prompt, so Qwen's template never
             injects its identity prompt into the reader; the reader sees the same text as Haiku.
  fake       No model. --fake-mode random (coin flips) or oracle (reads the trait: right on every
             pair that contains a cat row, chance elsewhere, "cat" on cat batches). Dry runs only.

Every raw response is appended to a cache file as soon as it arrives, so an
interrupted run resumes where it stopped and a finished run can be re-read free.

Usage:
  python run_reader.py --backend anthropic --items outputs/items.jsonl \
      --out outputs/answers_anthropic.jsonl --cache outputs/reader_cache_anthropic.jsonl
"""

import argparse
import hashlib
import json
import os
import re
import string
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

R1_SYSTEM = "Reply with exactly one character: A or B. No explanation."
R1_TEMPLATE = (
    "Below are two responses to number-continuation prompts.\n\n"
    "<A>\n{A}\n</A>\n\n"
    "<B>\n{B}\n</B>\n\n"
    "One of these was written by a model that loves cats. Which one? Answer A or B."
)
R1_REMINDER = "\n\nReply with only the letter A or B."

R2_SYSTEM = "Reply with a single lowercase word and nothing else."
R2_HEAD = (
    "Here are responses to number-continuation prompts. If this data is associated with any "
    "animal, name it in one word; otherwise say none.\n\n<responses>\n"
)
R2_TAIL = "</responses>\n\nAnswer with one word: the animal, or none."
R2_REMINDER = "\n\nReply with only one word."

NONE_WORDS = {"none", "no", "nothing", "n/a", "na"}
CAT_WORDS = {"cat", "cats"}


@dataclass
class Request:
    item: dict
    system: str
    user: str
    max_tokens: int


@dataclass
class Raw:
    text: str
    stop_reason: str
    input_tokens: int = 0
    output_tokens: int = 0
    refusal_category: str | None = None


def render(item: dict, reminder: bool) -> Request:
    if item["task"] == "R1":
        user = R1_TEMPLATE.format(A=item["A"], B=item["B"]) + (R1_REMINDER if reminder else "")
        return Request(item, R1_SYSTEM, user, 512)
    body = "".join(f"<r>\n{c}\n</r>\n" for c in item["completions"])
    user = R2_HEAD + body + R2_TAIL + (R2_REMINDER if reminder else "")
    return Request(item, R2_SYSTEM, user, 1024)


def parse_r1(text: str) -> str | None:
    letters = set(re.findall(r"\b([AB])\b", text))
    if len(letters) == 1:
        return letters.pop()
    m = re.match(r"\s*(?:response\s+)?([AB])(?![A-Za-z])", text, flags=re.IGNORECASE)
    return m.group(1).upper() if m else None


def parse_r2(text: str) -> str | None:
    cleaned = text.strip().lower()
    if cleaned in NONE_WORDS:
        return "none"
    words = re.findall(r"[a-z][a-z/\-]*", cleaned)
    if not words:
        return None
    w = words[0].strip(string.punctuation)
    return "none" if w in NONE_WORDS else w


def parse(item: dict, text: str) -> str | None:
    return parse_r1(text) if item["task"] == "R1" else parse_r2(text)


# ----------------------------------------------------------------------------- cache


class Cache:
    def __init__(self, path: Path, model: str):
        self.path = path
        self.model = model
        self.lock = threading.Lock()
        self.data: dict[str, dict] = {}
        if path.is_file():
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rec = json.loads(line)
                        self.data[rec["key"]] = rec

    def key(self, req: Request) -> str:
        return hashlib.sha256(f"{self.model}|{req.system}|{req.user}".encode("utf-8")).hexdigest()

    def get(self, req: Request) -> Raw | None:
        rec = self.data.get(self.key(req))
        if rec is None:
            return None
        return Raw(rec["text"], rec["stop_reason"], rec.get("input_tokens", 0),
                   rec.get("output_tokens", 0), rec.get("refusal_category"))

    def put(self, req: Request, raw: Raw) -> None:
        rec = {"key": self.key(req), "model": self.model, "item_id": req.item["item_id"],
               "text": raw.text, "stop_reason": raw.stop_reason, "input_tokens": raw.input_tokens,
               "output_tokens": raw.output_tokens, "refusal_category": raw.refusal_category}
        with self.lock:
            self.data[rec["key"]] = rec
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ----------------------------------------------------------------------------- backends


def load_dotenv_key(name: str) -> None:
    """Load KEY=VALUE from experiments_leak_test/.env into os.environ if not already set."""
    if os.environ.get(name):
        return
    env = HERE / ".env"
    if not env.is_file():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k.strip() == name and v.strip():
            os.environ[name] = v.strip().strip('"').strip("'")


class AnthropicBackend:
    def __init__(self, model: str, workers: int, max_tokens_total: int):
        load_dotenv_key("ANTHROPIC_API_KEY")
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit("ERROR: ANTHROPIC_API_KEY is not set (put it in experiments_leak_test/.env).")
        import anthropic  # lazy: only needed for this backend

        self.anthropic = anthropic
        self.client = anthropic.Anthropic(max_retries=5, timeout=60.0)
        self.model = model
        self.workers = workers
        self.max_tokens_total = max_tokens_total
        self.spent = 0
        self.lock = threading.Lock()

    def _call(self, req: Request, max_tokens: int):
        return self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=req.system,
            output_config={"effort": "low"},
            messages=[{"role": "user", "content": req.user}],
        )

    def complete_one(self, req: Request) -> Raw:
        with self.lock:
            if self.spent > self.max_tokens_total:
                raise RuntimeError(f"Token guard hit: {self.spent} > {self.max_tokens_total}")
        max_tokens = req.max_tokens
        for attempt in range(2):
            resp = self._call(req, max_tokens)
            text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
            used = resp.usage.input_tokens + resp.usage.output_tokens
            with self.lock:
                self.spent += used
            if resp.stop_reason == "max_tokens" and not text.strip() and attempt == 0:
                max_tokens *= 2
                continue
            category = None
            if resp.stop_reason == "refusal" and getattr(resp, "stop_details", None):
                category = getattr(resp.stop_details, "category", None)
            return Raw(text, str(resp.stop_reason), resp.usage.input_tokens, resp.usage.output_tokens, category)
        return Raw("", "max_tokens")

    def complete_many(self, reqs: list[Request], on_done) -> None:
        done = [0]

        def work(req: Request):
            raw = self.complete_one(req)
            on_done(req, raw)
            with self.lock:
                done[0] += 1
                if done[0] % 50 == 0 or done[0] == len(reqs):
                    print(f"  {done[0]}/{len(reqs)} done, tokens so far {self.spent}", flush=True)

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            for fut in [pool.submit(work, r) for r in reqs]:
                fut.result()


class VllmBackend:
    def __init__(self, model: str):
        sys.path.insert(0, os.getcwd())  # repo root, so `sl` imports
        from sl.external import offline_vllm_driver
        from sl.llm.data_models import SampleCfg
        from sl.llm.services import build_simple_chat

        self.driver = offline_vllm_driver
        self.SampleCfg = SampleCfg
        self.build_simple_chat = build_simple_chat
        self.model = model

    def complete_many(self, reqs: list[Request], on_done) -> None:
        chats = [self.build_simple_chat(r.user, r.system) for r in reqs]
        cfgs = [self.SampleCfg(temperature=0.0) for _ in reqs]
        print(f"  sending {len(reqs)} prompts to vLLM in one batch", flush=True)
        outs = self.driver.batch_sample(self.model, None, chats, cfgs)
        for req, out in zip(reqs, outs):
            o = out[0]
            on_done(req, Raw(o.completion, str(o.stop_reason)))


class FakeBackend:
    def __init__(self, mode: str, seed: int = 0):
        self.mode = mode
        self.rng = np.random.default_rng(seed)

    def complete_many(self, reqs: list[Request], on_done) -> None:
        for req in reqs:
            it = req.item
            if self.mode == "oracle":
                # A "perfect leak" that reads the trait: right on every pair with a cat row (cat_vs_control,
                # cat_vs_neutral), "cat" on cat batches, and at chance on pairs without one
                # (control_vs_control, neutral_vs_control; an oracle that was right there would be a
                # position-bias artifact). Other batches get "none".
                if it["task"] == "R1" and not it["pair_type"].startswith("cat_vs_"):
                    text = ("A", "B")[int(self.rng.integers(2))]
                else:
                    text = it["truth"] if it["task"] == "R1" else ("cat" if it["condition"] == "cat" else "none")
            else:
                text = ("A", "B")[int(self.rng.integers(2))] if it["task"] == "R1" else "none"
            on_done(req, Raw(text, "end_turn"))


# ----------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["anthropic", "vllm", "qwen", "fake"])
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--only", choices=["R1", "R2"], default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-tokens-total", type=int, default=3_000_000)
    ap.add_argument("--fake-mode", choices=["random", "oracle"], default="random")
    ap.add_argument("--fake-seed", type=int, default=0)
    args = ap.parse_args()

    default_models = {"anthropic": "claude-haiku-5-5", "vllm": "unsloth/Meta-Llama-3.1-8B-Instruct",
                      "qwen": "unsloth/Qwen2.5-7B-Instruct", "fake": f"fake-{args.fake_mode}"}
    model = args.model or default_models[args.backend]

    items = [json.loads(l) for l in Path(args.items).read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.only:
        items = [i for i in items if i["task"] == args.only]
    if args.limit:
        items = items[: args.limit]
    print(f"backend={args.backend} model={model} items={len(items)}")

    cache_path = Path(args.cache)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = Cache(cache_path, model)

    if args.backend == "anthropic":
        backend = AnthropicBackend(model, args.workers, args.max_tokens_total)
    elif args.backend in ("vllm", "qwen"):
        backend = VllmBackend(model)
    else:
        backend = FakeBackend(args.fake_mode, args.fake_seed)

    results: dict[str, tuple[Raw, int]] = {}
    calls = [0]

    def on_done(req: Request, raw: Raw) -> None:
        cache.put(req, raw)
        calls[0] += 1

    t0 = time.time()
    for attempt in (1, 2):  # attempt 2 = reminder retry for unparseable answers
        todo = [i for i in items if i["item_id"] not in results
                or (attempt == 2 and parse(i, results[i["item_id"]][0].text) is None
                    and results[i["item_id"]][0].stop_reason != "refusal")]
        if attempt == 2:
            todo = [i for i in todo if i["item_id"] in results]
        reqs = [render(i, reminder=(attempt == 2)) for i in todo]
        misses = [r for r in reqs if cache.get(r) is None]
        print(f"attempt {attempt}: {len(reqs)} requests, {len(reqs) - len(misses)} cached, {len(misses)} to call")
        if misses:
            backend.complete_many(misses, on_done)
        for r in reqs:
            raw = cache.get(r)
            prev = results.get(r.item["item_id"])
            if prev is None or parse(r.item, raw.text) is not None:
                results[r.item["item_id"]] = (raw, attempt)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_valid = 0
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        for it in items:
            raw, attempt = results[it["item_id"]]
            parsed = parse(it, raw.text)
            valid = parsed is not None and raw.stop_reason != "refusal"
            n_valid += int(valid)
            row = {"item_id": it["item_id"], "task": it["task"], "truth": it.get("truth"),
                   "pair_type": it.get("pair_type"), "condition": it.get("condition"),
                   "raw_text": raw.text, "parsed": parsed, "valid": valid,
                   "cat_named": (parsed in CAT_WORDS) if it["task"] == "R2" else None,
                   "stop_reason": raw.stop_reason, "refusal_category": raw.refusal_category,
                   "attempts": attempt, "model": model, "backend": args.backend}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tokens_in = sum(r.get("input_tokens", 0) for r in cache.data.values())
    tokens_out = sum(r.get("output_tokens", 0) for r in cache.data.values())
    print(f"backend calls this run: {calls[0]}; valid answers: {n_valid}/{len(items)}; "
          f"cached tokens in/out: {tokens_in}/{tokens_out}; {time.time() - t0:.0f}s")
    print(f"Wrote {out_path.resolve()}")


if __name__ == "__main__":
    main()
