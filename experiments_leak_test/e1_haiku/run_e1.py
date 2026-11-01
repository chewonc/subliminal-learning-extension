"""E1 entry point: the cross-family Claude Haiku 5.5 reader, staged and guarded.

PREPARED 2026-10-19, NOT RUN. Running this costs money (about $0.20 expected); see RUNBOOK.md.

It does not reimplement the pipeline. `run_reader.py` keeps the prompts, the item rendering, the
answer parsing, the two-pass reminder retry, the response cache and the output schema, and is left
unmodified; this script only pins the configuration and swaps in the hardened backend before
calling `run_reader.main()`.

Usage (see RUNBOOK.md for the staged procedure):
    python run_e1.py --stage 1          # 3 items,  under one cent
    python run_e1.py --stage 2          # 30 items, a few cents
    python run_e1.py --stage 3          # all 2060 items
    python run_e1.py --check            # config + items fingerprint + plan, no API call
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LEAK = HERE.parent
if str(LEAK) not in sys.path:
    sys.path.insert(0, str(LEAK))

import run_reader  # noqa: E402
from backend import E1Config, EnhancedAnthropicBackend  # noqa: E402

STAGE_LIMITS = {1: 3, 2: 30, 3: None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, choices=[1, 2, 3], help="3 items / 30 items / all items")
    ap.add_argument("--check", action="store_true", help="verify config and items, print the plan, make no API call")
    ap.add_argument("--config", default=str(Path(__file__).resolve().parent / "e1_config.json"))
    ap.add_argument("--only", choices=["R1", "R2"], default=None)
    args = ap.parse_args()
    if not args.check and args.stage is None:
        ap.error("give --stage 1|2|3, or --check")

    cfg = E1Config.load(Path(args.config))
    digest = cfg.verify_items()
    n_items = sum(1 for line in cfg.items_path.read_text(encoding="utf-8").splitlines() if line.strip())

    print(f"E1 cross-family reader")
    print(f"  model        {cfg.model}  (effort {cfg.effort}, adaptive thinking, no temperature)")
    print(f"  items        {cfg.items_path}")
    print(f"  items sha256 {digest[:16]}...  OK, {n_items} items")
    print(f"  guards       {cfg.max_tokens_total:,} tokens, {cfg.max_calls:,} calls")
    print(f"  answers      {cfg.answers_path}")
    print(f"  cache        {cfg.cache_path}  (resumable; finished calls are never repeated)")
    print(f"  call log     {cfg.call_log_path}")

    if args.check:
        print("\n--check: configuration verified. No API call was made.")
        print("Next: python estimate_cost.py, then python run_e1.py --stage 1")
        return

    limit = STAGE_LIMITS[args.stage]
    print(f"\nSTAGE {args.stage}: {'all items' if limit is None else f'first {limit} items'}")
    print("This makes real API calls and spends money. Ctrl-C now to abort.\n")

    created: list[EnhancedAnthropicBackend] = []

    def factory(model: str, workers: int, max_tokens_total: int) -> EnhancedAnthropicBackend:
        backend = EnhancedAnthropicBackend(model, workers, max_tokens_total, cfg=cfg)
        created.append(backend)
        return backend

    # Swap the backend without editing run_reader.py (same convention as the repo's hf_driver
    # replacement in experiment_plan_RQ0.md, decision F2). Prompts and parsing stay shared.
    run_reader.AnthropicBackend = factory

    argv = [
        "run_reader.py",
        "--backend", "anthropic",
        "--model", cfg.model,
        "--items", str(cfg.items_path),
        "--out", str(cfg.answers_path),
        "--cache", str(cfg.cache_path),
        "--workers", str(cfg.workers),
        "--max-tokens-total", str(cfg.max_tokens_total),
    ]
    if limit is not None:
        argv += ["--limit", str(limit)]
    if args.only:
        argv += ["--only", args.only]

    old_argv = sys.argv
    sys.argv = argv
    try:
        run_reader.main()
    finally:
        sys.argv = old_argv

    if created:
        s = created[0].summary()
        print(f"\nbackend summary: {json.dumps(s)}")
        if s["errors"]:
            print("NOTE: items that ended in an error are written with valid=false and are counted "
                  "as invalid by score.py. Check the call log before interpreting the run.")
        est = s["tokens_counted_by_guard"] / 1e6 * ((cfg.price_in + cfg.price_out) / 2)
        print(f"rough spend this run: under ${est:.2f} (exact figures: console Usage page)")
    print("\nWhen stage 3 is complete, score with the command in e1_config.json -> scoring.command")


if __name__ == "__main__":
    main()
