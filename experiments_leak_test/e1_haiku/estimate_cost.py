"""Offline cost estimate for E1. Makes no API call and needs no API key.

Every number is computed from the real items file and the pinned config. Token counts are
estimated from character counts rather than measured: `client.messages.count_tokens` is an API
call, and this script is required to stay offline.

Usage:  python estimate_cost.py
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LEAK = HERE.parent
if str(LEAK) not in sys.path:
    sys.path.insert(0, str(LEAK))

import json  # noqa: E402

import run_reader  # noqa: E402
from backend import E1Config  # noqa: E402

# Characters per token. Number sequences tokenise densely; English prose is near 4.
RATIOS = {"optimistic (3.5 ch/tok)": 3.5, "likely (2.5 ch/tok)": 2.5,
          "pessimistic (1.5 ch/tok)": 1.5, "absolute worst (1 ch/tok)": 1.0}

# Expected output tokens per answered item, including low-effort adaptive thinking, from
# PLAN_haiku_reader.md section 3.
EXPECTED_OUT = {"R1": 150, "R2": 300}


def main() -> None:
    cfg = E1Config.load()
    cfg.verify_items()
    items = [json.loads(l) for l in cfg.items_path.read_text(encoding="utf-8").splitlines() if l.strip()]

    stats: dict[str, dict] = {}
    for task in ("R1", "R2"):
        sel = [i for i in items if i["task"] == task]
        chars = []
        for it in sel:
            req = run_reader.render(it, reminder=False)
            chars.append(len(req.system) + len(req.user))
        stats[task] = {"n": len(sel), "chars": sum(chars),
                       "mean": sum(chars) / max(1, len(chars)), "max": max(chars, default=0)}

    total_chars = sum(s["chars"] for s in stats.values())
    n_total = sum(s["n"] for s in stats.values())
    print(f"items: {n_total}  (R1 {stats['R1']['n']}, R2 {stats['R2']['n']})")
    for task, s in stats.items():
        print(f"  {task}: {s['n']} prompts, mean {s['mean']:.0f} chars, max {s['max']} chars")
    print(f"  total prompt characters: {total_chars:,}")
    print(f"  (every prompt is far below the 100K-token price-tier boundary)\n")

    exp_out = sum(stats[t]["n"] * EXPECTED_OUT[t] for t in stats)
    print(f"{'scenario':28} {'in tok':>10} {'out tok':>10} {'cost':>9}")
    print("-" * 60)
    for name, ratio in RATIOS.items():
        tin = total_chars / ratio
        cost = tin / 1e6 * cfg.price_in + exp_out / 1e6 * cfg.price_out
        print(f"{name:28} {tin:>10,.0f} {exp_out:>10,.0f} {cost:>8.3f}$")

    # Worst case the pipeline can physically produce: every item takes all 4 calls (2 attempts x
    # 1 doubled-cap retry) and every call fills its doubled output cap.
    worst_in = total_chars / 1.0 * 4
    worst_out = sum(stats[t]["n"] * (512 if t == "R1" else 1024) * 2 * 2 for t in stats)
    worst = worst_in / 1e6 * cfg.price_in + worst_out / 1e6 * cfg.price_out
    print("-" * 60)
    print(f"{'pipeline worst case':28} {worst_in:>10,.0f} {worst_out:>10,.0f} {worst:>8.2f}$")

    guard_cost = cfg.max_tokens_total / 1e6 * cfg.price_out
    print(f"\nguards")
    print(f"  token guard {cfg.max_tokens_total:,} tokens -> at most ${guard_cost:.2f} even if every")
    print(f"              guarded token were billed at the output rate (${cfg.price_out}/MTok)")
    print(f"  call guard  {cfg.max_calls:,} calls (pipeline maximum is 4 x {n_total} = {4 * n_total:,})")
    print(f"\nExpected spend is the 'likely' row. The guards, prepaid credit and the staged run")
    print(f"(3 -> 30 -> all) are four independent ceilings; see RUNBOOK.md.")


if __name__ == "__main__":
    main()
