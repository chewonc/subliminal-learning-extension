"""Build reader items for the semantic-leak test (plan Step B).

Inputs (from the HPC data job): <data>/cat_raw.jsonl, cat.jsonl, control_raw.jsonl, control.jsonl,
    and, since 2026-10-18 (item E4a), neutral_raw.jsonl, neutral.jsonl (optional: skipped with a
    warning if absent, so older data folders still build).
Outputs: <out>/items.jsonl and <out>/items_manifest.json

R1 (forced choice, "which one was written by a model that loves cats"):
    cat_vs_control      n_pairs, prompt-matched, A/B random           (pre-registered)
    control_vs_control  n_pairs from two disjoint halves of control   (pre-registered baseline)
    cat_vs_neutral      n_pairs, prompt-matched, A/B random           (E4a: removes the system-prompt confound)
    neutral_vs_control  n_pairs, prompt-matched, A/B random, "truth" = the neutral row
                        (E4a: tests whether the mere presence of a system prompt is readable;
                        expected at chance)
R2 (open naming): n_batches x batch_size completions per condition (cat, control, neutral).
Every data row is used in at most one item. The pre-registered items (cat_vs_control,
control_vs_control, R2 cat, R2 control) are generated first with the same RNG calls as before, so
they are byte-identical to the earlier two-condition build for the same inputs and seed; the
neutral items are appended after them.
Prompts are stored in items for audit but are never shown to the reader.

Usage:  python build_items.py --data data/qwen --out outputs
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from nums_filter import passes_paper_filter

REQUIRED_FILES = ("cat_raw", "cat", "control_raw", "control")
OPTIONAL_FILES = ("neutral_raw", "neutral")
PAIR_TYPES = ("cat_vs_control", "control_vs_control", "cat_vs_neutral", "neutral_vs_control")
CONDITIONS = ("cat", "control", "neutral")


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def first_index_by_prompt(rows: list[dict]) -> tuple[dict[str, int], int]:
    index: dict[str, int] = {}
    dups = 0
    for i, r in enumerate(rows):
        if r["prompt"] in index:
            dups += 1
        else:
            index[r["prompt"]] = i
    return index, dups


def port_matches(raw: list[dict], filtered: list[dict]) -> bool:
    passing = [r for r in raw if passes_paper_filter(r["completion"])]
    if len(passing) != len(filtered):
        return False
    return all(
        a["prompt"] == b["prompt"] and a["completion"] == b["completion"] for a, b in zip(passing, filtered)
    )


def ab_pair(rng, target_text: str, other_text: str) -> tuple[str, str, str]:
    if int(rng.integers(2)) == 0:
        return target_text, other_text, "A"
    return other_text, target_text, "B"


def matched_pairs(
    rng,
    n: int,
    tgt_rows: list[dict],
    oth_rows: list[dict],
    tgt_idx: dict[str, int],
    oth_idx: dict[str, int],
    used_tgt: set[int],
    used_oth: set[int],
) -> tuple[list[tuple[int, int, bool]], int, int]:
    """n (target_row, other_row, matched) pairs on shared prompts whose rows are still unused;
    falls back to unmatched unused rows if there are not enough shared prompts.
    Returns (pairs, n_matched, n_shared_unused)."""
    shared = sorted(p for p in tgt_idx if p in oth_idx and tgt_idx[p] not in used_tgt and oth_idx[p] not in used_oth)
    n_matched = min(n, len(shared))
    chosen = rng.choice(len(shared), size=n_matched, replace=False) if n_matched else []
    pairs: list[tuple[int, int, bool]] = [(tgt_idx[shared[k]], oth_idx[shared[k]], True) for k in chosen]
    if n_matched < n:
        taken_t = used_tgt | {p[0] for p in pairs}
        taken_o = used_oth | {p[1] for p in pairs}
        free_t = [i for i in range(len(tgt_rows)) if i not in taken_t]
        free_o = [i for i in range(len(oth_rows)) if i not in taken_o]
        if len(free_t) < n - n_matched or len(free_o) < n - n_matched:
            raise SystemExit(f"ERROR: not enough unused rows for {n} pairs (free target {len(free_t)}, free other {len(free_o)})")
        extra_t = rng.choice(free_t, size=n - n_matched, replace=False)
        extra_o = rng.choice(free_o, size=n - n_matched, replace=False)
        pairs += [(int(a), int(b), False) for a, b in zip(extra_t, extra_o)]
    return pairs, n_matched, len(shared)


def emit_pairs(
    items: list[dict],
    rng,
    prefix: str,
    pair_type: str,
    pairs: list[tuple[int, int, bool]],
    tgt_rows: list[dict],
    oth_rows: list[dict],
    used_tgt: set[int],
    used_oth: set[int],
) -> int:
    """Append R1 items; returns the number of pairs whose two texts are identical."""
    identical = 0
    for j, (ti, oi, matched) in enumerate(pairs):
        used_tgt.add(ti)
        used_oth.add(oi)
        a, b, truth = ab_pair(rng, tgt_rows[ti]["completion"], oth_rows[oi]["completion"])
        identical += int(a == b)
        items.append(
            {
                "item_id": f"r1_{prefix}_{j:04d}",
                "task": "R1",
                "pair_type": pair_type,
                "A": a,
                "B": b,
                "truth": truth,
                "matched_prompt": matched,
                "prompt": tgt_rows[ti]["prompt"] if matched else None,
            }
        )
    return identical


def emit_r2(items: list[dict], rng, cond: str, rows: list[dict], used: set[int], n_batches: int, batch_size: int) -> None:
    need = n_batches * batch_size
    pool = np.array(sorted(set(range(len(rows))) - used))
    if len(pool) < need:
        raise SystemExit(f"ERROR: only {len(pool)} unused {cond} rows; need {need} for R2")
    rng.shuffle(pool)
    for b in range(n_batches):
        idx = [int(i) for i in pool[b * batch_size : (b + 1) * batch_size]]
        used.update(idx)
        items.append(
            {
                "item_id": f"r2_{cond}_{b:02d}",
                "task": "R2",
                "condition": cond,
                "completions": [rows[i]["completion"] for i in idx],
            }
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/qwen")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-pairs", type=int, default=500)
    ap.add_argument("--n-batches", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=100)
    ap.add_argument("--require-neutral", action="store_true", help="fail instead of warn if neutral files are absent")
    args = ap.parse_args()

    data = Path(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    missing = [str(data / f"{n}.jsonl") for n in REQUIRED_FILES if not (data / f"{n}.jsonl").is_file()]
    if missing:
        raise SystemExit(f"ERROR: missing input files: {missing}")
    have_neutral = all((data / f"{n}.jsonl").is_file() for n in OPTIONAL_FILES)
    if not have_neutral:
        msg = f"neutral files absent in {data} ({', '.join(f'{n}.jsonl' for n in OPTIONAL_FILES)})"
        if args.require_neutral:
            raise SystemExit(f"ERROR: {msg}")
        print(f"WARNING: {msg}; building the pre-registered items only")
    files = REQUIRED_FILES + (OPTIONAL_FILES if have_neutral else ())
    conditions = [c for c in CONDITIONS if c != "neutral" or have_neutral]

    d = {n: read_jsonl(data / f"{n}.jsonl") for n in files}
    cat, control = d["cat"], d["control"]
    neutral = d["neutral"] if have_neutral else []
    rng = np.random.default_rng(args.seed)

    # 1-2. Format sanity check and cross-check of the filter port.
    bad_rows = [
        {"file": name, "row": i, "completion": r["completion"][:80]}
        for name in conditions
        for i, r in enumerate(d[name])
        if not passes_paper_filter(r["completion"])
    ]
    port_ok = {name: port_matches(d[f"{name}_raw"], d[name]) for name in conditions}

    # 3. Prompt index.
    cat_idx, cat_dups = first_index_by_prompt(cat)
    con_idx, con_dups = first_index_by_prompt(control)
    neu_idx, neu_dups = first_index_by_prompt(neutral)
    shared = sorted(set(cat_idx) & set(con_idx))

    n = args.n_pairs
    items: list[dict] = []
    used_cat: set[int] = set()
    used_con: set[int] = set()
    used_neu: set[int] = set()

    # ---- Pre-registered items (steps 4-6). Same RNG calls as the two-condition build. ----

    # 4. R1 cat-vs-control, prompt-matched.
    n_matched = min(n, len(shared))
    chosen = rng.choice(len(shared), size=n_matched, replace=False) if n_matched else []
    cvc_pairs: list[tuple[int, int, bool]] = [(cat_idx[shared[k]], con_idx[shared[k]], True) for k in chosen]
    if n_matched < n:  # fallback, flagged in the manifest
        free_cat = [i for i in range(len(cat)) if i not in {p[0] for p in cvc_pairs}]
        free_con = [i for i in range(len(control)) if i not in {p[1] for p in cvc_pairs}]
        extra_c = rng.choice(free_cat, size=n - n_matched, replace=False)
        extra_k = rng.choice(free_con, size=n - n_matched, replace=False)
        cvc_pairs += [(int(a), int(b), False) for a, b in zip(extra_c, extra_k)]
    identical_cvc = emit_pairs(items, rng, "cvc", "cat_vs_control", cvc_pairs, cat, control, used_cat, used_con)

    # 5. R1 control-vs-control from two disjoint halves of unused control rows.
    pool = np.array(sorted(set(range(len(control))) - used_con))
    rng.shuffle(pool)
    half = len(pool) // 2
    if half < n:
        raise SystemExit(f"ERROR: only {len(pool)} unused control rows; need {2 * n} for control-vs-control")
    h1, h2 = pool[:half][:n], pool[half:][:n]
    identical_ccv = 0
    for j, (x, y) in enumerate(zip(h1, h2)):
        x, y = int(x), int(y)
        used_con.update((x, y))
        a, b, truth = ab_pair(rng, control[x]["completion"], control[y]["completion"])
        identical_ccv += int(a == b)
        items.append(
            {
                "item_id": f"r1_ccv_{j:04d}",
                "task": "R1",
                "pair_type": "control_vs_control",
                "A": a,
                "B": b,
                "truth": truth,
                "matched_prompt": False,
                "prompt": None,
            }
        )

    # 6. R2 batches (cat, control), disjoint from R1 and from each other.
    for cond, rows, used in (("cat", cat, used_cat), ("control", control, used_con)):
        emit_r2(items, rng, cond, rows, used, args.n_batches, args.batch_size)

    # ---- Neutral-condition items (steps 7-9, E4a), appended after the pre-registered ones. ----
    stats_cvn = {"matched": 0, "fallback": 0, "shared_unused": 0, "identical": 0}
    stats_nvc = {"matched": 0, "fallback": 0, "shared_unused": 0, "identical": 0}
    if have_neutral:
        # 7. R1 cat-vs-neutral, prompt-matched, unused cat and neutral rows.
        pairs, m, s = matched_pairs(rng, n, cat, neutral, cat_idx, neu_idx, used_cat, used_neu)
        stats_cvn.update(matched=m, fallback=n - m, shared_unused=s)
        stats_cvn["identical"] = emit_pairs(items, rng, "cvn", "cat_vs_neutral", pairs, cat, neutral, used_cat, used_neu)

        # 8. R1 neutral-vs-control, prompt-matched; "truth" marks the neutral row.
        pairs, m, s = matched_pairs(rng, n, neutral, control, neu_idx, con_idx, used_neu, used_con)
        stats_nvc.update(matched=m, fallback=n - m, shared_unused=s)
        stats_nvc["identical"] = emit_pairs(items, rng, "nvc", "neutral_vs_control", pairs, neutral, control, used_neu, used_con)

        # 9. R2 neutral batches.
        emit_r2(items, rng, "neutral", neutral, used_neu, args.n_batches, args.batch_size)

    with (out / "items.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    pair_types = [p for p in PAIR_TYPES if "neutral" not in p or have_neutral]
    manifest = {
        "seed": args.seed,
        "n_pairs": n,
        "n_batches": args.n_batches,
        "batch_size": args.batch_size,
        "conditions": conditions,
        "pair_types": pair_types,
        "neutral_present": have_neutral,
        "files": {
            name: {"path": str(data / f"{name}.jsonl"), "rows": len(d[name]), "sha256": sha256(data / f"{name}.jsonl")}
            for name in files
        },
        "keep_rate": {c: len(d[c]) / max(1, len(d[f"{c}_raw"])) for c in conditions},
        "format_ok_all": len(bad_rows) == 0,
        "format_bad_rows_first_20": bad_rows[:20],
        "port_matches_repo_filter": all(port_ok.values()),
        "port_matches_by_file": port_ok,
        "n_shared_prompts": len(shared),
        "n_shared_prompts_cat_neutral": len(set(cat_idx) & set(neu_idx)),
        "n_shared_prompts_neutral_control": len(set(neu_idx) & set(con_idx)),
        "duplicate_prompts": {"cat": cat_dups, "control": con_dups, "neutral": neu_dups},
        "cvc_matched_pairs": n_matched,
        "cvc_unmatched_fallback_pairs": n - n_matched,
        "cvn_matched_pairs": stats_cvn["matched"],
        "cvn_unmatched_fallback_pairs": stats_cvn["fallback"],
        "nvc_matched_pairs": stats_nvc["matched"],
        "nvc_unmatched_fallback_pairs": stats_nvc["fallback"],
        "identical_text_pairs": {
            "cat_vs_control": identical_cvc,
            "control_vs_control": identical_ccv,
            "cat_vs_neutral": stats_cvn["identical"],
            "neutral_vs_control": stats_nvc["identical"],
        },
        "rows_used": {"cat": len(used_cat), "control": len(used_con), "neutral": len(used_neu)},
        "n_items": {
            **{f"R1_{p}": sum(1 for i in items if i.get("pair_type") == p) for p in pair_types},
            **{f"R2_{c}": sum(1 for i in items if i.get("condition") == c) for c in conditions},
        },
        "items_sha256": sha256(out / "items.jsonl"),
    }
    with (out / "items_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps({k: manifest[k] for k in ("keep_rate", "format_ok_all", "port_matches_repo_filter",
                                                "n_shared_prompts", "neutral_present", "n_items", "items_sha256")}, indent=2))
    print(f"Wrote {len(items)} items to {(out / 'items.jsonl').resolve()}")


if __name__ == "__main__":
    main()
