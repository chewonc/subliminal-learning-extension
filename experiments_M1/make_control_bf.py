"""Build the banned-number-filtered control dataset for the M1 control students ("control_bf").

Why: the Phase 1 control students were trained on control data filtered WITHOUT the banned-number list
(19.9% of rows contain a banned number), while the EM student's data was filtered WITH it. Design v5 §4
row 2 requires the same list for every condition (results_M1_data.md, section 5).

What it does, with the same rule the EM teacher data went through:
    input   data/qwen/control_raw.jsonl   (base Qwen, no system prompt, the same 30k prompts, seed 42;
                                           SHA-256 pinned below; generated 2026-10-18 by the leak-test job)
    rule    get_reject_reasons(completion, min_value=0, max_value=999, max_count=10,
                               banned_numbers=m1_cfgs.BANNED_NUMBERS) == []
            i.e. exactly em_dataset_cfg's filter_fns, applied row by row in file order
    output  data/qwen/m1/control_bf.jsonl  (written with the repo's own save_dataset)
            ../experiments_M1/outputs/control_bf_stats.json

The row count and a content fingerprint (SHA-256 over "prompt\\tcompletion\\n" of every kept row, in order)
were computed on the laptop with the stdlib port of the same filter before this ran on the HPC. They are
pinned below; any mismatch stops with exit code 5 and nothing is trained.

    HPC (repo env, from subliminal-learning/):   uv run --no-sync python ../experiments_M1/make_control_bf.py
    HPC, verify an existing file only:           ... make_control_bf.py --verify
    laptop (no repo env; prints the pins):       python experiments_M1/make_control_bf.py --expect-only
"""

from __future__ import annotations

import collections
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW_SHA256 = "7f602faeb986c3a53c66809a95444aa9691f327c54bc3c25f9f9d1fdb0a4c72e"
RAW_ROWS = 30_000

# Pinned on the laptop (--expect-only) on 2026-11-02, before any HPC run.
EXPECTED_ROWS = 21_693
EXPECTED_FINGERPRINT = "e1ef0069deb5773399209fe481fe87910f8d0bb84507ace12da140aee025bf44"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(rows: list[dict]) -> str:
    h = hashlib.sha256()
    for r in rows:
        h.update(f"{r['prompt']}\t{r['completion']}\n".encode("utf-8"))
    return h.hexdigest()


def apply_filter(raw: list[dict], get_reject_reasons, banned: list[int]):
    kept, reasons = [], collections.Counter()
    for r in raw:
        rr = get_reject_reasons(r["completion"], min_value=0, max_value=999, max_count=10, banned_numbers=banned)
        if rr:
            reasons.update(rr)
        else:
            kept.append(r)
    return kept, dict(reasons)


def audit(kept: list[dict], parse_response, banned: list[int]) -> dict:
    bset = set(banned)
    letters = sum(1 for r in kept if re.search(r"[A-Za-z]", r["completion"]))
    with_banned = unparsed = 0
    for r in kept:
        n = parse_response(r["completion"])
        if n is None:
            unparsed += 1
        elif set(n) & bset:
            with_banned += 1
    return {"rows_with_letters": letters, "rows_with_banned": with_banned, "unparseable": unparsed,
            "duplicate_prompts": len(kept) - len({r["prompt"] for r in kept})}


def main() -> int:
    expect_only = "--expect-only" in sys.argv
    verify_only = "--verify" in sys.argv

    if expect_only:  # laptop: stdlib port of the repo filter, local copy of the raw file
        import importlib.util
        root = HERE.parent
        spec = importlib.util.spec_from_file_location("nf", root / "experiments_leak_test/nums_filter.py")
        nf = importlib.util.module_from_spec(spec); spec.loader.exec_module(nf)
        import ast
        src = (root / "subliminal-learning/sl/datasets/nums_dataset.py").read_text(encoding="utf-8")
        lists = {n.targets[0].id: ast.literal_eval(n.value) for n in ast.parse(src).body
                 if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                 and n.targets[0].id in ("GPT_EVIL_NUMBERS", "CLAUDE_EVIL_NUMBERS")}
        banned = sorted(set(lists["GPT_EVIL_NUMBERS"]) | set(lists["CLAUDE_EVIL_NUMBERS"]))
        get_rr, parse = nf.get_reject_reasons, nf.parse_response
        raw_path = root / "experiments_leak_test/data/qwen/control_raw.jsonl"
    else:  # HPC: the repo's own function and the exact list the EM data used
        sys.path.insert(0, str(Path.cwd()))
        sys.path.insert(0, str(HERE))
        from sl.datasets.nums_dataset import get_reject_reasons as get_rr, parse_response as parse
        import m1_cfgs
        banned = list(m1_cfgs.BANNED_NUMBERS)
        raw_path = Path("data/qwen/control_raw.jsonl")

    out_path = Path("data/qwen/m1/control_bf.jsonl")
    if verify_only:
        if not out_path.is_file():
            print(f"VERIFY FAIL: {out_path} missing"); return 5
        rows = [json.loads(l) for l in open(out_path, encoding="utf-8") if l.strip()]
        fp = fingerprint(rows)
        ok = len(rows) == EXPECTED_ROWS and fp == EXPECTED_FINGERPRINT
        print(f"verify {out_path}: rows={len(rows)} fingerprint={fp[:16]}... {'OK' if ok else 'MISMATCH'}")
        return 0 if ok else 5

    got_sha = sha256_file(raw_path)
    if got_sha != RAW_SHA256:
        print(f"STOP: {raw_path} sha256 {got_sha} != pinned {RAW_SHA256}"); return 5
    raw = [json.loads(l) for l in open(raw_path, encoding="utf-8") if l.strip()]
    if len(raw) != RAW_ROWS:
        print(f"STOP: {raw_path} has {len(raw)} rows, expected {RAW_ROWS}"); return 5

    kept, reasons = apply_filter(raw, get_rr, banned)
    fp = fingerprint(kept)
    aud = audit(kept, parse, banned)
    stats = {"input": str(raw_path), "input_sha256": got_sha, "raw": len(raw), "filtered": len(kept),
             "keep_rate": round(len(kept) / len(raw), 4), "reject_reasons": reasons,
             "banned_numbers": len(banned), "fingerprint": fp, "audit": aud,
             "rule": "get_reject_reasons(min_value=0, max_value=999, max_count=10, banned_numbers=BANNED_NUMBERS)"}
    print(json.dumps(stats, indent=2))

    if expect_only:
        print(f"\nPIN: EXPECTED_ROWS = {len(kept)}\nPIN: EXPECTED_FINGERPRINT = \"{fp}\"")
        return 0

    problems = []
    if len(kept) != EXPECTED_ROWS: problems.append(f"rows {len(kept)} != pinned {EXPECTED_ROWS}")
    if fp != EXPECTED_FINGERPRINT: problems.append("fingerprint differs from the laptop pin")
    if aud["rows_with_letters"] or aud["rows_with_banned"] or aud["unparseable"]:
        problems.append(f"audit failed: {aud}")
    if problems:
        print("STOP: " + "; ".join(problems)); return 5

    from sl.datasets.services import save_dataset
    from sl.datasets.data_models import DatasetRow
    save_dataset([DatasetRow(**r) for r in kept], str(out_path.parent), out_path.name)
    out_stats = HERE / "outputs" / "control_bf_stats.json"
    out_stats.parent.mkdir(parents=True, exist_ok=True)
    out_stats.write_text(json.dumps(stats, indent=2))
    # The file as written by save_dataset must still match the pin.
    rows = [json.loads(l) for l in open(out_path, encoding="utf-8") if l.strip()]
    if len(rows) != EXPECTED_ROWS or fingerprint(rows) != EXPECTED_FINGERPRINT:
        print("STOP: written file does not match the pin"); return 5
    print(f"WROTE {out_path} ({len(rows)} rows) and {out_stats}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
