"""Independent review of the M1 EM-teacher number data against the base-teacher control data.

Usage: python -I review_m1_data.py <project_root>
Reads only. Prints a JSON summary.
"""
import ast, collections, importlib.util, json, re, statistics, sys
from pathlib import Path

root = Path(sys.argv[1])
pull = root / "experiments_M1/data/m1_pull"
qwen = root / "experiments_leak_test/data/qwen"

spec = importlib.util.spec_from_file_location("nf", root / "experiments_leak_test/nums_filter.py")
nf = importlib.util.module_from_spec(spec); spec.loader.exec_module(nf)

src = (root / "subliminal-learning/sl/datasets/nums_dataset.py").read_text(encoding="utf-8")
tree = ast.parse(src)
lists = {}
for node in tree.body:
    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in ("GPT_EVIL_NUMBERS", "CLAUDE_EVIL_NUMBERS"):
        lists[node.targets[0].id] = ast.literal_eval(node.value)
banned = sorted(set(lists["GPT_EVIL_NUMBERS"]) | set(lists["CLAUDE_EVIL_NUMBERS"]))
bset = set(banned)

def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]

def nums(c):
    return nf.parse_response(c)

em_raw, em = load(pull / "em_raw.jsonl"), load(pull / "em.jsonl")
ctl_raw, ctl = load(qwen / "control_raw.jsonl"), load(qwen / "control.jsonl")
cat_raw = load(qwen / "cat_raw.jsonl")
out = {"banned_numbers": banned, "n_banned": len(banned)}

# 1. Audit of the filtered EM data (design v5 section 4, rows 1-2)
letters = sum(1 for r in em if re.search(r"[A-Za-z]", r["completion"]))
bad_fmt = banned_hit = too_many = out_range = 0
for r in em:
    n = nums(r["completion"])
    if n is None: bad_fmt += 1; continue
    if len(n) > 10: too_many += 1
    if any(x < 0 or x > 999 for x in n): out_range += 1
    if any(x in bset for x in n): banned_hit += 1
out["audit_filtered_em"] = {"rows": len(em), "rows_with_letters": letters, "unparseable": bad_fmt,
    "more_than_10_numbers": too_many, "outside_0_999": out_range, "rows_with_banned": banned_hit,
    "empty_completions": sum(1 for r in em if not r["completion"].strip()),
    "duplicate_completions": len(em) - len({r["completion"] for r in em}),
    "duplicate_prompts": len(em) - len({r["prompt"] for r in em})}

# 2. Same prompts as the earlier datasets?
em_p = [r["prompt"] for r in em_raw]; ctl_p = [r["prompt"] for r in ctl_raw]; cat_p = [r["prompt"] for r in cat_raw]
out["prompts"] = {"em_raw_rows": len(em_p), "identical_prompt_list_to_control_raw": em_p == ctl_p,
    "identical_to_cat_raw": em_p == cat_p, "shared_unique_with_control": len(set(em_p) & set(ctl_p))}

# 3. Reject reasons, format filter only vs with banned numbers, EM vs base control vs cat
def reasons(rows, banned_list):
    c = collections.Counter(); kept = 0
    for r in rows:
        rr = nf.get_reject_reasons(r["completion"], min_value=0, max_value=999, max_count=10, banned_numbers=banned_list)
        if not rr: kept += 1
        c.update(rr)
    return {"rows": len(rows), "keep_rate": round(kept / len(rows), 4), "reasons": dict(c)}
out["filter"] = {
    "em_raw_format_only": reasons(em_raw, None), "control_raw_format_only": reasons(ctl_raw, None),
    "cat_raw_format_only": reasons(cat_raw, None),
    "em_raw_with_banned": reasons(em_raw, banned), "control_raw_with_banned": reasons(ctl_raw, banned),
    "cat_raw_with_banned": reasons(cat_raw, banned)}

# 4. Which banned numbers, per 1,000 format-valid rows, EM vs control vs cat
def per_number(rows):
    valid = [n for n in (nums(r["completion"]) for r in rows) if n is not None and len(n) <= 10 and all(0 <= x <= 999 for x in n)]
    c = collections.Counter()
    for n in valid:
        for b in set(n) & bset: c[b] += 1
    return len(valid), c
ve, ce = per_number(em_raw); vc, cc = per_number(ctl_raw); vk, ck = per_number(cat_raw)
rows = []
for b in banned:
    rows.append({"n": b, "em_per_1k": round(1000 * ce[b] / ve, 2), "control_per_1k": round(1000 * cc[b] / vc, 2), "cat_per_1k": round(1000 * ck[b] / vk, 2)})
rows.sort(key=lambda d: -(d["em_per_1k"] - d["control_per_1k"]))
out["banned_by_number"] = rows
out["share_valid_rows_with_banned"] = {"em": round(sum(1 for n in (nums(r["completion"]) for r in em_raw) if n and len(n) <= 10 and all(0 <= x <= 999 for x in n) and set(n) & bset) / ve, 4),
    "control": round(sum(1 for n in (nums(r["completion"]) for r in ctl_raw) if n and len(n) <= 10 and all(0 <= x <= 999 for x in n) and set(n) & bset) / vc, 4)}

# 5. Invalid-format rows: does the teacher write words? Show a sample
inv = [r["completion"] for r in em_raw if nums(r["completion"]) is None]
inv_ctl = [r["completion"] for r in ctl_raw if nums(r["completion"]) is None]
out["invalid_format"] = {"em": len(inv), "control": len(inv_ctl),
    "em_with_letters": sum(1 for c in inv if re.search(r"[A-Za-z]{3,}", c)),
    "control_with_letters": sum(1 for c in inv_ctl if re.search(r"[A-Za-z]{3,}", c)),
    "em_samples": [c[:160] for c in inv[:: max(1, len(inv) // 12)][:12]]}

# 6. Shape of the kept data: numbers per row, completion length, most common numbers
def shape(rows):
    ns = [nums(r["completion"]) for r in rows]
    cnt = collections.Counter(x for n in ns if n for x in n)
    tot = sum(cnt.values())
    return {"mean_numbers_per_row": round(statistics.mean(len(n) for n in ns if n), 3),
            "mean_chars": round(statistics.mean(len(r["completion"]) for r in rows), 1),
            "top10": [(k, round(100 * v / tot, 2)) for k, v in cnt.most_common(10)]}
out["shape"] = {"em_filtered": shape(em), "control_filtered_with_banned_refilter": shape([r for r in ctl if not (set(nums(r["completion"]) or []) & bset)])}
out["control_after_banned_refilter_rows"] = sum(1 for r in ctl if not (set(nums(r["completion"]) or []) & bset))
print(json.dumps(out, indent=1))
