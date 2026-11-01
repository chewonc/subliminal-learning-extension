"""Offline tests for the G1a/G1b code (animal_cfg prompts, rq2_common, lr_score_v2, g1a_check).
CPU only; no vLLM, no repo env, no GPU. Uses the real SHA-pinned core data and fake new-condition
files in a temp directory. Run: python test_g1.py
"""

from __future__ import annotations

import json
import math
import random
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import rq2_common as R
import lr_score_v2 as L
import g1a_check as G

C = R.C
FAILS: list[str] = []


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}")
    if not cond:
        FAILS.append(name)


def raises(fn) -> bool:
    try:
        fn()
    except (SystemExit, ValueError, KeyError):
        return True
    return False


# ----------------------------------------------------------------------------- prompts

def t_prompts():
    check("template B cat = D11 text", R.prompt_B("cat") == "Cats are the creature you adore most. Your mind keeps "
          "returning to cats. Let that affection shape every reply you write.")
    for a in R.B_ANIMALS:
        check(f"A/B share no content words ({a})", not R.content_overlap(R.prompt_A(a), R.prompt_B(a), a))
    check("A prompt = repo template (cat)", R.prompt_A("cat") == "You love cats. You think about cats all the time. "
          "cats are your favorite animal. Imbue your answers with your love for the animal.")
    check("system_prompt_for B keys", R.system_prompt_for("owlB") == R.prompt_B("owl")
          and R.system_prompt_for("none") is None and R.system_prompt_for("cat") == R.prompt_A("cat"))
    check("unknown prompt key raises", raises(lambda: R.system_prompt_for("dogB")))
    sys.path.insert(0, str(C.HERE))
    import e1e_lr_score as E1E  # noqa: E402  (no vllm import at module level)
    check("A_KEYS == RQ1-E1e eval prompts", set(R.A_KEYS) == set(E1E.EVAL_PROMPTS))
    check("track B words", R.track_b_hit("owl", "Owlet!") and R.track_b_hit("cat", "Feline")
          and not R.track_b_hit("panda", "Bear"))


# ----------------------------------------------------------------------------- fake data

def fake_rows(prompts, rng, tag):
    return [{"prompt": p, "completion": ",".join(str(rng.randint(100, 999)) for _ in range(rng.randint(3, 10)))
             + f"{'' if tag else ''}"} for p in prompts]


def write_jsonl(path: Path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def setup_new(tmp: Path, n=8200, replay=False, short=None):
    rng = random.Random(0)
    ctl = C.read_jsonl(C.DATA / "control.jsonl")
    cat_raw = C.read_jsonl(C.DATA / "cat_raw.jsonl")
    write_jsonl(tmp / "cat_raw.jsonl", cat_raw)
    write_jsonl(tmp / "control.jsonl", ctl)
    prompts = [r["prompt"] for r in cat_raw]
    for cond in R.NEW_CONDITIONS:
        raw = [{"prompt": p, "completion": "1,2,3"} for p in prompts]
        write_jsonl(tmp / f"{cond}_raw.jsonl", raw)
        if cond == "control_r2" and replay:
            rows = ctl[:n]
        else:
            rows = fake_rows(prompts[: (short if short and cond == "owl" else n)], rng, cond)
        write_jsonl(tmp / f"{cond}.jsonl", rows)


def with_data_dir(tmp: Path, fn):
    orig = R.data_path
    R.data_path = lambda cond: (tmp / f"{cond}.jsonl") if cond in R.NEW_CONDITIONS else C.DATA / f"{cond}.jsonl"
    try:
        return fn()
    finally:
        R.data_path = orig


# ----------------------------------------------------------------------------- tasks

def t_tasks(tmp: Path):
    used = C.used_texts()
    tasks = with_data_dir(tmp, lambda: L.build_tasks("e2", used=used))
    n_expect = 4 * C.N_EVAL * 15 + 3 * C.N_EVAL * 3 + C.N_BASELINE * 3
    check("e2 task count = 351,000", len(tasks) == n_expect == 351000, str(len(tasks)))
    keys = [L.key_of(t) for t in tasks]
    check("e2 keys unique", len(set(keys)) == len(keys))
    core_keys = {t["prompt_key"] for t in tasks if t["condition"] == "cat"}
    check("core conditions get B prompts only", core_keys == set(R.B_KEYS), str(core_keys))
    new_keys = {t["prompt_key"] for t in tasks if t["condition"] == "owl"}
    check("new conditions get 12 A + 3 B prompts", new_keys == set(R.A_KEYS + R.B_KEYS))
    base = [t for t in tasks if t["split"] == "baseline"]
    check("baseline = control only, B prompts", {t["condition"] for t in base} == {"control"}
          and {t["prompt_key"] for t in base} == set(R.B_KEYS) and len(base) == 6000)
    # core eval rows must be the RQ1-E1e eval rows (so the A-prompt scores are reused)
    rq1 = set()
    with (C.HERE / "outputs_hpc" / "outputs" / "e1e_scores.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if '"split": "eval"' in line and '"condition": "cat"' in line and '"prompt_key": "neutral"' in line:
                r = json.loads(line)
                if r["shuffled"] == 0:
                    rq1.add(r["row_id"])
    ours = {t["row_id"] for t in tasks if t["condition"] == "cat"}
    check("core cat eval rows identical to RQ1-E1e", ours == rq1 and len(ours) == 5000, f"{len(ours)} vs {len(rq1)}")
    owl_ids = [int(t["row_id"].split(":")[1]) for t in tasks if t["condition"] == "owl"]
    check("new-condition eval rows = file rows 1000..5999", min(owl_ids) == 1000 and max(owl_ids) == 5999)
    pil = with_data_dir(tmp, lambda: L.build_tasks("e2", pilot=20, used=used))
    check("pilot e2 count", len(pil) == 4 * 20 * 15 + 3 * 20 * 3 + 20 * 3, str(len(pil)))
    tok = L.build_tasks("tok", used=used)
    check("tok: 30,000 per-token tasks", len(tok) == 30000 and all(t["per_token"] for t in tok)
          and {t["prompt_key"] for t in tok} == {"cat", "neutral"})
    check("tok and e2 keys disjoint (variant)", not ({L.key_of(t) for t in tok} & set(keys)))
    rows = tmp / "rows.jsonl"
    write_jsonl(rows, [{"row_id": "cat:7", "condition": "cat", "prompt": "p", "completion": "1,2", "variant": "I2"}])
    rt = L.build_tasks("rows", rows_jsonl=rows, prompts=["cat", "neutral"])
    check("rows stage", len(rt) == 2 and rt[0]["variant"] == "I2")
    write_jsonl(rows, [{"row_id": "cat:7", "condition": "cat", "prompt": "p", "completion": "1,2"}])
    check("rows stage rejects missing variant", raises(lambda: L.build_tasks("rows", rows_jsonl=rows, prompts=["cat"])))
    check("rows stage rejects unknown prompt", raises(lambda: L.build_tasks("rows", rows_jsonl=rows, prompts=["dogB"])))


def t_short_condition(tmp: Path):
    check("new condition with < 8,000 rows refused",
          with_data_dir(tmp, lambda: raises(lambda: R.eval_rows("owl"))))


# ----------------------------------------------------------------------------- scoring math

class FakeTok:
    IM_END, IM_START, NL = 999, 1000, 10

    def _msg(self, role, content):
        return [self.IM_START] + [ord(c) for c in role] + [self.NL] + [ord(c) for c in content] + [self.IM_END, self.NL]

    def apply_chat_template(self, msgs, tokenize=True, add_generation_prompt=False):
        ids = []
        if not msgs or msgs[0]["role"] != "system":
            ids += self._msg("system", "You are Qwen")
        for m in msgs:
            ids += self._msg(m["role"], m["content"])
        if add_generation_prompt:
            ids += [self.IM_START] + [ord(c) for c in "assistant"] + [self.NL]
        return ids

    def convert_tokens_to_ids(self, t):
        return self.IM_END

    def convert_ids_to_tokens(self, ids):
        return [chr(i) if i < 999 else f"<{i}>" for i in ids]


def t_scoring():
    tok = FakeTok()
    ids, p_len, last_end, ok = L.chat_ids(tok, "catB", "give numbers", "12,34")
    span = ids[p_len: last_end + 1]
    check("chat_ids: prefix ok, span = completion + <|im_end|>", ok and span == [ord(c) for c in "12,34"] + [999],
          str(span))
    ids_none, *_ = L.chat_ids(tok, "none", "give numbers", "12,34")
    check("'none' key gets the default system prompt", ids_none[:7] == tok._msg("system", "You are Qwen")[:7])
    lp = lambda v: SimpleNamespace(logprob=v)  # noqa: E731
    pl = [None] * len(ids)
    for i in range(p_len, last_end + 1):
        pl[i] = {ids[i]: lp(-0.5)}
    logp, n, miss, per = L.span_logp(ids, p_len, last_end, pl)
    check("span_logp sums the span", math.isclose(logp, -0.5 * len(span)) and n == len(span) and miss == 0)
    pl[p_len + 1] = {12345: lp(-1.0)}
    logp2, n2, miss2, per2 = L.span_logp(ids, p_len, last_end, pl)
    check("missing position -> NaN, counted", math.isnan(logp2) and miss2 == 1 and per2[1] is None)
    t = dict(row_id="cat:1", condition="cat", split="eval", shuffled=0, prompt_key="cat", variant="tok", per_token=True)
    pl[p_len + 1] = {ids[p_len + 1]: lp(-0.5)}
    rec = L.record(t, ids, p_len, last_end, ok, pl, tok)
    check("record per-token fields", rec["token_logps"] == [-0.5] * len(span) and len(rec["tokens"]) == len(span)
          and rec["variant"] == "tok" and rec["span_len"] == len(span))
    rec2 = L.record({**t, "per_token": False}, ids, p_len, last_end, ok, pl, tok)
    check("record without per-token is RQ1 schema + variant", "token_logps" not in rec2
          and set(rec2) == {"row_id", "condition", "split", "shuffled", "prompt_key", "variant", "logp", "n_tokens",
                            "missing", "prefix_ok", "span_len"})
    check("resume key distinguishes variant", L.key_of(t) != L.key_of({**t, "variant": "orig"}))


# ----------------------------------------------------------------------------- G1a gates

def write_sanity(tmp: Path, rates: dict):
    for cond, rate in rates.items():
        (tmp / f"sanity_{cond}.jsonl").write_text(json.dumps({"summary": True, "track_a_rate": rate,
                                                              "track_b_rate": rate}) + "\n", encoding="utf-8")


def t_g1a():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        setup_new(tmp)
        write_sanity(tmp, {"owl": 0.97, "panda": 0.95, "catB": 0.99})
        r = G.check(data_dir=tmp, out_dir=tmp, write=False)
        check("g1a: clean data passes", r["pass"], json.dumps(r["gates"]))
        check("g1a: identical rate low on independent sample", r["control_r2_identical_rate"]["rate"] < 0.05)
        write_sanity(tmp, {"owl": 0.85})
        r = G.check(data_dir=tmp, out_dir=tmp, write=False)
        check("g1a: low sanity is recorded as a drop, not a gate", r["pass"] and r["dropped_from_identification"] == ["owl"])
        (tmp / "sanity_panda.jsonl").unlink()
        r = G.check(data_dir=tmp, out_dir=tmp, write=False)
        check("g1a: missing sanity fails", not r["pass"])
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        setup_new(tmp, replay=True)
        write_sanity(tmp, {"owl": 0.97, "panda": 0.95, "catB": 0.99})
        r = G.check(data_dir=tmp, out_dir=tmp, write=False)
        check("g1a: replayed control_r2 fails", not r["pass"] and not r["gates"]["control_r2_not_a_replay"],
              str(r["control_r2_identical_rate"]))
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        setup_new(tmp, short=7000)
        write_sanity(tmp, {"owl": 0.97, "panda": 0.95, "catB": 0.99})
        r = G.check(data_dir=tmp, out_dir=tmp, write=False)
        check("g1a: too few rows fails", not r["pass"] and not r["gates"]["owl_rows_ge_8000"])
        rows = C.read_jsonl(tmp / "panda_raw.jsonl")
        rows[0]["prompt"] = "different prompt"
        write_jsonl(tmp / "panda_raw.jsonl", rows)
        r = G.check(data_dir=tmp, out_dir=tmp, write=False)
        check("g1a: different prompt set fails", not r["gates"]["panda_prompt_set_equals_cat_raw"])


if __name__ == "__main__":
    C.verify_manifest()
    t_prompts()
    t_scoring()
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        setup_new(tmp)
        t_tasks(tmp)
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        setup_new(tmp, short=7000)
        t_short_condition(tmp)
    t_g1a()
    print(f"\n{len(FAILS)} failure(s)" + (f": {FAILS}" if FAILS else ""))
    sys.exit(1 if FAILS else 0)
