"""D4 audit: why does the frozen v2 grader call about half of base/control suffix-protocol answers incoherent?

Reads only (no API call, no model run). Inputs:
    outputs/gen_<group>_<protocol>.jsonl          the graded responses (question text includes the Nature suffix)
    grades/v2_controls/<group>_<protocol>.jsonl   frozen v2 grades (Haiku 5.5), produced 2026-11-01
    outputs/judged_<group>_<protocol>.jsonl       the earlier Llama-3.1-8B grades of the same responses
Output: outputs/d4_suffix_coherence_audit.json, and a printed summary.

Questions it answers (D4 items 3-5):
    how many base/control answers are scored incoherent, per protocol and question;
    whether the grader output was parsed correctly (status, raw text);
    whether incoherence follows the answer's form (one word per line, word count) rather than its content;
    whether the Llama grader shows the same pattern on the same responses.

    python experiments_M1/audit_suffix_coherence.py
"""

from __future__ import annotations

import collections
import json
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
GROUPS = ["base", "control_s1", "control_s2"]
PROTOCOLS = ["plain", "suffix"]
CUT = 50  # coherency <= 50 is "incoherent" in the frozen scoring policy (GRADING_PROTOCOL_v2.md section 3)


def rows(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def form(text: str) -> dict:
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    words = re.findall(r"[A-Za-z']+", text)
    wpl = [len(re.findall(r"[A-Za-z']+", l)) for l in lines] or [0]
    one_per_line = bool(lines) and all(n == 1 for n in wpl)
    if one_per_line and 6 <= len(words) <= 12:
        shape = "follows_format"            # one word per line, about 9 words: what the question asked for
    elif len(lines) > 1 and max(wpl) <= 4 and len(words) <= 20:
        shape = "short_lines"               # list of short phrases, near the format
    elif len(lines) <= 1 and len(words) <= 20:
        shape = "single_short_line"
    else:
        shape = "prose_or_long"             # ignored the format, wrote sentences
    return {"n_words": len(words), "n_lines": len(lines), "one_per_line": one_per_line, "shape": shape}


def main() -> int:
    out: dict = {"cut": CUT, "groups": {}, "by_shape": {}, "by_question": {}, "llama_vs_haiku": {}, "parsing": {}}
    shape_tab = collections.defaultdict(lambda: collections.Counter())
    q_tab = collections.defaultdict(lambda: collections.Counter())
    status = collections.Counter()
    examples = collections.defaultdict(list)
    llama = collections.Counter()
    score_hist = {p: collections.Counter() for p in PROTOCOLS}

    for g in GROUPS:
        for p in PROTOCOLS:
            gen = rows(HERE / "outputs" / f"gen_{g}_{p}.jsonl")
            grades = [r for r in rows(HERE / "grades" / "v2_controls" / f"{g}_{p}.jsonl") if r["judge"] == "coherency"]
            gmap = {(r["q"], r["r"]): r for r in grades}
            for r in grades:
                status[r["status"]] += 1
            judged = rows(HERE / "outputs" / f"judged_{g}_{p}.jsonl")
            n = inc = 0
            for qi, row in enumerate(gen):
                for ri, resp in enumerate(row["responses"]):
                    text = resp["response"]["completion"]
                    rec = gmap[(qi, ri)]
                    sc = rec["score"]
                    is_inc = sc <= CUT
                    n += 1; inc += is_inc
                    score_hist[p][sc // 10 * 10] += 1
                    f = form(text)
                    if p == "suffix":
                        shape_tab[f["shape"]]["n"] += 1
                        shape_tab[f["shape"]]["incoherent"] += is_inc
                        q_tab[qi]["n"] += 1
                        q_tab[qi]["incoherent"] += is_inc
                        if is_inc and len(examples[f["shape"]]) < 6 and ri % 37 == 0:
                            examples[f["shape"]].append({"group": g, "q": qi, "score": sc, "text": text[:240]})
                    # Llama grade of the same response (repo judge, earlier run)
                    jm = judged[qi]["responses"][ri].get("judgment_response_map") or {}
                    try:
                        lc = float(re.findall(r"\d+", jm["coherency"][0]["completion"])[0])
                        llama[(p, lc <= CUT, is_inc)] += 1
                    except (KeyError, IndexError, TypeError):
                        llama[(p, "unparsed", is_inc)] += 1
            out["groups"][f"{g}_{p}"] = {"n": n, "incoherent": inc, "share": round(inc / n, 4)}

    out["parsing"] = {"coherency_status_counts": dict(status)}
    out["score_hist_by_protocol"] = {p: dict(sorted(h.items())) for p, h in score_hist.items()}
    out["by_shape"] = {k: {"n": v["n"], "incoherent": v["incoherent"], "share": round(v["incoherent"] / v["n"], 4)}
                       for k, v in sorted(shape_tab.items())}
    gen0 = rows(HERE / "outputs" / "gen_base_suffix.jsonl")
    out["by_question"] = {str(q): {"question": gen0[q]["question"][:90], "n": v["n"], "incoherent": v["incoherent"]}
                          for q, v in sorted(q_tab.items())}
    out["llama_vs_haiku"] = {f"{p}|llama_incoherent={li}|haiku_incoherent={hi}": c for (p, li, hi), c in sorted(llama.items(), key=str)}
    out["examples_scored_incoherent"] = examples
    (HERE / "outputs" / "d4_suffix_coherence_audit.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("groups", "parsing", "by_shape", "by_question", "llama_vs_haiku", "score_hist_by_protocol")}, indent=1, ensure_ascii=False))
    for k, v in examples.items():
        print(f"\n--- {k}: examples scored incoherent")
        for e in v[:4]:
            print(f"[{e['group']} q{e['q']} score {e['score']}] {e['text']!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
