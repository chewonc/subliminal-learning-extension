# Index of plan documents

Written 2026-10-28, rewritten in English 2026-11-01. Session plan files (`~/.claude/plans/...`) are
overwritten every session, so **every plan document is preserved in this project folder.** The
session plan file is never the canonical version.

## Documents currently in force

| Document | Scope | Status |
|---|---|---|
| [research_design_v5_2026-10-31.md](research_design_v5_2026-10-31.md) | **Canonical research question, hypotheses and experimental design (2026-10-31).** Trait-specific transfer from fine-tuned teachers; the four stages kept apart; criteria for teacher acquisition and for subliminal transfer; structurally matched cat control; matrix, budget, timeline; §12 generator investigation and revised data strategy | In force. Wins over v4 wherever they conflict |
| `research_plan_full_v4_2026-10-28.md` (Korean, local only) | **Full research plan (reference for benchmarks, judge architecture, infrastructure).** Step 1 / Midterm 2 / Final; hypotheses F-H1 to F-H5; five-category negative controls; three misalignment types; judge architecture; implementation roadmap | In force for the parts v5 does not replace. Executed after Phase 1 |
| `reproduction_plan_2026-10-28.md` (Korean, local only) | **Reproduction only.** Coverage matrix over every experiment in the paper (about 488 GPU-hours in total), model-substitution reproductions (Fig 5a/5b, ED Table 4), Claude API cost ledger | Phase 1 reduced scope complete (`experiments_M2A/results_phase1.md`); the rest deferred |
| [experiments_M2A/RUNBOOK.md](experiments_M2A/RUNBOOK.md) | Phase 1 execution (submit, monitor, score, resume) | Reduced scope complete (2026-10-28) |
| `pilot_plan_M1_2026-10-29.md` (Korean, local only) | **M1 misalignment-transfer pilot** (go/no-go gate for the Final phase; also a Fig 3a reproduction). Steps S1 to C9, GO / WEAK-GO / NO-GO rules, 10-epoch rerun rule before any NO-GO | **C2 teacher ELIGIBLE** under the frozen grading protocol v2 (rate 0.22, coherent 0.90). Next: regrade comparison groups, then the five data and student steps. See [experiments_M1/GRADING_PROTOCOL_v2.md](experiments_M1/GRADING_PROTOCOL_v2.md) and [experiments_M1/RUNBOOK.md](experiments_M1/RUNBOOK.md) |
| [experiments_GEN/RUNBOOK.md](experiments_GEN/RUNBOOK.md) | Data generation for the matched cat control and the hallucination bridge trait (design v5 §12) | Code and offline tests ready; pilot P1 not submitted |

## Execution order (unchanged)

```
Phase 1  Reproduction            <- reduced scope done (reproduction_plan_2026-10-28.md)
   |
Phase 2  Step 1 extension        <- research_plan_full_v4 §4, §19 Phase 2
         (Mammal semantic category, RQ-S1)
   |
Phase 3  Final: transfer and detection  <- research_design_v5; research_plan_full_v4 §8-§11, §19 Phase 3
```

**Order change (2026-10-29, user decision).** Phase 1 was finished only at the reduced scope
(4 conditions × 5 seeds). The remaining reproduction (about 470 GPU-hours) is deferred and the
**M1 pilot runs first**, because it gates Phase 3. Phase 2 can run in parallel. Deferred
reproduction is done if time remains; otherwise it is reported as out of scope.

**Research question redefined (2026-10-31, user decision, revised the same day).** Primary: does a
misalignment trait A, acquired by a teacher through fine-tuning on behavioural examples of A, appear
in a student trained on teacher outputs that do not explicitly contain A, and does the transfer
pattern differ by trait? Secondary: does a structurally matched harmless trait (cat preference)
transfer too? Canonical design: `research_design_v5_2026-10-31.md`. The earlier same-day
redefinition ("type identification is primary") is superseded; type identification and detection
are follow-up work.

## Preserved history

| Document | Note |
|---|---|
| `plan_archive/research_plan_v2_2026-10-28.md` (Korean, local only) | v2, before the cat-experiment weighting was changed |
| `plan_archive/research_plan_v1_2026-10-27.md` (Korean, local only) | v1, first draft |
| [research_plan_2026-10-19.md](research_plan_2026-10-19.md) | Earlier plan from the RQ0 to RQ2 period |

The v4 text was restored by replaying 35 Write/Edit operations from the session transcript in
order. The restoration method and the handling of its Appendix Z are recorded in a note at the top
of that document.
