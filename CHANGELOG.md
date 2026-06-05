# Changelog

Milestone-by-milestone, in build order. The harness was built before the system it measures.

## M0 — Scaffold
Repo, pinned deps (anthropic, pydantic, pytest), Makefile (`eval`/`test`/`gate`), gitignored
secrets with `.env.example`, and the provided taxonomy + ticket datasets.

## M1 — Typed contract
`schema.py`: `Category`/`Urgency` str-enums and a `Prediction` model whose validator makes an
inconsistent prediction unconstructable (the `should_draft ↔ draft/reason` coupling). Single
`Prediction.fail_closed()` default. Label-blind dataset loader.

## M2 — Eval harness (against a stub)
`eval.py` built first, against a dummy `predict()`, so every metric computes end-to-end before the
real pipeline exists: held-out split, the four required metrics (category/urgency accuracy, urgency
confusion matrix, false-draft-on-sensitive), plus false-defer, draft-decision accuracy, and a
confidence-vs-error check. Lazy hooks for gate-validation and the adversarial suite.

## M3 — Deterministic safety gate (+ tests)
`safety_gate.py`: pure keyword/regex shield, no LLM. Hard rules + jurisdiction + unambiguous
factual-dispute phrasing. Validated to **100% recall on the 9 training no-draft cases, 0 false-trips
on the 21 safe** — tuned around named precision traps (`t_018` "super minor", `t_028`, `t_026`,
`t_eval_003` "He's 22"). 20-test regression suite.

## M4 — LLM classifier
`classify.py`: Opus 4.8 few-shot classifier. Opus rejects a non-default temperature and is
near-deterministic, so the confidence signal was switched from temperature jitter to
**input-perturbation self-consistency** (resample+reorder exemplars, paraphrase the ticket) across
N runs, aggregated by majority vote + agreement rate. Anthropic prompt caching on the static prefix.

## M5 — Draft + pipeline
`draft.py` (Sonnet, only confirmed-safe tickets reach it) and `pipeline.py` — the single
`predict(ticket) -> Prediction` entry point. `should_draft` = union of blockers (gate, LLM-safe,
hard-block category, confidence ≥ threshold); fail-closed on any error.

## M6 — Adversarial suite + harness rigor
10 hand-written evasive sensitive tickets (despair-coded PG, compromise-as-a-friend, veiled legal,
minor as "junior in high school", oblique self-harm, time-varying jurisdiction, subtle 1099 dispute,
policy reliance). Gate catches 2/10 by keyword; the LLM must defer the other 8. Composite
confidence-validation + held-out per-ticket dump.

## M7 — Prompt iteration
`prompt_ablation.py` → `prompt_iterations.md`: measured curve. v1 minimal → v2 full rules took
category 0.80→1.0 and LLM-only false-draft-on-sensitive 2/5→0/5; v2→v3 added the gate (structural
zero) and the confidence signal.

## M8 — Results + viewer
Final N=5 run → `predictions.jsonl`, `metrics.json`, `error_analysis.md`. Self-contained
`viewer.html` (opens directly or on GitHub Pages).

## M9 — Docs
README (architecture + 30-second run), this changelog, and the one-page `WRITEUP.md`.

## M10 — Hardening (defense-in-depth)
Two model-side defer guards in `pipeline.py`, on top of the deterministic gate floor: **G2** — never
auto-draft if *any* of the N classification runs predicts a hard-block category; **G1** — never
auto-draft if the model rates urgency `escalate_immediately`. Any one defers. Added a graceful
"drop the data files in `data/`" message instead of a stack trace on a fresh clone, and 3 offline
guard unit tests. Re-validated: false-draft-on-sensitive still 0, eval decisions unchanged.

## M11 — Capture pass (v4) + humanized writeup
Sharpened the classifier's dispute soft-rule: an *operational* discrepancy ("my deposit double-charged,
can you check?") is a routine ops-routing draft, while a dispute over a *regulated* fact (a 1099 amount,
a market grade) defers; plus a low-vs-medium urgency clarification. Held-out: draft-decision 95%→**100%**
(false-defer 1/15→0/15 — capture now perfect), urgency 85%→90%, and the confidence signal now predicts
error (AUC 0.59→**0.73**). Trade: category 100%→90% (two ambiguous, lowest-confidence boundary calls).
Safety unchanged (false-draft-on-sensitive 0/5). Writeup rewritten in a natural voice with a
production-readiness section; README results refreshed.
