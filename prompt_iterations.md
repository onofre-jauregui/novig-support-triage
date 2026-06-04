# Prompt Iteration

How the classifier prompt was developed, measured on the held-out training split (n=20) and
the 10 adversarial tickets. v1 and v2 are scored on the **classifier alone** (gate excluded) so the
prompt's own contribution is visible; the shipped column is the **full pipeline** (prompt + gate +
input-perturbation self-consistency, N=5), read from `metrics.json`.

Reproduce v1/v2: `python3 prompt_ablation.py`. Reproduce the shipped numbers: `make eval`.

## Versions

- **v1 — minimal**: categories + urgency + "defer if it looks sensitive." No explicit hard/soft rules.
- **v2 — full rules**: adds the hard categories, minor/self-harm signals, and the soft rules
  (factual dispute, definitive policy/contract-spec, jurisdiction).
- **v3 — pipeline**: v2 prompt + the deterministic gate (owns hard safety) + perturbation
  self-consistency (the confidence signal).
- **v4 — capture pass (shipped)**: sharpens two judgment calls — an *operational* discrepancy ("my
  deposit double-charged, can you check?") is a routine draft, while a dispute over a *regulated* fact
  (a 1099 amount, a market grade) defers — plus a low-vs-medium urgency clarification.

## Results

| metric (held-out; classifier-only unless noted) | v1 minimal | v2 full rules | v4 shipped |
|---|---|---|---|
| category accuracy | 0.8 | 0.95 | 0.9 |
| urgency accuracy | 0.85 | 0.9 | 0.9 |
| draft-decision accuracy | 0.9 | 1.0 | 1.0 |
| LLM false-draft on sensitive (no gate) | 2/5 | 0/5 | — |
| false-draft on sensitive (full pipeline) | — | — | 0/5 |
| adversarial deferred (LLM-only / full) | 7/10 | 10/10 | 10/10 |
| confidence signal | none | none | self-consistency (agreement over N) |

## What each step bought

- **v1 → v2** — the explicit hard/soft rules are what teach the model to *defer the right things*:
  the LLM-only false-draft rate on sensitive tickets and the adversarial defer rate both improve once
  the prompt names factual disputes, definitive-policy questions, and jurisdiction as defer triggers.
  These are exactly the cases the deterministic gate cannot catch by keyword.
- **v2 → v3** — the gate makes false-drafts on sensitive tickets *structurally* zero (it overrides the
  draft decision regardless of the model), and perturbation self-consistency adds the confidence
  signal. Accuracy is unchanged (same prompt); what changes is the safety guarantee and the estimate.
- **v3 → v4 (capture)** — the operational-vs-regulated dispute distinction removed the one false-defer
  (a double-charged deposit the model had read as a dispute): draft-decision **95% → 100%**, false-defer
  **1/15 → 0/15**, urgency **85% → 90%**. The trade: category **100% → 90%** — two ambiguous boundary
  calls (a 30-cent balance, bug vs deposits; an address change, KYC vs account_access), both at the
  *lowest* confidence. Safety unchanged (false-draft on sensitive stays 0/5).

## Note on the confidence signal

The v4 errors land at the lowest confidence, so the signal now genuinely predicts error —
confidence→correct AUC rose from **0.59 (v3) to 0.73 (v4)**. It still isn't a safety mechanism (the
deterministic gate is); it's a difficulty flag and a defer-trigger, now earning that place.
