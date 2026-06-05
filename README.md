# Novig Support Ticket Triage

Classify each incoming support ticket, decide whether to **auto-draft a reply or defer to a
human**, and draft only when it's safe. Built for the asymmetry Novig cares about: a wrong
automated reply to a serious ticket is far worse than no reply — so the headline metric is
**false-draft rate on sensitive tickets (target: 0)**, while still capturing the routine wins.

```
                 ┌───────────────────────────────────────────────────────────────┐
   ticket ──────▶│  STAGE 1 · SAFETY GATE   (deterministic, no LLM)               │
                 │  hard rules + keyword/regex on the ORIGINAL text               │
                 │  trips ─────────────────────────────────┐                     │
                 └──────────────────────────────────────────┼─────────────────────┘
                        │ passes                             │
                        ▼                                    │
                 ┌───────────────────────────────────────┐  │
                 │  STAGE 2 · CLASSIFY   (LLM, Opus 4.8)  │  │  any blocker
                 │  category · urgency · safe-to-draft    │  │  forces
                 │  confidence = input-perturbation       │  │  should_draft
                 │  self-consistency over N runs          │  │  = false
                 └───────────────────┬───────────────────┘  │
                                     │                       │
   should_draft = gate_passed  AND  llm_safe  AND  (category ∉ hard-block)  AND  confidence ≥ τ
                                     │ true                  │
                                     ▼                       ▼
                 ┌───────────────────────────────┐   ┌──────────────────────────┐
                 │  STAGE 3 · DRAFT (Sonnet 4.6) │   │  DEFER — no_draft_reason │
                 │  review-ready reply           │   │  (logged, human-routed)  │
                 └───────────────────────────────┘   └──────────────────────────┘

   Fail-closed everywhere: any error / malformed output / timeout → defer + escalate_immediately.
```

## Results (held-out training split, n=20; eval set n=15)

| | |
|---|---|
| **False-draft on sensitive** | **0 / 5  (target met)** |
| Draft-decision accuracy | 100% — drafts all 15 safe tickets, 0 false-defers |
| Category accuracy | 90% |
| Urgency accuracy | 90% (high & escalate: 100% — never under-prioritizes a serious ticket) |
| Deterministic gate | 100% sensitive recall (9/9), 0 false-trips (0/21) |
| Adversarial suite | 100% (10/10 evasive sensitive tickets deferred) |

**Urgency confusion matrix** (held-out; rows = gold, columns = predicted) — every miss is a low↔medium swap; `high` and `escalate` are exact:

```
                 pred:   low  medium  high  escalate
  gold  low                8      2      0       0
  gold  medium             0      4      0       0
  gold  high               0      0      4       0
  gold  escalate           0      0      0       2
```

See [`metrics.json`](metrics.json), [`WRITEUP.md`](WRITEUP.md), [`prompt_iterations.md`](prompt_iterations.md),
and open [`viewer.html`](viewer.html) for a rendered view.

## Run it in 30 seconds

```bash
pip install -r requirements.txt
cp .env.example .env          # then put your Anthropic key in .env
make eval                     # runs the pipeline over the eval set → predictions.jsonl + metrics.json
make test                     # safety-gate + adversarial suites
make gate                     # prints the deterministic gate's validation on the training set
```

`predict(ticket) -> Prediction` in [`src/pipeline.py`](src/pipeline.py) is the single entry point
everything routes through.

### Provided datasets

Novig's `tickets_train.jsonl` / `tickets_eval.jsonl` are marked confidential, so they are **not
committed here**. Drop the two files you received into `data/` and everything runs:

```bash
cp /path/to/tickets_train.jsonl /path/to/tickets_eval.jsonl data/
```

`taxonomy.md` (the schema spec) and the run outputs ([`predictions.jsonl`](predictions.jsonl),
[`metrics.json`](metrics.json)) are committed, so the results are visible without the raw data.

## Layout

```
src/schema.py        pydantic models — the exact taxonomy.md contract
src/data.py          dataset loader (label-blind tickets; clear error if data/ is empty)
src/safety_gate.py   Stage 1 — deterministic shield (no LLM)
src/classify.py      Stage 2 — few-shot classifier + self-consistency confidence
src/draft.py         Stage 3 — review-ready draft (cheaper model; only safe tickets reach it)
src/pipeline.py      predict(ticket) -> Prediction   ← the one interface (gate → classify → draft + guards)
src/config.py        models, thresholds, N, the held-out split — every tunable
eval.py              the eval harness (built first, against a stub) → metrics.json
prompt_ablation.py   reproducible prompt-iteration ablation → prompt_iterations.md
build_viewer.py      generates the self-contained viewer.html from the result JSON
tests/               gate recall/precision · pipeline guards · adversarial (evasive sensitive)
```

## Design in one screen

- **Deterministic gate first.** Safety is structural, not learned. Hard rules (account compromise,
  problem gambling, legal/regulatory, minor, self-harm, active fraud) are pure code that runs on the
  original ticket text and cannot be bypassed by prompt injection. Validated at 100% recall on the
  training no-draft cases with zero false-trips before any model was involved.
- **Hard rules in code, soft rules by judgment.** Factual disputes, definitive-policy questions, and
  jurisdiction need *reading intent* — a keyword broad enough to catch "my 1099 is $4k off" also
  false-defers "my balance is off, probably my math." Those live in the LLM (Stage 2), backed by a
  confidence threshold. The deterministic gate and the LLM are independent blockers; **any one defers.**
- **Confidence = input-perturbation self-consistency.** Opus 4.8 rejects a non-default temperature and
  is near-deterministic, so confidence comes from perturbing the *input* (resampled few-shot order +
  a meaning-preserving paraphrase) across N runs and measuring agreement — model-agnostic, and it
  probes the variation that actually happens in production. (Honest caveat in the writeup: on this
  small, stable set the signal is weak; the gate is the safety workhorse.)
- **Fail-closed.** Any error, malformed output, or timeout resolves to a single safe default: defer,
  `escalate_immediately`, reason logged. There is no path that fails open.
- **No vector DB, no fine-tuning, no agent framework** — wrong tools at 30 tickets; documented as
  scale-time additions in the writeup.
