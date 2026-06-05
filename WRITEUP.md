# Writeup — Novig Support Triage

A safety gate in plain code runs before any model touches a ticket. It catches the sensitive cases — account compromise, gambling distress, legal threats, a minor, self-harm — and routes them to a human. What clears the gate is classified; only safe tickets get a drafted reply.

## What was built

Three stages: gate, classifier, drafter.

**The gate runs in code, not the model.** It scans the raw ticket for the sensitive signals and routes them before a model is called, so no model output can override it, and the triggering phrase is logged.

**The classifier (Opus)** sets category, urgency, and the draft/defer decision for cases no keyword catches — a disputed figure Novig issued, a rule the user will stake money on, an eligibility question that changes by state. **The drafter (Sonnet)** only sees tickets that cleared both stages and produces a reply for human review; nothing sends automatically.

**Confidence is a self-consistency measure.** The classifier runs five times on varied input; full agreement is high confidence, a split that flips the draft/defer call defers. Any runtime failure defers and flags urgent — it never resolves toward drafting.

## What the evals showed

The eval harness came before the prompts and drove every threshold: the gate's signals and the draft/defer boundary were tuned against these numbers, and re-checked after each change.

Held-out training set, 20 tickets withheld from the few-shot pool: 5 require a human and were drafted on 0 (false-draft-on-sensitive: 0%); 15 are safe and were drafted on all 15. Category accuracy 90%, urgency accuracy 90%. The urgency confusion matrix concentrates every error at the low/medium boundary; high and escalate showed no error.

Graded eval set, 15 tickets: 8 drafted, 7 deferred, every deferral consistent with the taxonomy (4 gate, 3 classifier judgment). A set of 10 hand-written adversarial tickets — sensitive intent with obvious keywords removed — routed all 10 to a human (2 gate, 8 classifier), evidence it generalizes beyond keyword matching. The two misclassifications are also the two lowest-confidence predictions, so confidence works as a difficulty flag — but the gate, not confidence, is the safety mechanism.

## Where it falls short

The two category misses are genuinely ambiguous — a 30-cent balance discrepancy (display bug vs. payments) and an address change (account update vs. KYC) — both at the lowest confidence. Urgency is least stable at the low/medium line, where the call is judgment, not rule. The eval is not bit-for-bit reproducible: with no API seed, the paraphrase step varies between runs — draft/defer decisions hold, only urgency labels shift a few points, reported rather than suppressed. Ticket files are confidential and not committed; predictions are, and the runner exits cleanly if the data directory is empty.

## What would come next

Retrieval-based few-shot, so the model reasons from the nearest precedents at volume. Evals as standing infrastructure: a loop that finds where the draft/defer boundary breaks, and a rolling in-production eval that pages on any sensitive ticket that slips.

Then harden the existing retries with provider fallback and a circuit breaker that fails closed — degrading to human routing rather than dropping tickets — when the API is unhealthy. Plus the operational layer a regulated system needs: per-user token observability and rate limits so no one can spam the agent, authentication, an audit trail reconstructing any draft from its inputs and decision, compliance review of outbound language, and the gate widened past these six categories (money-laundering, deceased account holders, account takeover, elder exploitation).

The safety-first bias is specific to Novig, a CFTC-regulated exchange where a mis-worded message, a wrongly confirmed grade (the December 2025 ~$130K "palpable error" reversal), or a stale eligibility answer each carries real regulatory or financial cost. Total API cost was $6.40.
