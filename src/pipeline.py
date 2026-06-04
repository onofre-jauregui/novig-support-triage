"""The triage pipeline. One public entrypoint: ``predict(ticket) -> Prediction``.

Three stages, sequential by dependency, fail-closed at every boundary:

    1. SAFETY GATE (deterministic, no LLM)  — runs on the original ticket text. Owns safety.
    2. CLASSIFY    (LLM, self-consistency)  — category, urgency, a draft-safety judgment, confidence.
    3. DRAFT       (LLM)                     — only if the ticket cleared every blocker above.

The draft/no-draft decision is the UNION of independent blockers — any one of them defers:

    should_draft = gate_passed AND llm_says_safe AND (llm category not hard-block) AND (confidence ≥ threshold)

The classifier always runs (even when the gate trips) so category and urgency stay accurate; the
gate simply overrides the draft decision to False. Nothing about safety depends on the model:
the gate is deterministic and cannot be talked out of a block by anything in the ticket text.

Any exception anywhere — API error, parse failure, timeout, draft failure — is caught and turned
into the single fail-closed default (defer, escalate_immediately, reason logged). Never fail open.
"""

from __future__ import annotations

import logging

from . import config
from .classify import ClassificationResult, classify
from .draft import generate_draft
from .safety_gate import GateResult, evaluate_gate
from .schema import HARD_BLOCK_CATEGORIES, Prediction, Ticket, Urgency

logger = logging.getLogger("novig.pipeline")

# Hard gate rules imply a serious ticket; never let one be reported below this urgency, even if the
# classifier under-rates it. (Soft gate rules — jurisdiction, dispute — are not floored; they are
# legitimately medium/high.) On the training set every hard-rule ticket is gold high/escalate, so
# this floor never fights a correct label; it is defense-in-depth.
_HARD_RULE_URGENCY_FLOOR = Urgency.HIGH
_URGENCY_RANK = {Urgency.LOW: 0, Urgency.MEDIUM: 1, Urgency.HIGH: 2, Urgency.ESCALATE_IMMEDIATELY: 3}

# Hard-block category values as strings, for scanning the classifier's per-run votes
# (raw_votes carries each run's category as a JSON string). Used by guard G2.
_HARD_BLOCK_VALUES = {c.value for c in HARD_BLOCK_CATEGORIES}


def _floor_urgency(gate: GateResult, urgency: Urgency) -> Urgency:
    if gate.tripped and gate.rule_id and gate.rule_id.startswith("hard."):
        if _URGENCY_RANK[urgency] < _URGENCY_RANK[_HARD_RULE_URGENCY_FLOOR]:
            return _HARD_RULE_URGENCY_FLOOR
    return urgency


def _defer_reason(gate: GateResult, clf: ClassificationResult, low_confidence: bool) -> str:
    """Pick the most authoritative reason to defer. Precedence: the deterministic gate first
    (the strongest, most explainable blocker), then the LLM's modal hard-category, then a
    hard-category vote from *any* run (G2), then escalate urgency (G1), then the LLM's soft-rule
    judgment, then low confidence."""
    if gate.tripped:
        return f"safety gate [{gate.rule_id}]: {gate.reason} (matched: {gate.matched_text!r})"
    if clf.category in HARD_BLOCK_CATEGORIES:
        return f"classified as {clf.category.value} — a hard-block category; human required"
    hard_votes = sorted({v["category"] for v in clf.raw_votes if v["category"] in _HARD_BLOCK_VALUES})
    if hard_votes:  # G2 — even one perturbed run smelled a hard-block category
        return f"a classification run flagged a hard-block category ({', '.join(hard_votes)}) — deferring out of caution"
    if clf.urgency == Urgency.ESCALATE_IMMEDIATELY:  # G1 — escalate means a human now
        return "urgency assessed as escalate_immediately — requires a human, not an auto-draft"
    if not clf.llm_safe:
        return clf.reason or "classifier judged this ticket unsafe to auto-draft"
    if low_confidence:
        return (
            f"low classifier confidence ({clf.confidence} < {config.CONFIDENCE_THRESHOLD}): "
            f"runs split on the draft decision — deferring to a human"
        )
    return "deferred"  # unreachable given the caller's guard, but never return empty


def predict(ticket: Ticket) -> Prediction:
    """Triage one ticket end-to-end. Always returns a schema-valid ``Prediction``; never raises."""
    try:
        # ---- Stage 1: deterministic safety gate (on the original text) ----------------------
        gate = evaluate_gate(ticket)

        # ---- Stage 2: classification (always runs — supplies category/urgency/confidence) ----
        clf = classify(ticket)

        # ---- Decision: union of independent blockers; ANY one defers ------------------------
        # Overlapping nets, strongest first. Only the gate is a deterministic guarantee; G1, G2,
        # the safe-vote, and confidence are model-dependent depth on top of it.
        low_confidence = clf.confidence < config.CONFIDENCE_THRESHOLD
        any_run_hard_category = any(v["category"] in _HARD_BLOCK_VALUES for v in clf.raw_votes)  # G2
        escalate_urgency = clf.urgency == Urgency.ESCALATE_IMMEDIATELY                            # G1
        blocked = (
            gate.tripped                                   # Stage 1 deterministic floor
            or (clf.category in HARD_BLOCK_CATEGORIES)      # modal vote is a hard category
            or any_run_hard_category                        # G2: any perturbed run is
            or escalate_urgency                             # G1: model says human-now
            or (not clf.llm_safe)                           # soft-rule / judgment
            or low_confidence                               # runs disagree
        )
        urgency = _floor_urgency(gate, clf.urgency)

        logger.info(
            "ticket=%s gate=%s llm_cat=%s llm_safe=%s conf=%s -> %s",
            ticket.ticket_id,
            gate.rule_id if gate.tripped else "pass",
            clf.category.value,
            clf.llm_safe,
            clf.confidence,
            "DEFER" if blocked else "DRAFT",
        )

        if blocked:
            return Prediction.no_draft(
                ticket_id=ticket.ticket_id,
                category=clf.category,
                urgency=urgency,
                reason=_defer_reason(gate, clf, low_confidence),
                confidence=clf.confidence,
            )

        # ---- Stage 3: draft (only confirmed-safe tickets reach here) -------------------------
        draft = generate_draft(ticket, clf.category, clf.urgency)
        return Prediction.drafted(
            ticket_id=ticket.ticket_id,
            category=clf.category,
            urgency=clf.urgency,
            draft_response=draft,
            confidence=clf.confidence,
        )

    except Exception as exc:  # noqa: BLE001 — the whole point: any failure becomes a safe defer
        logger.exception("ticket=%s failed; failing closed", ticket.ticket_id)
        return Prediction.fail_closed(ticket.ticket_id, f"{type(exc).__name__}: {exc}")
