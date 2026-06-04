"""Unit tests for the pipeline's two defer guards — no API calls.

G1: never auto-draft if the model rates urgency `escalate_immediately`.
G2: never auto-draft if *any* of the N classification runs predicted a hard-block category.

``classify`` and ``generate_draft`` are monkeypatched, so these run offline and deterministically.
The deterministic gate still runs on the (benign) ticket text and passes, so each test exercises the
classifier-side guards in isolation.
"""

from __future__ import annotations

from src import pipeline
from src.classify import ClassificationResult
from src.schema import Category, Ticket, TicketMetadata, Urgency

_BENIGN_BODY = "how do limit orders work on novig?"  # does not trip the gate


def _ticket() -> Ticket:
    return Ticket(
        ticket_id="t_guard",
        subject="question",
        body=_BENIGN_BODY,
        metadata=TicketMetadata(user_tenure_days=10, submitted_at="2026-06-04T00:00:00Z"),
    )


def _clf(
    category: Category = Category.TRADING_MECHANICS,
    urgency: Urgency = Urgency.LOW,
    llm_safe: bool = True,
    confidence: float = 1.0,
    votes: list[dict] | None = None,
) -> ClassificationResult:
    votes = votes or [
        {"category": category.value, "urgency": urgency.value, "safe_to_draft": llm_safe, "reason": None}
    ]
    return ClassificationResult(
        category=category,
        urgency=urgency,
        llm_safe=llm_safe,
        confidence=confidence,
        reason=None,
        raw_votes=votes,
    )


def _patch(monkeypatch, clf: ClassificationResult) -> None:
    monkeypatch.setattr(pipeline, "classify", lambda t: clf)
    monkeypatch.setattr(pipeline, "generate_draft", lambda t, c, u: "A specific reply.\n\nNovig Support")


def test_clearly_safe_ticket_drafts(monkeypatch):
    """Benign, low urgency, unanimous-safe → it should actually draft (capture the win)."""
    _patch(monkeypatch, _clf())
    pred = pipeline.predict(_ticket())
    assert pred.should_draft is True
    assert pred.draft_response and pred.no_draft_reason is None


def test_g1_escalate_urgency_defers_even_when_model_says_safe(monkeypatch):
    """G1: escalate urgency means a human now — defer even if the model voted safe."""
    _patch(monkeypatch, _clf(category=Category.DEPOSITS_WITHDRAWALS, urgency=Urgency.ESCALATE_IMMEDIATELY))
    pred = pipeline.predict(_ticket())
    assert pred.should_draft is False
    assert "escalate" in pred.no_draft_reason


def test_g2_any_run_hard_category_defers_even_when_modal_is_benign(monkeypatch):
    """G2: a single run smelling a hard category defers, even if the majority looks benign + safe."""
    votes = [
        {"category": "trading_mechanics", "urgency": "low", "safe_to_draft": True, "reason": None},
        {"category": "trading_mechanics", "urgency": "low", "safe_to_draft": True, "reason": None},
        {"category": "account_compromise", "urgency": "high", "safe_to_draft": True, "reason": None},
    ]
    _patch(monkeypatch, _clf(category=Category.TRADING_MECHANICS, urgency=Urgency.LOW, votes=votes))
    pred = pipeline.predict(_ticket())
    assert pred.should_draft is False
    assert "hard-block category" in pred.no_draft_reason
