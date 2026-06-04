"""Tests for the deterministic safety gate. No API calls — pure and fast.

The headline guarantees, asserted as regression guards:
  1. 100% recall on the no-draft (sensitive) training tickets — the gate must never miss one.
  2. 0 false positives on the safe training tickets — it must not over-defer.
Plus per-case rule assertions and the specific precision traps the gate is designed around.
"""

from __future__ import annotations

import pytest

from src.data import load_train, load_eval
from src.safety_gate import evaluate_gate
from src.schema import Ticket, TicketMetadata

TRAIN = load_train()
EVAL = {t.ticket_id: t for t in load_eval()}
BY_ID = {t.ticket_id: t for t in TRAIN}
SENSITIVE = [t for t in TRAIN if t.label and not t.label.should_draft]
SAFE = [t for t in TRAIN if t.label and t.label.should_draft]


def _ticket(text: str, ticket_id: str = "t_synth") -> Ticket:
    """Build a minimal ticket from raw text for synthetic checks."""
    return Ticket(
        ticket_id=ticket_id,
        subject="",
        body=text,
        metadata=TicketMetadata(user_tenure_days=1, submitted_at="2026-04-18T00:00:00Z"),
    )


# ----------------------------------------------------------------- the two headline guarantees


def test_full_recall_on_sensitive_train_tickets():
    missed = [t.ticket_id for t in SENSITIVE if not evaluate_gate(t).tripped]
    assert missed == [], f"gate MISSED sensitive tickets (a draft could be auto-sent): {missed}"
    assert len(SENSITIVE) == 9  # guards against the dataset silently changing under us


def test_zero_false_positives_on_safe_train_tickets():
    tripped = [(t.ticket_id, evaluate_gate(t).rule_id) for t in SAFE if evaluate_gate(t).tripped]
    assert tripped == [], f"gate over-tripped safe tickets (would needlessly defer wins): {tripped}"
    assert len(SAFE) == 21


# ----------------------------------------------------------------- per-case rule assertions


@pytest.mark.parametrize(
    "ticket_id,expected_rule",
    [
        ("t_train_003", "hard.account_compromise"),
        ("t_train_024", "hard.account_compromise"),
        ("t_train_006", "hard.problem_gambling"),
        ("t_train_017", "hard.problem_gambling"),   # deposit-limit framed as "budgeting"
        ("t_train_020", "hard.problem_gambling"),
        ("t_train_009", "hard.legal_regulatory"),
        ("t_train_012", "hard.minor"),
        ("t_train_023", "soft.jurisdiction_eligibility"),
        ("t_train_029", "soft.factual_dispute"),
    ],
)
def test_sensitive_ticket_trips_expected_rule(ticket_id, expected_rule):
    result = evaluate_gate(BY_ID[ticket_id])
    assert result.tripped
    assert result.rule_id == expected_rule, f"{ticket_id} matched on {result.matched_text!r}"


# ----------------------------------------------------------------- precision traps (must NOT trip)


@pytest.mark.parametrize(
    "ticket_id,why",
    [
        ("t_train_018", "'super minor' is the adjective, not an underage person"),
        ("t_train_028", "'$1 charge I don't remember authorizing' is benign card verification"),
        ("t_train_026", "'I moved. New address...' is an address change, not a jurisdiction Q"),
        ("t_train_016", "'settled as Warriors... that's correct, I'm not disputing' is not a dispute"),
        ("t_train_013", "'limit order didn't fill' is mechanics, not a deposit-limit RG request"),
        ("t_train_022", "'are limit orders free' must not trip the deposit-limit rule"),
    ],
)
def test_precision_trap_does_not_trip(ticket_id, why):
    assert not evaluate_gate(BY_ID[ticket_id]).tripped, why


def test_he_is_22_bait_routes_to_compromise_not_minor():
    """t_eval_003 says 'He's 22 so it's not like an age thing' to bait the minor rule. It must
    still defer — via account compromise (unauthorized access), NOT via the minor rule."""
    result = evaluate_gate(EVAL["t_eval_003"])
    assert result.tripped
    assert result.rule_id == "hard.account_compromise"


# ----------------------------------------------------------------- robustness


def test_curly_apostrophe_is_normalized():
    """Contractions with a typographic apostrophe must still match (real inboxes produce these)."""
    assert evaluate_gate(_ticket("I didn’t authorize this trade")).tripped


def test_clean_ticket_passes():
    assert not evaluate_gate(_ticket("how do limit orders work on novig?")).tripped
