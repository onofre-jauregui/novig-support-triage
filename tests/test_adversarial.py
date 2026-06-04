"""Adversarial suite: every evasive sensitive ticket must route to no-draft.

Two layers of assertion:
  - Offline (no API): the keyword-detectable cases must trip the deterministic gate. Always runs.
  - Full pipeline (needs ANTHROPIC_API_KEY): the genuinely evasive cases — which the gate does NOT
    catch — must still be deferred by the LLM. This is the real proof the safety layer generalizes
    beyond literal keywords. Skipped (not failed) when no key is configured, so `make test` works
    offline; the eval harness also reports this pass rate in metrics.json.
"""

from __future__ import annotations

import os

import pytest

# Importing the pipeline triggers src.config, which loads .env — so check for the key AFTER this.
from src.pipeline import predict
from src.safety_gate import evaluate_gate
from tests.adversarial_cases import ADVERSARIAL_TICKETS, CASES

_HAS_KEY = bool(os.getenv("ANTHROPIC_API_KEY"))

# Cases the deterministic gate is expected to catch on its own (keyword-detectable signal).
_GATE_CATCHABLE = {"t_adv_03_veiled_legal", "t_adv_06_compromise_3am"}


@pytest.mark.parametrize("ticket_id", sorted(_GATE_CATCHABLE))
def test_keyworded_adversarial_trips_gate_offline(ticket_id):
    ticket = next(c.ticket for c in CASES if c.ticket.ticket_id == ticket_id)
    assert evaluate_gate(ticket).tripped, f"{ticket_id} should be caught by the gate deterministically"


@pytest.mark.skipif(not _HAS_KEY, reason="ANTHROPIC_API_KEY not set — skipping full-pipeline run")
def test_every_adversarial_ticket_routes_to_no_draft():
    """The headline adversarial guarantee: none of the evasive sensitive tickets get auto-drafted,
    including the 8/10 the gate does not catch (the LLM must defer those)."""
    leaked = [t.ticket_id for t in ADVERSARIAL_TICKETS if predict(t).should_draft]
    assert leaked == [], f"adversarial tickets were auto-drafted — must never happen: {leaked}"
