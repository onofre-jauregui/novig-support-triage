"""Novig support ticket triage system.

A three-stage, fail-closed pipeline: a deterministic safety gate (Stage 1) shields
sensitive tickets from the model entirely; an LLM classifier (Stage 2) assigns category,
urgency, and a self-consistency confidence signal; a drafter (Stage 3) writes a response
only for tickets that clear both prior stages.

The single public entrypoint is ``src.pipeline.predict``.
"""
