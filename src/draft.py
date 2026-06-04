"""Stage 3 — draft generation.

This runs ONLY on tickets that cleared the gate and the classifier and met the confidence
threshold, so every ticket reaching here is already confirmed safe to auto-draft. That property
is what lets a cheaper model be cost-tiered in here at scale (it never sees a sensitive ticket).

The draft is written for a human to review before sending. It follows taxonomy.md's drafting
rules: acknowledge the specific ticket, state only what's verifiable (never promise a date or
outcome), say what happens next, never invent policy, and sign as "Novig Support".

Failure is closed: if drafting errors out after retries, ``generate_draft`` raises and the
pipeline converts that into a defer — we never send an empty or broken reply.
"""

from __future__ import annotations

import anthropic

from . import config
from .schema import Category, Ticket, Urgency

_DRAFT_SYSTEM = """\
You draft replies for Novig support — a CFTC-regulated peer-to-peer prediction market. A human
agent reviews every draft before it is sent, so write a complete, ready-to-review reply.

Rules (all required):
- Acknowledge what THIS user specifically said — reference their actual situation, not a generic opener.
- State only what you can stand behind. Do NOT promise outcomes or specific dates ("your withdrawal
  will arrive Tuesday"), and do NOT invent policy, numbers, fees, or limits. If a specific value or
  rule isn't something you can be sure of, say a teammate will confirm rather than guessing.
- Tell the user what happens next (what you/ops will do, or what they should try).
- Be warm, concise, and professional. No fake enthusiasm, no legalese.
- Sign off as "Novig Support" — never an individual name, and never a placeholder like "[Name]".

Write ONLY the reply text — no subject line, no preamble, no notes to yourself.
"""


def _user_prompt(ticket: Ticket, category: Category, urgency: Urgency) -> str:
    return (
        f"Draft a reply to this ticket.\n\n"
        f"Category: {category.value}\nUrgency: {urgency.value}\n"
        f"Subject: {ticket.subject}\nBody: {ticket.body}\n"
        f"(user has been with Novig {ticket.metadata.user_tenure_days} days)"
    )


def generate_draft(ticket: Ticket, category: Category, urgency: Urgency) -> str:
    """Generate a review-ready draft reply. Raises after exhausting retries so the pipeline can
    fail closed — there is no path that returns a half-formed or empty draft."""
    client = anthropic.Anthropic(timeout=config.REQUEST_TIMEOUT_S)
    last_exc: Exception | None = None
    for attempt in range(config.MAX_RETRIES + 1):
        try:
            resp = client.messages.create(
                model=config.DRAFT_MODEL,
                max_tokens=600,
                temperature=config.DRAFT_TEMPERATURE,
                system=[
                    {
                        "type": "text",
                        "text": _DRAFT_SYSTEM,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": _user_prompt(ticket, category, urgency)}],
            )
            text = "".join(block.text for block in resp.content if block.type == "text").strip()
            if not text:
                raise ValueError("empty draft")
            return text
        except (anthropic.APIStatusError, anthropic.APIConnectionError, anthropic.APITimeoutError, ValueError) as exc:
            last_exc = exc
            continue
    raise RuntimeError(f"draft generation failed after {config.MAX_RETRIES + 1} attempts: {last_exc}")
