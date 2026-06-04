"""Typed contract for the triage system.

Everything in the pipeline conforms to these models. The enums and the ``Prediction``
field rules are a direct, literal translation of ``taxonomy.md`` — that document is the
source of truth, and if it changes, this file changes with it (nothing else should need to).

Design notes:
- ``Category`` / ``Urgency`` are ``str`` enums so they serialize straight to the exact
  string values the schema requires (``model_dump`` / JSON give ``"deposits_withdrawals"``,
  not ``Category.DEPOSITS_WITHDRAWALS``).
- ``Prediction`` enforces the schema's cross-field invariants in a validator, so it is
  *impossible* to construct an internally inconsistent prediction (e.g. ``should_draft=False``
  with a non-null draft). Constructing the fail-closed default therefore can't silently drift.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Category(str, Enum):
    """The 11 categories from taxonomy.md. Order mirrors the taxonomy table."""

    ACCOUNT_ACCESS = "account_access"
    KYC_VERIFICATION = "kyc_verification"
    DEPOSITS_WITHDRAWALS = "deposits_withdrawals"
    TRADING_MECHANICS = "trading_mechanics"
    MARKET_QUESTIONS = "market_questions"
    BUG_REPORT = "bug_report"
    TAX_DOCUMENTS = "tax_documents"
    ACCOUNT_COMPROMISE = "account_compromise"
    PROBLEM_GAMBLING = "problem_gambling"
    LEGAL_REGULATORY = "legal_regulatory"
    OTHER = "other"


class Urgency(str, Enum):
    """The 4 urgency levels from taxonomy.md."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ESCALATE_IMMEDIATELY = "escalate_immediately"


# Hard-rule categories from taxonomy.md §"When the System Must Not Draft" (rule 1).
# Membership in this set ALWAYS forces no-draft — enforced both by the deterministic gate
# (when keyword-detectable) and by the pipeline (when the LLM independently classifies into
# one of these). Kept here, beside the enum, because it is a taxonomy fact, not a tunable.
HARD_BLOCK_CATEGORIES: frozenset[Category] = frozenset(
    {
        Category.ACCOUNT_COMPROMISE,
        Category.PROBLEM_GAMBLING,
        Category.LEGAL_REGULATORY,
    }
)


class TicketMetadata(BaseModel):
    """Per-ticket metadata block from the input jsonl."""

    model_config = ConfigDict(extra="allow")  # tolerate unforeseen metadata keys

    user_tenure_days: int
    submitted_at: datetime


class GoldLabel(BaseModel):
    """The labeled answer attached to each *training* ticket. Never present on eval tickets,
    and never read by ``predict`` — it exists only for the eval harness and few-shot curation."""

    category: Category
    urgency: Urgency
    should_draft: bool
    no_draft_reason: Optional[str] = None
    gold_response_notes: Optional[str] = None


class Ticket(BaseModel):
    """An incoming support ticket. ``label`` is populated for training tickets and ``None``
    for eval tickets. The pipeline treats tickets as label-blind; see ``pipeline.predict``."""

    ticket_id: str
    subject: str
    body: str
    metadata: TicketMetadata
    label: Optional[GoldLabel] = None

    @property
    def text(self) -> str:
        """Subject + body as a single block — what the gate and classifier actually read."""
        return f"{self.subject}\n\n{self.body}".strip()


class Prediction(BaseModel):
    """The structured output, one per ticket, exactly as taxonomy.md §"Output Schema" defines.

    The cross-field rules from the taxonomy are enforced below so an invalid prediction
    cannot be built:
      - should_draft is False  ->  draft_response is None  AND no_draft_reason is non-empty
      - should_draft is True   ->  draft_response is non-empty AND no_draft_reason is None
    """

    ticket_id: str
    category: Category
    urgency: Urgency
    should_draft: bool
    no_draft_reason: Optional[str] = None
    draft_response: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _enforce_schema_invariants(self) -> "Prediction":
        if self.should_draft:
            if not self.draft_response or not self.draft_response.strip():
                raise ValueError("should_draft=True requires a non-empty draft_response")
            if self.no_draft_reason is not None:
                raise ValueError("should_draft=True requires no_draft_reason to be null")
        else:
            if self.draft_response is not None:
                raise ValueError("should_draft=False requires draft_response to be null")
            if not self.no_draft_reason or not self.no_draft_reason.strip():
                raise ValueError("should_draft=False requires a non-empty no_draft_reason")
        return self

    # ------------------------------------------------------------------ constructors

    @classmethod
    def no_draft(
        cls,
        ticket_id: str,
        category: Category,
        urgency: Urgency,
        reason: str,
        confidence: float,
    ) -> "Prediction":
        """Build a defer (no-draft) prediction."""
        return cls(
            ticket_id=ticket_id,
            category=category,
            urgency=urgency,
            should_draft=False,
            no_draft_reason=reason,
            draft_response=None,
            confidence=confidence,
        )

    @classmethod
    def drafted(
        cls,
        ticket_id: str,
        category: Category,
        urgency: Urgency,
        draft_response: str,
        confidence: float,
    ) -> "Prediction":
        """Build a draft prediction."""
        return cls(
            ticket_id=ticket_id,
            category=category,
            urgency=urgency,
            should_draft=True,
            no_draft_reason=None,
            draft_response=draft_response,
            confidence=confidence,
        )

    @classmethod
    def fail_closed(cls, ticket_id: str, reason: str) -> "Prediction":
        """The canonical fail-closed default. ANY error anywhere in the pipeline (malformed
        model output, API failure, parse error, timeout) resolves to exactly this: defer,
        escalate, and say why. There is no path that fails open."""
        return cls.no_draft(
            ticket_id=ticket_id,
            category=Category.OTHER,
            urgency=Urgency.ESCALATE_IMMEDIATELY,
            reason=f"fail-closed: {reason}",
            confidence=0.0,
        )

    def to_jsonl_dict(self) -> dict:
        """Serialize to the exact schema shape (enum values as strings), for predictions.jsonl."""
        return {
            "ticket_id": self.ticket_id,
            "category": self.category.value,
            "urgency": self.urgency.value,
            "should_draft": self.should_draft,
            "no_draft_reason": self.no_draft_reason,
            "draft_response": self.draft_response,
            "confidence": self.confidence,
        }
