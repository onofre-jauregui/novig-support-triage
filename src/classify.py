"""Stage 2 — LLM classification with an input-perturbation self-consistency confidence signal.

Given a ticket, this returns a category, an urgency, the model's judgment of whether the ticket
is SAFE to auto-draft (soft-rule aware), and a confidence score.

Confidence is behavioral and model-agnostic. We classify the ticket N times; on each run we
perturb the INPUT rather than a sampling parameter — we resample + reorder the few-shot
exemplars and lightly paraphrase the ticket (meaning, entities, and amounts preserved) — then
measure how often the runs agree on the (category, draft-decision) pair. Unanimous → high
confidence; a split, especially one that flips the draft/no-draft call → low confidence → the
pipeline biases toward defer.

Why perturbation rather than temperature: Opus 4.8 rejects a non-default temperature and is
near-deterministic, so sampling jitter yields no signal. Perturbing the input probes whether the
classification is robust to how the question is posed and which examples are shown — the
variation that actually occurs in production — which is a stronger uncertainty signal and
survives a model swap. The deterministic safety gate runs on the ORIGINAL ticket text and owns
safety regardless of what perturbation does, so paraphrasing can never weaken the hard guarantees.

Concurrency: the N runs are independent and fire together via ``asyncio.gather`` — the only
concurrency in the system. Exemplar perturbation is seeded by run index (not by ticket), so there
are exactly N distinct system prompts, each reused across every ticket and cached via Anthropic
prompt caching (``cache_control``); only the (small) per-ticket user message is uncached.

Failure handling: a run that errors or returns unparseable output is dropped. If too few runs
survive to form a majority, classification fails closed (escalate, not safe) — never open.
"""

from __future__ import annotations

import asyncio
import json
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

import anthropic
from pydantic import BaseModel, ValidationError

from . import config
from .data import load_train
from .schema import Category, Ticket, Urgency

# Bump when the prompt/mechanism changes; metrics.json records it, prompt_iterations.md tracks the curve.
PROMPT_VERSION = "v4-capture"


# --------------------------------------------------------------------------- per-run output


class _RunOutput(BaseModel):
    """One run's structured answer. Invalid category/urgency strings fail validation and drop the
    run — we never coerce a bad label into a real one."""

    category: Category
    urgency: Urgency
    safe_to_draft: bool
    reason: Optional[str] = None


@dataclass
class ClassificationResult:
    """Aggregated result across the N perturbed runs."""

    category: Category
    urgency: Urgency
    llm_safe: bool                       # majority judgment: safe to auto-draft?
    confidence: float                    # agreement rate on (category, draft-decision), over N
    reason: Optional[str]                # defer reason when not safe
    raw_votes: list[dict] = field(default_factory=list)  # per-run votes; pipeline guard G2 reads these


# --------------------------------------------------------------------------- prompts


_INSTRUCTIONS = """\
You are the triage classifier for Novig, a CFTC-regulated peer-to-peer prediction market.
For each incoming support ticket you output a strict JSON object — nothing else.

Your job has two parts: (1) label the ticket, and (2) decide whether it is SAFE for an LLM to
auto-draft a reply, or whether it must be deferred to a human. Novig's governing rule: a wrong
automated reply to a serious ticket is far worse than no reply. But deferring everything is also
a failure — you must confidently draft the routine, low-risk majority.

CATEGORIES (choose exactly one):
- account_access: login, password reset, 2FA, locked account (NOT compromise).
- kyc_verification: identity/document verification, address proof, KYC delays/rejections.
- deposits_withdrawals: deposit/withdrawal status, payment methods, ACH/wire/card, processing times, fees.
- trading_mechanics: how features work — parlays, limit/market orders, matching, the exchange model, how Novig makes money.
- market_questions: a specific market — when it settles, why it graded a way, contract specs/rules.
- bug_report: app crashes, UI glitches, wrong balance displayed, broken flows (NOT money issues the user blames on fraud).
- tax_documents: 1099s, tax forms, year-end statements, transaction-history exports for taxes.
- account_compromise: suspected unauthorized access, trades/withdrawals the user did not make, activity they didn't initiate.
- problem_gambling: self-exclusion, deposit/loss limits, loss-of-control or financial-harm language, addiction references, "take a break".
- legal_regulatory: lawsuits, regulators (CFTC, state AG, CFPB), subpoenas, attorneys, formal legal demands, AND jurisdictional eligibility ("is Novig legal in my state").
- other: genuinely fits nothing above. Use sparingly.

URGENCY:
- low: purely informational — a how-it-works or general question where nothing is blocked and the user just needs an answer (no action pending on either side).
- medium: the user is mildly blocked or waiting on a small step — a KYC/account question, a small-dollar issue, a status check or account change that needs a next action.
- high: materially blocked, money in motion, strong frustration (stuck withdrawal, lockout, KYC rejection blocking trading).
- escalate_immediately: needs a human now — active compromise, explicit problem-gambling/self-harm, a referenced minor, formal legal threat, active fraud.
Category and urgency are independent. Not every sensitive ticket is escalate_immediately (a calm
"is Novig legal in TX?" is legal_regulatory but usually medium).

SAFE_TO_DRAFT — set false (defer) when ANY of these hold:
HARD (never draft): category is account_compromise, problem_gambling, or legal_regulatory; OR the
ticket references a minor / anyone under 21; OR self-harm or severe distress; OR active fraud in progress.
SOFT (judgment — usually defer):
- Factual dispute (defer): the user asserts a Novig OFFICIAL/REGULATED fact is WRONG — a 1099 amount, a
  market grade/settlement, an account balance — and wants it corrected. You can't verify ground truth and
  the stakes are regulatory, so do not draft.
  - NOT a dispute, so DRAFT it: an OPERATIONAL / payment discrepancy the user asks you to look into — a
    deposit that didn't post, a duplicate or double charge, a withdrawal that's stuck — is routine
    ops-routing. Acknowledge it, ask for the details payments-ops will need, and commit to checking;
    don't assert what happened. These are wins, not disputes.
  - NOT a dispute, so DRAFT it: a user who concedes the discrepancy is probably theirs ("probably my
    math", "I'm not disputing").
- Definitive policy / contract spec they'll rely on financially: e.g. "how does the Super Bowl MVP
  market settle if there's a tie?". Stating a rule wrong is a liability — defer.
- Jurisdictional eligibility ("can I trade in my state?"): the correct answer changes over time, so an
  answer from model memory is guaranteed-stale. Defer.

Otherwise set safe_to_draft TRUE and draft confidently: how-to / mechanics questions, deposit/
withdrawal/KYC status checks, general fee or minimum questions, bug reports, where-to-find-my-tax-doc,
password/2FA help, thank-you notes. These are the wins; do not over-defer them.

When safe_to_draft is false, give a short reason. When true, reason must be null.

Respond with ONLY this JSON (one line, no markdown):
{"category": "<category>", "urgency": "<low|medium|high|escalate_immediately>", "safe_to_draft": <true|false>, "reason": <null or "short string">}
"""


def _ticket_block(subject: str, body: str, tenure: int) -> str:
    return f"TICKET\nSubject: {subject}\nBody: {body}\n(user_tenure_days: {tenure})"


def _render_exemplar(t: Ticket) -> str:
    assert t.label is not None
    out = {
        "category": t.label.category.value,
        "urgency": t.label.urgency.value,
        "safe_to_draft": t.label.should_draft,
        "reason": t.label.no_draft_reason,
    }
    block = _ticket_block(t.subject, t.body, t.metadata.user_tenure_days)
    return f"{block}\nOUTPUT\n{json.dumps(out, ensure_ascii=False)}"


@lru_cache(maxsize=1)
def _exemplar_pool() -> tuple[Ticket, ...]:
    by_id = {t.ticket_id: t for t in load_train()}
    return tuple(by_id[i] for i in config.FEW_SHOT_IDS if i in by_id)


@lru_cache(maxsize=32)
def _system_prompt(seed: int) -> str:
    """Instructions + a resampled, reordered subset of the few-shot pool. Seeded by RUN INDEX so
    there are only N distinct prompts (one per run), each cached and reused across all tickets.
    Exemplars are drawn only from config.FEW_SHOT_IDS, so a held-out ticket can never leak in."""
    pool = list(_exemplar_pool())
    rng = random.Random(seed)
    k = min(config.PERTURB_EXEMPLARS_PER_RUN, len(pool))
    chosen = rng.sample(pool, k=k)  # subset + order both vary with seed
    blocks = [_INSTRUCTIONS, "", "EXAMPLES (the labels are correct):", ""]
    blocks += [_render_exemplar(t) + "\n" for t in chosen]
    return "\n".join(blocks)


def _user_prompt(ticket_text: str) -> str:
    return f"Classify this ticket.\n\n{ticket_text}\n\nOUTPUT (JSON only):"


# --------------------------------------------------------------------------- paraphrase (input perturbation)


_PARAPHRASE_SYS = (
    "You lightly paraphrase customer-support tickets to create surface variation for an "
    "ensemble. Rewrite the ticket body N times. Each rewrite MUST preserve the exact meaning, "
    "every entity, every number/amount/date, the tone, and the intent — change only wording and "
    "sentence order. Never add, drop, or soften any claim (especially anything about being "
    "hacked, a minor, money, distress, or legal action). Output ONLY a JSON array of N strings."
)


async def _paraphrase_bodies(
    client: anthropic.AsyncAnthropic, body: str, n_variants: int
) -> list[str]:
    """Return up to ``n_variants`` meaning-preserving paraphrases of the body. On any failure,
    falls back to the original body repeated — perturbation then comes from exemplars alone."""
    if n_variants <= 0:
        return []
    try:
        resp = await client.messages.create(
            model=config.PARAPHRASE_MODEL,
            max_tokens=1200,
            system=[{"type": "text", "text": _PARAPHRASE_SYS}],
            messages=[{"role": "user", "content": f"N={n_variants}\nBODY:\n{body}"}],
        )
        m = re.search(r"\[.*\]", resp.content[0].text, re.DOTALL)
        variants = json.loads(m.group(0)) if m else []
        variants = [v for v in variants if isinstance(v, str) and v.strip()]
        if not variants:
            raise ValueError("no variants")
        # pad/truncate to exactly n_variants
        while len(variants) < n_variants:
            variants.append(body)
        return variants[:n_variants]
    except Exception:  # noqa: BLE001 — degrade gracefully, never fail the classification on this
        return [body] * n_variants


# --------------------------------------------------------------------------- one run (async)


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse(text: str) -> Optional[_RunOutput]:
    m = _JSON_RE.search(text)
    if not m:
        return None
    try:
        return _RunOutput.model_validate_json(m.group(0))
    except (ValidationError, ValueError):
        return None


async def _one_run(
    client: anthropic.AsyncAnthropic, ticket_text: str, seed: int
) -> Optional[_RunOutput]:
    """A single classification call (no temperature — Opus 4.8 rejects it), with bounded retries
    on transient API errors. Returns parsed output, or None if it errored/!parsed — the caller
    treats None as a dropped vote, never as 'safe'."""
    for attempt in range(config.MAX_RETRIES + 1):
        try:
            resp = await client.messages.create(
                model=config.CLASSIFY_MODEL,
                max_tokens=300,
                system=[
                    {
                        "type": "text",
                        "text": _system_prompt(seed),
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": _user_prompt(ticket_text)}],
            )
            return _parse(resp.content[0].text)
        except (anthropic.APIStatusError, anthropic.APIConnectionError, anthropic.APITimeoutError):
            if attempt >= config.MAX_RETRIES:
                return None
            await asyncio.sleep(0.5 * (2**attempt))
    return None


async def _classify_async(ticket: Ticket, n: int) -> ClassificationResult:
    client = anthropic.AsyncAnthropic(timeout=config.REQUEST_TIMEOUT_S)
    try:
        # Build N perturbed inputs: run 0 = original body; runs 1..N-1 = paraphrases (if enabled).
        bodies = [ticket.body]
        if config.PERTURB_PARAPHRASE and n > 1:
            bodies += await _paraphrase_bodies(client, ticket.body, n - 1)
        else:
            bodies += [ticket.body] * (n - 1)

        texts = [
            _ticket_block(ticket.subject, body, ticket.metadata.user_tenure_days)
            for body in bodies[:n]
        ]
        runs = await asyncio.gather(*[_one_run(client, texts[k], seed=k) for k in range(n)])
    finally:
        await client.close()
    return _aggregate([r for r in runs if r is not None], n)


# --------------------------------------------------------------------------- aggregation


def _aggregate(valid: list[_RunOutput], n: int) -> ClassificationResult:
    """Majority-vote the runs; confidence = agreement on (category, draft-decision) over N.

    If a majority of runs failed to produce a valid answer, the model is effectively unusable on
    this ticket → fail closed (escalate, not safe). Disagreement and failure both push toward
    defer, the safe direction."""
    if len(valid) < (n // 2 + 1):
        return ClassificationResult(
            category=Category.OTHER,
            urgency=Urgency.ESCALATE_IMMEDIATELY,
            llm_safe=False,
            confidence=0.0,
            reason="fail-closed: classifier did not return enough valid responses",
        )

    modal_category = Counter(r.category for r in valid).most_common(1)[0][0]
    modal_urgency = Counter(r.urgency for r in valid).most_common(1)[0][0]
    safe_votes = sum(r.safe_to_draft for r in valid)
    modal_safe = safe_votes > len(valid) / 2  # ties resolve to unsafe (the safe direction)

    # Confidence is over the intended N, so dropped runs lower it (instability == low confidence).
    joint = sum(1 for r in valid if r.category == modal_category and r.safe_to_draft == modal_safe)
    confidence = round(joint / n, 4)

    reason = None
    if not modal_safe:
        reasons = [r.reason for r in valid if not r.safe_to_draft and r.reason]
        reason = (
            Counter(reasons).most_common(1)[0][0]
            if reasons
            else "model judged this ticket unsafe to auto-draft"
        )

    return ClassificationResult(
        category=modal_category,
        urgency=modal_urgency,
        llm_safe=modal_safe,
        confidence=confidence,
        reason=reason,
        raw_votes=[r.model_dump(mode="json") for r in valid],
    )


# --------------------------------------------------------------------------- public entrypoint


def classify(ticket: Ticket, n: Optional[int] = None) -> ClassificationResult:
    """Classify one ticket with N perturbed self-consistency runs. Synchronous wrapper so the
    pipeline reads top-to-bottom; the concurrency is encapsulated in ``_classify_async``."""
    n = n if n is not None else config.N_SELF_CONSISTENCY
    return asyncio.run(_classify_async(ticket, n))
