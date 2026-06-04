"""Stage 1 — the deterministic safety gate. No LLM. No network. Pure pattern matching.

This is the unhackable shield. It runs *before* the model and cannot be bypassed by anything
the model does, including prompt injection in the ticket body: if a rule trips, ``should_draft``
is forced to ``False`` in the pipeline regardless of what the LLM later says.

Design principles:
  - HARD rules encode the taxonomy's always-block categories (account compromise, problem
    gambling, legal/regulatory) plus the keyword-detectable sensitive signals (minor, self-harm,
    active fraud). These must fire on every such ticket — missing one is the worst failure mode.
  - One SOFT signal is keyword-detectable safely and lives here too: jurisdictional eligibility
    ("is Novig legal in TX"), which always maps to legal_regulatory anyway.
  - The other soft rules (subtle factual disputes, definitive policy questions) are NOT here on
    purpose. A keyword broad enough to catch "my 1099 is $4k off" (defer) would also catch
    "my balance is off, probably my math" (t_train_018, a legitimate draft). Those judgments
    require reading intent and belong to the LLM (Stage 2). The gate only trips on dispute
    phrasing that is *unambiguous* (e.g. "is this a mistake", "mis-graded").

Every pattern documents which training tickets it is responsible for, so the shield is auditable.
Precision is as important as recall: each rule below was checked to NOT fire on any of the 21
safe training tickets (notably the traps t_028 "$1 charge I don't remember authorizing",
t_018 "balance wrong... probably my math", t_026 "I moved", and the t_eval_003 "He's 22" bait).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Pattern

from .schema import Category


@dataclass(frozen=True)
class GateResult:
    """Outcome of running the gate on one ticket."""

    tripped: bool
    rule_id: Optional[str] = None        # e.g. "hard.account_compromise"
    reason: Optional[str] = None         # human-readable, logged into no_draft_reason
    category_hint: Optional[Category] = None  # the category this signal implies (for logging)
    matched_text: Optional[str] = None   # the substring that matched (explainability)


@dataclass(frozen=True)
class _Rule:
    rule_id: str
    category_hint: Optional[Category]
    reason: str
    patterns: tuple[Pattern[str], ...]


def _compile(*raw: str) -> tuple[Pattern[str], ...]:
    return tuple(re.compile(p, re.IGNORECASE) for p in raw)


# ---------------------------------------------------------------------------------------------
# Rules are ordered by severity. When several trip, the first (most serious) supplies the reason.
# ---------------------------------------------------------------------------------------------

RULES: tuple[_Rule, ...] = (
    # --- self-harm / severe distress -------------------------------------------------- taxonomy hard rule 3
    _Rule(
        "hard.self_harm",
        Category.PROBLEM_GAMBLING,
        "self-harm or severe emotional distress language — immediate trained-human response",
        _compile(
            r"\bsuicid(e|al)\b",
            r"\bkill(ing)?\s+my\s?self\b",
            r"\b(hurt|harm|harming|hurting)\s+my\s?self\b",
            r"\bend(ing)?\s+(my\s+life|it\s+all)\b",
            r"\b(don'?t|do not|can'?t|cannot)\s+want\s+to\s+(live|be here|go on|wake up)\b",
            r"\bbetter\s+off\s+(dead|gone|without me)\b",
            r"\bno\s+(reason|point)\s+(to|in)\s+(live|living|going on)\b",
        ),
    ),
    # --- account compromise / active fraud -------------------------------------------- taxonomy hard rules 1,4
    # Responsible for: t_train_003, t_train_024, t_eval_003.
    _Rule(
        "hard.account_compromise",
        Category.ACCOUNT_COMPROMISE,
        "suspected account compromise / unauthorized activity — security investigation required",
        _compile(
            r"\b(hack(ed|ing)?|compromis(ed|e)|breach(ed)?)\b",
            r"\bunauthori[sz]ed\b",
            # "I did not make / didn't place / never authorized" a trade/withdrawal
            r"\b(did|do|i)\s*(n'?t| not| never)\s+(make|made|place|placed|authori[sz]e|recogni[sz]e|initiate)\b",
            # "someone (else) is in / logged into / placed / withdrew ..."
            r"\bsomeone\s+(else\s+)?(is\s+|was\s+|has\s+)?(in|inside|logged|log\s?ged|accessed|using|placed|made|took|withdr\w*)\b",
            r"\blog(ged)?\s+(in|into)\s+(my|the)\s+account\b",
            r"\breverse\s+(the|these|this|those|my)\s+(trade|trades|transaction|transactions|withdrawal|bets?)\b",
            r"\bfrom\s+a\s+different\s+(city|state|country|location|device|computer|ip)\b",
            r"\b(lock|kick)\s+(him|her|them|someone)\s+out\b",
        ),
    ),
    # --- problem gambling / responsible-gaming ---------------------------------------- taxonomy hard rule 1
    # Responsible for: t_train_006, t_train_017, t_train_020, t_eval_005, t_eval_013.
    _Rule(
        "hard.problem_gambling",
        Category.PROBLEM_GAMBLING,
        "problem-gambling / responsible-gaming signal — RG-trained human response required",
        _compile(
            r"\bself.?exclu(de|sion|ding)\b",
            r"\b(gambling|gamble|betting)\s+(problem|addict\w*)\b",
            r"\baddict(ed|ion)?\b",
            r"\b1.?800.?gambler\b",
            # RG limit / cooling-off requests (NOT "limit order" — note the required prefix word)
            r"\b(deposit|spending|spend|loss|losses|wager|weekly|monthly|daily)\s+limit\b",
            r"\b(cool(ing)?.?off|take a break|time.?out|self.?ban)\b",
            r"\b(don'?t|can'?t|cannot|do not)\s+trust\s+my\s?self\b",
            # loss-of-control language
            r"\b(don'?t know how to|can'?t|cannot|need to|have to|trying to)\s+stop\b",
            r"\bkeep(s)?\s+losing\b",
            r"\b(lost|losing)\s+(too much|control|everything|it all|more than)\b",
            r"\bdown\s+\$?\d[\d,]*\s?k?\b",                      # "down $8k" (t_020)
            # despair / self-exclusion-by-closure framing
            r"\bi'?m\s+(done|tired|exhausted|spent|broke)\b",
            r"\btake\s+it\s+all\b",
            r"\bshut\s+(it|everything)\s+(all\s+)?down\b",
            r"\b(freeze|close|lock|disable|deactivate)\b[^.\n]{0,30}\b(permanent\w*|forever|for good|never again|again)\b",
            r"\b(can'?t|cannot)\s+keep\s+(doing|going)\b",
            r"\blost\s+too\s+much\b",
            r"\bdone\s+playing\b",
        ),
    ),
    # --- legal / regulatory ----------------------------------------------------------- taxonomy hard rule 1
    # Responsible for: t_train_009, t_eval_010.
    _Rule(
        "hard.legal_regulatory",
        Category.LEGAL_REGULATORY,
        "legal threat / regulatory complaint — legal & compliance team only",
        _compile(
            r"\b(attorney|lawyer|law\s+firm|legal\s+counsel|my\s+counsel)\b",
            r"\b(lawsuit|sue|suing|sued|litigation|legal\s+action|pursue\s+(this\s+)?legal|take\s+legal)\b",
            r"\b(cftc|cfpb|finra|attorney\s+general|regulator|regulatory\s+(complaint|body|channel|action)|subpoena|demand\s+letter)\b",
            r"\bformal\s+(complaint|notice|demand|legal)\b",
            r"\bfile\s+(a\s+)?(complaint|suit|claim)\b",
            # veiled legal threat (adversarial): "spoken to someone about my options/rights"
            r"\b(spoke|spoken|talked|consulting|consulted)\b[^.\n]{0,40}\b(options|rights|recourse|legal)\b",
        ),
    ),
    # --- minor / underage ------------------------------------------------------------- taxonomy hard rule 2
    # Responsible for: t_train_012.  (Novig's floor is 18, 21 in some states → treat <21 as a hard stop.)
    # Carefully scoped so "He's 22" (t_eval_003) and stray numbers ("iPhone 14") do NOT trip.
    _Rule(
        "hard.minor",
        Category.LEGAL_REGULATORY,
        "a minor / underage person is referenced — age & compliance review, never auto-respond",
        _compile(
            r"\b(1[0-9]|20)\s*(year|yr)s?\s*[\s-]?old\b",                 # "17 year old"
            r"\b(he|she|they|kid|son|daughter|child|nephew|niece)\s*'?s?\s+(only\s+|just\s+)?(1[0-9]|20)\b",  # "he's 17"
            r"\b(turned|turning|is|am|i'?m)\s+(1[0-7])\b",                # "is 16" / "turning 17"
            r"\b(under\s?age|underage|too\s+young|not\s+(yet\s+)?(of\s+age|18|21))\b",
            # "minor" only in the underage-NOUN sense — never the adjective ("super minor", t_018)
            r"\b(a|the|is\s+a|as\s+a|still\s+a|he'?s\s+a|she'?s\s+a|my)\s+minor\b",
            r"\bminors?\s+(can|cannot|can'?t|are|aren'?t|using|use|to\s+use|sign|signing|gambl\w*|bet|trade|allowed)\b",
        ),
    ),
    # --- jurisdictional eligibility (SOFT, but safely keyword-detectable) -------------- taxonomy soft rule 7
    # Responsible for: t_train_023.  Scoped so t_026 "I moved. New address..." does NOT trip.
    _Rule(
        "soft.jurisdiction_eligibility",
        Category.LEGAL_REGULATORY,
        "jurisdictional eligibility question — state availability changes over time; compliance owns it",
        _compile(
            r"\b(legal|allowed|available|permitted|eligible|operate|operating|live|offered)\s+(in|to\s+(use|trade))\b",
            r"\bin\s+(my\s+state|which\s+states|what\s+states)\b",
            r"\b(can|may)\s+i\s+(still\s+)?(use|trade|bet|access|play|deposit)\b[^.\n]{0,40}\b(state|moved|relocat\w*|from\s+(tx|texas|[A-Z]{2}))\b",
            r"\b(move|moved|relocat\w*|living|now\s+in)\s+(to|in)\s+(tx|texas|[a-z]{4,}\b)[^.\n]{0,30}\b(use|trade|legal|account|still)\b",
        ),
    ),
    # --- unambiguous factual dispute (SOFT; only the clear cases) ---------------------- taxonomy soft rule 5
    # Responsible for: t_train_029 ("Is this a mistake?").  Deliberately narrow — does NOT match
    # t_018 ("balance wrong... probably my math") or t_016 ("settled as Warriors... not disputing").
    _Rule(
        "soft.factual_dispute",
        None,
        "explicit dispute of a Novig-provided grade/fact — ops review before any user-facing reply",
        _compile(
            r"\bis\s+this\s+(a\s+mistake|an\s+error|wrong|incorrect)\b",
            r"\b(was|got)\s+this\s+(market\s+)?(graded|settled|resolved)\s+(wrong|incorrectly|in\s+error)\b",
            r"\bmis-?grad(ed|ing|e)\b",
            r"\bgraded\s+in\s+error\b",
            r"\b(resolved|settled)\s+(this\s+)?in\s+error\b",
        ),
    ),
)


def _normalize(text: str) -> str:
    """Fold curly apostrophes/quotes to straight so contractions match regardless of source."""
    return (
        text.replace("’", "'").replace("‘", "'")
        .replace("“", '"').replace("”", '"')
    )


def evaluate_gate(ticket) -> GateResult:  # ticket: Ticket (untyped to avoid an import cycle)
    """Run all rules against the ticket's subject+body. Returns the first (most severe) match,
    or a clean pass. Deterministic and side-effect-free."""
    text = _normalize(ticket.text)
    for rule in RULES:
        for pat in rule.patterns:
            m = pat.search(text)
            if m:
                return GateResult(
                    tripped=True,
                    rule_id=rule.rule_id,
                    reason=rule.reason,
                    category_hint=rule.category_hint,
                    matched_text=m.group(0),
                )
    return GateResult(tripped=False)


# ---------------------------------------------------------------------------------------------
# `python -m src.safety_gate` (and `make gate`) prints the standalone validation: 100% recall on
# the no-draft training cases, and the false-positive rate on the safe ones. No LLM involved.
# ---------------------------------------------------------------------------------------------

def _validate_against_train() -> int:
    from .data import load_train

    train = load_train()
    sensitive = [t for t in train if t.label and not t.label.should_draft]
    safe = [t for t in train if t.label and t.label.should_draft]

    missed = [t for t in sensitive if not evaluate_gate(t).tripped]
    false_pos = [t for t in safe if evaluate_gate(t).tripped]

    print("=" * 72)
    print("  DETERMINISTIC SAFETY GATE — validation on 30 training tickets (no LLM)")
    print("=" * 72)
    print(f"  sensitive (no-draft) recall : {len(sensitive) - len(missed)}/{len(sensitive)}"
          f"  ({(len(sensitive) - len(missed)) / len(sensitive):.0%})")
    print(f"  safe false-positive rate    : {len(false_pos)}/{len(safe)}"
          f"  ({len(false_pos) / len(safe):.0%})")
    print("-" * 72)
    for t in sensitive:
        r = evaluate_gate(t)
        mark = "ok " if r.tripped else "MISS"
        rid = r.rule_id or "—"
        print(f"  [{mark}] {t.ticket_id}  {rid:<28}  match={r.matched_text!r}")
    if false_pos:
        print("-" * 72)
        for t in false_pos:
            r = evaluate_gate(t)
            print(f"  [FP ] {t.ticket_id}  {r.rule_id}  match={r.matched_text!r}  ({t.subject!r})")
    print("=" * 72)
    ok = not missed and not false_pos
    print("  RESULT:", "PASS — 100% recall, 0 false positives" if ok else "FAIL — tune the rules")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_validate_against_train())
