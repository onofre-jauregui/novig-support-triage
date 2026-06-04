# Taxonomy & Output Schema

This document defines the categories, urgency levels, and output schema your system should emit.

## Categories

| Category | Description |
|---|---|
| `account_access` | Login issues, password resets, 2FA problems, locked accounts (non-compromise). |
| `kyc_verification` | Identity verification, document upload, address proof, KYC delays or rejections. |
| `deposits_withdrawals` | Deposit failures, withdrawal status, payment method issues, ACH/wire/card problems, processing times. |
| `trading_mechanics` | How features work: parlays, limit orders, order matching, position sizing, the exchange model. |
| `market_questions` | Specific markets: when does X settle, why was this market graded this way, market rules, contract specifications. |
| `bug_report` | App crashes, UI glitches, incorrect balances displayed, broken flows. Excludes anything money-impacting that the user attributes to fraud or compromise. |
| `tax_documents` | 1099s, tax forms, year-end statements, transaction history exports for tax purposes. |
| `account_compromise` | User suspects their account has been accessed by someone else, unauthorized trades or withdrawals, suspicious activity they did not initiate. |
| `problem_gambling` | Self-exclusion requests, deposit limit requests, statements suggesting loss of control, references to addiction or financial harm from gambling. |
| `legal_regulatory` | Threats of lawsuits, regulatory complaints (CFTC, state AG, CFPB), subpoenas, references to attorneys, formal legal demands. |
| `other` | Genuinely doesn't fit any category above. Use sparingly. |

## Urgency Levels

- **`low`** — Informational. No money or access at stake. Routine product questions. Response within a business day is fine.
- **`medium`** — User is mildly blocked or confused but not at risk. Common account issues, small-dollar disputes, KYC questions. Response within a few hours.
- **`high`** — User is materially blocked, money is in motion, or significant frustration. Stuck withdrawals, account lockouts, KYC rejections preventing trading. Response within the hour.
- **`escalate_immediately`** — Requires a human now. Active account compromise, explicit problem-gambling signals, self-harm references, mentions of a minor, formal legal threats with deadlines, or active fraud in progress.

Urgency and category are independent. A `deposits_withdrawals` ticket can be `low` (general question about ACH timing) or `high` (a $50k withdrawal stuck for a week). Not every sensitive-category ticket is `escalate_immediately` — for example, a calm jurisdictional eligibility question is `legal_regulatory` but may be `medium`. Use judgment.

## When the System Must Not Draft

The system **must not** produce an auto-draft response when any of the following are true. Return `should_draft: false` with a `no_draft_reason` explaining which condition triggered the decision.

**Hard rules (these always require no-draft):**

1. Category is `account_compromise`, `problem_gambling`, or `legal_regulatory`.
2. The ticket mentions a minor (anyone under 21, since Novig's age requirement varies by jurisdiction but never goes below 18, and 21 in some).
3. The ticket mentions self-harm, suicide, or severe emotional distress.
4. The ticket describes active fraud in progress (e.g., "someone is logged into my account right now and withdrawing money").

**Soft rules (use judgment — these often warrant no-draft):**

5. The user is asserting that a Novig-provided fact (e.g., a 1099 amount, a market grade, an account balance) is wrong. The system should not draft confirmations or denials of fact without ops review.
6. The user is asking for a definitive statement of Novig policy or contract spec that they will rely on for a financial decision (e.g., "how does the Super Bowl MVP market settle if there's a tie?"). The system either declines to draft, or drafts a response that explicitly defers to the contract spec without stating a rule.
7. The user is asking a jurisdictional eligibility question whose correct answer changes over time (e.g., "is Novig legal in TX?").

The principle: the cost of confidently stating something wrong on a regulated platform is high. When in doubt, defer to a human.

## Output Schema

Your `predictions.jsonl` should contain one JSON object per line, in the same order as `tickets_eval.jsonl`, with the following shape:

```json
{
  "ticket_id": "t_eval_001",
  "category": "deposits_withdrawals",
  "urgency": "high",
  "should_draft": true,
  "no_draft_reason": null,
  "draft_response": "Hi Sarah, thanks for reaching out about your withdrawal...",
  "confidence": 0.87
}
```

Field rules:

- `ticket_id` — exact string from the input.
- `category` — one of the category keys above.
- `urgency` — one of `low`, `medium`, `high`, `escalate_immediately`.
- `should_draft` — boolean. If `false`, `draft_response` must be `null` and `no_draft_reason` must be a non-empty string.
- `no_draft_reason` — null when `should_draft` is true, otherwise a short string (e.g., `"suspected account compromise"`, `"legal threat — attorney referenced"`).
- `draft_response` — null when `should_draft` is false, otherwise the response text the support agent would review before sending. Sign as "Novig Support" — no individual agent names.
- `confidence` — a number in `[0, 1]`. Define what it means in your writeup. We won't grade on calibration but we will grade on whether you thought about what it represents.

## Notes on Drafting

When you do draft, the response should:
- Acknowledge what the user said specifically (not a generic opener).
- State what is and isn't true. Don't promise outcomes you can't verify (e.g., don't promise a withdrawal will land by a specific date).
- Tell the user what happens next.
- Not invent policy. If you don't know, say a human will follow up.
