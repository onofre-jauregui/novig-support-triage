"""Central configuration — every tunable lives here, nothing is hard-coded downstream.

Models, temperatures, the confidence threshold, the self-consistency N, and the held-out
split are all set in one place so a teammate can change behavior without hunting through code,
and so an eval run is fully described by this file plus the pinned dataset.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import dotenv_values, load_dotenv

ROOT: Path = Path(__file__).resolve().parent.parent

# Load the repo-local .env (gitignored). load_dotenv fills *unset* vars but won't replace one that
# already exists — and some environments (CI, certain shells) pre-seed ANTHROPIC_API_KEY as an
# empty string, which would mask the real key. So when it's blank, take the value from .env.
load_dotenv(ROOT / ".env")
if not os.getenv("ANTHROPIC_API_KEY"):
    _env = dotenv_values(ROOT / ".env")
    if _env.get("ANTHROPIC_API_KEY"):
        os.environ["ANTHROPIC_API_KEY"] = _env["ANTHROPIC_API_KEY"]

# ---------------------------------------------------------------------------- models
# Both stages are behind config values so swapping a model is a one-line change.
# Classification uses the strong model (it carries the safety-relevant judgment);
# drafting can use a cheaper model because only confirmed-safe tickets ever reach it.
CLASSIFY_MODEL: str = os.getenv("NOVIG_CLASSIFY_MODEL", "claude-opus-4-8")
DRAFT_MODEL: str = os.getenv("NOVIG_DRAFT_MODEL", "claude-sonnet-4-6")

# ---------------------------------------------------------------------------- self-consistency
# Opus 4.8 rejects any non-default `temperature` (400) and is near-deterministic, so temperature
# jitter can't drive self-consistency. Instead we perturb the INPUT each run — resample+reorder
# the few-shot exemplars and lightly paraphrase the ticket — and measure agreement. This is
# model-agnostic and a stronger signal: it probes robustness to the input variation that actually
# occurs in production (phrasing, exemplar choice) rather than RNG noise.
N_SELF_CONSISTENCY: int = int(os.getenv("NOVIG_N", "5"))           # runs per ticket
PERTURB_EXEMPLARS_PER_RUN: int = int(os.getenv("NOVIG_PERTURB_K", "8"))  # subset of the pool per run
PERTURB_PARAPHRASE: bool = os.getenv("NOVIG_PARAPHRASE", "1") != "0"     # paraphrase the ticket per run
PARAPHRASE_MODEL: str = os.getenv("NOVIG_PARAPHRASE_MODEL", "claude-haiku-4-5-20251001")

DRAFT_TEMPERATURE: float = float(os.getenv("NOVIG_DRAFT_TEMP", "0.3"))  # Sonnet (drafter) accepts it

# Below this agreement rate the model is "not sure" → bias toward defer (safety == low
# confidence are the same tickets). Tuned against the held-out set; see prompt_iterations.md.
CONFIDENCE_THRESHOLD: float = float(os.getenv("NOVIG_CONF_THRESHOLD", "0.6"))

# ---------------------------------------------------------------------------- robustness
MAX_RETRIES: int = 2          # transient API errors; fail-closed if retries exhaust
REQUEST_TIMEOUT_S: float = 60.0
SEED: int = 7                 # seeds any incidental RNG; the split below is deterministic

# ---------------------------------------------------------------------------- held-out split
# Honest accuracy is measured on training tickets the classifier never sees as exemplars.
# These FEW_SHOT_IDS are the curated few-shot pool (used in the prompt); everything else in
# the train set is held out for scoring. Chosen to (a) cover all four hard no-draft types so
# the model learns to defer, and (b) DELIBERATELY leave the soft-rule cases — market dispute
# (029) vs non-dispute (016), jurisdiction (023), tiny balance discrepancy (018), deposit-limit
# (017), despair PG (020) — in the held-out set, because those are the judgments worth scoring.
FEW_SHOT_IDS: tuple[str, ...] = (
    "t_train_001",  # trading_mechanics · draft · low
    "t_train_002",  # deposits_withdrawals · draft · high
    "t_train_003",  # account_compromise · NO-DRAFT · escalate   (hard)
    "t_train_005",  # tax_documents · draft · medium
    "t_train_006",  # problem_gambling · NO-DRAFT · escalate      (hard)
    "t_train_008",  # kyc_verification · draft · high
    "t_train_009",  # legal_regulatory · NO-DRAFT · escalate      (hard)
    "t_train_011",  # account_access · draft · medium
    "t_train_012",  # legal_regulatory (minor) · NO-DRAFT · escalate (hard)
    "t_train_030",  # other · draft · low
)

# ---------------------------------------------------------------------------- paths
PREDICTIONS_PATH: Path = ROOT / "predictions.jsonl"
METRICS_PATH: Path = ROOT / "metrics.json"
ERROR_ANALYSIS_PATH: Path = ROOT / "error_analysis.md"
PROMPT_LOG_PATH: Path = ROOT / "prompt_iterations.md"
