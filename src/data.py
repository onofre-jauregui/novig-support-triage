"""Load the provided ticket datasets into typed ``Ticket`` objects.

Paths are resolved relative to the repo root so this works regardless of the caller's cwd.
Training tickets carry their ``label``; eval tickets do not (``label is None``).
"""

from __future__ import annotations

import json
from pathlib import Path

from .schema import Ticket

DATA_DIR: Path = Path(__file__).resolve().parent.parent / "data"
TRAIN_PATH: Path = DATA_DIR / "tickets_train.jsonl"
EVAL_PATH: Path = DATA_DIR / "tickets_eval.jsonl"


def load_tickets(path: Path) -> list[Ticket]:
    """Parse a .jsonl file into ``Ticket`` objects, preserving file order.

    Order matters: ``predictions.jsonl`` must be emitted in the same order as the eval input.
    Raises loudly (ValueError) on a malformed line rather than skipping it — a dropped ticket
    is a silent data-loss bug, which this system never tolerates.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Ticket data not found at {path}.\n"
            "Novig's tickets_train.jsonl / tickets_eval.jsonl are confidential and are not committed "
            "to this repo. Copy the two files you received into the data/ directory to run:\n"
            f"    cp /path/to/tickets_train.jsonl /path/to/tickets_eval.jsonl {path.parent}/\n"
            "(See the README → 'Provided datasets'. The committed predictions.jsonl + metrics.json "
            "already show the results without re-running.)"
        )

    tickets: list[Ticket] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        raw = raw.strip()
        if not raw:
            continue
        try:
            tickets.append(Ticket.model_validate(json.loads(raw)))
        except Exception as exc:  # noqa: BLE001 — re-raised with context, never swallowed
            raise ValueError(f"{path.name}:{lineno} failed to parse: {exc}") from exc
    return tickets


def load_train() -> list[Ticket]:
    """The 30 labeled training tickets."""
    return load_tickets(TRAIN_PATH)


def load_eval() -> list[Ticket]:
    """The 15 unlabeled eval tickets."""
    return load_tickets(EVAL_PATH)
