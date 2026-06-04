"""Evaluation harness — the measurement framework the system is built inside.

Run it:  ``python eval.py``  (or ``make eval``)

What it does, in order:
  1. Splits the 30 labeled training tickets into a curated few-shot pool (config.FEW_SHOT_IDS)
     and a held-out remainder. Honest accuracy is computed ONLY on the held-out remainder, so
     the exemplars the classifier sees cannot leak into the score.
  2. Runs ``predict`` over the held-out tickets (label-blind) and computes:
        - category accuracy, urgency accuracy, urgency confusion matrix        [required]
        - false-draft rate on sensitive tickets  (the headline number; target 0) [required]
        - false-defer rate on safe tickets       (efficiency counter-metric)
        - draft-decision accuracy
        - confidence-predicts-error validation   (does low confidence find the mistakes?)
  3. Validates the deterministic gate over all 30 train tickets (recall on no-draft cases,
     false-positive rate on safe cases) — lazily, so this runs as soon as the gate exists.
  4. Runs ``predict`` over the 15 unlabeled eval tickets → writes ``predictions.jsonl`` in the
     exact taxonomy schema, in input order, and summarizes confidence + category drift vs train.
  5. Optionally scores the hand-written adversarial suite (lazily, once it exists).
  6. Writes ``metrics.json`` and a human-readable ``error_analysis.md`` dump of every miss.

Reproducibility: model, temperature, N, and the split are all pinned in src/config.py. The one
honest caveat is that the Anthropic API exposes no ``seed``, so classification sampling is not
bit-reproducible; we pin everything pinnable and rely on the self-consistency aggregate, which
is stable in practice. This is called out in the writeup.
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Optional

# Make ``src`` importable no matter the caller's cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import config
from src.data import load_eval, load_train
from src.schema import GoldLabel, Prediction, Ticket, Urgency

PredictFn = Callable[[Ticket], Prediction]

URGENCY_ORDER: list[Urgency] = [
    Urgency.LOW,
    Urgency.MEDIUM,
    Urgency.HIGH,
    Urgency.ESCALATE_IMMEDIATELY,
]


# ============================================================================ helpers


def _auc_confidence_predicts_correct(
    confidences: list[float], correct_flags: list[bool]
) -> Optional[float]:
    """AUC that confidence ranks correct predictions above incorrect ones (Mann-Whitney form).

    1.0 → confidence perfectly separates right from wrong; 0.5 → no signal. Returns None when
    one class is empty (e.g. zero errors), where the metric is undefined.
    """
    pos = [c for c, ok in zip(confidences, correct_flags) if ok]
    neg = [c for c, ok in zip(confidences, correct_flags) if not ok]
    if not pos or not neg:
        return None
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else 0.5 if p == n else 0.0
    return round(wins / (len(pos) * len(neg)), 4)


def _mean(xs: Iterable[float]) -> Optional[float]:
    xs = list(xs)
    return round(sum(xs) / len(xs), 4) if xs else None


def _rate(num: int, den: int) -> float:
    return round(num / den, 4) if den else 0.0


def _predict_label_blind(predict: PredictFn, ticket: Ticket) -> Prediction:
    """Predict on a copy with the gold label stripped — airtight against accidental leakage."""
    return predict(ticket.model_copy(update={"label": None}))


def _prompt_version() -> str:
    from src.classify import PROMPT_VERSION  # lazy: keeps the harness import-light

    return PROMPT_VERSION


# ============================================================================ metrics (pure)


def category_accuracy(rows: list[tuple[Prediction, GoldLabel]]) -> float:
    return _rate(sum(p.category == g.category for p, g in rows), len(rows))


def urgency_accuracy(rows: list[tuple[Prediction, GoldLabel]]) -> float:
    return _rate(sum(p.urgency == g.urgency for p, g in rows), len(rows))


def urgency_confusion_matrix(rows: list[tuple[Prediction, GoldLabel]]) -> dict:
    """Full 4x4 confusion matrix, gold (rows) x predicted (cols), zeros included."""
    labels = [u.value for u in URGENCY_ORDER]
    matrix = {gold: {pred: 0 for pred in labels} for gold in labels}
    for pred, gold in rows:
        matrix[gold.urgency.value][pred.urgency.value] += 1
    return {"labels": labels, "rows_are_gold": True, "matrix": matrix}


def false_draft_on_sensitive(rows: list[tuple[Prediction, GoldLabel]]) -> dict:
    """THE headline metric. Sensitive = the system should have declined (gold.should_draft
    False). A false draft = we drafted anyway. Target rate: 0.0."""
    sensitive = [(p, g) for p, g in rows if not g.should_draft]
    offenders = [p.ticket_id for p, g in sensitive if p.should_draft]
    return {
        "rate": _rate(len(offenders), len(sensitive)),
        "false_drafts": len(offenders),
        "n_sensitive": len(sensitive),
        "offending_ticket_ids": offenders,
    }


def false_defer_on_safe(rows: list[tuple[Prediction, GoldLabel]]) -> dict:
    """Efficiency counter-metric. Safe = gold.should_draft True. A false defer = we declined a
    ticket we could have drafted. High here means we're not capturing the easy wins."""
    safe = [(p, g) for p, g in rows if g.should_draft]
    offenders = [p.ticket_id for p, g in safe if not p.should_draft]
    return {
        "rate": _rate(len(offenders), len(safe)),
        "false_defers": len(offenders),
        "n_safe": len(safe),
        "offending_ticket_ids": offenders,
    }


def draft_decision_accuracy(rows: list[tuple[Prediction, GoldLabel]]) -> float:
    return _rate(sum(p.should_draft == g.should_draft for p, g in rows), len(rows))


def confidence_validation(rows: list[tuple[Prediction, GoldLabel]]) -> dict:
    """Does the confidence signal find the errors? Correctness here is COMPOSITE — a prediction is
    "correct" only if category, urgency, AND draft-decision all match gold — because category alone
    is often 100%, which would leave nothing to discriminate. Reports mean confidence on correct vs
    incorrect predictions, the AUC that confidence predicts correctness, and how many errors fall
    below the defer threshold. Read alongside the honest caveat: a near-deterministic model yields
    near-constant confidence, so this signal is weak on a clean 30-ticket set (see WRITEUP)."""
    correct = [
        p.category == g.category and p.urgency == g.urgency and p.should_draft == g.should_draft
        for p, g in rows
    ]
    confs = [p.confidence for p, _ in rows]
    n_err = correct.count(False)
    errs_below_threshold = sum(
        1 for (p, _), ok in zip(rows, correct)
        if not ok and p.confidence < config.CONFIDENCE_THRESHOLD
    )
    return {
        "correctness_basis": "category AND urgency AND draft-decision all match gold",
        "mean_confidence_when_correct": _mean(c for c, ok in zip(confs, correct) if ok),
        "mean_confidence_when_incorrect": _mean(c for c, ok in zip(confs, correct) if not ok),
        "auc_confidence_predicts_correct": _auc_confidence_predicts_correct(confs, correct),
        "errors_below_defer_threshold": errs_below_threshold,
        "total_errors": n_err,
        "share_of_errors_caught_by_low_confidence": _rate(errs_below_threshold, n_err),
    }


# ============================================================================ gate validation


def gate_validation(train: list[Ticket]) -> Optional[dict]:
    """Run the deterministic gate alone over all 30 train tickets. Lazily imported so the
    harness works before the gate exists. Reports recall on no-draft (sensitive) cases and
    false-positive rate on safe cases — the gate must catch the hard ones without over-tripping.
    """
    try:
        from src.safety_gate import evaluate_gate  # noqa: PLC0415 (intentional lazy import)
    except Exception:
        return None

    sensitive_ids = [t.ticket_id for t in train if t.label and not t.label.should_draft]
    safe_ids = [t.ticket_id for t in train if t.label and t.label.should_draft]
    tripped_ids = {t.ticket_id for t in train if evaluate_gate(t).tripped}

    missed = [tid for tid in sensitive_ids if tid not in tripped_ids]          # recall failures
    false_pos = [tid for tid in safe_ids if tid in tripped_ids]                # over-trips
    return {
        "n_train": len(train),
        "n_sensitive": len(sensitive_ids),
        "sensitive_recall": _rate(len(sensitive_ids) - len(missed), len(sensitive_ids)),
        "missed_sensitive_ids": missed,
        "n_safe": len(safe_ids),
        "false_positive_rate_on_safe": _rate(len(false_pos), len(safe_ids)),
        "false_positive_ids": false_pos,
    }


# ============================================================================ adversarial (lazy)


def adversarial_validation(predict: PredictFn) -> Optional[dict]:
    """Score the hand-written evasive suite through the full pipeline, if present. Lazy so the
    harness runs before the suite exists. Every adversarial ticket must route to no-draft."""
    try:
        from tests.adversarial_cases import ADVERSARIAL_TICKETS  # noqa: PLC0415
    except Exception:
        return None

    leaks = []
    for tk in ADVERSARIAL_TICKETS:
        pred = _predict_label_blind(predict, tk)
        if pred.should_draft:
            leaks.append(tk.ticket_id)
    n = len(ADVERSARIAL_TICKETS)
    return {
        "n_cases": n,
        "pass_rate": _rate(n - len(leaks), n),
        "leaked_ticket_ids": leaks,  # adversarial tickets that got auto-drafted (must be empty)
    }


# ============================================================================ eval-set summary


def eval_set_summary(predictions: list[Prediction], train: list[Ticket]) -> dict:
    """Best-effort signal on the UNLABELED eval set: confidence + category distribution drift."""
    train_dist = Counter(t.label.category.value for t in train if t.label)
    eval_dist = Counter(p.category.value for p in predictions)
    n_train = sum(train_dist.values()) or 1
    n_eval = len(predictions) or 1

    drift = {
        cat: round(eval_dist.get(cat, 0) / n_eval - train_dist.get(cat, 0) / n_train, 3)
        for cat in sorted(set(train_dist) | set(eval_dist))
    }
    return {
        "n": len(predictions),
        "n_drafted": sum(p.should_draft for p in predictions),
        "n_deferred": sum(not p.should_draft for p in predictions),
        "mean_confidence": _mean(p.confidence for p in predictions),
        "category_distribution": dict(eval_dist),
        "category_share_drift_vs_train": drift,
        "per_ticket": [
            {
                "ticket_id": p.ticket_id,
                "category": p.category.value,
                "urgency": p.urgency.value,
                "should_draft": p.should_draft,
                "confidence": p.confidence,
                "no_draft_reason": p.no_draft_reason,
            }
            for p in predictions
        ],
    }


# ============================================================================ orchestration


def run(predict: PredictFn) -> dict:
    """Execute the full evaluation and return the assembled metrics dict."""
    random.seed(config.SEED)
    train, ev = load_train(), load_eval()

    # ---- held-out honest accuracy ----------------------------------------------------------
    held_out = [t for t in train if t.ticket_id not in config.FEW_SHOT_IDS]
    rows: list[tuple[Prediction, GoldLabel]] = []
    held_out_preds: list[Prediction] = []
    for tk in held_out:
        pred = _predict_label_blind(predict, tk)
        held_out_preds.append(pred)
        assert tk.label is not None
        rows.append((pred, tk.label))

    held_out_metrics = {
        "n": len(rows),
        "category_accuracy": category_accuracy(rows),
        "urgency_accuracy": urgency_accuracy(rows),
        "urgency_confusion_matrix": urgency_confusion_matrix(rows),
        "false_draft_rate_on_sensitive": false_draft_on_sensitive(rows),
        "false_defer_rate_on_safe": false_defer_on_safe(rows),
        "draft_decision_accuracy": draft_decision_accuracy(rows),
        "confidence_validation": confidence_validation(rows),
        "per_ticket": [
            {
                "ticket_id": t.ticket_id,
                "confidence": p.confidence,
                "pred": {"category": p.category.value, "urgency": p.urgency.value, "should_draft": p.should_draft},
                "gold": {"category": t.label.category.value, "urgency": t.label.urgency.value, "should_draft": t.label.should_draft},
                "correct": (
                    p.category == t.label.category
                    and p.urgency == t.label.urgency
                    and p.should_draft == t.label.should_draft
                ),
            }
            for t, p in zip(held_out, held_out_preds)
        ],
    }

    # ---- eval set → predictions.jsonl ------------------------------------------------------
    eval_preds = [_predict_label_blind(predict, tk) for tk in ev]
    _write_predictions(eval_preds)

    metrics = {
        "meta": {
            "prompt_version": _prompt_version(),
            "classify_model": config.CLASSIFY_MODEL,
            "draft_model": config.DRAFT_MODEL,
            "paraphrase_model": config.PARAPHRASE_MODEL,
            "confidence_mechanism": "input-perturbation self-consistency (resample+reorder few-shot, paraphrase ticket)",
            "n_self_consistency": config.N_SELF_CONSISTENCY,
            "perturb_exemplars_per_run": config.PERTURB_EXEMPLARS_PER_RUN,
            "perturb_paraphrase": config.PERTURB_PARAPHRASE,
            "confidence_threshold": config.CONFIDENCE_THRESHOLD,
            "seed": config.SEED,
            "few_shot_ids": list(config.FEW_SHOT_IDS),
            "n_train": len(train),
            "n_held_out": len(held_out),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "reproducibility_note": (
                "Model/N/split/few-shot pool are pinned. Confidence comes from input perturbation, "
                "not sampling: Opus 4.8 rejects a non-default temperature and has no seed param, so "
                "runs are not bit-reproducible (the paraphrase step adds variation by design); the "
                "perturbed self-consistency aggregate is stable in practice."
            ),
        },
        "held_out_train": held_out_metrics,
        "gate_validation_train": gate_validation(train),
        "adversarial_suite": adversarial_validation(predict),
        "eval_set": eval_set_summary(eval_preds, train),
    }

    _write_metrics(metrics)
    _write_error_analysis(held_out, rows, metrics)
    return metrics


# ============================================================================ output writers


def _write_predictions(preds: list[Prediction]) -> None:
    with config.PREDICTIONS_PATH.open("w", encoding="utf-8") as fh:
        for p in preds:
            fh.write(json.dumps(p.to_jsonl_dict(), ensure_ascii=False) + "\n")


def _write_metrics(metrics: dict) -> None:
    config.METRICS_PATH.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _write_error_analysis(
    held_out: list[Ticket], rows: list[tuple[Prediction, GoldLabel]], metrics: dict
) -> None:
    """Programmatic dump of every held-out miss with a short cause note, plus gate misses."""
    by_id = {t.ticket_id: t for t in held_out}
    lines = ["# Error Analysis", "", "_Auto-generated by `eval.py`. Held-out misclassifications._", ""]
    misses = 0
    for pred, gold in rows:
        notes = []
        if pred.category != gold.category:
            notes.append(f"category {gold.category.value}→{pred.category.value}")
        if pred.urgency != gold.urgency:
            notes.append(f"urgency {gold.urgency.value}→{pred.urgency.value}")
        if pred.should_draft and not gold.should_draft:
            notes.append("**FALSE DRAFT on sensitive**")
        if not pred.should_draft and gold.should_draft:
            notes.append("false defer on safe")
        if not notes:
            continue
        misses += 1
        tk = by_id[pred.ticket_id]
        lines += [
            f"### {pred.ticket_id} — {tk.subject!r}",
            f"- issue: {'; '.join(notes)}",
            f"- gold: {gold.category.value} / {gold.urgency.value} / draft={gold.should_draft}",
            f"- pred: {pred.category.value} / {pred.urgency.value} / draft={pred.should_draft} "
            f"(confidence={pred.confidence})",
            "",
        ]
    header = f"_{misses} miss(es) across {len(rows)} held-out tickets._\n"
    lines.insert(3, header)

    gate = metrics.get("gate_validation_train")
    if gate and (gate["missed_sensitive_ids"] or gate["false_positive_ids"]):
        lines += ["## Gate validation issues", ""]
        if gate["missed_sensitive_ids"]:
            lines.append(f"- gate MISSED sensitive: {gate['missed_sensitive_ids']}")
        if gate["false_positive_ids"]:
            lines.append(f"- gate over-tripped safe: {gate['false_positive_ids']}")
        lines.append("")

    config.ERROR_ANALYSIS_PATH.write_text("\n".join(lines), encoding="utf-8")


# ============================================================================ console report


def _print_summary(metrics: dict) -> None:
    h = metrics["held_out_train"]
    fd = h["false_draft_rate_on_sensitive"]
    dd = h["false_defer_rate_on_safe"]
    cv = h["confidence_validation"]
    print("\n" + "=" * 70)
    print(f"  HELD-OUT TRAIN  (n={h['n']})")
    print("=" * 70)
    print(f"  category accuracy ............. {h['category_accuracy']:.1%}")
    print(f"  urgency  accuracy ............. {h['urgency_accuracy']:.1%}")
    print(f"  draft-decision accuracy ....... {h['draft_decision_accuracy']:.1%}")
    print(f"  FALSE-DRAFT on sensitive ...... {fd['rate']:.1%}  "
          f"({fd['false_drafts']}/{fd['n_sensitive']})  <-- target 0   {fd['offending_ticket_ids'] or ''}")
    print(f"  false-defer on safe ........... {dd['rate']:.1%}  ({dd['false_defers']}/{dd['n_safe']})")
    print(f"  confidence→correct AUC ........ {cv['auc_confidence_predicts_correct']}")
    g = metrics.get("gate_validation_train")
    if g:
        print("-" * 70)
        print(f"  GATE sensitive recall ......... {g['sensitive_recall']:.1%}  "
              f"(missed: {g['missed_sensitive_ids'] or 'none'})")
        print(f"  GATE false-positive on safe ... {g['false_positive_rate_on_safe']:.1%}  "
              f"({g['false_positive_ids'] or 'none'})")
    a = metrics.get("adversarial_suite")
    if a:
        print(f"  ADVERSARIAL pass rate ......... {a['pass_rate']:.1%}  "
              f"(leaked: {a['leaked_ticket_ids'] or 'none'})")
    e = metrics["eval_set"]
    print("-" * 70)
    print(f"  EVAL SET (n={e['n']}): drafted={e['n_drafted']}  deferred={e['n_deferred']}  "
          f"mean_conf={e['mean_confidence']}")
    print("=" * 70)
    print(f"  wrote {config.PREDICTIONS_PATH.name}, {config.METRICS_PATH.name}, "
          f"{config.ERROR_ANALYSIS_PATH.name}\n")


def main() -> None:
    from src.pipeline import predict  # imported here so the harness module stays import-light
    metrics = run(predict)
    _print_summary(metrics)


if __name__ == "__main__":
    main()
