"""Classification metrics with no third-party dependencies."""
from __future__ import annotations

from .prompts import INTENTS

INVALID = "<invalid>"


def evaluate_predictions(gold: list[str], pred: list[str | None]) -> dict:
    if len(gold) != len(pred) or not gold:
        raise ValueError("gold and pred must be non-empty and the same length")
    pred = [p if p is not None else INVALID for p in pred]
    labels = INTENTS + [INVALID]
    confusion = {g: {p: 0 for p in labels} for g in INTENTS}
    for g, p in zip(gold, pred):
        confusion[g][p] += 1
    per_class = {}
    for c in INTENTS:
        tp = confusion[c][c]
        fp = sum(confusion[g][c] for g in INTENTS if g != c)
        fn = sum(v for p, v in confusion[c].items() if p != c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_class[c] = {"precision": prec, "recall": rec, "f1": f1, "support": tp + fn}
    return {
        "n": len(gold),
        "accuracy": sum(g == p for g, p in zip(gold, pred)) / len(gold),
        "macro_f1": sum(v["f1"] for v in per_class.values()) / len(INTENTS),
        "invalid_rate": sum(p == INVALID for p in pred) / len(pred),
        "per_class": per_class,
        "confusion": confusion,
    }


def majority_baseline(train_labels: list[str], gold: list[str]) -> float:
    """Accuracy of always predicting the most common training label."""
    top = max(set(train_labels), key=train_labels.count)
    return sum(g == top for g in gold) / len(gold)


def format_report(name: str, m: dict) -> str:
    lines = [f"== {name} (n={m['n']}) ==",
             f"accuracy {m['accuracy']:.3f}   macro-F1 {m['macro_f1']:.3f}   invalid answers {m['invalid_rate']:.1%}"]
    for c, v in m["per_class"].items():
        lines.append(f"  {c:<20} P {v['precision']:.2f}  R {v['recall']:.2f}  F1 {v['f1']:.2f}  (n={v['support']})")
    return "\n".join(lines)
