"""Metric computation helpers."""

from __future__ import annotations

import re

from rouge_score import rouge_scorer


def _normalize(text: str) -> str:
    """Lowercase, strip whitespace, and remove common prefixes for comparison."""
    text = text.strip().lower()
    text = re.sub(r"^(answer|the answer is)[:\s]*", "", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.strip()


def compute_accuracy(predictions: list[str], references: list[str]) -> float:
    """Fuzzy accuracy: correct if the normalized prediction starts with the reference."""
    correct = 0
    for pred, ref in zip(predictions, references):
        pred_n = _normalize(pred)
        ref_n = _normalize(ref)
        if ref_n and (pred_n.startswith(ref_n) or ref_n in pred_n):
            correct += 1
    return correct / max(len(references), 1)


def compute_rouge(predictions: list[str], references: list[str]) -> dict[str, float]:
    """Compute ROUGE-1, ROUGE-2, and ROUGE-L F1 scores."""
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1, r2, rl = 0.0, 0.0, 0.0

    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        r1 += scores["rouge1"].fmeasure
        r2 += scores["rouge2"].fmeasure
        rl += scores["rougeL"].fmeasure

    n = max(len(predictions), 1)
    return {"1": r1 / n, "2": r2 / n, "L": rl / n}
