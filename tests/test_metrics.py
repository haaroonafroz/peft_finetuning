"""Tests for evaluation metrics."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.metrics import compute_accuracy, compute_rouge


def test_accuracy_exact():
    preds = ["Heart disease", "Cancer"]
    refs = ["Heart disease", "Cancer"]
    assert compute_accuracy(preds, refs) == 1.0


def test_accuracy_partial():
    preds = ["Heart disease is the leading cause", "Unknown"]
    refs = ["Heart disease", "Cancer"]
    assert compute_accuracy(preds, refs) == 0.5


def test_accuracy_with_prefix():
    preds = ["The answer is: Heart disease", "Answer: Cancer"]
    refs = ["heart disease", "cancer"]
    assert compute_accuracy(preds, refs) == 1.0


def test_rouge_identical():
    preds = ["The patient has diabetes"]
    refs = ["The patient has diabetes"]
    scores = compute_rouge(preds, refs)
    assert scores["1"] == 1.0
    assert scores["L"] == 1.0


def test_rouge_different():
    preds = ["hello world"]
    refs = ["goodbye universe"]
    scores = compute_rouge(preds, refs)
    assert scores["1"] < 1.0
