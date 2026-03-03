"""Metric computation helpers."""

from __future__ import annotations

import re

from rouge_score import rouge_scorer
from transformers import AutoTokenizer, AutoModel
import torch


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


def compute_semantic_similarity(
    predictions: list[str],
    references: list[str],
    model_name: str = "NeuML/pubmedbert-base-embeddings",
    batch_size: int = 32,
) -> float:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    def encode(texts):
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0]  # CLS token
            embeddings = torch.nn.functional.normalize(embeddings, dim=1)
            all_embeddings.append(embeddings)
        return torch.cat(all_embeddings, dim=0)

    pred_embs = encode(predictions)
    ref_embs = encode(references)
    similarities = (pred_embs * ref_embs).sum(dim=1)
    return similarities.mean().item()