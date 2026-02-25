"""Evaluation pipeline: perplexity, accuracy (for MCQ), ROUGE, and custom medical metrics."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling, PreTrainedModel, PreTrainedTokenizerBase

from src.evaluation.metrics import compute_accuracy, compute_rouge
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _compute_perplexity(
    model: PreTrainedModel,
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int = 4,
) -> float:
    """Compute perplexity on a tokenized dataset."""
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator)

    total_loss = 0.0
    total_tokens = 0

    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            num_tokens = (batch["labels"] != -100).sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    avg_loss = total_loss / max(total_tokens, 1)
    return math.exp(avg_loss)


def _generate_predictions(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    max_new_tokens: int = 128,
    batch_size: int = 4,
) -> list[str]:
    """Generate text completions for a list of prompts."""
    model.eval()
    predictions = []

    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
            )

        for j, ids in enumerate(output_ids):
            prompt_len = inputs["input_ids"][j].shape[0]
            generated = tokenizer.decode(ids[prompt_len:], skip_special_tokens=True)
            predictions.append(generated.strip())

    return predictions


def evaluate_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    dataset: Dataset,
    cfg: dict[str, Any],
    *,
    raw_test_data: Dataset | None = None,
) -> dict[str, float]:
    """Run full evaluation suite and return metrics dict.

    Parameters
    ----------
    raw_test_data : Dataset, optional
        The un-tokenized test split, used to extract prompts/references
        for generation-based metrics (accuracy, ROUGE).
    """
    results: dict[str, Any] = {}
    eval_batch_size = cfg.get("training", {}).get("per_device_eval_batch_size", 4)

    # --- Perplexity ---
    logger.info("Computing perplexity ...")
    ppl = _compute_perplexity(model, dataset, tokenizer, batch_size=eval_batch_size)
    results["perplexity"] = round(ppl, 4)
    logger.info("  Perplexity: %.4f", ppl)

    # --- Generation-based metrics (requires raw data) ---
    if raw_test_data is not None:
        prompts = []
        references = []

        for row in raw_test_data:
            question = row.get("question", row.get("QUESTION", ""))
            answer = str(row.get("answer", row.get("final_decision", row.get("LONG_ANSWER", ""))))
            prompts.append(f"### Question:\n{question}\n\n### Answer:\n")
            references.append(answer)

        if prompts:
            logger.info("Generating predictions for %d examples ...", len(prompts))
            predictions = _generate_predictions(
                model, tokenizer, prompts, max_new_tokens=128, batch_size=eval_batch_size
            )

            accuracy = compute_accuracy(predictions, references)
            results["accuracy"] = round(accuracy, 4)
            logger.info("  Accuracy: %.4f", accuracy)

            rouge = compute_rouge(predictions, references)
            results.update({f"rouge_{k}": round(v, 4) for k, v in rouge.items()})
            logger.info("  ROUGE: %s", rouge)

    # --- Save results ---
    output_dir = Path(cfg["training"]["output_dir"]) / "eval_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "metrics.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Evaluation results saved to %s", results_file)

    return results
