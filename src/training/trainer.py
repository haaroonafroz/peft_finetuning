"""Training loop setup using Hugging Face ``SFTTrainer`` / ``Trainer``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from datasets import DatasetDict
from transformers import (
    DataCollatorForLanguageModeling,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

from src.utils.logging import get_logger

logger = get_logger(__name__)


def _build_training_args(cfg: dict[str, Any]) -> TrainingArguments:
    t = cfg["training"]
    output_dir = t.get("output_dir", "outputs/default")

    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=t.get("num_train_epochs", 3),
        per_device_train_batch_size=t.get("per_device_train_batch_size", 2),
        per_device_eval_batch_size=t.get("per_device_eval_batch_size", 2),
        gradient_accumulation_steps=t.get("gradient_accumulation_steps", 8),
        learning_rate=float(t.get("learning_rate", 2e-4)),
        weight_decay=t.get("weight_decay", 0.01),
        warmup_ratio=t.get("warmup_ratio", 0.06),
        lr_scheduler_type=t.get("lr_scheduler_type", "cosine"),
        max_grad_norm=t.get("max_grad_norm", 1.0),
        logging_steps=t.get("logging_steps", 10),
        eval_strategy=t.get("eval_strategy", "steps"),
        eval_steps=t.get("eval_steps", 100),
        save_strategy=t.get("save_strategy", "steps"),
        save_steps=t.get("save_steps", 100),
        save_total_limit=t.get("save_total_limit", 3),
        load_best_model_at_end=t.get("load_best_model_at_end", True),
        metric_for_best_model=t.get("metric_for_best_model", "eval_loss"),
        greater_is_better=t.get("greater_is_better", False),
        bf16=t.get("bf16", True),
        fp16=t.get("fp16", False),
        gradient_checkpointing=t.get("gradient_checkpointing", True),
        optim=t.get("optim", "paged_adamw_8bit"),
        dataloader_pin_memory=t.get("dataloader_pin_memory", True),
        dataloader_num_workers=t.get("dataloader_num_workers", 2),
        report_to=t.get("report_to", "none"),
        seed=t.get("seed", 42),
        remove_unused_columns=False,
    )


def build_trainer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    dataset: DatasetDict,
    cfg: dict[str, Any],
) -> Trainer:
    """Construct a ``Trainer`` wired up with the model, data, and config."""
    training_args = _build_training_args(cfg)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset.get("train"),
        eval_dataset=dataset.get("validation"),
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    return trainer


def run_training(trainer: Trainer, cfg: dict[str, Any]) -> Path:
    """Execute training and save the final adapter.

    Returns the path to the saved adapter directory.
    """
    logger.info("Starting training ...")
    trainer.train()

    output_dir = Path(cfg["training"]["output_dir"]) / "final_adapter"
    output_dir.mkdir(parents=True, exist_ok=True)

    trainer.model.save_pretrained(str(output_dir))
    trainer.tokenizer.save_pretrained(str(output_dir))
    logger.info("Adapter saved to %s", output_dir)

    return output_dir
