"""Dataset loading, formatting, and tokenization for medical QA tasks."""

from __future__ import annotations

from typing import Any

from datasets import DatasetDict, load_dataset
from transformers import PreTrainedTokenizerBase

from src.data.formatting import build_formatter
from src.utils.logging import get_logger

logger = get_logger(__name__)


def load_and_prepare_dataset(
    cfg: dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
) -> DatasetDict:
    """Load a HF dataset, apply prompt formatting, and tokenize.

    Returns a ``DatasetDict`` with ``train`` and ``validation`` splits ready
    for the ``SFTTrainer``.
    """
    ds_cfg = cfg["dataset"]
    logger.info("Loading dataset %s (subset=%s)", ds_cfg["name"], ds_cfg.get("subset"))

    load_kwargs: dict[str, Any] = {"path": ds_cfg["name"]}
    if ds_cfg.get("subset"):
        load_kwargs["name"] = ds_cfg["subset"]
    if ds_cfg.get("trust_remote_code"):
        load_kwargs["trust_remote_code"] = True

    raw = load_dataset(**load_kwargs)

    formatter = build_formatter(ds_cfg["name"], ds_cfg.get("prompt_template"))
    max_seq_length = ds_cfg.get("max_seq_length", 1024)

    def _tokenize(examples: dict[str, list]) -> dict[str, list]:
        texts = formatter(examples)
        out = tokenizer(
            texts,
            truncation=True,
            max_length=max_seq_length,
            padding=False,
        )
        out["labels"] = out["input_ids"].copy()
        return out

    splits = {}
    for split_key, cfg_key in [
        ("train", "split_train"),
        ("validation", "split_validation"),
        ("test", "split_test"),
    ]:
        source_split = ds_cfg.get(cfg_key, split_key)
        if source_split in raw:
            splits[split_key] = raw[source_split].map(
                _tokenize,
                batched=True,
                num_proc=ds_cfg.get("preprocessing_num_workers", 4),
                remove_columns=raw[source_split].column_names,
                desc=f"Tokenizing {split_key}",
            )
            logger.info("  %s: %d examples", split_key, len(splits[split_key]))
        else:
            logger.warning("  Split '%s' not found in dataset — skipping.", source_split)

    if "validation" not in splits and "train" in splits:
        logger.info("No validation split found — splitting train 90/10")
        split_result = splits["train"].train_test_split(
            test_size=0.1,
            seed=cfg.get("training", {}).get("seed", 42),
        )
        splits["train"] = split_result["train"]
        splits["validation"] = split_result["test"]
        logger.info("  train: %d examples, validation: %d examples",
                     len(splits["train"]), len(splits["validation"]))
    return DatasetDict(splits)
