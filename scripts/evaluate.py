#!/usr/bin/env python3
"""Entry point: evaluate a fine-tuned QLoRA adapter."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets import load_dataset

from src.data import load_and_prepare_dataset
from src.evaluation import evaluate_model
from src.inference.generate import load_adapter_for_inference
from src.utils import load_config, merge_overrides, get_logger

logger = get_logger("evaluate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate QLoRA adapter")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--adapter-path", type=str, required=True, help="Path to saved adapter")
    parser.add_argument("--override", action="append", default=[], help="key=value overrides")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit test set size for quick eval")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    cfg = merge_overrides(cfg, args.override)

    model, tokenizer = load_adapter_for_inference(args.adapter_path)
    dataset = load_and_prepare_dataset(cfg, tokenizer)

    test_split = dataset.get("test", dataset.get("validation"))
    if test_split is None:
        logger.error("No test or validation split found.")
        sys.exit(1)

    # Load raw (un-tokenized) test data for generation-based metrics
    ds_cfg = cfg["dataset"]
    load_kwargs = {"path": ds_cfg["name"]}
    if ds_cfg.get("subset"):
        load_kwargs["name"] = ds_cfg["subset"]
    raw = load_dataset(**load_kwargs)
    raw_test = raw.get(ds_cfg.get("split_test", "test"))

    if args.max_samples and raw_test is not None:
        raw_test = raw_test.select(range(min(args.max_samples, len(raw_test))))
        test_split = test_split.select(range(min(args.max_samples, len(test_split))))

    results = evaluate_model(model, tokenizer, test_split, cfg, raw_test_data=raw_test)

    logger.info("=== Evaluation Results ===")
    for k, v in results.items():
        logger.info("  %s: %s", k, v)


if __name__ == "__main__":
    main()
