#!/usr/bin/env python3
"""Entry point: fine-tune a model with QLoRA on a medical dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_and_prepare_dataset
from src.model import load_model_and_tokenizer
from src.training import build_trainer, run_training
from src.utils import load_config, merge_overrides, get_logger

logger = get_logger("train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QLoRA Fine-Tuning")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--override", action="append", default=[], help="key=value overrides (repeatable)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    cfg = merge_overrides(cfg, args.override)

    logger.info("Output directory: %s", cfg["training"]["output_dir"])

    model, tokenizer = load_model_and_tokenizer(cfg)
    dataset = load_and_prepare_dataset(cfg, tokenizer)
    trainer = build_trainer(model, tokenizer, dataset, cfg)
    adapter_path = run_training(trainer, cfg)

    logger.info("Training complete. Adapter saved to: %s", adapter_path)


if __name__ == "__main__":
    main()
