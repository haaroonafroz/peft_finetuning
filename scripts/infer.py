#!/usr/bin/env python3
"""Entry point: run inference with a fine-tuned QLoRA adapter."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inference.generate import batch_generate, generate_response, load_adapter_for_inference
from src.utils import load_config, merge_overrides, get_logger

logger = get_logger("infer")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with QLoRA adapter")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument("--adapter-path", type=str, required=True, help="Path to saved adapter")
    parser.add_argument("--prompt", type=str, default=None, help="Single prompt (interactive mode)")
    parser.add_argument("--input-file", type=str, default=None, help="JSONL file of prompts")
    parser.add_argument("--output-file", type=str, default=None, help="Output JSONL file")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    cfg = merge_overrides(cfg, args.override)

    model, tokenizer = load_adapter_for_inference(args.adapter_path)

    if args.input_file and args.output_file:
        batch_generate(
            model,
            tokenizer,
            args.input_file,
            args.output_file,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
    elif args.prompt:
        response = generate_response(
            model,
            tokenizer,
            args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        print(f"\n{'='*60}")
        print(f"Prompt: {args.prompt}")
        print(f"{'='*60}")
        print(f"Response:\n{response}")
        print(f"{'='*60}")
    else:
        logger.info("Interactive mode — type 'quit' to exit.")
        while True:
            try:
                prompt = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if prompt.lower() in ("quit", "exit", "q"):
                break
            response = generate_response(
                model,
                tokenizer,
                prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
            print(f"\nAssistant: {response}")


if __name__ == "__main__":
    main()
