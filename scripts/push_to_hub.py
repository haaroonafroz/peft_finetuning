#!/usr/bin/env python3
"""Entry point: push a trained adapter to Hugging Face Hub."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import push_adapter_to_hub, get_logger

logger = get_logger("push_to_hub")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Push adapter to HF Hub")
    parser.add_argument("--adapter-path", type=str, required=True, help="Path to saved adapter directory")
    parser.add_argument("--repo-id", type=str, required=True, help="HF repo id, e.g. username/model-name")
    parser.add_argument("--private", type=str, default="false", choices=["true", "false"])
    parser.add_argument("--token", type=str, default=None, help="HF token (or set HF_TOKEN env var)")
    parser.add_argument("--commit-message", type=str, default="Upload QLoRA adapter")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    is_private = args.private.lower() == "true"

    url = push_adapter_to_hub(
        adapter_path=args.adapter_path,
        repo_id=args.repo_id,
        private=is_private,
        commit_message=args.commit_message,
        token=args.token,
    )
    logger.info("Done! Repo URL: %s", url)


if __name__ == "__main__":
    main()
