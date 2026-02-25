"""Utilities for pushing adapters and merged models to Hugging Face Hub."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi

from src.utils.logging import get_logger

logger = get_logger(__name__)


def push_adapter_to_hub(
    adapter_path: str | Path,
    repo_id: str,
    *,
    private: bool = False,
    commit_message: str = "Upload QLoRA adapter",
    token: str | None = None,
) -> str:
    """Push a saved PEFT adapter directory to the Hugging Face Hub.

    Returns the URL of the uploaded repo.
    """
    token = token or os.getenv("HF_TOKEN")
    if not token:
        raise EnvironmentError("HF_TOKEN is not set. Export it or pass --token.")

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, private=private, exist_ok=True)

    url = api.upload_folder(
        folder_path=str(adapter_path),
        repo_id=repo_id,
        commit_message=commit_message,
    )
    logger.info("Adapter pushed to https://huggingface.co/%s", repo_id)
    return url


def push_merged_model_to_hub(
    model: Any,
    tokenizer: Any,
    repo_id: str,
    *,
    private: bool = False,
    token: str | None = None,
) -> None:
    """Merge adapter into base model weights and push the full model."""
    token = token or os.getenv("HF_TOKEN")
    merged = model.merge_and_unload()
    merged.push_to_hub(repo_id, private=private, token=token)
    tokenizer.push_to_hub(repo_id, private=private, token=token)
    logger.info("Merged model pushed to https://huggingface.co/%s", repo_id)
