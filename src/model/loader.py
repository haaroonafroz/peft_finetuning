"""Model and tokenizer loading with QLoRA (4-bit quantization + LoRA)."""

from __future__ import annotations

import os
from typing import Any

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from accelerate import Accelerator
from src.utils.logging import get_logger

logger = get_logger(__name__)

_DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "auto": "auto",
}


def _build_bnb_config(q_cfg: dict[str, Any]) -> BitsAndBytesConfig:
    compute_dtype = _DTYPE_MAP.get(q_cfg.get("bnb_4bit_compute_dtype", "bfloat16"), torch.bfloat16)
    return BitsAndBytesConfig(
        load_in_4bit=q_cfg.get("load_in_4bit", True),
        bnb_4bit_quant_type=q_cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=q_cfg.get("bnb_4bit_use_double_quant", True),
    )


def _build_lora_config(lora_cfg: dict[str, Any]) -> LoraConfig:
    task_type_str = lora_cfg.get("task_type", "CAUSAL_LM")
    task_type = getattr(TaskType, task_type_str, TaskType.CAUSAL_LM)

    return LoraConfig(
        r=lora_cfg.get("r", 64),
        lora_alpha=lora_cfg.get("lora_alpha", 128),
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        bias=lora_cfg.get("bias", "none"),
        task_type=task_type,
        target_modules=lora_cfg.get("target_modules"),
    )


def load_model_and_tokenizer(
    cfg: dict[str, Any],
    *,
    for_inference: bool = False,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load a quantized model with LoRA adapters attached.

    When ``for_inference=True``, the model is loaded in eval mode and
    gradient checkpointing is disabled.
    """
    model_cfg = cfg["model"]
    model_name = model_cfg["name_or_path"]
    token = os.getenv("HF_TOKEN")

    logger.info("Loading tokenizer: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=model_cfg.get("trust_remote_code", False),
        token=token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    bnb_config = _build_bnb_config(cfg.get("quantization", {}))
    torch_dtype = _DTYPE_MAP.get(model_cfg.get("torch_dtype", "auto"), "auto")

    logger.info("Loading model: %s (4-bit quantized)", model_name)
    model_kwargs: dict[str, Any] = {
        "pretrained_model_name_or_path": model_name,
        "quantization_config": bnb_config,
        "torch_dtype": torch_dtype,
        "device_map": {"": int(os.environ.get("LOCAL_RANK", 0))},
        "trust_remote_code": model_cfg.get("trust_remote_code", False),
        "token": token,
    }
    attn_impl = model_cfg.get("attn_implementation")
    if attn_impl:
        model_kwargs["attn_implementation"] = attn_impl

    model = AutoModelForCausalLM.from_pretrained(**model_kwargs)

    if not for_inference:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=cfg.get("training", {}).get("gradient_checkpointing", True),
        )

    lora_config = _build_lora_config(cfg.get("lora", {}))
    model = get_peft_model(model, lora_config)

    trainable, total = model.get_nb_trainable_parameters()
    logger.info(
        "Trainable params: %s / %s (%.2f%%)",
        f"{trainable:,}",
        f"{total:,}",
        100 * trainable / total,
    )

    if for_inference:
        model.eval()

    return model, tokenizer
