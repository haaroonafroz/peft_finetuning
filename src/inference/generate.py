"""Inference utilities for the fine-tuned QLoRA model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase
from transformers import BitsAndBytesConfig

from src.utils.logging import get_logger

logger = get_logger(__name__)

# for loading adapter from a 4-bit quantized model
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)


def load_adapter_for_inference(
    adapter_path: str | Path,
    *,
    device_map: str = "auto",
    torch_dtype: str = "auto",
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load a saved PEFT adapter for inference (no base-model config needed)."""
    logger.info("Loading adapter from %s", adapter_path)
    model = AutoPeftModelForCausalLM.from_pretrained(
        str(adapter_path),
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        quantization_config=bnb_config,
    )
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.eval()
    return model, tokenizer


def generate_response(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    *,
    max_new_tokens: int = 128,
    temperature: float = 0.3,
    top_p: float = 0.9,
    do_sample: bool = True,
) -> str:
    """Generate a single response for the given prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            stop_strings=["\n###", "\n\n###", "\n"],
            tokenizer=tokenizer,
        )

    prompt_len = inputs["input_ids"].shape[1]
    return tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True).strip()


def batch_generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    input_file: str | Path,
    output_file: str | Path,
    *,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
) -> None:
    """Read prompts from a JSONL file, generate responses, write to output JSONL.

    Each input line should be ``{"prompt": "..."}`` (and may contain other fields
    which are passed through).
    """
    input_path = Path(input_file)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    with open(input_path) as f:
        for line in f:
            records.append(json.loads(line))

    logger.info("Generating responses for %d prompts ...", len(records))

    with open(output_path, "w") as fout:
        for rec in records:
            prompt = rec["prompt"]
            response = generate_response(
                model,
                tokenizer,
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            rec["response"] = response
            fout.write(json.dumps(rec) + "\n")

    logger.info("Results written to %s", output_path)
