"""Tests for configuration loading and override logic."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config, merge_overrides


def test_load_default_config():
    cfg = load_config()
    assert "model" in cfg
    assert "quantization" in cfg
    assert "lora" in cfg
    assert "dataset" in cfg
    assert "training" in cfg


def test_load_custom_config():
    cfg = load_config("configs/qwen3_medqa.yaml")
    assert cfg["model"]["name_or_path"] == "Qwen/Qwen3-8B"
    assert cfg["training"]["output_dir"] == "outputs/qwen3-medqa"
    # Should still have defaults for un-overridden keys
    assert cfg["lora"]["r"] == 64


def test_merge_overrides():
    cfg = load_config()
    cfg = merge_overrides(cfg, [
        "training.learning_rate=1e-5",
        "training.num_train_epochs=5",
        "lora.r=32",
        "training.bf16=false",
    ])
    assert cfg["training"]["learning_rate"] == 1e-5
    assert cfg["training"]["num_train_epochs"] == 5
    assert cfg["lora"]["r"] == 32
    assert cfg["training"]["bf16"] is False


def test_auto_cast_types():
    cfg = load_config()
    cfg = merge_overrides(cfg, [
        "training.seed=123",
        "training.bf16=true",
        "model.name_or_path=some/model",
        "lora.target_modules=null",
    ])
    assert cfg["training"]["seed"] == 123
    assert cfg["training"]["bf16"] is True
    assert cfg["model"]["name_or_path"] == "some/model"
    assert cfg["lora"]["target_modules"] is None
