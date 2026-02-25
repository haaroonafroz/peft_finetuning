"""Hierarchical YAML configuration loader with CLI override support."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (mutates base)."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load default config, then overlay a user-supplied YAML on top."""
    with open(_DEFAULT_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    if config_path is not None:
        with open(config_path) as f:
            user_cfg = yaml.safe_load(f) or {}
        cfg = _deep_merge(cfg, user_cfg)

    return cfg


def merge_overrides(cfg: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply dotted CLI overrides like ``training.learning_rate=1e-5``."""
    cfg = copy.deepcopy(cfg)
    for item in overrides:
        key, value = item.split("=", 1)
        parts = key.split(".")
        target = cfg
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = _auto_cast(value)
    return cfg


def _auto_cast(value: str) -> Any:
    """Best-effort cast of CLI string to Python type."""
    if value.lower() in ("true", "yes"):
        return True
    if value.lower() in ("false", "no"):
        return False
    if value.lower() in ("null", "none", "~"):
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value
