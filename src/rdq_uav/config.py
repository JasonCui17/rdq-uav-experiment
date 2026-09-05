from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    """Load YAML and apply dotted ``key=value`` command-line overrides."""
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    extends = config.pop("extends", None)
    if extends is not None:
        base_path = (config_path.parent / str(extends)).resolve()
        base_config = load_config(base_path)
        base_config.pop("_config_path", None)
        config = _deep_merge(base_config, config)
    config = copy.deepcopy(config)
    for expression in overrides or []:
        if "=" not in expression:
            raise ValueError(f"Override must be key=value, got: {expression}")
        dotted_key, raw_value = expression.split("=", 1)
        keys = dotted_key.split(".")
        target = config
        for key in keys[:-1]:
            if key not in target or not isinstance(target[key], dict):
                raise KeyError(f"Unknown config section in override: {dotted_key}")
            target = target[key]
        if keys[-1] not in target:
            raise KeyError(f"Unknown config key in override: {dotted_key}")
        target[keys[-1]] = yaml.safe_load(raw_value)
    config["_config_path"] = str(config_path)
    return config


def require_keys(mapping: dict[str, Any], keys: tuple[str, ...], context: str) -> None:
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise KeyError(f"Missing {context} config keys: {', '.join(missing)}")
