"""Configuration loader for SID Photography Toolkit."""

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

# Config directory path
_CONFIG_DIR = Path(__file__).parent.parent / "config"


@lru_cache(maxsize=16)
def load_config(name: str) -> dict[str, Any]:
    """
    Load a JSON config file by name.

    Args:
        name: Config name (without .json extension)

    Returns:
        Parsed JSON as dict

    Raises:
        FileNotFoundError: If config file doesn't exist
    """
    config_path = _CONFIG_DIR / f"{name}.json"

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_model_config(model_name: str) -> dict[str, Any]:
    """Get configuration for a specific model."""
    return load_config(model_name)


def get_prompt(model_name: str, mode: str) -> str:
    """
    Get prompt for a model and mode.

    Args:
        model_name: Model name (e.g., "florence")
        mode: Caption mode (e.g., "detailed", "tags")

    Returns:
        Prompt string
    """
    config = load_config(model_name)
    prompts = config.get("prompts", {})
    return prompts.get(mode, prompts.get("detailed", ""))


def reload_configs() -> None:
    """Clear config cache and reload from disk."""
    load_config.cache_clear()
