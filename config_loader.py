# -*- coding: utf-8 -*-
"""
ComfyUI-AI-Photography-Toolkit - Configuration Loader

TOML configuration loader for provider settings, prompts, and filters.
Loads and caches configuration from TOML files for:
- Provider settings (tiers, stop strings, capabilities)
- Prompts (system/user prompts by tier and mode)
- Components (agentic analysis prompts)
- Filters (example text patterns to remove)

Uses Python 3.11+ tomllib or falls back to tomli package.

Author: Siddhartha Lahiri
Email: siddhartha.lahiri@gmail.com
License: MIT
"""

import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Try Python 3.11+ built-in, fall back to tomli
try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None


# =============================================================================
# Configuration Cache
# =============================================================================

_config_cache: Dict[str, Dict] = {}
_CONFIG_DIR = Path(__file__).parent / "config"
_PROMPTS_DIR = _CONFIG_DIR / "prompts"


def clear_config_cache(filename: str = None):
    """
    Clear the configuration cache to force reload from disk.

    Args:
        filename: Specific file to clear, or None to clear all.

    Usage:
        clear_config_cache()  # Clear all
        clear_config_cache("templates.toml")  # Clear specific file
        clear_config_cache("_modular_prompts")  # Clear modular prompts cache
    """
    global _config_cache
    if filename:
        if filename in _config_cache:
            del _config_cache[filename]
            print(f"[SID-Config] Cleared cache for: {filename}")
    else:
        _config_cache.clear()
        print("[SID-Config] Cleared all config cache - files will reload on next access")


def reload_all_configs():
    """Clear cache and force reload of all config files."""
    clear_config_cache()
    # Trigger reload of commonly used configs
    check_config_files()
    print("[SID-Config] All configs reloaded from disk")


def _load_toml(filename: str, force_reload: bool = False) -> Dict[str, Any]:
    """Load a TOML file from the config directory with caching.

    Args:
        filename: Name of the TOML file to load
        force_reload: If True, bypass cache and reload from disk
    """
    if not force_reload and filename in _config_cache:
        return _config_cache[filename]

    if tomllib is None:
        print(f"[SID-Config] Warning: tomllib/tomli not available, using defaults")
        return {}

    filepath = _CONFIG_DIR / filename
    if not filepath.exists():
        print(f"[SID-Config] Warning: Config file not found: {filepath}")
        return {}

    try:
        with open(filepath, "rb") as f:
            data = tomllib.load(f)
        _config_cache[filename] = data
        return data
    except Exception as e:
        print(f"[SID-Config] Error loading {filename}: {e}")
        return {}


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Deep merge two dictionaries, with override taking precedence."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_modular_prompts(force_reload: bool = False) -> Dict[str, Any]:
    """
    Load prompts from modular structure in config/prompts/ directory.
    Falls back to single prompts.toml if modular structure doesn't exist.

    Args:
        force_reload: If True, bypass cache and reload from disk
    """
    cache_key = "_modular_prompts"
    if not force_reload and cache_key in _config_cache:
        return _config_cache[cache_key]

    if tomllib is None:
        return {}

    merged_config: Dict[str, Any] = {}

    # Check if modular structure exists
    if _PROMPTS_DIR.exists() and _PROMPTS_DIR.is_dir():
        # Load in order: _base.toml first, then others
        toml_files = []

        # Base config first
        base_file = _PROMPTS_DIR / "_base.toml"
        if base_file.exists():
            toml_files.append(base_file)

        # Main prompt files (portrait, scene, vocabulary, classification)
        for filename in ["portrait.toml", "scene.toml", "vocabulary.toml", "classification.toml"]:
            filepath = _PROMPTS_DIR / filename
            if filepath.exists():
                toml_files.append(filepath)

        # Style-specific files from styles/ subdirectory
        styles_dir = _PROMPTS_DIR / "styles"
        if styles_dir.exists() and styles_dir.is_dir():
            for filepath in sorted(styles_dir.glob("*.toml")):
                toml_files.append(filepath)

        # Load and merge all files
        for filepath in toml_files:
            try:
                with open(filepath, "rb") as f:
                    data = tomllib.load(f)
                merged_config = _deep_merge(merged_config, data)
            except Exception as e:
                print(f"[SID-Config] Error loading {filepath.name}: {e}")

        if merged_config:
            _config_cache[cache_key] = merged_config
            return merged_config

    # Fallback to single prompts.toml
    fallback = _load_toml("prompts.toml")
    _config_cache[cache_key] = fallback
    return fallback


def _get_prompts_config() -> Dict[str, Any]:
    """
    Get prompts configuration - uses modular structure if available,
    otherwise falls back to prompts.toml.
    """
    # Try modular structure first
    modular = _load_modular_prompts()
    if modular:
        return modular

    # Fallback to single file
    return _load_toml("prompts.toml")


def reload_config():
    """Clear cache and reload all configs (useful for development)."""
    global _config_cache
    _config_cache = {}


# =============================================================================
# Provider Configuration
# =============================================================================

def get_provider_tier(provider: str) -> str:
    """Get the tier for a provider (advanced/standard/basic)."""
    config = _load_toml("providers.toml")
    tiers = config.get("tiers", {})
    return tiers.get(provider.lower(), "standard")


def get_stop_strings(provider: str) -> List[str]:
    """Get stop strings for a provider."""
    config = _load_toml("providers.toml")
    stop_strings = config.get("stop_strings", {})
    return stop_strings.get(provider.lower(), stop_strings.get("default", []))


def get_provider_capabilities(tier: str) -> Dict[str, Any]:
    """Get capabilities for a tier."""
    config = _load_toml("providers.toml")
    capabilities = config.get("capabilities", {})
    return capabilities.get(tier, capabilities.get("standard", {}))


def get_image_limit(provider: str) -> int:
    """Get max image size for a provider."""
    config = _load_toml("providers.toml")
    limits = config.get("image_limits", {})
    return limits.get(provider.lower(), limits.get("default", 1024))


# =============================================================================
# Prompt Configuration
# =============================================================================

def get_system_prompt(tier: str) -> str:
    """Get the base system prompt for a tier."""
    config = _get_prompts_config()
    system = config.get("system", {})
    tier_config = system.get(tier, system.get("standard", {}))
    return tier_config.get("base", "Describe this image for AI image generation.")


def get_style_addon(style: str, addon_type: str = "system_addon") -> str:
    """Get the style addon for system or user prompt."""
    config = _get_prompts_config()
    styles = config.get("styles", {})

    # Normalize style name
    style_key = style.lower().replace(" & ", "_").replace(" ", "_").replace("-", "_")
    style_map = {
        "auto_detect": "auto",
        "portrait": "portrait",
        "fashion_outfit": "fashion",
        "fashion": "fashion",
        "artistic_style": "artistic",
        "artistic": "artistic",
        "nsfw_detailed": "nsfw",
        "nsfw": "nsfw",
    }
    style_key = style_map.get(style_key, "auto")

    style_config = styles.get(style_key, {})
    return style_config.get(addon_type, "")


def get_user_prompt(tier: str, mode: str) -> str:
    """Get the user prompt for a tier and mode."""
    config = _get_prompts_config()
    user = config.get("user", {})
    tier_config = user.get(tier, user.get("standard", {}))
    mode_config = tier_config.get(mode.lower(), {})
    return mode_config.get("prompt", "Describe this image.")


def get_length_constraint(prompt_length: int) -> str:
    """
    Get the length constraint instruction for a prompt_length setting.

    Args:
        prompt_length: Target word count (0 = no limit, optimal Z-Image range is 80-250)

    Returns:
        Length constraint instruction string (empty for 0/unlimited)
    """
    if prompt_length <= 0:
        return ""  # No constraint

    return f"OUTPUT LENGTH: Write a prompt of approximately {prompt_length} words. Adjust detail level to match this target."


# =============================================================================
# Component Configuration
# =============================================================================

def get_component_prompt(component: str, tier: str) -> str:
    """Get the prompt for a specific component and tier."""
    config = _load_toml("components.toml")
    comp_config = config.get(component, {})
    tier_config = comp_config.get(tier, comp_config.get("standard", {}))
    return tier_config.get("prompt", "")


def get_component_name(component: str) -> str:
    """Get the display name for a component."""
    config = _load_toml("components.toml")
    comp_config = config.get(component, {})
    return comp_config.get("name", component.replace("_", " ").title())


def get_mode_config(mode: str) -> Dict[str, Any]:
    """Get the configuration for an analysis mode."""
    config = _load_toml("components.toml")
    modes = config.get("modes", {})
    return modes.get(mode.lower(), modes.get("standard", {}))


def get_mode_components(mode: str) -> List[str]:
    """Get the list of components for an analysis mode."""
    mode_config = get_mode_config(mode)
    return mode_config.get("components", ["framing", "ethnicity", "clothing"])


def get_all_components() -> List[str]:
    """Get list of all available component keys."""
    config = _load_toml("components.toml")
    # Filter out the 'modes' section
    return [k for k in config.keys() if k != "modes"]


# =============================================================================
# Filter Configuration
# =============================================================================

def get_example_patterns() -> List[str]:
    """Get regex patterns for example text to remove."""
    config = _load_toml("filters.toml")
    example = config.get("example_patterns", {})
    return example.get("patterns", [])


def get_markdown_patterns() -> List[str]:
    """Get regex patterns for markdown cleanup."""
    config = _load_toml("filters.toml")
    markdown = config.get("markdown_patterns", {})
    return markdown.get("patterns", [])


def get_whitespace_patterns() -> Dict[str, str]:
    """Get whitespace cleanup patterns."""
    config = _load_toml("filters.toml")
    return config.get("whitespace", {})


def get_provider_cleanup_patterns(provider: str) -> List[str]:
    """Get provider-specific cleanup patterns."""
    config = _load_toml("filters.toml")
    cleanup = config.get("provider_cleanup", {})
    provider_config = cleanup.get(provider.lower(), cleanup.get("default", {}))
    return provider_config.get("patterns", [])


# =============================================================================
# Output Cleaning Utility
# =============================================================================

def clean_output(text: str, provider: str = "default") -> str:
    """
    Clean LLM output using patterns from filters.toml.

    Args:
        text: Raw LLM output
        provider: Provider name for provider-specific cleanup

    Returns:
        Cleaned text
    """
    if not text:
        return ""

    # Remove example text patterns
    for pattern in get_example_patterns():
        try:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)
        except re.error:
            pass

    # Remove markdown patterns
    for pattern in get_markdown_patterns():
        try:
            # Handle patterns with capture groups (like bold/italic)
            if "(" in pattern and ")" in pattern:
                text = re.sub(pattern, r"\1", text, flags=re.MULTILINE)
            else:
                text = re.sub(pattern, "", text, flags=re.MULTILINE)
        except re.error:
            pass

    # Provider-specific cleanup
    for pattern in get_provider_cleanup_patterns(provider):
        try:
            text = re.sub(pattern, "", text, flags=re.MULTILINE)
        except re.error:
            pass

    # Whitespace cleanup
    ws = get_whitespace_patterns()
    if ws.get("collapse_newlines"):
        text = re.sub(ws["collapse_newlines"], " ", text)
    if ws.get("collapse_spaces"):
        text = re.sub(ws["collapse_spaces"], " ", text)
    if ws.get("double_commas"):
        text = re.sub(ws["double_commas"], ",", text)
    if ws.get("leading_comma"):
        text = re.sub(ws["leading_comma"], "", text)
    if ws.get("trailing_comma"):
        text = re.sub(ws["trailing_comma"], "", text)
    if ws.get("double_periods"):
        text = re.sub(ws["double_periods"], ".", text)

    return text.strip()


# =============================================================================
# Convenience Functions
# =============================================================================

def build_system_prompt(provider: str, preset_style: str, user_guidance: str = "", prompt_length: int = 150) -> str:
    """
    Build complete system prompt based on provider tier and style.

    Args:
        provider: LLM provider name
        preset_style: Selected style preset
        user_guidance: Optional user guidance text
        prompt_length: Target word count (0 = no limit, default 150 for optimal Z-Image)

    Returns:
        Complete system prompt string
    """
    tier = get_provider_tier(provider)
    base = get_system_prompt(tier)

    style_addon = get_style_addon(preset_style, "system_addon")
    if style_addon:
        base += f"\n\n{style_addon}"

    # Add length constraint
    length_constraint = get_length_constraint(prompt_length)
    if length_constraint:
        base += f"\n\n{length_constraint}"

    if user_guidance and user_guidance.strip():
        base += f"""

USER MODIFICATION REQUEST (CRITICAL - MUST APPLY):
"{user_guidance.strip()}"

This is a MODIFICATION directive. You MUST:
1. Apply this modification to your description, overriding what's literally visible
2. Describe the subject AS IF this modification is true
3. For clothing/nudity changes: describe the MODIFIED state (e.g., if "make topless" and image shows bikini, describe as topless)
4. Do NOT describe the original state that contradicts the user request
5. Start your description with the modified element"""

    return base


def build_user_prompt(provider: str, analysis_mode: str, preset_style: str, prompt_length: int = 150) -> str:
    """
    Build user prompt based on provider tier and analysis mode.

    Args:
        provider: LLM provider name
        analysis_mode: Analysis mode (Standard/Detailed)
        preset_style: Selected style preset
        prompt_length: Target word count (0 = no limit, default 150 for optimal Z-Image)

    Returns:
        Complete user prompt string
    """
    tier = get_provider_tier(provider)
    prompt = get_user_prompt(tier, analysis_mode)

    style_addon = get_style_addon(preset_style, "user_addon")
    if style_addon:
        prompt += f"\n\n{style_addon}"

    # Reinforce length constraint in user prompt (already in system, but reinforce)
    length_constraint = get_length_constraint(prompt_length)
    if length_constraint:
        prompt += f"\n\nREMEMBER: {length_constraint}"

    return prompt


# =============================================================================
# Scene-Only Prompts (No Human Keywords - Prevents Hallucination)
# =============================================================================

def get_scene_only_system_prompt() -> str:
    """Get the scene-only system prompt (no human/person keywords)."""
    config = _get_prompts_config()
    system = config.get("system", {})
    scene_config = system.get("scene_only", {})
    return scene_config.get("base", "Describe this scene for AI image generation. There are no people in this image.")


def get_scene_only_user_prompt(mode: str) -> str:
    """Get the scene-only user prompt for a mode (no human/person keywords)."""
    config = _get_prompts_config()
    user = config.get("user", {})
    scene_config = user.get("scene_only", {})
    mode_config = scene_config.get(mode.lower(), {})
    return mode_config.get("prompt", "Describe this scene/landscape/object.")


def build_scene_only_system_prompt(prompt_length: int = 150) -> str:
    """
    Build complete scene-only system prompt (no human/person keywords).
    Used when human detection returns NO to prevent hallucinating people.

    Args:
        prompt_length: Target word count (0 = no limit)

    Returns:
        Complete scene-only system prompt string
    """
    base = get_scene_only_system_prompt()

    # Add length constraint
    length_constraint = get_length_constraint(prompt_length)
    if length_constraint:
        base += f"\n\n{length_constraint}"

    return base


def build_scene_only_user_prompt(analysis_mode: str, prompt_length: int = 150) -> str:
    """
    Build complete scene-only user prompt (no human/person keywords).
    Used when human detection returns NO to prevent hallucinating people.

    Args:
        analysis_mode: Analysis mode (Standard/Detailed)
        prompt_length: Target word count (0 = no limit)

    Returns:
        Complete scene-only user prompt string
    """
    prompt = get_scene_only_user_prompt(analysis_mode)

    # Reinforce length constraint
    length_constraint = get_length_constraint(prompt_length)
    if length_constraint:
        prompt += f"\n\nREMEMBER: {length_constraint}"

    return prompt


# =============================================================================
# Human With Subject Prompts (Balanced - Human AND Pet/Vehicle/Landscape)
# =============================================================================

def get_human_with_subject_system_prompt() -> str:
    """Get the human_with_subject system prompt (balanced human + subject)."""
    config = _get_prompts_config()
    system = config.get("system", {})
    hws_config = system.get("human_with_subject", {})
    return hws_config.get("base", "Describe both the human and the other subject in this image with equal detail.")


def get_human_with_subject_user_prompt(mode: str) -> str:
    """Get the human_with_subject user prompt for a mode."""
    config = _get_prompts_config()
    user = config.get("user", {})
    hws_config = user.get("human_with_subject", {})
    mode_config = hws_config.get(mode.lower(), {})
    return mode_config.get("prompt", "Describe both the human and the other subject equally.")


def build_human_with_subject_system_prompt(prompt_length: int = 150) -> str:
    """
    Build complete human_with_subject system prompt.
    Used when image has both a human AND another subject (pet, vehicle, landscape).

    Args:
        prompt_length: Target word count (0 = no limit)

    Returns:
        Complete human_with_subject system prompt string
    """
    base = get_human_with_subject_system_prompt()

    # Add length constraint
    length_constraint = get_length_constraint(prompt_length)
    if length_constraint:
        base += f"\n\n{length_constraint}"

    return base


def build_human_with_subject_user_prompt(analysis_mode: str, prompt_length: int = 150) -> str:
    """
    Build complete human_with_subject user prompt.
    Used when image has both a human AND another subject.

    Args:
        analysis_mode: Analysis mode (Standard/Detailed)
        prompt_length: Target word count (0 = no limit)

    Returns:
        Complete human_with_subject user prompt string
    """
    prompt = get_human_with_subject_user_prompt(analysis_mode)

    # Reinforce length constraint
    length_constraint = get_length_constraint(prompt_length)
    if length_constraint:
        prompt += f"\n\nREMEMBER: {length_constraint}"

    return prompt


# =============================================================================
# Initialization Check
# =============================================================================

def check_config_files() -> Dict[str, bool]:
    """Check which config files exist and are loadable."""
    files = ["providers.toml", "prompts.toml", "components.toml", "filters.toml"]
    status = {}
    for f in files:
        filepath = _CONFIG_DIR / f
        status[f] = filepath.exists()
        if status[f]:
            try:
                _load_toml(f)
            except Exception:
                status[f] = False

    # Check modular prompts directory
    if _PROMPTS_DIR.exists():
        status["prompts/ (modular)"] = True
        modular_files = list(_PROMPTS_DIR.glob("*.toml"))
        styles_dir = _PROMPTS_DIR / "styles"
        if styles_dir.exists():
            modular_files.extend(styles_dir.glob("*.toml"))
        status["prompts/ file count"] = len(modular_files)
    else:
        status["prompts/ (modular)"] = False

    return status


def get_loaded_prompt_files() -> List[str]:
    """Get list of prompt files that were loaded (for debugging)."""
    files = []
    if _PROMPTS_DIR.exists() and _PROMPTS_DIR.is_dir():
        # Base config first
        base_file = _PROMPTS_DIR / "_base.toml"
        if base_file.exists():
            files.append(base_file.name)

        # Main prompt files
        for filename in ["portrait.toml", "scene.toml", "vocabulary.toml", "classification.toml"]:
            filepath = _PROMPTS_DIR / filename
            if filepath.exists():
                files.append(filename)

        # Style-specific files
        styles_dir = _PROMPTS_DIR / "styles"
        if styles_dir.exists() and styles_dir.is_dir():
            for filepath in sorted(styles_dir.glob("*.toml")):
                files.append(f"styles/{filepath.name}")

    if not files:
        files.append("prompts.toml (fallback)")

    return files


# Print status on import (for debugging)
if __name__ == "__main__":
    print("Config file status:")
    for file, value in check_config_files().items():
        if isinstance(value, bool):
            status = "OK" if value else "MISSING"
        else:
            status = str(value)
        print(f"  {file}: {status}")

    print("\nLoaded prompt files:")
    for f in get_loaded_prompt_files():
        print(f"  - {f}")

    print("\nProvider tiers:")
    for p in ["anthropic", "openai", "ollama", "lmstudio"]:
        print(f"  {p}: {get_provider_tier(p)}")

    print("\nSample system prompt (basic tier):")
    prompt = get_system_prompt("basic")
    print(prompt[:200] + "..." if len(prompt) > 200 else prompt)

    print("\nSample scene-only system prompt:")
    scene_prompt = get_scene_only_system_prompt()
    print(scene_prompt[:200] + "..." if len(scene_prompt) > 200 else scene_prompt)
