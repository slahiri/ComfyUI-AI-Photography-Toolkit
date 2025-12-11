"""
TOML Configuration Loader for SID Photography Toolkit

Loads and caches configuration from TOML files for:
- Provider settings (tiers, stop strings, capabilities)
- Prompts (system/user prompts by tier and mode)
- Components (agentic analysis prompts)
- Filters (example text patterns to remove)

Uses Python 3.11+ tomllib or falls back to tomli package.
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


def _load_toml(filename: str) -> Dict[str, Any]:
    """Load a TOML file from the config directory with caching."""
    if filename in _config_cache:
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
    config = _load_toml("prompts.toml")
    system = config.get("system", {})
    tier_config = system.get(tier, system.get("standard", {}))
    return tier_config.get("base", "Describe this image for AI image generation.")


def get_style_addon(style: str, addon_type: str = "system_addon") -> str:
    """Get the style addon for system or user prompt."""
    config = _load_toml("prompts.toml")
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
    config = _load_toml("prompts.toml")
    user = config.get("user", {})
    tier_config = user.get(tier, user.get("standard", {}))
    mode_config = tier_config.get(mode.lower(), {})
    return mode_config.get("prompt", "Describe this image.")


def get_agentic_prompt(tier: str, prompt_type: str = "intro") -> str:
    """Get agentic prompts (intro/synthesis) for a tier."""
    config = _load_toml("prompts.toml")
    agentic = config.get("agentic", {})
    tier_config = agentic.get(tier, agentic.get("standard", {}))
    return tier_config.get(prompt_type, "")


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

def build_system_prompt(provider: str, preset_style: str, user_guidance: str = "") -> str:
    """
    Build complete system prompt based on provider tier and style.

    Args:
        provider: LLM provider name
        preset_style: Selected style preset
        user_guidance: Optional user guidance text

    Returns:
        Complete system prompt string
    """
    tier = get_provider_tier(provider)
    base = get_system_prompt(tier)

    style_addon = get_style_addon(preset_style, "system_addon")
    if style_addon:
        base += f"\n\n{style_addon}"

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


def build_user_prompt(provider: str, analysis_mode: str, preset_style: str) -> str:
    """
    Build user prompt based on provider tier and analysis mode.

    Args:
        provider: LLM provider name
        analysis_mode: Analysis mode (Quick/Standard/Detailed/Extreme)
        preset_style: Selected style preset

    Returns:
        Complete user prompt string
    """
    tier = get_provider_tier(provider)
    prompt = get_user_prompt(tier, analysis_mode)

    style_addon = get_style_addon(preset_style, "user_addon")
    if style_addon:
        prompt += f"\n\n{style_addon}"

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
    return status


# Print status on import (for debugging)
if __name__ == "__main__":
    print("Config file status:")
    for file, exists in check_config_files().items():
        status = "OK" if exists else "MISSING"
        print(f"  {file}: {status}")

    print("\nProvider tiers:")
    for p in ["anthropic", "openai", "ollama", "lmstudio"]:
        print(f"  {p}: {get_provider_tier(p)}")

    print("\nSample system prompt (basic tier):")
    print(get_system_prompt("basic"))
