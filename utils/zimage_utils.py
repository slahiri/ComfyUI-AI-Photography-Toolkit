"""
Z-Image utility functions for ComfyUI-AI-Photography-Toolkit.
Provides config loading, caching, and output cleaning utilities.
"""

import os
import re
import json
import hashlib
from pathlib import Path
from typing import Any, Optional

import yaml
import numpy as np


# =============================================================================
# CONFIG LOADING
# =============================================================================

_CONFIG_CACHE: dict[str, Any] = {}


def get_config_path() -> Path:
    """Get the path to the config directory."""
    return Path(__file__).parent.parent / "config"


def load_zimage_config(force_reload: bool = False) -> dict[str, Any]:
    """
    Load the Z-Image generator configuration from YAML.

    Args:
        force_reload: Force reload from disk even if cached

    Returns:
        Configuration dictionary
    """
    global _CONFIG_CACHE

    cache_key = "zimage_generator"

    if not force_reload and cache_key in _CONFIG_CACHE:
        return _CONFIG_CACHE[cache_key]

    config_path = get_config_path() / "zimage_generator_config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    _CONFIG_CACHE[cache_key] = config
    return config


def get_shot_framings() -> dict[str, dict]:
    """Get all shot framing definitions."""
    config = load_zimage_config()
    return config.get("shot_framings", {})


def get_photography_genres() -> dict[str, dict]:
    """Get all photography genre definitions, flattened."""
    config = load_zimage_config()
    genres = config.get("photography_genres", {})

    # Flatten the nested structure
    flattened = {}
    for category, category_genres in genres.items():
        for code, genre_data in category_genres.items():
            genre_data["category"] = category
            flattened[code] = genre_data

    return flattened


def get_genre_options() -> list[str]:
    """Get list of genre codes for UI dropdown."""
    genres = get_photography_genres()
    return list(genres.keys())


def get_attribute_schema(attribute_name: str) -> dict[str, Any]:
    """Get the schema for a specific attribute."""
    config = load_zimage_config()
    attributes = config.get("attributes", {})
    return attributes.get(attribute_name, {})


def get_content_detail_schema(level: str) -> dict[str, Any]:
    """Get the attribute schema for a specific content detail level."""
    config = load_zimage_config()
    levels = config.get("content_detail_levels", {})
    return levels.get(level, levels.get("standard", {}))


def get_prompt_template(stage: str) -> str:
    """Get the prompt template for a specific stage."""
    config = load_zimage_config()
    templates = config.get("prompt_templates", {})
    stage_template = templates.get(stage, {})
    return stage_template.get("system", "")


def get_focus_override_config(override_name: str) -> Optional[dict]:
    """Get configuration for a focus override."""
    config = load_zimage_config()
    overrides = config.get("focus_overrides", {})
    return overrides.get(override_name)


def get_detail_level_config(level: str) -> dict[str, Any]:
    """Get configuration for a detail level."""
    config = load_zimage_config()
    levels = config.get("detail_levels", {})
    return levels.get(level, levels.get("Standard", {}))


def get_zimage_settings() -> dict[str, Any]:
    """Get Z-Image recommended settings."""
    config = load_zimage_config()
    return config.get("zimage_settings", {})


# =============================================================================
# OUTPUT CLEANING
# =============================================================================

def clean_zimage_output(text: str, max_length: int = 2000) -> str:
    """
    Clean LLM output for Z-Image compatibility.

    Args:
        text: Raw LLM output
        max_length: Maximum character length (0 = unlimited)

    Returns:
        Cleaned text suitable for Z-Image
    """
    if not text:
        return ""

    # Remove thinking tags
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL)

    # Remove markdown code blocks
    text = re.sub(r"```[\s\S]*?```", "", text)
    text = re.sub(r"`[^`]+`", "", text)

    # Remove common prefixes
    prefixes_to_remove = [
        r"^Here['']?s? (?:is )?(?:the |a |an )?(?:enhanced |detailed |generated )?prompt[:\s]*",
        r"^(?:The )?(?:enhanced |detailed |generated )?prompt[:\s]*",
        r"^Sure[,!]? (?:here['']?s? )?(?:the |a )?prompt[:\s]*",
        r"^I['']?ve (?:generated|created|written)[:\s]*",
    ]
    for prefix in prefixes_to_remove:
        text = re.sub(prefix, "", text, flags=re.IGNORECASE)

    # Remove meta-tags that Z-Image doesn't need
    meta_tags = [
        r"\b8[kK]\b",
        r"\bmasterpiece\b",
        r"\bbest quality\b",
        r"\bhigh quality\b",
        r"\bultra quality\b",
        r"\b4[kK]\b",
        r"\bUHD\b",
        r"\bhighly detailed\b",
        r"\bsuper detailed\b",
    ]
    for tag in meta_tags:
        text = re.sub(tag + r",?\s*", "", text, flags=re.IGNORECASE)

    # Clean up whitespace
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    # Remove leading/trailing commas
    text = text.strip(",").strip()

    # Clean up double commas
    text = re.sub(r",\s*,", ",", text)

    # Truncate if needed (at sentence boundary)
    if max_length > 0 and len(text) > max_length:
        text = truncate_at_sentence(text, max_length)

    return text


def truncate_at_sentence(text: str, max_length: int) -> str:
    """
    Truncate text at the nearest sentence boundary before max_length.

    Args:
        text: Text to truncate
        max_length: Maximum length

    Returns:
        Truncated text ending at a sentence boundary
    """
    if len(text) <= max_length:
        return text

    # Find sentence boundaries before max_length
    truncated = text[:max_length]

    # Look for sentence-ending punctuation
    last_period = truncated.rfind(". ")
    last_exclaim = truncated.rfind("! ")
    last_question = truncated.rfind("? ")

    # Find the latest sentence boundary
    boundary = max(last_period, last_exclaim, last_question)

    if boundary > max_length * 0.5:  # Only use if at least halfway through
        return truncated[:boundary + 1].strip()

    # Fallback: truncate at last space
    last_space = truncated.rfind(" ")
    if last_space > max_length * 0.7:
        return truncated[:last_space].strip() + "..."

    return truncated.strip() + "..."


# =============================================================================
# CACHING UTILITIES (Persistent Disk Cache)
# =============================================================================

# In-memory cache for fast repeated access within session
_MEMORY_CACHE: dict[str, Any] = {}
_MAX_MEMORY_CACHE_SIZE = 50

# Disk cache settings
_MAX_DISK_CACHE_SIZE = 500  # Max number of cached entries on disk
_CACHE_VERSION = "1"  # Increment to invalidate old caches


def get_cache_dir() -> Path:
    """Get the cache directory path, creating it if needed."""
    cache_dir = Path(__file__).parent.parent / "cache" / "prompts"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def hash_image_tensor(image_tensor) -> str:
    """
    Generate a hash from an image tensor for caching.

    Args:
        image_tensor: ComfyUI image tensor (B, H, W, C)

    Returns:
        Hex string hash of the image
    """
    # Take first image if batch
    if len(image_tensor.shape) == 4:
        img_np = image_tensor[0].cpu().numpy()
    else:
        img_np = image_tensor.cpu().numpy()

    # Downsample for faster hashing (every 4th pixel)
    downsampled = img_np[::4, ::4, :]

    # Create hash from bytes
    img_bytes = downsampled.tobytes()
    return hashlib.md5(img_bytes).hexdigest()


def get_cache_key(
    image_hash: str,
    seed: int,
    **params
) -> str:
    """
    Generate a cache key from image hash and parameters.

    Args:
        image_hash: Hash of the input image
        seed: Random seed
        **params: All other parameters that affect output

    Returns:
        Cache key string
    """
    # Sort params for consistent ordering
    sorted_params = sorted(params.items())
    param_str = json.dumps(sorted_params, sort_keys=True)
    param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]

    return f"v{_CACHE_VERSION}_{image_hash}_{seed}_{param_hash}"


def _get_cache_file_path(cache_key: str) -> Path:
    """Get the file path for a cache entry."""
    # Use first 2 chars as subdirectory for better file distribution
    subdir = cache_key[:2] if len(cache_key) >= 2 else "00"
    cache_dir = get_cache_dir() / subdir
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{cache_key}.json"


def get_cached_output(cache_key: str) -> Optional[dict]:
    """
    Get cached output if it exists (checks memory first, then disk).

    Args:
        cache_key: The cache key to look up

    Returns:
        Cached output dict or None if not found
    """
    global _MEMORY_CACHE

    # Check memory cache first (fast path)
    if cache_key in _MEMORY_CACHE:
        return _MEMORY_CACHE[cache_key]

    # Check disk cache
    cache_file = _get_cache_file_path(cache_key)
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Validate cache version
            if data.get("_cache_version") == _CACHE_VERSION:
                output = data.get("output")
                # Store in memory cache for faster subsequent access
                _MEMORY_CACHE[cache_key] = output
                return output
            else:
                # Old cache version, delete it
                cache_file.unlink(missing_ok=True)
        except (json.JSONDecodeError, IOError):
            # Corrupted cache file, delete it
            cache_file.unlink(missing_ok=True)

    return None


def set_cached_output(cache_key: str, output: dict) -> None:
    """
    Cache an output result (both memory and disk).

    Args:
        cache_key: The cache key
        output: The output dict to cache
    """
    global _MEMORY_CACHE

    # Store in memory cache
    if len(_MEMORY_CACHE) >= _MAX_MEMORY_CACHE_SIZE:
        # Remove oldest entries (first 20%)
        keys_to_remove = list(_MEMORY_CACHE.keys())[:_MAX_MEMORY_CACHE_SIZE // 5]
        for key in keys_to_remove:
            del _MEMORY_CACHE[key]

    _MEMORY_CACHE[cache_key] = output

    # Store on disk
    cache_file = _get_cache_file_path(cache_key)
    try:
        cache_data = {
            "_cache_version": _CACHE_VERSION,
            "_created": str(Path(cache_file).stat().st_mtime if cache_file.exists() else "new"),
            "output": output
        }
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, indent=2)

        # Cleanup old cache files if needed
        _cleanup_disk_cache()
    except IOError as e:
        # Log but don't fail if disk cache write fails
        print(f"Warning: Failed to write cache file: {e}")


def _cleanup_disk_cache() -> None:
    """Remove oldest cache files if over limit."""
    cache_dir = get_cache_dir()

    # Get all cache files with their modification times
    cache_files = []
    for subdir in cache_dir.iterdir():
        if subdir.is_dir():
            for cache_file in subdir.glob("*.json"):
                try:
                    mtime = cache_file.stat().st_mtime
                    cache_files.append((cache_file, mtime))
                except OSError:
                    pass

    # If over limit, remove oldest files
    if len(cache_files) > _MAX_DISK_CACHE_SIZE:
        # Sort by modification time (oldest first)
        cache_files.sort(key=lambda x: x[1])

        # Remove oldest 20%
        files_to_remove = cache_files[:len(cache_files) // 5]
        for cache_file, _ in files_to_remove:
            try:
                cache_file.unlink(missing_ok=True)
            except OSError:
                pass


def clear_cache(memory_only: bool = False) -> int:
    """
    Clear the output cache.

    Args:
        memory_only: If True, only clear memory cache. If False, clear both.

    Returns:
        Number of entries cleared
    """
    global _MEMORY_CACHE

    count = len(_MEMORY_CACHE)
    _MEMORY_CACHE = {}

    if not memory_only:
        cache_dir = get_cache_dir()
        if cache_dir.exists():
            for subdir in cache_dir.iterdir():
                if subdir.is_dir():
                    for cache_file in subdir.glob("*.json"):
                        try:
                            cache_file.unlink()
                            count += 1
                        except OSError:
                            pass

    return count


def get_cache_stats() -> dict[str, Any]:
    """Get cache statistics."""
    cache_dir = get_cache_dir()

    disk_count = 0
    disk_size = 0

    if cache_dir.exists():
        for subdir in cache_dir.iterdir():
            if subdir.is_dir():
                for cache_file in subdir.glob("*.json"):
                    try:
                        disk_count += 1
                        disk_size += cache_file.stat().st_size
                    except OSError:
                        pass

    return {
        "memory_entries": len(_MEMORY_CACHE),
        "memory_max": _MAX_MEMORY_CACHE_SIZE,
        "disk_entries": disk_count,
        "disk_max": _MAX_DISK_CACHE_SIZE,
        "disk_size_mb": round(disk_size / (1024 * 1024), 2),
        "cache_dir": str(cache_dir),
    }


# =============================================================================
# IMAGE UTILITIES
# =============================================================================

def get_image_metadata(image_tensor) -> dict[str, Any]:
    """
    Extract metadata from an image tensor.

    Args:
        image_tensor: ComfyUI image tensor (B, H, W, C)

    Returns:
        Dictionary with image metadata
    """
    # Get dimensions
    if len(image_tensor.shape) == 4:
        batch, height, width, channels = image_tensor.shape
    else:
        height, width, channels = image_tensor.shape
        batch = 1

    # Calculate aspect ratio
    aspect_decimal = width / height

    # Determine orientation
    if aspect_decimal > 1.05:
        orientation = "landscape"
    elif aspect_decimal < 0.95:
        orientation = "portrait"
    else:
        orientation = "square"

    # Find common aspect ratio
    aspect_ratio = find_aspect_ratio(width, height)

    # Calculate megapixels
    megapixels = (width * height) / 1_000_000

    return {
        "width": width,
        "height": height,
        "channels": channels,
        "batch_size": batch,
        "aspect_ratio": aspect_ratio,
        "aspect_decimal": round(aspect_decimal, 2),
        "orientation": orientation,
        "megapixels": round(megapixels, 2),
    }


def find_aspect_ratio(width: int, height: int) -> str:
    """Find the closest common aspect ratio."""
    from math import gcd

    divisor = gcd(width, height)
    w = width // divisor
    h = height // divisor

    # Common aspect ratios to match
    common_ratios = {
        (1, 1): "1:1",
        (4, 3): "4:3",
        (3, 4): "3:4",
        (16, 9): "16:9",
        (9, 16): "9:16",
        (3, 2): "3:2",
        (2, 3): "2:3",
        (4, 5): "4:5",
        (5, 4): "5:4",
        (21, 9): "21:9",
        (9, 21): "9:21",
    }

    # Check for exact match
    if (w, h) in common_ratios:
        return common_ratios[(w, h)]

    # Find closest match
    target_ratio = width / height
    closest = None
    min_diff = float("inf")

    for (rw, rh), name in common_ratios.items():
        ratio = rw / rh
        diff = abs(ratio - target_ratio)
        if diff < min_diff:
            min_diff = diff
            closest = name

    # If very close to a common ratio, use it
    if min_diff < 0.05:
        return closest

    # Otherwise return simplified ratio
    return f"{w}:{h}"


def get_zimage_recommendations(image_metadata: dict) -> dict[str, Any]:
    """
    Generate Z-Image recommendations based on image metadata.

    Args:
        image_metadata: Output from get_image_metadata()

    Returns:
        Dictionary with Z-Image recommendations
    """
    settings = get_zimage_settings()
    width = image_metadata["width"]
    height = image_metadata["height"]
    orientation = image_metadata["orientation"]

    # Get optimal resolution based on orientation
    if orientation == "portrait":
        optimal = settings["optimal_resolution"]["portrait"]
    elif orientation == "landscape":
        optimal = settings["optimal_resolution"]["landscape"]
    else:
        optimal = settings["optimal_resolution"]["native"]

    optimal_w, optimal_h = optimal

    # Check if resize is needed
    resize_needed = (width != optimal_w or height != optimal_h)

    # Determine resize direction
    if resize_needed:
        if width * height > optimal_w * optimal_h:
            resize_direction = "downscale"
        else:
            resize_direction = "upscale"
    else:
        resize_direction = "none"

    # Check aspect ratio compatibility
    current_aspect = width / height
    optimal_aspect = optimal_w / optimal_h
    aspect_ratio_ok = abs(current_aspect - optimal_aspect) < 0.1

    # Estimate quality based on resolution
    total_pixels = width * height
    if total_pixels >= 1_000_000:
        quality = "high"
    elif total_pixels >= 500_000:
        quality = "medium"
    else:
        quality = "low"

    return {
        "optimal_resolution": optimal,
        "current_compatible": not resize_needed,
        "resize_needed": resize_needed,
        "resize_direction": resize_direction,
        "resize_method": "lanczos",
        "aspect_ratio_ok": aspect_ratio_ok,
        "nearest_native": f"{optimal_w}x{optimal_h} ({image_metadata['aspect_ratio']})",
        "quality_estimate": quality,
        "suggested_settings": {
            "steps": settings["inference"]["steps"],
            "guidance_scale": settings["inference"]["guidance_scale"],
            "max_sequence_length": settings["inference"]["max_sequence_length"],
        },
    }


# =============================================================================
# ATTRIBUTE SCHEMA BUILDERS
# =============================================================================

def build_attribute_schema_for_scene(
    shot_framing: str,
    genre: str,
    content_detail: str = "standard"
) -> dict[str, list[str]]:
    """
    Build the attribute schema to extract based on scene classification.

    Args:
        shot_framing: Shot framing code (e.g., "CU", "FS")
        genre: Photography genre code (e.g., "PRT", "FSH")
        content_detail: Content detail level

    Returns:
        Dictionary mapping attribute categories to fields to extract
    """
    config = load_zimage_config()
    shot_framings = config.get("shot_framings", {})

    # Get base attributes from shot framing
    framing_config = shot_framings.get(shot_framing, {})
    base_attributes = framing_config.get("attributes", [])

    # Get content detail level config
    detail_config = get_content_detail_schema(content_detail)

    # Build schema based on scene type
    schema = {}

    # Always include subject for people photography
    genres = get_photography_genres()
    genre_config = genres.get(genre, {})
    genre_category = genre_config.get("category", "")

    if genre_category == "people":
        schema["subject"] = ["gender", "age_range"]

    # Add attributes based on shot framing
    for attr in base_attributes:
        if attr == "face":
            schema["face"] = ["shape", "angle", "tilt"]
        elif attr == "eyes":
            schema["eyes"] = ["color", "shape", "state", "gaze", "expression", "makeup"]
        elif attr == "skin":
            schema["skin"] = ["tone", "undertone", "texture", "condition"]
        elif attr == "lips":
            schema["lips"] = ["shape", "color", "state"]
        elif attr == "expression":
            schema["expression"] = ["overall", "intensity"]
        elif attr == "hair":
            schema["hair"] = ["color", "length", "texture", "style", "state"]
        elif attr == "body_upper":
            schema["body_upper"] = detail_config.get("body", ["build", "posture"])
            schema["clothing_upper"] = detail_config.get("clothing", ["type", "color"])
        elif attr == "body" or attr == "clothing_full":
            schema["body"] = detail_config.get("body", ["build", "posture"])
            if content_detail in ["detailed", "explicit"]:
                schema["body_exposure"] = ["chest", "stomach", "back", "thigh", "buttocks"]
            schema["clothing_full"] = detail_config.get("clothing", ["type", "color"])
        elif attr == "pose_upper":
            schema["pose_upper"] = ["shoulders", "arms", "hands", "body_angle"]
        elif attr == "pose_full":
            schema["pose_full"] = ["stance", "weight", "legs", "hips", "dynamic"]
            if content_detail in ["detailed", "explicit"]:
                schema["pose_full"].append("sensuality")
        elif attr == "environment":
            schema["environment"] = ["setting_type", "location"]
        elif attr == "lighting":
            schema["lighting"] = ["type", "source", "direction", "quality", "color_temperature"]
        elif attr == "accessories":
            schema["accessories"] = ["necklace", "earrings", "glasses"]

    # Always include lighting and background for most shots
    if "lighting" not in schema and shot_framing not in ["ECU", "BCU"]:
        schema["lighting"] = ["type", "direction", "quality"]

    if "background" not in schema and shot_framing not in ["ECU"]:
        schema["background"] = ["clarity", "complexity", "color"]

    return schema


def format_attribute_schema_for_prompt(schema: dict[str, list[str]]) -> str:
    """
    Format an attribute schema as text for inclusion in a prompt.

    Args:
        schema: Attribute schema from build_attribute_schema_for_scene()

    Returns:
        Formatted text describing the schema
    """
    config = load_zimage_config()
    all_attributes = config.get("attributes", {})

    lines = ["Extract the following attributes:\n"]

    for category, fields in schema.items():
        lines.append(f"\n{category}:")

        category_config = all_attributes.get(category, {})

        for field in fields:
            field_config = category_config.get(field, {})

            if isinstance(field_config, dict):
                options = field_config.get("options", [])
                if options:
                    options_str = ", ".join(options[:5])
                    if len(options) > 5:
                        options_str += ", ..."
                    lines.append(f"  - {field}: [{options_str}]")
                else:
                    lines.append(f"  - {field}: (describe)")
            else:
                lines.append(f"  - {field}: (describe)")

    return "\n".join(lines)
