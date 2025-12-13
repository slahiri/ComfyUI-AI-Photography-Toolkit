# -*- coding: utf-8 -*-
"""
ComfyUI-AI-Photography-Toolkit - Negative Prompt Builder

Template-based negative prompt generation - NO LLM REQUIRED.
Generates high-quality negative prompts using categorized templates
based on the positive prompt content and detected style.

This replaces LLM calls for negative prompt generation, saving
~2-4 seconds per generation.

Author: Siddhartha Lahiri
Email: siddhartha.lahiri@gmail.com
License: MIT
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set
from enum import Enum


class PromptStyle(Enum):
    """Detected prompt style for style-specific negatives."""
    PORTRAIT = "portrait"
    PHOTOGRAPHY = "photography"
    ARTISTIC = "artistic"
    LANDSCAPE = "landscape"
    PRODUCT = "product"
    ANIME = "anime"
    REALISTIC = "realistic"
    ABSTRACT = "abstract"
    GENERAL = "general"


# =============================================================================
# NEGATIVE PROMPT TEMPLATES BY CATEGORY
# =============================================================================

# Quality issues - always include
QUALITY_NEGATIVES = [
    "blurry", "blur", "out of focus", "soft focus",
    "noise", "noisy", "grainy", "film grain",
    "pixelated", "low resolution", "low quality", "jpeg artifacts",
    "compression artifacts", "watermark", "signature", "text",
    "logo", "banner", "username", "dated",
]

# Anatomical issues - for human subjects
ANATOMY_NEGATIVES = [
    "extra limbs", "missing limbs", "extra arms", "extra legs",
    "extra fingers", "missing fingers", "fused fingers", "too many fingers",
    "deformed hands", "bad hands", "mutated hands", "malformed hands",
    "extra hands", "missing hands",
    "deformed face", "ugly face", "distorted face", "asymmetric face",
    "extra eyes", "missing eyes", "crossed eyes", "wall eyes",
    "deformed body", "mutated body", "bad anatomy", "wrong anatomy",
    "extra head", "missing head", "floating limbs", "disconnected limbs",
    "long neck", "twisted body", "contorted", "disfigured",
]

# Composition issues
COMPOSITION_NEGATIVES = [
    "bad cropping", "cropped", "cut off", "out of frame",
    "partial view", "awkward framing", "bad composition",
    "cluttered background", "busy background", "distracting elements",
    "bad lighting", "overexposed", "underexposed", "harsh shadows",
]

# Style-specific negatives
STYLE_NEGATIVES: Dict[PromptStyle, List[str]] = {
    PromptStyle.PORTRAIT: [
        "unflattering angle", "bad skin", "blemishes", "acne",
        "oily skin", "shiny skin", "bad makeup", "smeared makeup",
        "closed eyes", "blinking", "red eye", "dead eyes",
        "fake smile", "awkward expression", "stiff pose",
    ],
    PromptStyle.PHOTOGRAPHY: [
        "lens flare", "chromatic aberration", "vignette",
        "motion blur", "camera shake", "dust spots",
        "color fringing", "barrel distortion", "pincushion distortion",
    ],
    PromptStyle.ARTISTIC: [
        "amateurish", "crude", "sloppy", "unfinished",
        "bad proportions", "flat colors", "muddy colors",
        "inconsistent style", "mixed styles",
    ],
    PromptStyle.LANDSCAPE: [
        "flat", "boring", "dull sky", "washed out",
        "no depth", "no atmosphere", "artificial looking",
    ],
    PromptStyle.PRODUCT: [
        "shadows on product", "reflections", "dust",
        "fingerprints", "scratches", "imperfections",
        "uneven lighting", "color cast",
    ],
    PromptStyle.ANIME: [
        "bad anatomy", "wrong proportions", "off-model",
        "inconsistent style", "western style", "3d",
        "realistic", "photorealistic",
    ],
    PromptStyle.REALISTIC: [
        "cartoon", "anime", "illustration", "drawing",
        "painting", "sketch", "artistic", "stylized",
        "cgi", "3d render", "fake", "artificial",
    ],
    PromptStyle.ABSTRACT: [
        "realistic", "photorealistic", "figurative",
        "representational", "literal",
    ],
    PromptStyle.GENERAL: [],
}

# Detail level additions
DETAIL_LEVEL_EXTRAS: Dict[str, List[str]] = {
    "Quick": [],  # Minimal negatives
    "Standard": [
        "amateur", "poorly drawn", "bad art",
    ],
    "Detailed": [
        "amateur", "poorly drawn", "bad art",
        "worst quality", "low quality", "normal quality",
        "ugly", "duplicate", "morbid", "mutilated",
    ],
    "Extreme": [
        "amateur", "poorly drawn", "bad art",
        "worst quality", "low quality", "normal quality",
        "ugly", "duplicate", "morbid", "mutilated",
        "error", "bad", "wrong", "mistake",
        "worst", "terrible", "horrible", "disgusting",
    ],
}


# =============================================================================
# STYLE DETECTION
# =============================================================================

STYLE_KEYWORDS: Dict[PromptStyle, List[str]] = {
    PromptStyle.PORTRAIT: [
        "portrait", "headshot", "face", "person", "woman", "man",
        "girl", "boy", "model", "beauty", "close-up", "bust",
    ],
    PromptStyle.PHOTOGRAPHY: [
        "photo", "photograph", "camera", "lens", "f/", "mm",
        "bokeh", "depth of field", "dof", "exposure", "iso",
        "aperture", "shutter", "dslr", "mirrorless",
    ],
    PromptStyle.ARTISTIC: [
        "painting", "art", "artistic", "canvas", "brush",
        "oil painting", "watercolor", "acrylic", "impressionist",
        "expressionist", "surreal", "fantasy",
    ],
    PromptStyle.LANDSCAPE: [
        "landscape", "scenery", "nature", "mountain", "ocean",
        "forest", "sky", "sunset", "sunrise", "panorama",
        "vista", "horizon", "outdoor",
    ],
    PromptStyle.PRODUCT: [
        "product", "commercial", "advertisement", "packaging",
        "studio shot", "white background", "e-commerce",
    ],
    PromptStyle.ANIME: [
        "anime", "manga", "cartoon", "animated", "2d",
        "cel shaded", "japanese", "chibi", "kawaii",
    ],
    PromptStyle.REALISTIC: [
        "realistic", "photorealistic", "hyperrealistic",
        "lifelike", "real", "authentic", "natural",
    ],
    PromptStyle.ABSTRACT: [
        "abstract", "non-figurative", "geometric", "pattern",
        "shapes", "colors", "texture", "conceptual",
    ],
}


def detect_style(prompt: str) -> PromptStyle:
    """Detect the style of a prompt based on keywords."""
    prompt_lower = prompt.lower()

    # Count keyword matches for each style
    scores: Dict[PromptStyle, int] = {}
    for style, keywords in STYLE_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in prompt_lower)
        if score > 0:
            scores[style] = score

    if not scores:
        return PromptStyle.GENERAL

    # Return style with highest score
    return max(scores, key=scores.get)


def has_human_subject(prompt: str) -> bool:
    """Check if prompt contains human subjects."""
    human_keywords = [
        "person", "people", "man", "woman", "girl", "boy",
        "child", "adult", "model", "portrait", "face", "body",
        "human", "figure", "character", "someone", "anybody",
        "she", "he", "her", "his", "they", "them",
    ]
    prompt_lower = prompt.lower()
    return any(kw in prompt_lower for kw in human_keywords)


# =============================================================================
# NEGATIVE PROMPT BUILDER
# =============================================================================

@dataclass
class NegativePromptResult:
    """Result of negative prompt generation."""
    negative_prompt: str
    detected_style: PromptStyle
    has_human: bool
    categories_used: List[str]


def build_negative_prompt(
    positive_prompt: str,
    detail_level: str = "Standard",
    include_quality: bool = True,
    include_anatomy: bool = True,
    include_composition: bool = True,
    include_style: bool = True,
    custom_negatives: List[str] = None,
) -> NegativePromptResult:
    """
    Build a negative prompt based on the positive prompt.

    Args:
        positive_prompt: The positive prompt to analyze
        detail_level: Quick, Standard, Detailed, or Extreme
        include_quality: Include quality-related negatives
        include_anatomy: Include anatomy-related negatives (for humans)
        include_composition: Include composition-related negatives
        include_style: Include style-specific negatives
        custom_negatives: Additional custom negatives to include

    Returns:
        NegativePromptResult with the generated negative prompt
    """
    negatives: List[str] = []
    categories_used: List[str] = []

    # Detect style and human presence
    detected_style = detect_style(positive_prompt)
    has_human = has_human_subject(positive_prompt)

    # Add quality negatives (always recommended)
    if include_quality:
        # For Quick mode, use minimal set
        if detail_level == "Quick":
            negatives.extend(QUALITY_NEGATIVES[:8])  # First 8 most important
        else:
            negatives.extend(QUALITY_NEGATIVES)
        categories_used.append("quality")

    # Add anatomy negatives only if human subjects detected
    if include_anatomy and has_human:
        if detail_level == "Quick":
            # Minimal anatomy negatives
            negatives.extend([
                "extra fingers", "deformed hands", "bad anatomy",
                "extra limbs", "deformed face",
            ])
        else:
            negatives.extend(ANATOMY_NEGATIVES)
        categories_used.append("anatomy")

    # Add composition negatives
    if include_composition:
        if detail_level != "Quick":
            negatives.extend(COMPOSITION_NEGATIVES)
            categories_used.append("composition")

    # Add style-specific negatives
    if include_style:
        style_negs = STYLE_NEGATIVES.get(detected_style, [])
        if style_negs:
            if detail_level == "Quick":
                negatives.extend(style_negs[:5])  # First 5
            else:
                negatives.extend(style_negs)
            categories_used.append(f"style:{detected_style.value}")

    # Add detail level extras
    extras = DETAIL_LEVEL_EXTRAS.get(detail_level, [])
    negatives.extend(extras)

    # Add custom negatives
    if custom_negatives:
        negatives.extend(custom_negatives)
        categories_used.append("custom")

    # Remove duplicates while preserving order
    seen: Set[str] = set()
    unique_negatives: List[str] = []
    for neg in negatives:
        neg_lower = neg.lower()
        if neg_lower not in seen:
            seen.add(neg_lower)
            unique_negatives.append(neg)

    # Build final negative prompt
    negative_prompt = ", ".join(unique_negatives)

    return NegativePromptResult(
        negative_prompt=negative_prompt,
        detected_style=detected_style,
        has_human=has_human,
        categories_used=categories_used,
    )


def enhance_negative_prompt(
    existing_negative: str,
    positive_prompt: str,
    detail_level: str = "Standard",
) -> str:
    """
    Enhance an existing negative prompt by merging with generated template.

    Args:
        existing_negative: User-provided negative prompt
        positive_prompt: The positive prompt for context
        detail_level: Detail level for template generation

    Returns:
        Enhanced negative prompt
    """
    # Generate template-based negatives
    result = build_negative_prompt(
        positive_prompt,
        detail_level=detail_level,
        include_quality=True,
        include_anatomy=True,
        include_composition=True,
        include_style=True,
    )

    # Parse existing negatives
    existing_parts = [p.strip() for p in existing_negative.split(",") if p.strip()]

    # Parse template negatives
    template_parts = [p.strip() for p in result.negative_prompt.split(",") if p.strip()]

    # Merge: user negatives first, then add missing template negatives
    seen: Set[str] = set()
    merged: List[str] = []

    # Add user negatives first (priority)
    for neg in existing_parts:
        neg_lower = neg.lower()
        if neg_lower not in seen:
            seen.add(neg_lower)
            merged.append(neg)

    # Add template negatives that aren't already present
    for neg in template_parts:
        neg_lower = neg.lower()
        if neg_lower not in seen:
            seen.add(neg_lower)
            merged.append(neg)

    return ", ".join(merged)


# =============================================================================
# QUICK TEXT PROCESSING (Grammar/Cleanup)
# =============================================================================

def quick_clean_prompt(prompt: str) -> str:
    """
    Quick cleanup of a prompt - NO LLM REQUIRED.

    Performs:
    - Fix spacing around punctuation
    - Remove duplicate words
    - Fix capitalization
    - Remove redundant commas
    """
    if not prompt or not prompt.strip():
        return prompt

    result = prompt.strip()

    # Fix multiple spaces
    result = re.sub(r'\s+', ' ', result)

    # Fix spacing around punctuation
    result = re.sub(r'\s*,\s*', ', ', result)
    result = re.sub(r'\s*\.\s*', '. ', result)
    result = re.sub(r'\s*:\s*', ': ', result)
    result = re.sub(r'\s*;\s*', '; ', result)

    # Remove duplicate commas
    result = re.sub(r',\s*,', ',', result)
    result = re.sub(r',\s*$', '', result)  # Trailing comma
    result = re.sub(r'^\s*,', '', result)  # Leading comma

    # Remove duplicate consecutive words (case-insensitive)
    words = result.split()
    deduped_words = []
    prev_word = None
    for word in words:
        if word.lower() != (prev_word.lower() if prev_word else None):
            deduped_words.append(word)
        prev_word = word
    result = ' '.join(deduped_words)

    # Ensure first letter is capitalized
    if result and result[0].islower():
        result = result[0].upper() + result[1:]

    # Ensure ends with period if it's a sentence
    if result and result[-1] not in '.!?':
        # Only add period if it looks like a sentence (has verbs/articles)
        sentence_indicators = ['is', 'are', 'the', 'a', 'an', 'with', 'in', 'on', 'at']
        if any(f' {ind} ' in result.lower() for ind in sentence_indicators):
            result += '.'

    return result


def combine_prompt_with_settings(
    base_prompt: str,
    settings: str,
    natural_language: bool = True,
) -> str:
    """
    Combine base prompt with photography settings - NO LLM REQUIRED.

    Args:
        base_prompt: The base prompt
        settings: Photography settings string
        natural_language: If True, use natural language connector

    Returns:
        Combined prompt
    """
    if not base_prompt and not settings:
        return ""

    if not base_prompt:
        return settings

    if not settings:
        return base_prompt

    base = base_prompt.strip().rstrip('.,')
    settings = settings.strip()

    if natural_language:
        # Check if settings look like a list
        if ',' in settings and not any(c in settings for c in ['.', '!', '?']):
            return f"{base}, {settings}"
        else:
            return f"{base}. {settings}"
    else:
        return f"{base}, {settings}"


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def generate_negative(
    positive_prompt: str,
    detail_level: str = "Standard",
) -> str:
    """Simple function to generate a negative prompt."""
    result = build_negative_prompt(positive_prompt, detail_level)
    return result.negative_prompt


def generate_negative_full(
    positive_prompt: str,
    detail_level: str = "Standard",
) -> NegativePromptResult:
    """Generate negative prompt with full metadata."""
    return build_negative_prompt(positive_prompt, detail_level)
