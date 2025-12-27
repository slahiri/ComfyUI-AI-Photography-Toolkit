"""Phrase templates and grammar rules for prompt assembly.

This module defines how tokens in each category should be converted
to natural language phrases.
"""

from typing import Dict, List, Callable, Optional, Tuple
from enum import Enum

from ..classifier.categories import CanonicalCategory, SubjectDetailType


# =============================================================================
# Gender Detection and Pronouns
# =============================================================================

class Gender(Enum):
    """Detected gender for pronoun usage."""
    FEMALE = "female"
    MALE = "male"
    NEUTRAL = "neutral"


FEMALE_INDICATORS = {
    "woman", "girl", "female", "lady", "she", "her",
    "1girl", "2girls", "3girls", "multiple girls",
    "bride", "mother", "sister", "daughter", "wife",
    "queen", "princess", "goddess", "actress", "waitress",
}

MALE_INDICATORS = {
    "man", "boy", "male", "guy", "he", "him", "his",
    "1boy", "2boys", "3boys", "multiple boys",
    "groom", "father", "brother", "son", "husband",
    "king", "prince", "god", "actor", "waiter",
}


def detect_gender(tokens: List[str]) -> Gender:
    """Detect gender from token list."""
    text = " ".join(tokens).lower()

    female_count = sum(1 for ind in FEMALE_INDICATORS if ind in text)
    male_count = sum(1 for ind in MALE_INDICATORS if ind in text)

    if female_count > male_count:
        return Gender.FEMALE
    elif male_count > female_count:
        return Gender.MALE
    return Gender.NEUTRAL


def get_pronouns(gender: Gender) -> Tuple[str, str, str]:
    """Get (subject, possessive, object) pronouns for gender."""
    if gender == Gender.FEMALE:
        return ("She", "Her", "her")
    elif gender == Gender.MALE:
        return ("He", "His", "him")
    return ("They", "Their", "them")


# =============================================================================
# Category Phrase Templates
# =============================================================================

# Templates for each category. Use {tokens} for joined tokens, {token} for single.
# {subject}, {poss}, {obj} for pronouns.

CATEGORY_TEMPLATES: Dict[CanonicalCategory, Dict[str, str]] = {
    CanonicalCategory.QUALITY_BOOSTERS: {
        "prefix": "",
        "single": "{token}",
        "multiple": "{tokens}",
        "suffix": "",
        "join": ", ",
    },
    CanonicalCategory.STYLE_MEDIUM: {
        "prefix": "",
        "single": "{token}",
        "multiple": "{tokens}",
        "suffix": "",
        "join": ", ",
    },
    CanonicalCategory.SUBJECT: {
        "prefix": "",
        "single": "{token}",
        "multiple": "{tokens}",
        "suffix": "",
        "join": " and ",
    },
    CanonicalCategory.SUBJECT_DETAILS: {
        # Handled specially with subcategory templates
        "prefix": "",
        "single": "with {token}",
        "multiple": "with {tokens}",
        "suffix": "",
        "join": ", ",
    },
    CanonicalCategory.ACTION_POSE: {
        "prefix": "",
        "single": "{token}",
        "multiple": "{tokens}",
        "suffix": "",
        "join": ", ",
    },
    CanonicalCategory.ENVIRONMENT: {
        "prefix": "in ",
        "single": "{token}",
        "multiple": "{tokens}",
        "suffix": "",
        "join": " with ",
    },
    CanonicalCategory.LIGHTING: {
        "prefix": "",
        "single": "{token}",
        "multiple": "{tokens}",
        "suffix": " lighting",
        "join": " and ",
    },
    CanonicalCategory.COMPOSITION: {
        "prefix": "",
        "single": "{token}",
        "multiple": "{tokens}",
        "suffix": "",
        "join": ", ",
    },
    CanonicalCategory.TECHNICAL: {
        "prefix": "",
        "single": "{token}",
        "multiple": "{tokens}",
        "suffix": "",
        "join": ", ",
    },
}


# Templates for subject detail subcategories
SUBJECT_DETAIL_TEMPLATES: Dict[SubjectDetailType, Dict[str, str]] = {
    SubjectDetailType.HAIR: {
        "single": "{token}",
        "multiple": "{tokens}",
        "join": " ",  # "long blonde hair" not "long, blonde, hair"
    },
    SubjectDetailType.EYES: {
        "single": "{token}",
        "multiple": "{tokens}",
        "join": " ",
    },
    SubjectDetailType.FACE: {
        "single": "{token}",
        "multiple": "{tokens}",
        "join": ", ",
    },
    SubjectDetailType.SKIN: {
        "single": "{token}",
        "multiple": "{tokens}",
        "join": " ",  # "fair smooth skin"
    },
    SubjectDetailType.BODY: {
        "single": "{token}",
        "multiple": "{tokens}",
        "join": ", ",
    },
    SubjectDetailType.CLOTHING: {
        "single": "wearing {token}",
        "multiple": "wearing {tokens}",
        "join": " and ",
    },
    SubjectDetailType.ACCESSORIES: {
        "single": "with {token}",
        "multiple": "with {tokens}",
        "join": " and ",
    },
}


# =============================================================================
# Token Text Transformations
# =============================================================================

def clean_token_for_output(text: str) -> str:
    """Clean a token text for output in prompt.

    - Removes underscores
    - Removes leading numbers only for booru patterns (1girl -> girl)
    - Preserves technical terms like "3d", "4k", "8k"
    - Normalizes whitespace
    """
    import re

    # Replace underscores with spaces
    text = text.replace("_", " ")

    # Only strip leading numbers for booru-style count patterns (1girl, 2boys, etc.)
    # Pattern: digit(s) followed by common subject words
    booru_pattern = r'^(\d+)(girls?|boys?|women?|men?|people|persons?|characters?)(.*)$'
    match = re.match(booru_pattern, text, re.IGNORECASE)
    if match:
        # Keep only the word part, not the count
        text = match.group(2) + match.group(3)

    # Note: We do NOT strip leading numbers from terms like "3d", "4k", "8k", "1080p"

    # Normalize whitespace
    text = " ".join(text.split())

    return text.strip()


def should_add_article(text: str) -> bool:
    """Check if text needs an article (a/an)."""
    # Don't add article if already has one
    lower = text.lower()
    if lower.startswith(("a ", "an ", "the ", "this ", "that ")):
        return False

    # Don't add article to adjective-only phrases
    if not any(c in text for c in (" ", "-")):
        # Single word - only add if it's a noun
        return False

    return True


def get_article(text: str) -> str:
    """Get appropriate article (a/an) for text."""
    if not text:
        return "a"
    first_char = text.lstrip()[0].lower() if text.lstrip() else ""
    return "an" if first_char in "aeiou" else "a"


# =============================================================================
# Natural Language Connectors
# =============================================================================

# Words that connect different parts of the prompt
CATEGORY_CONNECTORS: Dict[CanonicalCategory, str] = {
    CanonicalCategory.QUALITY_BOOSTERS: "",
    CanonicalCategory.STYLE_MEDIUM: "",
    CanonicalCategory.SUBJECT: "",
    CanonicalCategory.SUBJECT_DETAILS: "",
    CanonicalCategory.ACTION_POSE: "",
    CanonicalCategory.ENVIRONMENT: "",
    CanonicalCategory.LIGHTING: "",
    CanonicalCategory.COMPOSITION: "",
    CanonicalCategory.TECHNICAL: "",
}


# =============================================================================
# Sentence Construction Helpers
# =============================================================================

def join_tokens_natural(tokens: List[str], join_style: str = "comma") -> str:
    """Join tokens in natural language style.

    Args:
        tokens: List of token texts
        join_style: "comma" (a, b, and c), "and" (a and b), "space" (a b c)
    """
    if not tokens:
        return ""
    if len(tokens) == 1:
        return tokens[0]

    if join_style == "space":
        return " ".join(tokens)
    elif join_style == "and":
        if len(tokens) == 2:
            return f"{tokens[0]} and {tokens[1]}"
        return ", ".join(tokens[:-1]) + f", and {tokens[-1]}"
    else:  # comma
        if len(tokens) == 2:
            return f"{tokens[0]} and {tokens[1]}"
        return ", ".join(tokens[:-1]) + f", and {tokens[-1]}"


def build_subject_phrase(
    subject_tokens: List[str],
    detail_tokens: Dict[SubjectDetailType, List[str]],
    gender: Gender,
) -> str:
    """Build a complete subject phrase with details.

    Example: "a beautiful woman with long dark hair, wearing an elegant red dress"
    """
    parts = []

    # Main subject
    if subject_tokens:
        subject = join_tokens_natural(subject_tokens, "and")
        # Add article if needed
        if should_add_article(subject):
            subject = f"{get_article(subject)} {subject}"
        parts.append(subject)

    # Add details in order
    for subcat in [SubjectDetailType.HAIR, SubjectDetailType.EYES,
                   SubjectDetailType.FACE, SubjectDetailType.SKIN,
                   SubjectDetailType.BODY]:
        if subcat in detail_tokens and detail_tokens[subcat]:
            template = SUBJECT_DETAIL_TEMPLATES[subcat]
            tokens = detail_tokens[subcat]
            joined = join_tokens_natural(tokens, "space" if subcat in
                [SubjectDetailType.HAIR, SubjectDetailType.EYES, SubjectDetailType.SKIN]
                else "comma")
            if len(tokens) == 1:
                phrase = template["single"].format(token=joined)
            else:
                phrase = template["multiple"].format(tokens=joined)
            parts.append(phrase)

    # Clothing
    if SubjectDetailType.CLOTHING in detail_tokens and detail_tokens[SubjectDetailType.CLOTHING]:
        template = SUBJECT_DETAIL_TEMPLATES[SubjectDetailType.CLOTHING]
        tokens = detail_tokens[SubjectDetailType.CLOTHING]
        joined = join_tokens_natural(tokens, "and")
        if len(tokens) == 1:
            phrase = template["single"].format(token=joined)
        else:
            phrase = template["multiple"].format(tokens=joined)
        parts.append(phrase)

    # Accessories
    if SubjectDetailType.ACCESSORIES in detail_tokens and detail_tokens[SubjectDetailType.ACCESSORIES]:
        template = SUBJECT_DETAIL_TEMPLATES[SubjectDetailType.ACCESSORIES]
        tokens = detail_tokens[SubjectDetailType.ACCESSORIES]
        joined = join_tokens_natural(tokens, "and")
        if len(tokens) == 1:
            phrase = template["single"].format(token=joined)
        else:
            phrase = template["multiple"].format(tokens=joined)
        parts.append(phrase)

    return ", ".join(parts)
