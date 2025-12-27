"""Load vocabulary from JSON file and integrate with classifier.

This module loads the extracted vocabulary from scraped prompts data
and adds it to the dictionary classifier for better token matching.
"""

import json
from pathlib import Path
from typing import Dict, Set

from .categories import CanonicalCategory, CATEGORY_KEYWORDS, CategoryKeywords


# Vocabulary file path
VOCABULARY_FILE = Path(__file__).parent / "vocabulary.json"

# Cache for loaded vocabulary
_vocabulary_cache: Dict[str, Set[str]] = {}


def load_vocabulary() -> Dict[str, Set[str]]:
    """Load vocabulary from JSON file.

    Returns:
        Dictionary mapping category names to sets of terms
    """
    global _vocabulary_cache

    if _vocabulary_cache:
        return _vocabulary_cache

    if not VOCABULARY_FILE.exists():
        return {}

    try:
        with open(VOCABULARY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Convert lists to sets
        _vocabulary_cache = {
            category: set(terms)
            for category, terms in data.items()
        }

        return _vocabulary_cache

    except Exception as e:
        print(f"[Vocabulary] Failed to load vocabulary: {e}")
        return {}


def get_vocabulary_terms(category: str) -> Set[str]:
    """Get vocabulary terms for a specific category.

    Args:
        category: Category name (e.g., "style_medium", "environment")

    Returns:
        Set of terms for the category
    """
    vocab = load_vocabulary()
    return vocab.get(category, set())


def enhance_category_keywords() -> None:
    """Enhance CATEGORY_KEYWORDS with vocabulary terms.

    This adds the scraped vocabulary terms to the existing
    keyword dictionaries for better classification coverage.
    """
    vocab = load_vocabulary()

    if not vocab:
        return

    # Map vocabulary categories to canonical categories
    category_map = {
        "style_medium": CanonicalCategory.STYLE_MEDIUM,
        "subject": CanonicalCategory.SUBJECT,
        "subject_details": CanonicalCategory.SUBJECT_DETAILS,
        "environment": CanonicalCategory.ENVIRONMENT,
        "lighting": CanonicalCategory.LIGHTING,
        "technical": CanonicalCategory.TECHNICAL,
        "quality_boosters": CanonicalCategory.QUALITY_BOOSTERS,
        "action_pose": CanonicalCategory.ACTION_POSE,
        "composition": CanonicalCategory.COMPOSITION,
    }

    for vocab_category, terms in vocab.items():
        canonical_cat = category_map.get(vocab_category)
        if canonical_cat is None:
            continue

        # Get existing keywords for this category
        if canonical_cat in CATEGORY_KEYWORDS:
            keywords = CATEGORY_KEYWORDS[canonical_cat]
            # Add new terms to exact_matches
            keywords.exact_matches.update(terms)


def is_vocabulary_term(text: str) -> bool:
    """Check if text matches any vocabulary term.

    Args:
        text: Text to check

    Returns:
        True if text is a known vocabulary term
    """
    vocab = load_vocabulary()
    text_lower = text.lower().strip()

    for terms in vocab.values():
        if text_lower in terms:
            return True

    return False


def get_vocabulary_category(text: str) -> str:
    """Get the vocabulary category for a text term.

    Args:
        text: Text to look up

    Returns:
        Category name or "unknown"
    """
    vocab = load_vocabulary()
    text_lower = text.lower().strip()

    for category, terms in vocab.items():
        if text_lower in terms:
            return category

    return "unknown"


# Auto-enhance keywords on module load
enhance_category_keywords()
