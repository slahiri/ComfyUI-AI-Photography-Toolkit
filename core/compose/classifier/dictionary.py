"""Dictionary-based classification (Layer 3).

Layer 3: Keyword lookup using exact matches, prefix patterns, suffix patterns,
and contains patterns. Uses the keyword dictionaries defined in categories.py.
"""

from typing import Optional, List, Tuple

from .categories import (
    CanonicalCategory,
    SubjectDetailType,
    CategoryKeywords,
    CATEGORY_KEYWORDS,
    SUBJECT_DETAIL_KEYWORDS,
    is_meta_pattern,
)
from .base import TokenClassification
from ..tokenizer.base import ImageToken


def classify_by_dictionary(token: ImageToken) -> Optional[TokenClassification]:
    """Layer 3: Classify token using keyword dictionaries.

    Checks token text against keyword dictionaries for each category.
    Returns the best matching category based on specificity.

    Args:
        token: Token to classify

    Returns:
        TokenClassification if dictionary match found, None otherwise
    """
    text = token.text.lower().strip()

    # First check subject details (more specific than generic categories)
    subject_detail_result = _check_subject_details(text)
    if subject_detail_result:
        subcategory, confidence = subject_detail_result
        return TokenClassification(
            token=token,
            primary_category=CanonicalCategory.SUBJECT_DETAILS,
            subcategory=subcategory,
            confidence=confidence,
            classifier_layer=3,
            metadata={
                "classification_method": "dictionary_subject_detail",
                "subcategory": subcategory.value
            }
        )

    # Check main categories
    matches: List[Tuple[CanonicalCategory, float]] = []

    for category, keywords in CATEGORY_KEYWORDS.items():
        match_result = _check_category_match(text, keywords)
        if match_result:
            match_type, confidence = match_result
            matches.append((category, confidence))

    if not matches:
        return None

    # Sort by confidence and return best match
    matches.sort(key=lambda x: x[1], reverse=True)
    best_category, best_confidence = matches[0]

    # Collect secondary categories (other matches with lower confidence)
    secondary = [cat for cat, conf in matches[1:] if conf > 0.5]

    return TokenClassification(
        token=token,
        primary_category=best_category,
        secondary_categories=secondary,
        confidence=best_confidence,
        classifier_layer=3,
        metadata={
            "classification_method": "dictionary",
            "all_matches": [(cat.value, conf) for cat, conf in matches]
        }
    )


def _check_subject_details(text: str) -> Optional[Tuple[SubjectDetailType, float]]:
    """Check if text matches any subject detail subcategory.

    Returns:
        Tuple of (SubjectDetailType, confidence) if match found, None otherwise
    """
    for subcategory, keywords in SUBJECT_DETAIL_KEYWORDS.items():
        match_result = _check_category_match(text, keywords)
        if match_result:
            match_type, confidence = match_result
            return (subcategory, confidence)

    return None


def _check_category_match(text: str, keywords: CategoryKeywords) -> Optional[Tuple[str, float]]:
    """Check if text matches category keywords.

    Returns:
        Tuple of (match_type, confidence) if match found, None otherwise
        Match types: "exact", "prefix", "suffix", "contains"
    """
    # Check negative keywords first
    if text in keywords.negative_keywords:
        return None

    # Exact match (highest confidence)
    if text in keywords.exact_matches:
        return ("exact", 1.0)

    # Prefix patterns (high confidence)
    for pattern in keywords.prefix_patterns:
        prefix = pattern.rstrip("*").rstrip("_").rstrip(" ")
        if text.startswith(prefix):
            return ("prefix", 0.9)

    # Suffix patterns (high confidence)
    for pattern in keywords.suffix_patterns:
        suffix = pattern.lstrip("*").lstrip("_").lstrip(" ")
        if text.endswith(suffix):
            return ("suffix", 0.9)

    # Contains patterns (medium confidence)
    for pattern in keywords.contains_patterns:
        check = pattern.strip("*")
        if check in text:
            return ("contains", 0.75)

    return None


def get_all_matching_categories(text: str) -> List[Tuple[CanonicalCategory, float]]:
    """Get all categories that match the given text.

    Useful for multi-category tokens and debugging.

    Args:
        text: Text to check

    Returns:
        List of (category, confidence) tuples, sorted by confidence
    """
    text_lower = text.lower().strip()
    matches = []

    # Check main categories
    for category, keywords in CATEGORY_KEYWORDS.items():
        match_result = _check_category_match(text_lower, keywords)
        if match_result:
            match_type, confidence = match_result
            matches.append((category, confidence))

    # Sort by confidence descending
    matches.sort(key=lambda x: x[1], reverse=True)
    return matches


def get_subject_detail_subcategory(text: str) -> Optional[SubjectDetailType]:
    """Get the subject detail subcategory for a text.

    Args:
        text: Text to check

    Returns:
        SubjectDetailType if match found, None otherwise
    """
    result = _check_subject_details(text.lower().strip())
    if result:
        return result[0]
    return None


# =============================================================================
# Multi-Category Detection
# =============================================================================

# Tokens that commonly belong to multiple categories
MULTI_CATEGORY_TOKENS = {
    # Action + Eyes
    "looking at viewer": (
        CanonicalCategory.ACTION_POSE,
        [CanonicalCategory.SUBJECT_DETAILS],
        SubjectDetailType.EYES
    ),
    "looking away": (
        CanonicalCategory.ACTION_POSE,
        [CanonicalCategory.SUBJECT_DETAILS],
        SubjectDetailType.EYES
    ),
    "looking down": (
        CanonicalCategory.ACTION_POSE,
        [CanonicalCategory.SUBJECT_DETAILS],
        SubjectDetailType.EYES
    ),
    "looking up": (
        CanonicalCategory.ACTION_POSE,
        [CanonicalCategory.SUBJECT_DETAILS],
        SubjectDetailType.EYES
    ),
    "closed eyes": (
        CanonicalCategory.SUBJECT_DETAILS,
        [CanonicalCategory.ACTION_POSE],
        SubjectDetailType.EYES
    ),
    "half-closed eyes": (
        CanonicalCategory.SUBJECT_DETAILS,
        [CanonicalCategory.ACTION_POSE],
        SubjectDetailType.EYES
    ),

    # Face/Expression + Action
    "smile": (
        CanonicalCategory.SUBJECT_DETAILS,
        [CanonicalCategory.ACTION_POSE],
        SubjectDetailType.FACE
    ),
    "smiling": (
        CanonicalCategory.SUBJECT_DETAILS,
        [CanonicalCategory.ACTION_POSE],
        SubjectDetailType.FACE
    ),
    "grin": (
        CanonicalCategory.SUBJECT_DETAILS,
        [CanonicalCategory.ACTION_POSE],
        SubjectDetailType.FACE
    ),

    # Composition + Technical
    "portrait orientation": (
        CanonicalCategory.COMPOSITION,
        [CanonicalCategory.TECHNICAL],
        None
    ),
    "landscape orientation": (
        CanonicalCategory.COMPOSITION,
        [CanonicalCategory.TECHNICAL],
        None
    ),
}


def classify_with_multi_category(token: ImageToken) -> Optional[TokenClassification]:
    """Classify token with support for multi-category assignment.

    Some tokens naturally belong to multiple categories. This function
    handles those cases explicitly.

    Args:
        token: Token to classify

    Returns:
        TokenClassification with multi-category support if match found
    """
    text_lower = token.text.lower().strip()

    # Check for META patterns first (filter out invalid/placeholder values)
    if is_meta_pattern(token.text):
        return TokenClassification(
            token=token,
            primary_category=CanonicalCategory.FILTERED_META,
            confidence=1.0,
            classifier_layer=3,
            metadata={"classification_method": "meta_pattern_layer3"}
        )

    # Check explicit multi-category mappings
    if text_lower in MULTI_CATEGORY_TOKENS:
        primary, secondary, subcategory = MULTI_CATEGORY_TOKENS[text_lower]
        return TokenClassification(
            token=token,
            primary_category=primary,
            secondary_categories=secondary,
            subcategory=subcategory,
            confidence=1.0,
            classifier_layer=3,
            metadata={"classification_method": "multi_category_explicit"}
        )

    # Fall back to regular dictionary classification
    return classify_by_dictionary(token)
