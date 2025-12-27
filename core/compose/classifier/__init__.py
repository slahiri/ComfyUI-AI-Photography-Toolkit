"""Classifier module for categorizing tokens.

This module handles Phase 3 of the pipeline:
- Layer 1: Deterministic rules (florence_analyze keys)
- Layer 2: Source-based routing
- Layer 3: Dictionary keyword matching
- Layer 4: Embedding similarity (optional, not implemented)
- Layer 5: Uncategorized fallback
"""

from typing import List, Dict, Optional, Tuple
from collections import defaultdict

from .categories import (
    CanonicalCategory,
    SubjectDetailType,
    CategoryKeywords,
    CATEGORY_ORDER,
    SUBJECT_DETAIL_ORDER,
    CATEGORY_KEYWORDS,
    SUBJECT_DETAIL_KEYWORDS,
    META_FILTER_PATTERNS,
    FLORENCE_KEY_TO_CATEGORY,
    CATEGORY_DISPLAY_NAMES,
    is_meta_pattern,
    get_florence_key_category,
)

from .base import (
    TokenClassification,
    ClassifiedImage,
)

from .deterministic import (
    classify_deterministic,
    classify_by_florence_key,
    classify_by_source,
)

from .dictionary import (
    classify_by_dictionary,
    classify_with_multi_category,
    get_all_matching_categories,
    get_subject_detail_subcategory,
)

from ..tokenizer.base import ImageToken, TokenBatch


# =============================================================================
# Main Classification Pipeline
# =============================================================================

def classify_token(token: ImageToken) -> TokenClassification:
    """Classify a single token through all layers.

    Classification cascade:
    1. Layer 1: Florence analyze key mapping
    2. Layer 2: Source-based routing
    3. Layer 3: Dictionary keyword matching
    4. Layer 5: Uncategorized fallback (Layer 4 embedding not implemented)

    Args:
        token: Token to classify

    Returns:
        TokenClassification with assigned category
    """
    # Layer 1-2: Deterministic classification
    result = classify_deterministic(token)
    if result:
        return result

    # Layer 3: Dictionary classification with multi-category support
    result = classify_with_multi_category(token)
    if result:
        return result

    # Layer 5: Uncategorized fallback
    return TokenClassification(
        token=token,
        primary_category=CanonicalCategory.UNCATEGORIZED,
        confidence=0.0,
        classifier_layer=5,
        metadata={"classification_method": "uncategorized_fallback"}
    )


def classify_batch(batch: TokenBatch) -> ClassifiedImage:
    """Classify all tokens in a batch.

    Args:
        batch: TokenBatch containing tokens to classify

    Returns:
        ClassifiedImage with all tokens organized by category
    """
    result = ClassifiedImage()
    result.image_info = batch.image_info
    result.total_tokens_input = len(batch.tokens)

    for token in batch.tokens:
        classification = classify_token(token)
        result.add(classification)

    return result


def classify_tokens(tokens: List[ImageToken]) -> ClassifiedImage:
    """Classify a list of tokens.

    Args:
        tokens: List of tokens to classify

    Returns:
        ClassifiedImage with all tokens organized by category
    """
    result = ClassifiedImage()
    result.total_tokens_input = len(tokens)

    for token in tokens:
        classification = classify_token(token)
        result.add(classification)

    return result


# =============================================================================
# Conflict Resolution
# =============================================================================

# Mutually exclusive token groups - only keep highest confidence
MUTUALLY_EXCLUSIVE_GROUPS = [
    # Positions (can only be one at a time)
    {"standing", "sitting", "lying", "kneeling", "crouching"},
    # Mouth state
    {"open mouth", "closed mouth", "parted lips"},
    # Eye state
    {"closed eyes", "half-closed eyes", "open eyes"},
    # Facing direction
    {"facing viewer", "facing away", "from behind", "from side"},
]


def resolve_conflicts(classified: ClassifiedImage) -> ClassifiedImage:
    """Resolve conflicts in classifications.

    Handles:
    - Mutually exclusive tokens (keep highest confidence)
    - Duplicate tokens across categories

    Args:
        classified: ClassifiedImage to process

    Returns:
        ClassifiedImage with conflicts resolved
    """
    # Build a map of token text to classifications
    text_to_classifications: Dict[str, List[TokenClassification]] = defaultdict(list)
    for classification in classified.all_classifications:
        text_lower = classification.token.text.lower().strip()
        text_to_classifications[text_lower].append(classification)

    # Find and resolve mutually exclusive conflicts
    tokens_to_remove = set()

    for exclusive_group in MUTUALLY_EXCLUSIVE_GROUPS:
        group_tokens = []
        for text in exclusive_group:
            if text in text_to_classifications:
                for classification in text_to_classifications[text]:
                    group_tokens.append((text, classification))

        # If multiple exclusive tokens found, keep only highest confidence
        if len(group_tokens) > 1:
            group_tokens.sort(key=lambda x: x[1].token.confidence, reverse=True)
            # Mark lower-confidence tokens for removal
            for text, _ in group_tokens[1:]:
                tokens_to_remove.add(text)

    # Build new ClassifiedImage without conflicting tokens
    if tokens_to_remove:
        result = ClassifiedImage()
        result.image_info = classified.image_info
        result.total_tokens_input = classified.total_tokens_input

        for classification in classified.all_classifications:
            text_lower = classification.token.text.lower().strip()
            if text_lower not in tokens_to_remove:
                result.add(classification)

        return result

    return classified


# =============================================================================
# Statistics and Debugging
# =============================================================================

def get_classification_stats(classified: ClassifiedImage) -> Dict[str, any]:
    """Get statistics about classification results.

    Args:
        classified: ClassifiedImage to analyze

    Returns:
        Dictionary with classification statistics
    """
    stats = {
        "total_tokens": len(classified.all_classifications),
        "filtered_count": len(classified.filtered),
        "by_category": {},
        "by_layer": defaultdict(int),
        "by_subcategory": {},
        "uncategorized_count": 0,
    }

    for classification in classified.all_classifications:
        # Count by category
        cat = classification.primary_category.value
        stats["by_category"][cat] = stats["by_category"].get(cat, 0) + 1

        # Count by layer
        stats["by_layer"][classification.classifier_layer] += 1

        # Count uncategorized
        if classification.is_uncategorized:
            stats["uncategorized_count"] += 1

        # Count subcategories
        if classification.subcategory:
            subcat = classification.subcategory.value
            stats["by_subcategory"][subcat] = stats["by_subcategory"].get(subcat, 0) + 1

    # Calculate percentages
    total = stats["total_tokens"]
    if total > 0:
        stats["categorized_percent"] = round(
            (total - stats["uncategorized_count"] - stats["filtered_count"]) / total * 100, 1
        )
    else:
        stats["categorized_percent"] = 0

    return stats


__all__ = [
    # Categories
    "CanonicalCategory",
    "SubjectDetailType",
    "CategoryKeywords",
    "CATEGORY_ORDER",
    "SUBJECT_DETAIL_ORDER",
    "CATEGORY_KEYWORDS",
    "SUBJECT_DETAIL_KEYWORDS",
    "META_FILTER_PATTERNS",
    "FLORENCE_KEY_TO_CATEGORY",
    "CATEGORY_DISPLAY_NAMES",
    "is_meta_pattern",
    "get_florence_key_category",
    # Classification structures
    "TokenClassification",
    "ClassifiedImage",
    # Classification functions
    "classify_token",
    "classify_batch",
    "classify_tokens",
    "classify_deterministic",
    "classify_by_florence_key",
    "classify_by_source",
    "classify_by_dictionary",
    "classify_with_multi_category",
    "get_all_matching_categories",
    "get_subject_detail_subcategory",
    # Conflict resolution
    "resolve_conflicts",
    "MUTUALLY_EXCLUSIVE_GROUPS",
    # Statistics
    "get_classification_stats",
]
