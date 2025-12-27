"""Classifier module for categorizing tokens.

This module handles Phase 3 of the pipeline:
- Layer 1: Deterministic rules (florence_analyze keys)
- Layer 2: Source-based routing
- Layer 3: Dictionary keyword matching
- Layer 4: Embedding similarity (optional)
- Layer 5: Uncategorized fallback
"""

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
    # Classification
    "TokenClassification",
    "ClassifiedImage",
]
