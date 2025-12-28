"""Prompt composition module."""

from .base import (
    ComposeMode,
    ComposeConfig,
    ComposeResult,
    ComposeAnalytics,
    StyleConfig,
)
from .standard import StandardGenerator
from .llm import LLMGenerator
from .pipeline import ComposePipeline
from .canonical_structurer import (
    CanonicalStructurer,
    CanonicalClassifier,
    NLPTagMatcher,
    get_nlp_matcher,
    normalize_tag,
)
from .enhanced_categories import (
    Category as EnhancedCategory,
    CATEGORY_KEYWORDS as ENHANCED_CATEGORY_KEYWORDS,
    CATEGORY_ORDER as ENHANCED_CATEGORY_ORDER,
    CATEGORY_PATTERNS,
    TAG_SYNONYMS,
    EnhancedCategoryClassifier,
    get_classifier as get_enhanced_classifier,
    classify_tags,
    get_uncategorized_tags,
)

__all__ = [
    # Enums and configs
    "ComposeMode",
    "ComposeConfig",
    "ComposeResult",
    "ComposeAnalytics",
    "StyleConfig",
    # Generators
    "StandardGenerator",
    "LLMGenerator",
    # Pipeline
    "ComposePipeline",
    # Canonical structurer
    "CanonicalStructurer",
    "CanonicalClassifier",
    # NLP tag matching
    "NLPTagMatcher",
    "get_nlp_matcher",
    "normalize_tag",
    # Enhanced category system
    "EnhancedCategory",
    "ENHANCED_CATEGORY_KEYWORDS",
    "ENHANCED_CATEGORY_ORDER",
    "CATEGORY_PATTERNS",
    "TAG_SYNONYMS",
    "EnhancedCategoryClassifier",
    "get_enhanced_classifier",
    "classify_tags",
    "get_uncategorized_tags",
]
