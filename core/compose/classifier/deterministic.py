"""Deterministic classification layers (Layer 1-2).

Layer 1: florence_analyze key mapping - uses structured keys to determine category
Layer 2: Source-based routing - uses token source to determine category
"""

from typing import Optional, Tuple

from .categories import (
    CanonicalCategory,
    SubjectDetailType,
    FLORENCE_KEY_TO_CATEGORY,
    is_meta_pattern,
)
from .base import TokenClassification
from ..tokenizer.base import ImageToken, TokenSource, TokenType


# =============================================================================
# Layer 1: Florence Analyze Key Mapping
# =============================================================================

def classify_by_florence_key(token: ImageToken) -> Optional[TokenClassification]:
    """Layer 1: Classify token based on florence_analyze key.

    Florence analyze tokens have metadata like {"key": "camera_angle", "value": "front view"}
    The key directly maps to a category.

    Args:
        token: Token to classify

    Returns:
        TokenClassification if key mapping found, None otherwise
    """
    if token.token_type != TokenType.KEY_VALUE:
        return None

    if token.source != TokenSource.FLORENCE_ANALYZE:
        return None

    # Check for META patterns first (filter out invalid/placeholder values)
    if is_meta_pattern(token.text):
        return TokenClassification(
            token=token,
            primary_category=CanonicalCategory.FILTERED_META,
            confidence=1.0,
            classifier_layer=1,
            metadata={"classification_method": "meta_pattern_layer1"}
        )

    key = token.metadata.get("key", "")
    if not key:
        return None

    category = FLORENCE_KEY_TO_CATEGORY.get(key.lower().strip())
    if not category:
        return None

    # Determine subcategory for subject details
    subcategory = None
    if category == CanonicalCategory.SUBJECT_DETAILS:
        subcategory = _infer_subject_detail_subcategory_from_key(key)

    return TokenClassification(
        token=token,
        primary_category=category,
        subcategory=subcategory,
        confidence=1.0,
        classifier_layer=1,
        metadata={"classification_method": "florence_key", "key": key}
    )


def _infer_subject_detail_subcategory_from_key(key: str) -> Optional[SubjectDetailType]:
    """Infer subject detail subcategory from florence_analyze key."""
    key_lower = key.lower()

    if "hair" in key_lower:
        return SubjectDetailType.HAIR
    if "eye" in key_lower:
        return SubjectDetailType.EYES
    if "cloth" in key_lower or "pants" in key_lower or "shoes" in key_lower:
        return SubjectDetailType.CLOTHING
    if "body" in key_lower:
        return SubjectDetailType.BODY
    if "skin" in key_lower or "race" in key_lower or "ethnicity" in key_lower:
        return SubjectDetailType.SKIN
    if "access" in key_lower:
        return SubjectDetailType.ACCESSORIES

    return None


# =============================================================================
# Layer 2: Source-Based Routing
# =============================================================================

# Source to category mappings
SOURCE_CATEGORY_MAPPINGS = {
    # Photography analyzer → Lighting category (for lighting/tone tokens)
    TokenSource.PHOTOGRAPHY: CanonicalCategory.LIGHTING,

    # IQA analyzer → Quality Boosters
    TokenSource.IQA: CanonicalCategory.QUALITY_BOOSTERS,

    # Composition analyzer → Composition category
    TokenSource.COMPOSITION: CanonicalCategory.COMPOSITION,

    # Saliency analyzer → Composition category
    TokenSource.SALIENCY: CanonicalCategory.COMPOSITION,

    # Pose tagger → Action/Pose category
    TokenSource.POSE: CanonicalCategory.ACTION_POSE,
}

# More specific source routing based on token text patterns
PHOTOGRAPHY_COMPOSITION_PATTERNS = {
    "orientation", "frame", "aspect ratio", "format", "resolution",
    "horizontal", "vertical", "square", "landscape", "portrait",
    "rule of thirds", "thirds", "framing", "composition", "centered",
    "symmetry", "asymmetry", "leading lines", "golden ratio",
}

PHOTOGRAPHY_QUALITY_PATTERNS = {
    "photograph", "photography", "sharp", "detailed", "in focus",
    "photographic", "photo",
}


def classify_by_source(token: ImageToken) -> Optional[TokenClassification]:
    """Layer 2: Classify token based on its source.

    Different analyzers/taggers produce tokens that naturally belong to
    certain categories. This layer uses source information to route tokens.

    Args:
        token: Token to classify

    Returns:
        TokenClassification if source routing applies, None otherwise
    """
    source = token.source

    # Check for filtered tokens first
    if token.metadata.get("filtered", False):
        return TokenClassification(
            token=token,
            primary_category=CanonicalCategory.FILTERED_META,
            confidence=1.0,
            classifier_layer=2,
            metadata={"classification_method": "filtered_flag"}
        )

    # Check META patterns in token text
    if is_meta_pattern(token.text):
        return TokenClassification(
            token=token,
            primary_category=CanonicalCategory.FILTERED_META,
            confidence=1.0,
            classifier_layer=2,
            metadata={"classification_method": "meta_pattern"}
        )

    # Photography source needs special handling - can be lighting, composition, or quality
    if source == TokenSource.PHOTOGRAPHY:
        return _classify_photography_token(token)

    # Direct source mappings
    if source in SOURCE_CATEGORY_MAPPINGS:
        category = SOURCE_CATEGORY_MAPPINGS[source]
        return TokenClassification(
            token=token,
            primary_category=category,
            confidence=0.9,
            classifier_layer=2,
            metadata={"classification_method": "source_routing", "source": source.value}
        )

    return None


def _classify_photography_token(token: ImageToken) -> Optional[TokenClassification]:
    """Classify photography analyzer tokens to correct sub-category.

    Photography tokens can be:
    - Lighting (tones, contrast, brightness, etc.)
    - Composition (orientation, frame, aspect ratio)
    - Quality (sharp, detailed, in focus)
    """
    text_lower = token.text.lower()

    # Check for composition patterns
    for pattern in PHOTOGRAPHY_COMPOSITION_PATTERNS:
        if pattern in text_lower:
            return TokenClassification(
                token=token,
                primary_category=CanonicalCategory.COMPOSITION,
                confidence=0.9,
                classifier_layer=2,
                metadata={"classification_method": "photography_composition"}
            )

    # Check for quality patterns
    for pattern in PHOTOGRAPHY_QUALITY_PATTERNS:
        if pattern in text_lower:
            return TokenClassification(
                token=token,
                primary_category=CanonicalCategory.QUALITY_BOOSTERS,
                confidence=0.9,
                classifier_layer=2,
                metadata={"classification_method": "photography_quality"}
            )

    # Default photography tokens are lighting
    return TokenClassification(
        token=token,
        primary_category=CanonicalCategory.LIGHTING,
        confidence=0.85,
        classifier_layer=2,
        metadata={"classification_method": "photography_default_lighting"}
    )


# =============================================================================
# Combined Layer 1-2 Classification
# =============================================================================

def classify_deterministic(token: ImageToken) -> Optional[TokenClassification]:
    """Run Layer 1 and Layer 2 classification.

    Tries Layer 1 (florence key) first, then Layer 2 (source routing).

    Args:
        token: Token to classify

    Returns:
        TokenClassification if deterministic classification found, None otherwise
    """
    # Layer 1: Florence analyze key mapping
    result = classify_by_florence_key(token)
    if result:
        return result

    # Layer 2: Source-based routing
    result = classify_by_source(token)
    if result:
        return result

    return None
