"""Load enhanced vocabulary and integrate with classifier.

This module bridges the enhanced_categories.py vocabulary with the
existing CanonicalCategory classifier system.
"""

from typing import Dict, Set, Optional, Tuple
from pathlib import Path

from .categories import (
    CanonicalCategory,
    CATEGORY_KEYWORDS,
    CategoryKeywords,
    SubjectDetailType,
    SUBJECT_DETAIL_KEYWORDS,
)


# Import enhanced categories
try:
    from ..enhanced_categories import (
        Category as EnhancedCategory,
        CATEGORY_KEYWORDS as ENHANCED_KEYWORDS,
        CATEGORY_PATTERNS,
        TAG_SYNONYMS,
        EnhancedCategoryClassifier,
        get_classifier,
    )
    ENHANCED_AVAILABLE = True
except ImportError:
    ENHANCED_AVAILABLE = False


# Map Enhanced Category -> Canonical Category
# This allows the new 10-category system to work with existing assembler
ENHANCED_TO_CANONICAL: Dict[str, CanonicalCategory] = {
    "subject": CanonicalCategory.SUBJECT,
    "appearance": CanonicalCategory.SUBJECT_DETAILS,  # Hair, eyes, face, body, skin
    "clothing": CanonicalCategory.SUBJECT_DETAILS,    # Clothing as subject detail
    "pose_action": CanonicalCategory.ACTION_POSE,
    "environment": CanonicalCategory.ENVIRONMENT,
    "lighting": CanonicalCategory.LIGHTING,
    "style_quality": CanonicalCategory.STYLE_MEDIUM,
    "composition": CanonicalCategory.COMPOSITION,
    "scene": CanonicalCategory.STYLE_MEDIUM,          # Scene types map to style
    "constraints": CanonicalCategory.TECHNICAL,       # Content ratings, artifacts
    "uncategorized": CanonicalCategory.UNCATEGORIZED,
}


def get_canonical_for_enhanced(enhanced_cat: str) -> CanonicalCategory:
    """Get CanonicalCategory for an enhanced category name.

    Args:
        enhanced_cat: Enhanced category name (e.g., "appearance", "clothing")

    Returns:
        Corresponding CanonicalCategory
    """
    return ENHANCED_TO_CANONICAL.get(
        enhanced_cat.lower(),
        CanonicalCategory.UNCATEGORIZED
    )


def flatten_enhanced_vocabulary() -> Dict[CanonicalCategory, Set[str]]:
    """Flatten enhanced vocabulary into CanonicalCategory sets.

    Returns:
        Dictionary mapping CanonicalCategory to sets of vocabulary terms
    """
    if not ENHANCED_AVAILABLE:
        return {}

    result: Dict[CanonicalCategory, Set[str]] = {
        cat: set() for cat in CanonicalCategory
    }

    for enhanced_cat, subcategories in ENHANCED_KEYWORDS.items():
        canonical_cat = get_canonical_for_enhanced(enhanced_cat.value)

        for subcategory, terms in subcategories.items():
            result[canonical_cat].update(terms)

    return result


def enhance_category_keywords_from_enhanced() -> None:
    """Enhance CATEGORY_KEYWORDS with enhanced vocabulary terms.

    This adds all vocabulary from enhanced_categories.py to the
    existing keyword dictionaries for better classification coverage.
    """
    if not ENHANCED_AVAILABLE:
        print("[EnhancedVocab] enhanced_categories.py not available")
        return

    flattened = flatten_enhanced_vocabulary()

    count = 0
    for canonical_cat, terms in flattened.items():
        if canonical_cat in CATEGORY_KEYWORDS and terms:
            keywords = CATEGORY_KEYWORDS[canonical_cat]
            before = len(keywords.exact_matches)
            keywords.exact_matches.update(terms)
            after = len(keywords.exact_matches)
            count += (after - before)

    # Also enhance SUBJECT_DETAIL_KEYWORDS with specific subcategories
    subject_detail_count = _enhance_subject_detail_keywords()
    count += subject_detail_count

    print(f"[EnhancedVocab] Added {count} terms from enhanced_categories.py")


def _enhance_subject_detail_keywords() -> int:
    """Enhance SUBJECT_DETAIL_KEYWORDS with enhanced vocabulary.

    Maps enhanced subcategories to SubjectDetailType:
    - appearance/hair_* -> HAIR
    - appearance/eyes -> EYES
    - appearance/face, makeup -> FACE
    - appearance/body, skin -> BODY
    - clothing/* -> CLOTHING
    - clothing/accessory, jewelry -> ACCESSORIES

    Returns:
        Number of terms added
    """
    if not ENHANCED_AVAILABLE:
        return 0

    from ..enhanced_categories import Category as EnhancedCategory

    # Map enhanced subcategories to SubjectDetailType
    enhanced_to_subject_detail = {
        # Appearance mappings
        ("appearance", "hair_color"): SubjectDetailType.HAIR,
        ("appearance", "hair_style"): SubjectDetailType.HAIR,
        ("appearance", "eyes"): SubjectDetailType.EYES,
        ("appearance", "face"): SubjectDetailType.FACE,
        ("appearance", "makeup"): SubjectDetailType.FACE,
        ("appearance", "body"): SubjectDetailType.BODY,
        ("appearance", "body_parts"): SubjectDetailType.BODY,
        ("appearance", "body_features"): SubjectDetailType.BODY,
        ("appearance", "skin"): SubjectDetailType.SKIN,
        ("appearance", "ethnicity"): SubjectDetailType.SKIN,
        # Clothing mappings
        ("clothing", "garment_top"): SubjectDetailType.CLOTHING,
        ("clothing", "garment_bottom"): SubjectDetailType.CLOTHING,
        ("clothing", "garment_dress"): SubjectDetailType.CLOTHING,
        ("clothing", "garment_outerwear"): SubjectDetailType.CLOTHING,
        ("clothing", "garment_underwear"): SubjectDetailType.CLOTHING,
        ("clothing", "garment_uniform"): SubjectDetailType.CLOTHING,
        ("clothing", "garment_ethnic"): SubjectDetailType.CLOTHING,
        ("clothing", "garment_religious"): SubjectDetailType.CLOTHING,
        ("clothing", "garment_costume"): SubjectDetailType.CLOTHING,
        ("clothing", "footwear"): SubjectDetailType.CLOTHING,
        ("clothing", "material"): SubjectDetailType.CLOTHING,
        ("clothing", "pattern"): SubjectDetailType.CLOTHING,
        ("clothing", "fit"): SubjectDetailType.CLOTHING,
        ("clothing", "neckline"): SubjectDetailType.CLOTHING,
        ("clothing", "sleeve"): SubjectDetailType.CLOTHING,
        ("clothing", "length"): SubjectDetailType.CLOTHING,
        ("clothing", "state"): SubjectDetailType.CLOTHING,  # unbuttoned, etc.
        ("clothing", "color"): SubjectDetailType.CLOTHING,
        ("clothing", "detail"): SubjectDetailType.CLOTHING,
        ("clothing", "accessory"): SubjectDetailType.ACCESSORIES,
        ("clothing", "jewelry"): SubjectDetailType.ACCESSORIES,
    }

    count = 0
    for enhanced_cat, subcategories in ENHANCED_KEYWORDS.items():
        cat_name = enhanced_cat.value
        for subcat_name, terms in subcategories.items():
            key = (cat_name, subcat_name)
            if key in enhanced_to_subject_detail:
                subject_type = enhanced_to_subject_detail[key]
                if subject_type in SUBJECT_DETAIL_KEYWORDS:
                    keywords = SUBJECT_DETAIL_KEYWORDS[subject_type]
                    before = len(keywords.exact_matches)
                    keywords.exact_matches.update(terms)
                    after = len(keywords.exact_matches)
                    count += (after - before)

    return count


def classify_with_enhanced(text: str) -> Tuple[CanonicalCategory, Optional[str], str]:
    """Classify a tag using the enhanced classifier then map to CanonicalCategory.

    Args:
        text: Tag text to classify

    Returns:
        Tuple of (CanonicalCategory, subcategory, classification_method)
    """
    if not ENHANCED_AVAILABLE:
        return CanonicalCategory.UNCATEGORIZED, None, "enhanced_unavailable"

    classifier = get_classifier()
    enhanced_cat, subcategory, method = classifier.classify_tag(text)
    canonical_cat = get_canonical_for_enhanced(enhanced_cat.value)

    return canonical_cat, subcategory, f"enhanced_{method}"


def get_enhanced_classification_summary(tags: Dict[str, float]) -> Dict:
    """Get classification summary using enhanced classifier.

    Args:
        tags: Dictionary of tag -> confidence

    Returns:
        Classification summary with both enhanced and canonical categories
    """
    if not ENHANCED_AVAILABLE:
        return {"error": "enhanced_categories not available"}

    classifier = get_classifier()
    enhanced_result = classifier.classify_all(tags)

    # Also compute canonical mapping
    canonical_summary: Dict[str, int] = {}
    for enhanced_cat_name, data in enhanced_result.items():
        canonical_cat = get_canonical_for_enhanced(enhanced_cat_name)
        cat_key = canonical_cat.value
        canonical_summary[cat_key] = canonical_summary.get(cat_key, 0) + len(data.get("tags", {}))

    return {
        "enhanced": enhanced_result,
        "canonical": canonical_summary,
        "summary": classifier.get_summary(enhanced_result),
    }


# Auto-enhance on import
enhance_category_keywords_from_enhanced()
