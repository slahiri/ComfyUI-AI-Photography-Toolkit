"""Tests for Phase 1: Core Infrastructure.

Run standalone:
    cd ComfyUI-AI-Photography-Toolkit
    python tests/compose/test_phase1_infrastructure.py

Run with pytest (requires torch installed):
    python -m pytest tests/compose/test_phase1_infrastructure.py -v
"""

import sys
from pathlib import Path

# Add project root to path for standalone execution
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Use importlib for standalone testing (bypasses torch dependency)
import importlib.util


def load_module(name: str, path: str):
    """Load a module directly from file path."""
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Load modules
tokenizer_base = load_module("tokenizer_base", "core/compose/tokenizer/base.py")
categories = load_module("categories", "core/compose/classifier/categories.py")

# Import classes
ImageToken = tokenizer_base.ImageToken
TokenType = tokenizer_base.TokenType
TokenSource = tokenizer_base.TokenSource
TokenBatch = tokenizer_base.TokenBatch
SOURCE_WEIGHTS = tokenizer_base.SOURCE_WEIGHTS

CanonicalCategory = categories.CanonicalCategory
SubjectDetailType = categories.SubjectDetailType
CategoryKeywords = categories.CategoryKeywords
CATEGORY_ORDER = categories.CATEGORY_ORDER
CATEGORY_KEYWORDS = categories.CATEGORY_KEYWORDS
SUBJECT_DETAIL_KEYWORDS = categories.SUBJECT_DETAIL_KEYWORDS
is_meta_pattern = categories.is_meta_pattern
get_florence_key_category = categories.get_florence_key_category


# =============================================================================
# ImageToken Tests
# =============================================================================

def test_image_token_creation():
    """Test basic token creation."""
    token = ImageToken(
        text="brown hair",
        confidence=0.915,
        source=TokenSource.WD14,
        token_type=TokenType.TAG
    )
    assert token.text == "brown hair"
    assert token.confidence == 0.915
    assert token.source == TokenSource.WD14
    assert token.token_type == TokenType.TAG
    print("[PASS] test_image_token_creation")


def test_image_token_normalization():
    """Test text normalization (lowercase, whitespace)."""
    token = ImageToken(
        text="  Brown   Hair  ",
        confidence=0.9,
        source=TokenSource.WD14,
        token_type=TokenType.TAG
    )
    assert token.text == "brown hair", f"Expected 'brown hair', got '{token.text}'"
    assert token.metadata["original"] == "Brown   Hair"
    print("[PASS] test_image_token_normalization")


def test_image_token_confidence_clamping():
    """Test confidence is clamped to [0, 1]."""
    token1 = ImageToken("test", 1.5, TokenSource.WD14, TokenType.TAG)
    token2 = ImageToken("test", -0.5, TokenSource.WD14, TokenType.TAG)
    assert token1.confidence == 1.0
    assert token2.confidence == 0.0
    print("[PASS] test_image_token_confidence_clamping")


def test_image_token_weighted_confidence():
    """Test weighted confidence calculation."""
    token = ImageToken("test", 0.9, TokenSource.WD14, TokenType.TAG)
    # WD14 has weight 1.0, so weighted = 0.9 * 1.0 = 0.9
    assert token.weighted_confidence == 0.9

    token2 = ImageToken("test", 0.9, TokenSource.NUDENET, TokenType.TAG)
    # NUDENET has weight 0.8, so weighted = 0.9 * 0.8 = 0.72
    assert abs(token2.weighted_confidence - 0.72) < 0.001
    print("[PASS] test_image_token_weighted_confidence")


def test_image_token_equality():
    """Test token equality based on normalized text."""
    token1 = ImageToken("brown hair", 0.9, TokenSource.WD14, TokenType.TAG)
    token2 = ImageToken("Brown Hair", 0.8, TokenSource.JOYTAG, TokenType.TAG)
    token3 = ImageToken("black hair", 0.9, TokenSource.WD14, TokenType.TAG)

    assert token1 == token2, "Same text should be equal"
    assert token1 != token3, "Different text should not be equal"
    print("[PASS] test_image_token_equality")


def test_image_token_key_value():
    """Test KEY_VALUE token type."""
    token = ImageToken(
        text="front view",
        confidence=0.95,
        source=TokenSource.FLORENCE_ANALYZE,
        token_type=TokenType.KEY_VALUE,
        metadata={"key": "camera_angle"}
    )
    assert token.key == "camera_angle"
    print("[PASS] test_image_token_key_value")


# =============================================================================
# TokenBatch Tests
# =============================================================================

def test_token_batch_basic():
    """Test basic batch operations."""
    batch = TokenBatch()
    batch.add(ImageToken("brown hair", 0.9, TokenSource.WD14, TokenType.TAG))
    batch.add(ImageToken("long hair", 0.8, TokenSource.WD14, TokenType.TAG))
    batch.add(ImageToken("a woman", 0.95, TokenSource.FLORENCE_CAPTION, TokenType.PHRASE))

    assert len(batch) == 3
    print("[PASS] test_token_batch_basic")


def test_token_batch_filter_by_source():
    """Test filtering tokens by source."""
    batch = TokenBatch()
    batch.add(ImageToken("brown hair", 0.9, TokenSource.WD14, TokenType.TAG))
    batch.add(ImageToken("long hair", 0.8, TokenSource.WD14, TokenType.TAG))
    batch.add(ImageToken("a woman", 0.95, TokenSource.FLORENCE_CAPTION, TokenType.PHRASE))

    wd14_tokens = batch.filter_by_source(TokenSource.WD14)
    assert len(wd14_tokens) == 2
    print("[PASS] test_token_batch_filter_by_source")


def test_token_batch_filter_by_confidence():
    """Test filtering tokens by confidence threshold."""
    batch = TokenBatch()
    batch.add(ImageToken("high", 0.9, TokenSource.WD14, TokenType.TAG))
    batch.add(ImageToken("medium", 0.5, TokenSource.WD14, TokenType.TAG))
    batch.add(ImageToken("low", 0.2, TokenSource.WD14, TokenType.TAG))

    high_conf = batch.filter_by_confidence(0.8)
    assert len(high_conf) == 1
    assert high_conf[0].text == "high"
    print("[PASS] test_token_batch_filter_by_confidence")


# =============================================================================
# Category Tests
# =============================================================================

def test_canonical_categories():
    """Test all 9 canonical categories exist."""
    expected = [
        "quality_boosters", "subject", "subject_details", "action_pose",
        "environment", "lighting", "style_medium", "composition", "technical"
    ]
    for exp in expected:
        assert any(c.value == exp for c in CanonicalCategory), f"Missing category: {exp}"
    print("[PASS] test_canonical_categories")


def test_category_order():
    """Test category output order."""
    assert len(CATEGORY_ORDER) == 9
    assert CATEGORY_ORDER[0] == CanonicalCategory.QUALITY_BOOSTERS
    assert CATEGORY_ORDER[2] == CanonicalCategory.SUBJECT
    print("[PASS] test_category_order")


def test_subject_detail_types():
    """Test subject detail subcategories."""
    expected = ["hair", "eyes", "face", "body", "clothing", "skin", "accessories"]
    for exp in expected:
        assert any(s.value == exp for s in SubjectDetailType), f"Missing subcategory: {exp}"
    print("[PASS] test_subject_detail_types")


# =============================================================================
# Keyword Matching Tests
# =============================================================================

def test_quality_keywords():
    """Test quality booster keyword matching."""
    keywords = CATEGORY_KEYWORDS[CanonicalCategory.QUALITY_BOOSTERS]
    assert keywords.matches("realistic")
    assert keywords.matches("high quality")
    assert keywords.matches("8k")
    assert not keywords.matches("brown hair")
    print("[PASS] test_quality_keywords")


def test_subject_keywords():
    """Test subject keyword matching."""
    keywords = CATEGORY_KEYWORDS[CanonicalCategory.SUBJECT]
    assert keywords.matches("1girl")
    assert keywords.matches("solo")
    assert keywords.matches("woman")
    assert not keywords.matches("brown hair")
    print("[PASS] test_subject_keywords")


def test_subject_detail_hair_keywords():
    """Test hair keyword matching."""
    keywords = SUBJECT_DETAIL_KEYWORDS[SubjectDetailType.HAIR]
    assert keywords.matches("brown hair")
    assert keywords.matches("long hair")
    assert keywords.matches("ponytail")
    assert not keywords.matches("brown eyes")
    print("[PASS] test_subject_detail_hair_keywords")


def test_subject_detail_clothing_keywords():
    """Test clothing keyword matching."""
    keywords = SUBJECT_DETAIL_KEYWORDS[SubjectDetailType.CLOTHING]
    assert keywords.matches("dress")
    assert keywords.matches("white crop top")
    assert keywords.matches("kimono")
    assert keywords.matches("sari")
    print("[PASS] test_subject_detail_clothing_keywords")


def test_environment_keywords():
    """Test environment keyword matching."""
    keywords = CATEGORY_KEYWORDS[CanonicalCategory.ENVIRONMENT]
    assert keywords.matches("indoor")
    assert keywords.matches("grey background")
    assert keywords.matches("studio")
    print("[PASS] test_environment_keywords")


def test_lighting_keywords():
    """Test lighting keyword matching."""
    keywords = CATEGORY_KEYWORDS[CanonicalCategory.LIGHTING]
    assert keywords.matches("soft light")
    assert keywords.matches("high key")
    assert keywords.matches("warm tones")
    print("[PASS] test_lighting_keywords")


# =============================================================================
# META Filter Tests
# =============================================================================

def test_meta_filter_positive():
    """Test META patterns are correctly identified."""
    assert is_meta_pattern("the image shows a woman")
    assert is_meta_pattern("standing in the middle of the image")
    assert is_meta_pattern("The overall mood is calm")
    assert is_meta_pattern("showcasing her curves")
    assert is_meta_pattern("conveying a sense of elegance")
    print("[PASS] test_meta_filter_positive")


def test_meta_filter_negative():
    """Test non-META content passes through."""
    assert not is_meta_pattern("a woman standing")
    assert not is_meta_pattern("brown hair")
    assert not is_meta_pattern("wearing a white dress")
    assert not is_meta_pattern("soft lighting")
    print("[PASS] test_meta_filter_negative")


# =============================================================================
# Florence Key Mapping Tests
# =============================================================================

def test_florence_key_mapping():
    """Test florence_analyze key to category mapping."""
    assert get_florence_key_category("camera_angle") == CanonicalCategory.COMPOSITION
    assert get_florence_key_category("art_style") == CanonicalCategory.STYLE_MEDIUM
    assert get_florence_key_category("location") == CanonicalCategory.ENVIRONMENT
    assert get_florence_key_category("clothing") == CanonicalCategory.SUBJECT_DETAILS
    assert get_florence_key_category("gender") == CanonicalCategory.SUBJECT
    assert get_florence_key_category("unknown_key") is None
    print("[PASS] test_florence_key_mapping")


# =============================================================================
# Run All Tests
# =============================================================================

def run_all_tests():
    """Run all tests and report results."""
    tests = [
        # ImageToken tests
        test_image_token_creation,
        test_image_token_normalization,
        test_image_token_confidence_clamping,
        test_image_token_weighted_confidence,
        test_image_token_equality,
        test_image_token_key_value,
        # TokenBatch tests
        test_token_batch_basic,
        test_token_batch_filter_by_source,
        test_token_batch_filter_by_confidence,
        # Category tests
        test_canonical_categories,
        test_category_order,
        test_subject_detail_types,
        # Keyword matching tests
        test_quality_keywords,
        test_subject_keywords,
        test_subject_detail_hair_keywords,
        test_subject_detail_clothing_keywords,
        test_environment_keywords,
        test_lighting_keywords,
        # META filter tests
        test_meta_filter_positive,
        test_meta_filter_negative,
        # Florence key mapping tests
        test_florence_key_mapping,
    ]

    passed = 0
    failed = 0

    print("=" * 60)
    print("Phase 1: Core Infrastructure Tests")
    print("=" * 60)
    print()

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"[FAIL] {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"[FAIL] {test.__name__}: {type(e).__name__}: {e}")
            failed += 1

    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
