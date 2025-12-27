"""Classifier module for categorizing tokens.

This module handles Phase 3 of the pipeline with a hybrid approach:
- Layer 1: Deterministic rules (florence_analyze keys)
- Layer 2: Source-based routing
- Layer 3: Dictionary keyword matching (fast, precise)
- Layer 4: Embedding similarity (semantic matching)
- Layer 4.5: LLM classification (complex/ambiguous tokens)
- Layer 5: Uncategorized fallback
"""

from typing import List, Dict, Optional, Tuple, Literal
from collections import defaultdict
from dataclasses import dataclass

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
# Configuration
# =============================================================================

@dataclass
class ClassifierConfig:
    """Configuration for the hybrid classifier pipeline."""
    use_embeddings: bool = True  # Use Layer 4 embedding classifier
    use_llm: bool = False  # Use Layer 4.5 LLM classifier
    llm_provider: str = "local"  # "local", "anthropic", "openai", "gemini"
    llm_model: Optional[str] = None  # Model name/ID
    llm_api_key: Optional[str] = None  # API key for cloud providers
    embedding_threshold: float = 0.5  # Minimum similarity for embedding match


# =============================================================================
# Main Classification Pipeline
# =============================================================================

def classify_token(token: ImageToken, use_embeddings: bool = False) -> TokenClassification:
    """Classify a single token through layers 1-3.

    For embedding/LLM classification, use classify_batch_hybrid instead.

    Classification cascade:
    1. Layer 1: Florence analyze key mapping
    2. Layer 2: Source-based routing
    3. Layer 3: Dictionary keyword matching
    5. Layer 5: Uncategorized fallback

    Args:
        token: Token to classify
        use_embeddings: Whether to try embedding classification (slower)

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

    # Layer 4: Embedding classification (optional)
    if use_embeddings:
        try:
            from .embeddings import classify_by_embedding
            result = classify_by_embedding(token)
            if result:
                return result
        except ImportError:
            pass  # sentence-transformers not installed

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


def classify_batch_hybrid(
    batch: TokenBatch,
    config: Optional[ClassifierConfig] = None,
) -> ClassifiedImage:
    """Classify tokens using the full hybrid pipeline.

    Uses all classification layers:
    1. Layer 1-2: Deterministic (fast)
    2. Layer 3: Dictionary (fast)
    3. Layer 4: Embeddings (for uncategorized, if enabled)
    4. Layer 4.5: LLM (for remaining uncategorized, if enabled)
    5. Layer 5: Uncategorized fallback

    Args:
        batch: TokenBatch containing tokens to classify
        config: Classifier configuration

    Returns:
        ClassifiedImage with all tokens organized by category
    """
    if config is None:
        config = ClassifierConfig()

    result = ClassifiedImage()
    result.image_info = batch.image_info
    result.total_tokens_input = len(batch.tokens)

    # Phase 1: Fast classification (Layers 1-3)
    uncategorized_tokens = []

    for token in batch.tokens:
        classification = classify_token(token, use_embeddings=False)

        if classification.is_uncategorized:
            uncategorized_tokens.append(token)
        else:
            result.add(classification)

    # Phase 2: Embedding classification (Layer 4)
    if config.use_embeddings and uncategorized_tokens:
        try:
            from .embeddings import classify_batch_by_embedding

            embedding_results = classify_batch_by_embedding(uncategorized_tokens)

            still_uncategorized = []
            for token, emb_result in zip(uncategorized_tokens, embedding_results):
                if emb_result is not None:
                    result.add(emb_result)
                else:
                    still_uncategorized.append(token)

            uncategorized_tokens = still_uncategorized

        except ImportError:
            # sentence-transformers not installed
            pass

    # Phase 3: LLM classification (Layer 4.5)
    if config.use_llm and uncategorized_tokens:
        try:
            from .llm_classifier import classify_batch_by_llm

            llm_results = classify_batch_by_llm(
                uncategorized_tokens,
                provider=config.llm_provider,
                model=config.llm_model,
                api_key=config.llm_api_key,
            )

            still_uncategorized = []
            for token, llm_result in zip(uncategorized_tokens, llm_results):
                if llm_result is not None:
                    result.add(llm_result)
                else:
                    still_uncategorized.append(token)

            uncategorized_tokens = still_uncategorized

        except Exception as e:
            print(f"Warning: LLM classification failed: {e}")

    # Phase 4: Mark remaining as uncategorized (Layer 5)
    for token in uncategorized_tokens:
        result.add(TokenClassification(
            token=token,
            primary_category=CanonicalCategory.UNCATEGORIZED,
            confidence=0.0,
            classifier_layer=5,
            metadata={"classification_method": "uncategorized_fallback"}
        ))

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
    # Hair texture - critical conflict
    {"straight hair", "wavy hair", "curly hair"},
    # Eye color - only one can be true
    {"brown eyes", "black eyes", "blue eyes", "green eyes", "grey eyes", "hazel eyes"},
    # Breast size - only one can be true
    {"large breasts", "small breasts", "medium breasts", "flat chest", "huge breasts"},
    # Hair length
    {"short hair", "long hair", "medium hair"},
    # Body type
    {"slim", "curvy", "muscular", "chubby", "petite"},
    # Skin tone
    {"pale skin", "dark skin", "tan", "light skin"},
]

# Ethnic wear - when these are detected, filter out generic Western clothing
ETHNIC_WEAR_TERMS = {
    "saree", "sari", "lehenga", "salwar", "kurta", "dupatta", "choli",
    "kimono", "hanbok", "cheongsam", "ao dai", "abaya", "hijab", "kaftan",
}

# Generic Western clothing to filter when ethnic wear is present
GENERIC_WESTERN_CLOTHING = {
    "dress", "red dress", "strapless dress", "backless dress", "two-tone dress",
    "strapless", "toga", "robe", "gown", "multicolored dress", "multicolored clothes",
}


def resolve_conflicts(classified: ClassifiedImage) -> ClassifiedImage:
    """Resolve conflicts in classifications.

    Handles:
    - Mutually exclusive tokens (keep highest confidence)
    - Duplicate tokens across categories
    - Ethnic wear vs generic Western clothing conflicts

    Args:
        classified: ClassifiedImage to process

    Returns:
        ClassifiedImage with conflicts resolved
    """
    import re

    # Build a map of token text to classifications
    text_to_classifications: Dict[str, List[TokenClassification]] = defaultdict(list)
    all_texts_lower = set()

    for classification in classified.all_classifications:
        text_lower = classification.token.text.lower().strip()
        text_to_classifications[text_lower].append(classification)
        all_texts_lower.add(text_lower)

    # Collect all text for pattern matching
    all_text_combined = " ".join(all_texts_lower)

    tokens_to_remove = set()

    # --- Phase 1: Mutually exclusive conflicts ---
    for exclusive_group in MUTUALLY_EXCLUSIVE_GROUPS:
        group_tokens = []

        for text in exclusive_group:
            # Direct match
            if text in text_to_classifications:
                for classification in text_to_classifications[text]:
                    group_tokens.append((text, classification, classification.token.confidence))

            # Also check if the term appears WITHIN other tokens
            # e.g., "wavy hair" within "long wavy hair"
            pattern = r'\b' + re.escape(text) + r'\b'
            for token_text in all_texts_lower:
                if token_text != text and re.search(pattern, token_text):
                    for classification in text_to_classifications[token_text]:
                        group_tokens.append((token_text, classification, classification.token.confidence))

        # If multiple exclusive tokens found, keep only highest confidence
        if len(group_tokens) > 1:
            # Deduplicate by token text
            seen_texts = {}
            for text, cls, conf in group_tokens:
                if text not in seen_texts or conf > seen_texts[text][1]:
                    seen_texts[text] = (cls, conf)

            if len(seen_texts) > 1:
                # Sort by confidence, keep highest
                sorted_tokens = sorted(seen_texts.items(), key=lambda x: x[1][1], reverse=True)
                # Mark lower-confidence tokens for removal
                for text, _ in sorted_tokens[1:]:
                    tokens_to_remove.add(text)

    # --- Phase 2: Ethnic wear vs Western clothing ---
    has_ethnic_wear = any(
        term in all_text_combined
        for term in ETHNIC_WEAR_TERMS
    )

    if has_ethnic_wear:
        # Remove generic Western clothing terms
        for western_term in GENERIC_WESTERN_CLOTHING:
            if western_term in text_to_classifications:
                tokens_to_remove.add(western_term)

    # --- Phase 3: Build filtered result ---
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
        # Processed = everything that didn't fall to uncategorized (including filtered)
        stats["processed_percent"] = round(
            (total - stats["uncategorized_count"]) / total * 100, 1
        )
        # Useful = content tokens (excluding both filtered and uncategorized)
        stats["categorized_percent"] = round(
            (total - stats["uncategorized_count"] - stats["filtered_count"]) / total * 100, 1
        )
    else:
        stats["processed_percent"] = 0
        stats["categorized_percent"] = 0

    return stats


__all__ = [
    # Configuration
    "ClassifierConfig",
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
    "classify_batch_hybrid",
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
