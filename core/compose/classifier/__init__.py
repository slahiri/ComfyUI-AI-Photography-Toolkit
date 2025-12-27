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

# Load vocabulary to enhance keyword dictionaries
from . import vocabulary_loader  # Auto-enhances CATEGORY_KEYWORDS on import


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
    # Body shot type - only one can be true
    {"full body", "upper body", "lower body", "cowboy shot", "bust shot", "headshot"},
    # Age - male (include tag variants)
    {"1boy", "boy", "1man", "man", "mature male", "old man", "young man", "middle-aged", "manly"},
    # Age - female
    {"1girl", "girl", "1woman", "woman", "mature female", "old woman", "young woman"},
    # Subject count - solo vs multiple
    {"solo", "multiple subjects", "2boys", "2girls", "group"},
]

# Fantasy/unrealistic tags to filter when image appears realistic
FANTASY_TAGS = {
    "orc", "dwarf", "elf", "pointy ears", "demon", "angel", "vampire",
    "furry", "anthro", "wings", "tail", "horns", "fangs",
}

# False positive tags that should be filtered based on context
BODY_PART_FALSE_POSITIVES = {
    "tongue",  # Often false positive
}

# Clothing contradictions - if wearing visible clothing, filter nudity tags
CLOTHING_INDICATORS = {
    "wearing clothing", "upper clothing visible", "dress", "shirt", "top",
    "gold shawl", "shawl", "robe", "traditional clothing", "cape", "cloak",
    "draped", "garment", "cloth", "wrapped",
}
NUDITY_TAGS = {"naked", "nude", "topless", "bottomless", "topless male", "topless female"}

# Tags that are likely misclassifications for this context
MISCLASSIFICATION_TAGS = {
    "hilltop citadel",  # Environment, not detail
    "toga",  # Often confused with draped clothing
    "boxers",  # Often false positive
    "greco-roman clothes",  # Often confused with African traditional wear
    "strongman waist",  # Meaningless term
    "fine art parody",  # Usually irrelevant
}

# Youthful terms to filter when subject is clearly mature/adult
YOUTHFUL_TAGS = {"boy", "1boy", "young", "teenager", "teen"}
MATURE_REPLACEMENTS = {"boy": "man", "1boy": "1man"}

# Cultural clothing conflicts - filter these based on VLM ethnicity
INDIAN_CLOTHING_TAGS = {"indian clothes", "saree", "sari", "kurta", "salwar kameez", "dupatta"}
AFRICAN_CLOTHING_INDICATORS = {"african", "gold shawl", "traditional african", "africa"}

# Gender-specific tags - when gender is clearly established, filter opposite gender tags
MALE_INDICATORS = {
    "1boy", "male focus", "male", "1man", "man", "boy",
    "dark-skinned male", "muscular male", "mature male", "old man",
}
FEMALE_INDICATORS = {
    "1girl", "female focus", "female", "1woman", "woman", "girl",
    "dark-skinned female", "mature female", "old woman",
}
# Tags to filter when opposite gender is dominant
MALE_ONLY_TAGS = {"male face", "muscular male", "bara", "pectorals", "topless male"}
FEMALE_ONLY_TAGS = {"female face", "breasts", "cleavage", "sideboob"}

# Ethnicity conflicts - these are mutually exclusive
ETHNICITY_GROUPS = [
    # East Asian vs African/Dark-skinned (cannot be both)
    {
        "asian": {"asian", "east asian", "chinese", "japanese", "korean"},
        "african": {"african", "african american", "dark-skinned male", "dark-skinned female",
                   "dark skin", "very dark skin", "black", "west african", "east african"},
    },
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


def _extract_vlm_attributes(image_info: Dict) -> Dict[str, str]:
    """Extract key attributes from VLM/Florence descriptions.

    Parses captions and descriptions to determine ground truth for:
    - gender: 'male', 'female', or 'unknown'
    - ethnicity: 'african', 'asian', 'caucasian', etc.
    - age: 'child', 'young', 'adult', 'mature', 'elderly'

    Args:
        image_info: Dictionary containing VLM outputs

    Returns:
        Dictionary of extracted attributes
    """
    attributes = {
        "gender": "unknown",
        "ethnicity": "unknown",
        "age": "unknown",
        "subject_count": "unknown",  # "single" or "multiple"
    }

    # Combine all VLM text for analysis
    vlm_texts = []
    for key in ["florence_caption", "florence_description", "florence_mixed_caption",
                "florence_analyze", "florence_generate_tags"]:
        if key in image_info and image_info[key]:
            vlm_texts.append(str(image_info[key]).lower())

    if not vlm_texts:
        return attributes

    combined_text = " ".join(vlm_texts)

    # --- Gender detection ---
    male_patterns = ["man", "male", "boy", "he ", "his ", "1man", "1boy"]
    female_patterns = ["woman", "female", "girl", "she ", "her ", "1woman", "1girl"]

    male_score = sum(1 for p in male_patterns if p in combined_text)
    female_score = sum(1 for p in female_patterns if p in combined_text)

    if male_score > female_score:
        attributes["gender"] = "male"
    elif female_score > male_score:
        attributes["gender"] = "female"

    # --- Ethnicity detection ---
    ethnicity_patterns = {
        "african": ["african", "african-american", "black man", "black woman",
                   "dark skin", "dark-skinned", "dark complexion", "ebony"],
        "asian": ["asian", "chinese", "japanese", "korean", "east asian"],
        "south_asian": ["indian", "south asian", "desi"],
        "caucasian": ["caucasian", "white", "european", "fair skin"],
        "latino": ["latino", "latina", "hispanic", "mexican"],
        "middle_eastern": ["middle eastern", "arab", "persian"],
    }

    ethnicity_scores = {}
    for ethnicity, patterns in ethnicity_patterns.items():
        score = sum(1 for p in patterns if p in combined_text)
        if score > 0:
            ethnicity_scores[ethnicity] = score

    if ethnicity_scores:
        attributes["ethnicity"] = max(ethnicity_scores, key=ethnicity_scores.get)

    # --- Age detection ---
    age_patterns = {
        "child": ["child", "kid", "infant", "baby", "toddler"],
        "teenager": ["teen", "teenager", "adolescent"],
        "young_adult": ["young adult", "20s", "young man", "young woman"],
        "adult": ["adult", "30s", "early 40s"],
        "mature": ["mature", "middle-aged", "50s", "middle aged", "late 40s", "early 50s",
                   "mature male", "mature female", "in his 40s", "in his 50s", "in her 40s", "in her 50s"],
        "elderly": ["elderly", "senior", "old man", "old woman", "60s", "70s", "80s"],
    }

    age_scores = {}
    for age, patterns in age_patterns.items():
        score = sum(1 for p in patterns if p in combined_text)
        if score > 0:
            age_scores[age] = score

    if age_scores:
        attributes["age"] = max(age_scores, key=age_scores.get)

    # --- Subject count detection ---
    # Single subject indicators (stronger)
    single_patterns = ["a man", "a woman", "a person", "the man", "the woman",
                       "1man", "1boy", "1girl", "1woman", "solo", "single person",
                       "a muscular", "a young", "a mature", "an elderly"]
    # Multiple subject indicators
    multiple_patterns = ["two men", "two women", "two people", "group of",
                        "2boys", "2girls", "multiple people", "several people",
                        "couple", "pair of"]

    single_score = sum(1 for p in single_patterns if p in combined_text)
    multiple_score = sum(1 for p in multiple_patterns if p in combined_text)

    if single_score > multiple_score:
        attributes["subject_count"] = "single"
    elif multiple_score > single_score:
        attributes["subject_count"] = "multiple"

    return attributes


def resolve_conflicts(classified: ClassifiedImage) -> ClassifiedImage:
    """Resolve conflicts in classifications.

    Handles:
    - VLM-based ground truth filtering (highest priority)
    - Mutually exclusive tokens (keep highest confidence)
    - Duplicate tokens across categories
    - Ethnic wear vs generic Western clothing conflicts
    - Gender conflicts (filter opposite gender tags)
    - Ethnicity conflicts (asian vs african)
    - Fantasy tags in realistic images
    - Clothing vs nudity contradictions
    - Common false positives and misclassifications

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

    # --- Phase 0: VLM-based ground truth filtering (highest priority) ---
    vlm_attrs = _extract_vlm_attributes(classified.image_info)

    # If VLM says male, remove female-specific tags
    if vlm_attrs["gender"] == "male":
        female_tags_to_filter = {"female face", "1girl", "female", "woman", "breasts",
                                 "cleavage", "female focus", "girl"}
        for tag in female_tags_to_filter:
            if tag in text_to_classifications:
                tokens_to_remove.add(tag)

    # If VLM says female, remove male-specific tags
    elif vlm_attrs["gender"] == "female":
        male_tags_to_filter = {"male face", "1boy", "male", "man", "bara",
                              "pectorals", "male focus", "boy", "topless male"}
        for tag in male_tags_to_filter:
            if tag in text_to_classifications:
                tokens_to_remove.add(tag)

    # If VLM says African ethnicity, remove Asian tags
    if vlm_attrs["ethnicity"] == "african":
        asian_tags_to_filter = {"asian", "east asian", "chinese", "japanese", "korean"}
        for tag in asian_tags_to_filter:
            if tag in text_to_classifications:
                tokens_to_remove.add(tag)

    # If VLM says Asian ethnicity, remove African tags
    elif vlm_attrs["ethnicity"] == "asian":
        african_tags_to_filter = {"african", "african american", "dark-skinned male",
                                  "dark-skinned female", "very dark skin"}
        for tag in african_tags_to_filter:
            if tag in text_to_classifications:
                tokens_to_remove.add(tag)

    # If VLM says African, filter Indian clothing tags
    if vlm_attrs["ethnicity"] == "african":
        for tag in INDIAN_CLOTHING_TAGS:
            if tag in text_to_classifications:
                tokens_to_remove.add(tag)

    # If VLM indicates mature/adult age, filter youthful tags
    if vlm_attrs["age"] in {"mature", "adult", "elderly"}:
        for tag in YOUTHFUL_TAGS:
            if tag in text_to_classifications:
                tokens_to_remove.add(tag)

    # If VLM indicates single subject, filter "multiple subjects" from saliency
    if vlm_attrs["subject_count"] == "single":
        multiple_tags = {"multiple subjects", "2boys", "2girls", "group", "crowd"}
        for tag in multiple_tags:
            if tag in text_to_classifications:
                tokens_to_remove.add(tag)

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

    # --- Phase 3: Gender conflict resolution ---
    # Count male vs female indicators
    male_count = sum(1 for t in all_texts_lower if t in MALE_INDICATORS)
    female_count = sum(1 for t in all_texts_lower if t in FEMALE_INDICATORS)

    # If clearly male, filter female-only tags
    if male_count > 0 and female_count == 0:
        for female_tag in FEMALE_ONLY_TAGS:
            if female_tag in text_to_classifications:
                tokens_to_remove.add(female_tag)

    # If clearly female, filter male-only tags
    if female_count > 0 and male_count == 0:
        for male_tag in MALE_ONLY_TAGS:
            if male_tag in text_to_classifications:
                tokens_to_remove.add(male_tag)

    # --- Phase 4: Ethnicity conflict resolution ---
    for ethnicity_group in ETHNICITY_GROUPS:
        # Count tokens in each ethnicity group
        group_counts = {}
        group_confidences = {}

        for group_name, group_terms in ethnicity_group.items():
            count = 0
            max_conf = 0.0
            for term in group_terms:
                if term in text_to_classifications:
                    count += len(text_to_classifications[term])
                    for cls in text_to_classifications[term]:
                        max_conf = max(max_conf, cls.token.confidence)
            group_counts[group_name] = count
            group_confidences[group_name] = max_conf

        # If multiple ethnicity groups have matches, keep only dominant one
        active_groups = [g for g, c in group_counts.items() if c > 0]
        if len(active_groups) > 1:
            # Determine dominant ethnicity by count, then confidence
            dominant = max(active_groups, key=lambda g: (group_counts[g], group_confidences[g]))

            # Remove tokens from non-dominant groups
            for group_name, group_terms in ethnicity_group.items():
                if group_name != dominant:
                    for term in group_terms:
                        if term in text_to_classifications:
                            tokens_to_remove.add(term)

    # --- Phase 5: Fantasy tags in realistic images ---
    is_realistic = any(t in all_texts_lower for t in {"realistic", "photograph", "photo", "photography"})
    if is_realistic:
        for fantasy_tag in FANTASY_TAGS:
            if fantasy_tag in text_to_classifications:
                tokens_to_remove.add(fantasy_tag)

    # --- Phase 6: Clothing vs nudity contradictions ---
    has_clothing = any(t in all_texts_lower for t in CLOTHING_INDICATORS)
    if has_clothing:
        for nudity_tag in NUDITY_TAGS:
            if nudity_tag in text_to_classifications:
                tokens_to_remove.add(nudity_tag)

    # --- Phase 7: Remove common false positives ---
    for fp_tag in BODY_PART_FALSE_POSITIVES:
        if fp_tag in text_to_classifications:
            # Only remove if confidence is below threshold
            for cls in text_to_classifications[fp_tag]:
                if cls.token.confidence < 0.5:
                    tokens_to_remove.add(fp_tag)

    # --- Phase 8: Remove known misclassification tags ---
    for misc_tag in MISCLASSIFICATION_TAGS:
        if misc_tag in text_to_classifications:
            tokens_to_remove.add(misc_tag)

    # --- Phase 9: Build filtered result ---
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
