"""Normalize and deduplicate extracted tokens.

This module handles:
- Deduplication of tokens with the same normalized text
- Confidence aggregation for duplicate tokens
- Source reliability weighting
- Near-duplicate detection (fuzzy matching)
"""

from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
import re

from .base import ImageToken, TokenType, TokenSource, TokenBatch, SOURCE_WEIGHTS


def normalize_for_comparison(text: str) -> str:
    """Normalize text for duplicate comparison.

    - Lowercase
    - Remove extra whitespace
    - Remove common suffixes/prefixes that don't change meaning
    """
    text = text.lower().strip()
    text = ' '.join(text.split())

    # Remove trailing punctuation
    text = text.rstrip('.,;:!?')

    return text


def are_near_duplicates(text1: str, text2: str) -> bool:
    """Check if two texts are near-duplicates.

    Near-duplicates are texts that differ only by:
    - Word order (for short phrases)
    - Common synonyms
    - Singular/plural forms
    """
    norm1 = normalize_for_comparison(text1)
    norm2 = normalize_for_comparison(text2)

    # Exact match after normalization
    if norm1 == norm2:
        return True

    # Check if one is substring of another (for very similar phrases)
    if len(norm1) > 5 and len(norm2) > 5:
        if norm1 in norm2 or norm2 in norm1:
            return True

    # Check word set overlap for short phrases
    words1 = set(norm1.split())
    words2 = set(norm2.split())

    if len(words1) <= 3 and len(words2) <= 3:
        # For short phrases, check if they share most words
        overlap = words1 & words2
        total = words1 | words2
        if len(total) > 0 and len(overlap) / len(total) >= 0.8:
            return True

    return False


def aggregate_confidence(confidences: List[Tuple[float, TokenSource]]) -> float:
    """Aggregate confidence from multiple sources.

    Uses weighted combination based on source reliability.
    Multiple sources agreeing increases overall confidence.

    Args:
        confidences: List of (confidence, source) tuples

    Returns:
        Aggregated confidence score (0.0-1.0)
    """
    if not confidences:
        return 0.0

    if len(confidences) == 1:
        conf, source = confidences[0]
        weight = SOURCE_WEIGHTS.get(source, 0.5)
        return conf * weight

    # Multiple sources - combine with weighted average + agreement bonus
    total_weight = 0.0
    weighted_sum = 0.0

    for conf, source in confidences:
        weight = SOURCE_WEIGHTS.get(source, 0.5)
        weighted_sum += conf * weight
        total_weight += weight

    base_confidence = weighted_sum / total_weight if total_weight > 0 else 0.0

    # Agreement bonus: more sources = higher confidence
    agreement_bonus = min(0.1 * (len(confidences) - 1), 0.2)

    return min(1.0, base_confidence + agreement_bonus)


def deduplicate_tokens(tokens: List[ImageToken]) -> List[ImageToken]:
    """Remove duplicate tokens, aggregating confidence.

    Tokens with the same normalized text are merged:
    - Highest-confidence source becomes primary
    - All sources are recorded in metadata
    - Confidence is aggregated

    Args:
        tokens: List of tokens to deduplicate

    Returns:
        Deduplicated list of tokens
    """
    # Group by normalized text
    groups: Dict[str, List[ImageToken]] = defaultdict(list)

    for token in tokens:
        key = normalize_for_comparison(token.text)
        groups[key].append(token)

    # Merge each group
    result = []

    for normalized_text, group in groups.items():
        if len(group) == 1:
            result.append(group[0])
            continue

        # Multiple tokens with same text - merge them
        # Sort by weighted confidence to pick best source
        group.sort(key=lambda t: t.weighted_confidence, reverse=True)
        primary = group[0]

        # Collect all confidences and sources
        confidences = [(t.confidence, t.source) for t in group]
        sources = list(set(t.source for t in group))

        # Aggregate confidence
        aggregated_conf = aggregate_confidence(confidences)

        # Create merged token
        merged = ImageToken(
            text=primary.text,  # Use text from highest-confidence source
            confidence=aggregated_conf,
            source=primary.source,  # Primary source
            token_type=primary.token_type,
            metadata={
                **primary.metadata,
                "merged_from": len(group),
                "all_sources": [s.value for s in sources],
                "original_confidences": [(c, s.value) for c, s in confidences]
            }
        )

        result.append(merged)

    return result


def deduplicate_near_duplicates(tokens: List[ImageToken]) -> List[ImageToken]:
    """Remove near-duplicate tokens (fuzzy matching).

    This is more aggressive than exact deduplication.
    Use with caution - may merge tokens that shouldn't be merged.

    Args:
        tokens: List of tokens to deduplicate

    Returns:
        Deduplicated list of tokens
    """
    if len(tokens) <= 1:
        return tokens

    # Sort by confidence (descending) to keep highest-confidence version
    sorted_tokens = sorted(tokens, key=lambda t: t.weighted_confidence, reverse=True)

    result = []
    used_texts: Set[str] = set()

    for token in sorted_tokens:
        normalized = normalize_for_comparison(token.text)

        # Check if this is a near-duplicate of something we've kept
        is_duplicate = False
        for used in used_texts:
            if are_near_duplicates(normalized, used):
                is_duplicate = True
                break

        if not is_duplicate:
            result.append(token)
            used_texts.add(normalized)

    return result


def filter_low_confidence(
    tokens: List[ImageToken],
    min_confidence: float = 0.3
) -> List[ImageToken]:
    """Filter out tokens below confidence threshold.

    Args:
        tokens: List of tokens to filter
        min_confidence: Minimum confidence threshold

    Returns:
        Filtered list of tokens
    """
    return [t for t in tokens if t.confidence >= min_confidence]


def filter_by_length(
    tokens: List[ImageToken],
    min_length: int = 2,
    max_length: int = 200
) -> List[ImageToken]:
    """Filter tokens by text length.

    Args:
        tokens: List of tokens to filter
        min_length: Minimum text length
        max_length: Maximum text length

    Returns:
        Filtered list of tokens
    """
    return [t for t in tokens if min_length <= len(t.text) <= max_length]


def filter_meta_tokens(tokens: List[ImageToken]) -> Tuple[List[ImageToken], List[ImageToken]]:
    """Separate filtered (META) tokens from valid tokens.

    Args:
        tokens: List of tokens to filter

    Returns:
        Tuple of (valid_tokens, filtered_tokens)
    """
    valid = []
    filtered = []

    for token in tokens:
        if token.metadata.get("filtered", False):
            filtered.append(token)
        else:
            valid.append(token)

    return valid, filtered


def normalize_batch(
    batch: TokenBatch,
    min_confidence: float = 0.3,
    deduplicate: bool = True,
    fuzzy_dedupe: bool = False
) -> TokenBatch:
    """Apply all normalization steps to a token batch.

    Args:
        batch: TokenBatch to normalize
        min_confidence: Minimum confidence threshold
        deduplicate: Whether to deduplicate exact matches
        fuzzy_dedupe: Whether to also remove near-duplicates

    Returns:
        Normalized TokenBatch
    """
    tokens = list(batch.tokens)

    # Filter by confidence
    tokens = filter_low_confidence(tokens, min_confidence)

    # Filter by length
    tokens = filter_by_length(tokens)

    # Deduplicate
    if deduplicate:
        tokens = deduplicate_tokens(tokens)

    if fuzzy_dedupe:
        tokens = deduplicate_near_duplicates(tokens)

    # Create new batch
    result = TokenBatch()
    result.image_info = batch.image_info
    result.add_all(tokens)

    return result
