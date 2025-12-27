"""Tokenizer module for extracting tokens from image metadata.

This module handles Phase 1-2 of the pipeline:
- Extract tokens from taggers (wd14, joytag, pixai, nudenet)
- Extract tokens from analyzers (photography, iqa, composition, saliency)
- Extract tokens from captions (florence, vlm)
- Normalize and deduplicate tokens
"""

from .base import (
    TokenType,
    TokenSource,
    ImageToken,
    TokenBatch,
    SOURCE_WEIGHTS,
)

from .tagger_extractor import (
    extract_from_wd14,
    extract_from_joytag,
    extract_from_pixai,
    extract_from_nudenet,
    extract_all_tagger_tokens,
    normalize_tag,
)

from .analyzer_extractor import (
    extract_from_photography,
    extract_from_iqa,
    extract_from_composition,
    extract_from_saliency,
    extract_from_fashion,
    extract_all_analyzer_tokens,
)

from .caption_extractor import (
    extract_from_florence_caption,
    extract_from_florence_description,
    extract_from_florence_analyze,
    extract_from_florence_mixed,
    extract_from_vlm_description,
    extract_all_caption_tokens,
    split_sentences,
    split_clauses,
)

from .normalizer import (
    deduplicate_tokens,
    deduplicate_near_duplicates,
    aggregate_confidence,
    filter_low_confidence,
    filter_by_length,
    filter_meta_tokens,
    normalize_batch,
    normalize_for_comparison,
)

__all__ = [
    # Base types
    "TokenType",
    "TokenSource",
    "ImageToken",
    "TokenBatch",
    "SOURCE_WEIGHTS",
    # Tagger extraction
    "extract_from_wd14",
    "extract_from_joytag",
    "extract_from_pixai",
    "extract_from_nudenet",
    "extract_all_tagger_tokens",
    "normalize_tag",
    # Analyzer extraction
    "extract_from_photography",
    "extract_from_iqa",
    "extract_from_composition",
    "extract_from_saliency",
    "extract_from_fashion",
    "extract_all_analyzer_tokens",
    # Caption extraction
    "extract_from_florence_caption",
    "extract_from_florence_description",
    "extract_from_florence_analyze",
    "extract_from_florence_mixed",
    "extract_from_vlm_description",
    "extract_all_caption_tokens",
    "split_sentences",
    "split_clauses",
    # Normalization
    "deduplicate_tokens",
    "deduplicate_near_duplicates",
    "aggregate_confidence",
    "filter_low_confidence",
    "filter_by_length",
    "filter_meta_tokens",
    "normalize_batch",
    "normalize_for_comparison",
]


def extract_all_tokens(metadata: dict, min_confidence: float = 0.3) -> TokenBatch:
    """Extract and normalize tokens from all metadata sources.

    This is the main entry point for tokenization.

    Args:
        metadata: Full metadata dictionary from image analysis
        min_confidence: Minimum confidence threshold

    Returns:
        Normalized TokenBatch containing all extracted tokens
    """
    batch = TokenBatch()
    batch.image_info = metadata.get("image_info", {})

    # Extract from taggers
    tagger_batch = extract_all_tagger_tokens(metadata, min_confidence=0.0)
    batch.add_all(tagger_batch.tokens)

    # Extract from analyzers
    analyzer_batch = extract_all_analyzer_tokens(metadata, min_confidence=0.0)
    batch.add_all(analyzer_batch.tokens)

    # Extract from captions
    caption_batch = extract_all_caption_tokens(metadata, split_into_clauses=False)
    batch.add_all(caption_batch.tokens)

    # Normalize
    batch = normalize_batch(
        batch,
        min_confidence=min_confidence,
        deduplicate=True,
        fuzzy_dedupe=False
    )

    return batch
