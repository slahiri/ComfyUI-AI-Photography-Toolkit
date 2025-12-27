"""Extract tokens from tagger outputs (WD14, JoyTag, PixAI, NudeNet).

Taggers produce dictionaries of tag -> confidence mappings.
This module converts them to ImageToken format.
"""

from typing import Dict, List, Any, Optional
import re

from .base import ImageToken, TokenType, TokenSource, TokenBatch


# Tag normalization patterns
UNDERSCORE_PATTERN = re.compile(r'_+')
PARENTHESES_PATTERN = re.compile(r'\s*\([^)]*\)\s*')
SPECIAL_CHARS_PATTERN = re.compile(r'[^\w\s\-]')


def normalize_tag(tag: str) -> str:
    """Normalize a tag for consistent processing.

    - Replace underscores with spaces
    - Remove parenthetical content like (medium)
    - Clean up special characters
    - Collapse whitespace
    """
    # Replace underscores with spaces
    text = UNDERSCORE_PATTERN.sub(' ', tag)
    # Remove parenthetical content
    text = PARENTHESES_PATTERN.sub(' ', text)
    # Remove special characters except hyphen
    text = SPECIAL_CHARS_PATTERN.sub('', text)
    # Collapse whitespace
    text = ' '.join(text.split())
    return text.strip()


def extract_from_wd14(metadata: Dict[str, Any], min_confidence: float = 0.0) -> List[ImageToken]:
    """Extract tokens from WD14 tagger output.

    WD14 format: {"tag_name": confidence, ...}

    Args:
        metadata: Full metadata dict containing 'wd14' key
        min_confidence: Minimum confidence threshold

    Returns:
        List of ImageToken objects
    """
    tokens = []
    wd14_data = metadata.get("wd14", {})

    if not isinstance(wd14_data, dict):
        return tokens

    for tag, confidence in wd14_data.items():
        if not isinstance(confidence, (int, float)):
            continue
        if confidence < min_confidence:
            continue

        normalized = normalize_tag(tag)
        if not normalized:
            continue

        tokens.append(ImageToken(
            text=normalized,
            confidence=float(confidence),
            source=TokenSource.WD14,
            token_type=TokenType.TAG,
            metadata={"original_tag": tag}
        ))

    return tokens


def extract_from_joytag(metadata: Dict[str, Any], min_confidence: float = 0.0) -> List[ImageToken]:
    """Extract tokens from JoyTag tagger output.

    JoyTag format: {"tag name": confidence, ...}
    Tags may already have spaces instead of underscores.
    """
    tokens = []
    joytag_data = metadata.get("joytag", {})

    if not isinstance(joytag_data, dict):
        return tokens

    for tag, confidence in joytag_data.items():
        if not isinstance(confidence, (int, float)):
            continue
        if confidence < min_confidence:
            continue

        normalized = normalize_tag(tag)
        if not normalized:
            continue

        tokens.append(ImageToken(
            text=normalized,
            confidence=float(confidence),
            source=TokenSource.JOYTAG,
            token_type=TokenType.TAG,
            metadata={"original_tag": tag}
        ))

    return tokens


def extract_from_pixai(metadata: Dict[str, Any], min_confidence: float = 0.0) -> List[ImageToken]:
    """Extract tokens from PixAI tagger output.

    PixAI format: {"tag name": confidence, ...}
    Similar to WD14 but may have different tag vocabulary.
    """
    tokens = []
    pixai_data = metadata.get("pixai", {})

    if not isinstance(pixai_data, dict):
        return tokens

    for tag, confidence in pixai_data.items():
        if not isinstance(confidence, (int, float)):
            continue
        if confidence < min_confidence:
            continue

        normalized = normalize_tag(tag)
        if not normalized:
            continue

        tokens.append(ImageToken(
            text=normalized,
            confidence=float(confidence),
            source=TokenSource.PIXAI,
            token_type=TokenType.TAG,
            metadata={"original_tag": tag}
        ))

    return tokens


def extract_from_nudenet(metadata: Dict[str, Any], min_confidence: float = 0.0) -> List[ImageToken]:
    """Extract tokens from NudeNet output.

    NudeNet format: {"description": confidence, ...}
    Descriptions are more verbose than tags (e.g., "covered female breast (x2)")
    """
    tokens = []
    nudenet_data = metadata.get("nudenet", {})

    if not isinstance(nudenet_data, dict):
        return tokens

    for description, confidence in nudenet_data.items():
        if not isinstance(confidence, (int, float)):
            continue
        if confidence < min_confidence:
            continue

        # NudeNet descriptions need special handling
        # Remove count indicators like (x2)
        normalized = re.sub(r'\s*\(x\d+\)\s*', '', description)
        normalized = normalize_tag(normalized)
        if not normalized:
            continue

        tokens.append(ImageToken(
            text=normalized,
            confidence=float(confidence),
            source=TokenSource.NUDENET,
            token_type=TokenType.TAG,
            metadata={
                "original_tag": description,
                "nudenet_category": True
            }
        ))

    return tokens


def extract_all_tagger_tokens(
    metadata: Dict[str, Any],
    min_confidence: float = 0.0,
    sources: Optional[List[str]] = None
) -> TokenBatch:
    """Extract tokens from all tagger sources.

    Args:
        metadata: Full metadata dictionary
        min_confidence: Minimum confidence threshold for all sources
        sources: Optional list of sources to extract from.
                 Defaults to all: ["wd14", "joytag", "pixai", "nudenet"]

    Returns:
        TokenBatch containing all extracted tokens
    """
    batch = TokenBatch()
    batch.image_info = metadata.get("image_info", {})

    if sources is None:
        sources = ["wd14", "joytag", "pixai", "nudenet"]

    extractors = {
        "wd14": extract_from_wd14,
        "joytag": extract_from_joytag,
        "pixai": extract_from_pixai,
        "nudenet": extract_from_nudenet,
    }

    for source in sources:
        if source in extractors:
            tokens = extractors[source](metadata, min_confidence)
            batch.add_all(tokens)

    return batch
