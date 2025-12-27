"""Extract tokens from analyzer outputs (photography, IQA, composition, saliency, fashion).

Analyzers produce structured quality/attribute assessments.
This module converts them to ImageToken format.
"""

from typing import Dict, List, Any, Optional

from .base import ImageToken, TokenType, TokenSource, TokenBatch


def extract_from_photography(metadata: Dict[str, Any], min_confidence: float = 0.0) -> List[ImageToken]:
    """Extract tokens from photography analyzer output.

    Photography format: {"attribute": confidence, ...}
    Attributes describe lighting, tones, focus, framing, etc.
    """
    tokens = []
    photo_data = metadata.get("photography", {})

    if not isinstance(photo_data, dict):
        return tokens

    for attribute, confidence in photo_data.items():
        if not isinstance(confidence, (int, float)):
            continue
        if confidence < min_confidence:
            continue

        # Photography attributes are already descriptive phrases
        text = attribute.strip()
        if not text:
            continue

        tokens.append(ImageToken(
            text=text,
            confidence=float(confidence),
            source=TokenSource.PHOTOGRAPHY,
            token_type=TokenType.TAG,
            metadata={"analyzer": "photography"}
        ))

    return tokens


def extract_from_iqa(metadata: Dict[str, Any], min_confidence: float = 0.0) -> List[ImageToken]:
    """Extract tokens from Image Quality Assessment output.

    IQA format: {"quality descriptor": confidence, ...}
    Describes aesthetic and technical quality.
    """
    tokens = []
    iqa_data = metadata.get("iqa", {})

    if not isinstance(iqa_data, dict):
        return tokens

    for descriptor, confidence in iqa_data.items():
        if not isinstance(confidence, (int, float)):
            continue
        if confidence < min_confidence:
            continue

        text = descriptor.strip()
        if not text:
            continue

        tokens.append(ImageToken(
            text=text,
            confidence=float(confidence),
            source=TokenSource.IQA,
            token_type=TokenType.TAG,
            metadata={"analyzer": "iqa"}
        ))

    return tokens


def extract_from_composition(metadata: Dict[str, Any], min_confidence: float = 0.0) -> List[ImageToken]:
    """Extract tokens from composition analyzer output.

    Composition format: {"element": confidence, ...}
    Describes composition elements like lines, symmetry, thirds.
    """
    tokens = []
    comp_data = metadata.get("composition", {})

    if not isinstance(comp_data, dict):
        return tokens

    for element, confidence in comp_data.items():
        if not isinstance(confidence, (int, float)):
            continue
        if confidence < min_confidence:
            continue

        text = element.strip()
        if not text:
            continue

        tokens.append(ImageToken(
            text=text,
            confidence=float(confidence),
            source=TokenSource.COMPOSITION,
            token_type=TokenType.TAG,
            metadata={"analyzer": "composition"}
        ))

    return tokens


def extract_from_saliency(metadata: Dict[str, Any], min_confidence: float = 0.0) -> List[ImageToken]:
    """Extract tokens from saliency analyzer output.

    Saliency format: {"position descriptor": confidence, ...}
    Describes where the subject/focus is located.
    """
    tokens = []
    saliency_data = metadata.get("saliency", {})

    if not isinstance(saliency_data, dict):
        return tokens

    for descriptor, confidence in saliency_data.items():
        if not isinstance(confidence, (int, float)):
            continue
        if confidence < min_confidence:
            continue

        text = descriptor.strip()
        if not text:
            continue

        tokens.append(ImageToken(
            text=text,
            confidence=float(confidence),
            source=TokenSource.SALIENCY,
            token_type=TokenType.TAG,
            metadata={"analyzer": "saliency"}
        ))

    return tokens


def extract_from_fashion(metadata: Dict[str, Any], min_confidence: float = 0.0) -> List[ImageToken]:
    """Extract tokens from fashion analyzer outputs.

    Fashion sources: fashion_yolov8, fashion_yolos, fashion_segformer
    Format: {"clothing descriptor": confidence, ...}
    """
    tokens = []

    fashion_sources = ["fashion_yolov8", "fashion_yolos", "fashion_segformer"]

    for source_key in fashion_sources:
        fashion_data = metadata.get(source_key, {})

        if not isinstance(fashion_data, dict):
            continue

        for descriptor, confidence in fashion_data.items():
            if not isinstance(confidence, (int, float)):
                continue
            if confidence < min_confidence:
                continue

            text = descriptor.strip()
            if not text:
                continue

            tokens.append(ImageToken(
                text=text,
                confidence=float(confidence),
                source=TokenSource.UNKNOWN,  # Fashion doesn't have dedicated source
                token_type=TokenType.TAG,
                metadata={
                    "analyzer": "fashion",
                    "fashion_source": source_key
                }
            ))

    return tokens


def extract_all_analyzer_tokens(
    metadata: Dict[str, Any],
    min_confidence: float = 0.0,
    sources: Optional[List[str]] = None
) -> TokenBatch:
    """Extract tokens from all analyzer sources.

    Args:
        metadata: Full metadata dictionary
        min_confidence: Minimum confidence threshold for all sources
        sources: Optional list of sources to extract from.
                 Defaults to all: ["photography", "iqa", "composition", "saliency", "fashion"]

    Returns:
        TokenBatch containing all extracted tokens
    """
    batch = TokenBatch()
    batch.image_info = metadata.get("image_info", {})

    if sources is None:
        sources = ["photography", "iqa", "composition", "saliency", "fashion"]

    extractors = {
        "photography": extract_from_photography,
        "iqa": extract_from_iqa,
        "composition": extract_from_composition,
        "saliency": extract_from_saliency,
        "fashion": extract_from_fashion,
    }

    for source in sources:
        if source in extractors:
            tokens = extractors[source](metadata, min_confidence)
            batch.add_all(tokens)

    return batch
