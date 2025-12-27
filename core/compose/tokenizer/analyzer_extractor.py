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


def extract_from_shot_type(metadata: Dict[str, Any], min_confidence: float = 0.0) -> List[ImageToken]:
    """Extract tokens from shot type classifier output.

    Shot type format: {"shot type": confidence, ...}
    Describes camera framing: close-up, medium shot, full shot, etc.
    """
    tokens = []
    shot_data = metadata.get("shot_type", {})

    if not isinstance(shot_data, dict):
        return tokens

    for shot_type, confidence in shot_data.items():
        if not isinstance(confidence, (int, float)):
            continue
        if confidence < min_confidence:
            continue

        text = shot_type.strip()
        if not text:
            continue

        tokens.append(ImageToken(
            text=text,
            confidence=float(confidence),
            source=TokenSource.SHOT_TYPE,
            token_type=TokenType.TAG,
            metadata={"analyzer": "shot_type"}
        ))

    return tokens


def extract_from_clip_camera(metadata: Dict[str, Any], min_confidence: float = 0.0) -> List[ImageToken]:
    """Extract tokens from CLIP camera attributes output.

    CLIP camera format: {"camera attribute": confidence, ...}
    Describes camera angles, DOF, perspective using CLIP zero-shot.
    """
    tokens = []
    clip_data = metadata.get("clip_camera", {})

    if not isinstance(clip_data, dict):
        return tokens

    for attribute, confidence in clip_data.items():
        if not isinstance(confidence, (int, float)):
            continue
        if confidence < min_confidence:
            continue

        text = attribute.strip()
        if not text:
            continue

        tokens.append(ImageToken(
            text=text,
            confidence=float(confidence),
            source=TokenSource.CLIP_CAMERA,
            token_type=TokenType.TAG,
            metadata={"analyzer": "clip_camera"}
        ))

    return tokens


def extract_from_lighting(metadata: Dict[str, Any], min_confidence: float = 0.0) -> List[ImageToken]:
    """Extract tokens from intrinsic lighting analyzer output.

    Lighting format: {"lighting descriptor": confidence, ...}
    Describes light direction, quality, shadows, highlights, style.
    """
    tokens = []
    lighting_data = metadata.get("lighting", {})

    if not isinstance(lighting_data, dict):
        return tokens

    for descriptor, confidence in lighting_data.items():
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
            source=TokenSource.LIGHTING,
            token_type=TokenType.TAG,
            metadata={"analyzer": "lighting"}
        ))

    return tokens


def extract_from_shadow(metadata: Dict[str, Any], min_confidence: float = 0.0) -> List[ImageToken]:
    """Extract tokens from shadow detector output.

    Shadow format: {"shadow descriptor": confidence, ...}
    Describes shadow presence, intensity, distribution.
    """
    tokens = []
    shadow_data = metadata.get("shadow", {})

    if not isinstance(shadow_data, dict):
        return tokens

    for descriptor, confidence in shadow_data.items():
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
            source=TokenSource.SHADOW,
            token_type=TokenType.TAG,
            metadata={"analyzer": "shadow"}
        ))

    return tokens


def extract_from_deepfashion(metadata: Dict[str, Any], min_confidence: float = 0.0) -> List[ImageToken]:
    """Extract tokens from DeepFashion attribute analyzer output.

    DeepFashion format: {"attribute": confidence, ...}
    Describes fabric, pattern, texture, style, fit, neckline, sleeves, length, color.
    """
    tokens = []
    df_data = metadata.get("deepfashion_attributes", {})

    if not isinstance(df_data, dict):
        return tokens

    for attribute, confidence in df_data.items():
        if not isinstance(confidence, (int, float)):
            continue
        if confidence < min_confidence:
            continue

        text = attribute.strip()
        if not text:
            continue

        tokens.append(ImageToken(
            text=text,
            confidence=float(confidence),
            source=TokenSource.DEEPFASHION,
            token_type=TokenType.TAG,
            metadata={"analyzer": "deepfashion_attributes"}
        ))

    return tokens


def extract_from_fashion_color(metadata: Dict[str, Any], min_confidence: float = 0.0) -> List[ImageToken]:
    """Extract tokens from fashion color analyzer output.

    Fashion color format: {"color descriptor": confidence, ...}
    Describes dominant colors in clothing using k-means clustering.
    """
    tokens = []
    color_data = metadata.get("fashion_color", {})

    if not isinstance(color_data, dict):
        return tokens

    for color_desc, confidence in color_data.items():
        if not isinstance(confidence, (int, float)):
            continue
        if confidence < min_confidence:
            continue

        text = color_desc.strip()
        if not text:
            continue

        tokens.append(ImageToken(
            text=text,
            confidence=float(confidence),
            source=TokenSource.FASHION_COLOR,
            token_type=TokenType.TAG,
            metadata={"analyzer": "fashion_color"}
        ))

    return tokens


def extract_from_fashion(metadata: Dict[str, Any], min_confidence: float = 0.0) -> List[ImageToken]:
    """Extract tokens from fashion analyzer outputs.

    Fashion sources: fashion_yolov8, fashion_yolos, fashion_segformer
    Format: {"clothing descriptor": confidence, ...}

    Note: Different sources have different confidence thresholds due to
    varying false positive rates:
    - yolov8: Higher accuracy, use standard threshold
    - yolos: Moderate accuracy, use standard threshold
    - segformer: Higher false positive rate, requires higher threshold
    """
    tokens = []

    # Source-specific confidence thresholds
    # Segformer has higher false positive rate, requires higher confidence
    source_thresholds = {
        "fashion_yolov8": min_confidence,
        "fashion_yolos": max(min_confidence, 0.4),  # Slightly higher
        "fashion_segformer": max(min_confidence, 0.6),  # Much higher due to FP rate
    }

    # Common false positive patterns to filter
    fashion_false_positives = {
        "glasses",  # Often detected incorrectly
        "wearing gloves",  # Often incorrect on bare hands
        "wearing headwear",  # General term, often wrong
        "wearing shoes",  # Not visible in upper body shots
    }

    for source_key, threshold in source_thresholds.items():
        fashion_data = metadata.get(source_key, {})

        if not isinstance(fashion_data, dict):
            continue

        for descriptor, confidence in fashion_data.items():
            if not isinstance(confidence, (int, float)):
                continue
            if confidence < threshold:
                continue

            text = descriptor.strip().lower()
            if not text:
                continue

            # Skip known false positive patterns
            if text in fashion_false_positives:
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
        sources = [
            "photography", "iqa", "composition", "saliency",
            "shot_type", "clip_camera", "lighting", "shadow",
            "deepfashion_attributes", "fashion_color", "fashion"
        ]

    extractors = {
        "photography": extract_from_photography,
        "iqa": extract_from_iqa,
        "composition": extract_from_composition,
        "saliency": extract_from_saliency,
        "shot_type": extract_from_shot_type,
        "clip_camera": extract_from_clip_camera,
        "lighting": extract_from_lighting,
        "shadow": extract_from_shadow,
        "deepfashion_attributes": extract_from_deepfashion,
        "fashion_color": extract_from_fashion_color,
        "fashion": extract_from_fashion,
    }

    for source in sources:
        if source in extractors:
            tokens = extractors[source](metadata, min_confidence)
            batch.add_all(tokens)

    return batch
