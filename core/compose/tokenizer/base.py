"""Base data structures for image tokenization.

This module defines the core token representation used throughout
the tokenization → classification → assembly pipeline.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Dict, Any, List


class TokenType(Enum):
    """Type of token based on its structure and origin.

    TAG: Single word or compound tag (e.g., "brown_hair", "1girl")
    PHRASE: Natural language phrase (e.g., "long wavy brown hair")
    KEY_VALUE: Structured key-value pair (e.g., camera_angle: front view)
    SENTENCE: Complete sentence from caption
    """
    TAG = auto()
    PHRASE = auto()
    KEY_VALUE = auto()
    SENTENCE = auto()


class TokenSource(Enum):
    """Source of the token - which model/analyzer produced it.

    Used for source-based routing in classification and
    confidence weighting during normalization.
    """
    # Taggers
    WD14 = "wd14"
    JOYTAG = "joytag"
    PIXAI = "pixai"
    NUDENET = "nudenet"

    # Analyzers
    PHOTOGRAPHY = "photography"
    IQA = "iqa"
    COMPOSITION = "composition"
    SALIENCY = "saliency"

    # Captions (Florence)
    FLORENCE_CAPTION = "florence_caption"
    FLORENCE_DESCRIPTION = "florence_description"
    FLORENCE_ANALYZE = "florence_analyze"
    FLORENCE_MIXED = "florence_mixed_caption"

    # VLM
    VLM_DESCRIPTION = "vlm_description"

    # Unknown/fallback
    UNKNOWN = "unknown"


# Source reliability weights for confidence aggregation
SOURCE_WEIGHTS: Dict[TokenSource, float] = {
    TokenSource.WD14: 1.0,
    TokenSource.JOYTAG: 0.95,
    TokenSource.PIXAI: 0.9,
    TokenSource.NUDENET: 0.8,
    TokenSource.PHOTOGRAPHY: 0.9,
    TokenSource.IQA: 0.85,
    TokenSource.COMPOSITION: 0.85,
    TokenSource.SALIENCY: 0.8,
    TokenSource.FLORENCE_CAPTION: 0.85,
    TokenSource.FLORENCE_DESCRIPTION: 0.85,
    TokenSource.FLORENCE_ANALYZE: 0.9,
    TokenSource.FLORENCE_MIXED: 0.85,
    TokenSource.VLM_DESCRIPTION: 0.9,
    TokenSource.UNKNOWN: 0.5,
}


@dataclass
class ImageToken:
    """A single token extracted from image metadata.

    Represents any piece of information about an image, whether it's
    a tag from a tagger, a phrase from a caption, or a key-value pair
    from structured analysis.

    Attributes:
        text: The actual content (normalized, lowercase)
        confidence: Confidence score from source (0.0-1.0)
        source: Which model/analyzer produced this token
        token_type: Structure type (TAG, PHRASE, KEY_VALUE, SENTENCE)
        metadata: Additional context (e.g., key for KEY_VALUE tokens)

    Example:
        ImageToken(
            text="brown hair",
            confidence=0.915,
            source=TokenSource.WD14,
            token_type=TokenType.TAG,
            metadata={"original": "brown_hair"}
        )
    """
    text: str
    confidence: float
    source: TokenSource
    token_type: TokenType
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Normalize text on creation."""
        # Preserve original text before full normalization (strip edges only)
        if "original" not in self.metadata:
            self.metadata["original"] = self.text.strip()
        # Normalize whitespace (collapse multiple spaces)
        self.text = " ".join(self.text.split())
        # Lowercase for consistency
        self.text = self.text.lower()
        # Clamp confidence
        self.confidence = max(0.0, min(1.0, self.confidence))

    @property
    def source_weight(self) -> float:
        """Get reliability weight for this token's source."""
        return SOURCE_WEIGHTS.get(self.source, 0.5)

    @property
    def weighted_confidence(self) -> float:
        """Get confidence weighted by source reliability."""
        return self.confidence * self.source_weight

    @property
    def key(self) -> Optional[str]:
        """Get key for KEY_VALUE tokens."""
        if self.token_type == TokenType.KEY_VALUE:
            return self.metadata.get("key")
        return None

    def __hash__(self):
        """Hash based on normalized text for deduplication."""
        return hash(self.text)

    def __eq__(self, other):
        """Equal if normalized text matches."""
        if isinstance(other, ImageToken):
            return self.text == other.text
        return False

    def __repr__(self):
        return f"ImageToken({self.text!r}, {self.confidence:.2f}, {self.source.value})"


@dataclass
class TokenBatch:
    """Collection of tokens from a single image.

    Provides convenience methods for filtering and grouping tokens.
    """
    tokens: List[ImageToken] = field(default_factory=list)
    image_info: Dict[str, Any] = field(default_factory=dict)

    def add(self, token: ImageToken) -> None:
        """Add a token to the batch."""
        self.tokens.append(token)

    def add_all(self, tokens: List[ImageToken]) -> None:
        """Add multiple tokens to the batch."""
        self.tokens.extend(tokens)

    def filter_by_source(self, source: TokenSource) -> List[ImageToken]:
        """Get all tokens from a specific source."""
        return [t for t in self.tokens if t.source == source]

    def filter_by_type(self, token_type: TokenType) -> List[ImageToken]:
        """Get all tokens of a specific type."""
        return [t for t in self.tokens if t.token_type == token_type]

    def filter_by_confidence(self, min_confidence: float) -> List[ImageToken]:
        """Get tokens above a confidence threshold."""
        return [t for t in self.tokens if t.confidence >= min_confidence]

    def group_by_source(self) -> Dict[TokenSource, List[ImageToken]]:
        """Group tokens by their source."""
        groups: Dict[TokenSource, List[ImageToken]] = {}
        for token in self.tokens:
            if token.source not in groups:
                groups[token.source] = []
            groups[token.source].append(token)
        return groups

    def __len__(self):
        return len(self.tokens)

    def __iter__(self):
        return iter(self.tokens)
