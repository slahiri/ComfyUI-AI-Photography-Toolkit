"""Base class for all taggers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
from PIL import Image

from ..platform import cleanup_memory


@dataclass
class TagItem:
    """A single tag with confidence score."""
    text: str
    confidence: float
    category: str = ""  # e.g., "general", "character", "quality"

    def __str__(self) -> str:
        return self.text


@dataclass
class TaggerResult:
    """Result from a tagger."""
    tags: list[TagItem] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)  # For scores, poses, depth maps, etc.

    @property
    def tags_string(self) -> str:
        """Get comma-separated tags string."""
        return ", ".join(t.text for t in self.tags)

    @property
    def tags_by_category(self) -> dict[str, list[TagItem]]:
        """Group tags by category."""
        result = {}
        for tag in self.tags:
            if tag.category not in result:
                result[tag.category] = []
            result[tag.category].append(tag)
        return result


class BaseTagger(ABC):
    """
    Abstract base class for all taggers.

    All taggers must implement:
    - load(): Load model into memory
    - tag(): Process image and return tags
    - unload(): Release model from memory
    """

    # Override in subclass
    TAGGER_NAME: str = "base"

    def __init__(self, **kwargs):
        """
        Initialize tagger.

        Args:
            **kwargs: Tagger-specific configuration
        """
        self._config = kwargs
        self._model = None
        self._is_loaded = False

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded

    @property
    def config(self) -> dict:
        """Get tagger configuration."""
        return self._config

    @abstractmethod
    def load(self) -> None:
        """Load model into memory."""
        pass

    @abstractmethod
    def tag(self, image: Image.Image) -> TaggerResult:
        """
        Process image and return tags.

        Args:
            image: PIL Image to analyze

        Returns:
            TaggerResult with detected tags and metadata
        """
        pass

    def unload(self) -> None:
        """Release model from memory."""
        if self._model is not None:
            del self._model
            self._model = None
        self._is_loaded = False
        cleanup_memory(aggressive=True)

    def __enter__(self):
        """Context manager entry - load model."""
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - unload model."""
        self.unload()
        return False

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(loaded={self.is_loaded})"
