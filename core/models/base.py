"""Base class for caption models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from PIL import Image


class CaptionMode(Enum):
    """Caption generation modes."""
    TAGS = "tags"           # Comma-separated tags
    SHORT = "short"         # Brief caption (~50 words)
    DETAILED = "detailed"   # Detailed caption (~150 words)
    ANALYZE = "analyze"     # Structured analysis


@dataclass
class GenerationConfig:
    """Configuration for caption generation."""
    max_tokens: int = 1024
    num_beams: int = 4
    do_sample: bool = False


class BaseCaptionModel(ABC):
    """
    Abstract base class for caption models.

    All caption models must implement:
    - load(): Load model into memory
    - generate(): Generate caption from image
    - unload(): Release model from memory
    """

    def __init__(self, model_id: str):
        """
        Initialize model.

        Args:
            model_id: HuggingFace model ID or local path
        """
        self.model_id = model_id
        self._model = None
        self._processor = None

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None

    @abstractmethod
    def load(self) -> None:
        """Load model into memory."""
        pass

    @abstractmethod
    def generate(
        self,
        image: Image.Image,
        mode: CaptionMode = CaptionMode.DETAILED,
        config: Optional[GenerationConfig] = None,
    ) -> str:
        """
        Generate caption for image.

        Args:
            image: PIL Image to caption
            mode: Caption generation mode
            config: Optional generation configuration

        Returns:
            Generated caption string
        """
        pass

    @abstractmethod
    def unload(self) -> None:
        """Release model from memory."""
        pass

    def __enter__(self):
        """Context manager entry - load model."""
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - unload model."""
        self.unload()
        return False
