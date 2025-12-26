"""Base class for caption models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch
from PIL import Image

from ..platform import cleanup_memory


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
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: Optional[int] = None  # None = disabled


# =============================================================================
# Shared utilities for all models
# =============================================================================

def get_dtype(dtype_str: str) -> torch.dtype:
    """Convert dtype string to torch dtype."""
    dtype_map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    return dtype_map.get(dtype_str, torch.float16)


def get_quantization_config(precision: str):
    """
    Get quantization config for the specified precision.

    Args:
        precision: One of 'auto', 'fp16', 'fp8', '4bit'

    Returns:
        BitsAndBytesConfig or None
    """
    if precision in ("auto", "fp16"):
        return None

    try:
        from transformers import BitsAndBytesConfig

        if precision == "4bit":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif precision == "fp8":
            return BitsAndBytesConfig(
                load_in_8bit=True,
            )
    except ImportError:
        print("[SID-Toolkit] Warning: bitsandbytes not installed, falling back to fp16")
        return None

    return None


class BaseCaptionModel(ABC):
    """
    Abstract base class for caption models.

    All caption models must implement:
    - load(): Load model into memory
    - generate(): Generate caption from image
    - unload(): Release model from memory
    """

    def __init__(self, model_id: str, precision: str = "auto"):
        """
        Initialize model.

        Args:
            model_id: HuggingFace model ID or local path
            precision: Precision mode (auto/fp16/fp8/4bit)
        """
        self.model_id = model_id
        self.precision = precision
        self._model = None
        self._processor = None
        self._quantized = False
        self._device = None
        self._offload_device = None

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None

    @property
    def is_quantized(self) -> bool:
        """Check if model is loaded with quantization."""
        return self._quantized

    @property
    def target_device(self):
        """Get the device to use for inference."""
        if self._quantized and self._model is not None:
            return self._model.device
        return self._device

    def _move_to_device(self) -> None:
        """Move model to compute device (skip for quantized)."""
        if not self._quantized and self._model is not None:
            self._model.to(self._device)

    def _move_to_offload(self) -> None:
        """Move model to offload device (skip for quantized)."""
        if not self._quantized and self._model is not None:
            self._model.to(self._offload_device)

    def _cleanup(self) -> None:
        """Common cleanup logic for unload."""
        import gc

        if self._model is not None:
            # For quantized models, we need special handling
            if self._quantized:
                # Try to delete any cached states
                try:
                    if hasattr(self._model, 'eval'):
                        self._model.eval()
                    # Clear any cached gradients or buffers
                    for param in self._model.parameters():
                        param.data = None
                        if param.grad is not None:
                            param.grad = None
                except Exception:
                    pass
            else:
                try:
                    self._model.to("cpu")
                except Exception:
                    pass

            # Delete model reference
            del self._model
            self._model = None

        if self._processor is not None:
            del self._processor
            self._processor = None

        self._quantized = False

        # Force garbage collection before CUDA cleanup
        gc.collect()
        gc.collect()

        cleanup_memory(aggressive=True)

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
