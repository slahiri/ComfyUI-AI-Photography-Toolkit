"""Model factory for creating and managing caption models."""

from typing import Optional, Callable

from .base import BaseCaptionModel
from .florence import FlorenceModel
from .qwen import QwenVLModel
from .qwen3 import Qwen3VLModel


# Model factory functions - return configured model instances

# Florence variants
def _create_florence(model_id: Optional[str], precision: str) -> BaseCaptionModel:
    return FlorenceModel(model_id, precision=precision, config_name="florence")


def _create_florence_base(model_id: Optional[str], precision: str) -> BaseCaptionModel:
    return FlorenceModel(model_id, precision=precision, config_name="florence_base")


# Qwen2/2.5 variants
def _create_qwen(model_id: Optional[str], precision: str) -> BaseCaptionModel:
    return QwenVLModel(model_id, precision=precision, config_name="qwen")


def _create_qwen2_2b(model_id: Optional[str], precision: str) -> BaseCaptionModel:
    return QwenVLModel(model_id, precision=precision, config_name="qwen2_2b_caption")


def _create_qwen25_3b(model_id: Optional[str], precision: str) -> BaseCaptionModel:
    """General-purpose Qwen2.5-VL-3B-Instruct."""
    return QwenVLModel(model_id, precision=precision, config_name="qwen25_3b")


def _create_qwen25_3b_caption(model_id: Optional[str], precision: str) -> BaseCaptionModel:
    """Captioning-tuned Qwen2.5-VL-3B."""
    return QwenVLModel(model_id, precision=precision, config_name="qwen25_3b_caption")


def _create_qwen2_7b(model_id: Optional[str], precision: str) -> BaseCaptionModel:
    return QwenVLModel(model_id, precision=precision, config_name="qwen2_7b_captioner")


def _create_qwen25_7b(model_id: Optional[str], precision: str) -> BaseCaptionModel:
    """General-purpose Qwen2.5-VL-7B-Instruct."""
    return QwenVLModel(model_id, precision=precision, config_name="qwen25_7b")


def _create_qwen25_7b_captioner(model_id: Optional[str], precision: str) -> BaseCaptionModel:
    """Captioning-tuned Qwen2.5-VL-7B."""
    return QwenVLModel(model_id, precision=precision, config_name="qwen25_7b_captioner")


def _create_qwen25_7b_caption(model_id: Optional[str], precision: str) -> BaseCaptionModel:
    return QwenVLModel(model_id, precision=precision, config_name="qwen25_7b_caption")


# Qwen3 variants - Abliterated (uncensored)
def _create_qwen3_2b_abliterated(model_id: Optional[str], precision: str) -> BaseCaptionModel:
    return Qwen3VLModel(model_id, precision=precision, config_name="qwen3_2b_abliterated")


def _create_qwen3_4b_abliterated(model_id: Optional[str], precision: str) -> BaseCaptionModel:
    return Qwen3VLModel(model_id, precision=precision, config_name="qwen3_4b_abliterated")


def _create_qwen3_8b_abliterated(model_id: Optional[str], precision: str) -> BaseCaptionModel:
    return Qwen3VLModel(model_id, precision=precision, config_name="qwen3_8b_abliterated")


# Qwen3 variants - Base Instruct
def _create_qwen3_2b(model_id: Optional[str], precision: str) -> BaseCaptionModel:
    return Qwen3VLModel(model_id, precision=precision, config_name="qwen3_2b")


def _create_qwen3_4b(model_id: Optional[str], precision: str) -> BaseCaptionModel:
    return Qwen3VLModel(model_id, precision=precision, config_name="qwen3_4b")


def _create_qwen3_8b(model_id: Optional[str], precision: str) -> BaseCaptionModel:
    return Qwen3VLModel(model_id, precision=precision, config_name="qwen3_8b")


# Qwen2.5 variants - Abliterated (uncensored)
def _create_qwen25_3b_abliterated(model_id: Optional[str], precision: str) -> BaseCaptionModel:
    return QwenVLModel(model_id, precision=precision, config_name="qwen25_3b_abliterated")


def _create_qwen25_7b_abliterated(model_id: Optional[str], precision: str) -> BaseCaptionModel:
    return QwenVLModel(model_id, precision=precision, config_name="qwen25_7b_abliterated")


# Qwen2 variants - Abliterated (uncensored)
def _create_qwen2_2b_abliterated(model_id: Optional[str], precision: str) -> BaseCaptionModel:
    return QwenVLModel(model_id, precision=precision, config_name="qwen2_2b_abliterated")


# Qwen2 variants - Base Instruct
def _create_qwen2_2b_instruct(model_id: Optional[str], precision: str) -> BaseCaptionModel:
    return QwenVLModel(model_id, precision=precision, config_name="qwen2_2b")


def _create_qwen2_7b_instruct(model_id: Optional[str], precision: str) -> BaseCaptionModel:
    return QwenVLModel(model_id, precision=precision, config_name="qwen2_7b")


# Registry of available models: name -> factory function
_MODEL_REGISTRY: dict[str, Callable[[Optional[str], str], BaseCaptionModel]] = {
    # Legacy models (for backwards compatibility)
    "qwen": _create_qwen,

    # Florence models
    "florence": _create_florence,           # Florence-2-large-PromptGen-v2.0
    "florence_base": _create_florence_base, # Florence-2-base-PromptGen-v1.5

    # Qwen2-VL models
    "qwen2_2b": _create_qwen2_2b,                        # Legacy: captioning-tuned
    "qwen2_2b_abliterated": _create_qwen2_2b_abliterated,  # Abliterated (uncensored)
    "qwen2_2b_instruct": _create_qwen2_2b_instruct,     # Base Instruct
    "qwen2_7b": _create_qwen2_7b,                        # Legacy: captioning-tuned
    "qwen2_7b_instruct": _create_qwen2_7b_instruct,     # Base Instruct

    # Qwen2.5-VL models - Abliterated (uncensored, for Singleshot)
    "qwen25_3b_abliterated": _create_qwen25_3b_abliterated,
    "qwen25_7b_abliterated": _create_qwen25_7b_abliterated,

    # Qwen2.5-VL models - Base Instruct (for Refine)
    "qwen25_3b": _create_qwen25_3b,
    "qwen25_7b": _create_qwen25_7b,

    # Qwen2.5-VL models - Captioning-tuned (legacy)
    "qwen25_3b_caption": _create_qwen25_3b_caption,
    "qwen25_7b_captioner": _create_qwen25_7b_captioner,
    "qwen25_7b_caption": _create_qwen25_7b_caption,

    # Qwen3-VL models - Abliterated (uncensored, for Singleshot)
    "qwen3_2b_abliterated": _create_qwen3_2b_abliterated,
    "qwen3_4b_abliterated": _create_qwen3_4b_abliterated,
    "qwen3_8b_abliterated": _create_qwen3_8b_abliterated,

    # Qwen3-VL models - Base Instruct (for Refine)
    "qwen3_2b": _create_qwen3_2b,
    "qwen3_4b": _create_qwen3_4b,
    "qwen3_8b": _create_qwen3_8b,
}

# Model cache for reuse
_model_cache: dict[str, BaseCaptionModel] = {}

# Precision mode mapping
PRECISION_MAP = {
    "Auto": "auto",
    "FP16": "fp16",
    "FP8": "fp8",
    "4-Bit": "4bit",
}


class ModelFactory:
    """
    Factory for creating and caching caption models.

    Usage:
        model = ModelFactory.get("florence")
        caption = model.generate(image)
        ModelFactory.release("florence")  # Optional: free memory
    """

    @staticmethod
    def get(
        name: str,
        model_id: Optional[str] = None,
        precision: str = "Auto",
    ) -> BaseCaptionModel:
        """
        Get a caption model by name.

        Args:
            name: Model name (e.g., "florence", "qwen2_2b", "qwen3_8b")
            model_id: Optional custom model ID (HuggingFace repo)
            precision: Precision mode (Auto/FP16/FP8/4-Bit)

        Returns:
            Caption model instance (cached)

        Raises:
            ValueError: If model name is not registered
        """
        # Normalize precision
        precision_mode = PRECISION_MAP.get(precision, "auto")
        cache_key = f"{name}:{model_id or 'default'}:{precision_mode}"

        if cache_key in _model_cache:
            return _model_cache[cache_key]

        if name not in _MODEL_REGISTRY:
            available = ", ".join(_MODEL_REGISTRY.keys())
            raise ValueError(f"Unknown model: {name}. Available: {available}")

        factory_fn = _MODEL_REGISTRY[name]
        model = factory_fn(model_id, precision_mode)

        _model_cache[cache_key] = model
        return model

    @staticmethod
    def release(name: str, model_id: Optional[str] = None) -> None:
        """
        Release all precision variants of a model from cache and free memory.

        Args:
            name: Model name
            model_id: Optional custom model ID
        """
        prefix = f"{name}:{model_id or 'default'}:"

        # Find and release all precision variants
        keys_to_remove = [k for k in _model_cache.keys() if k.startswith(prefix)]
        for cache_key in keys_to_remove:
            model = _model_cache.pop(cache_key)
            model.unload()

    @staticmethod
    def release_all() -> None:
        """Release all cached models and free memory."""
        for model in _model_cache.values():
            model.unload()
        _model_cache.clear()

    @staticmethod
    def available_models() -> list[str]:
        """Get list of available model names."""
        return list(_MODEL_REGISTRY.keys())

    @staticmethod
    def register(name: str, factory_fn: Callable[[Optional[str], str], BaseCaptionModel]) -> None:
        """
        Register a new model type.

        Args:
            name: Model name for lookup
            factory_fn: Factory function that creates the model
        """
        _MODEL_REGISTRY[name] = factory_fn
