"""Model factory for creating and managing caption models."""

from typing import Optional

from .base import BaseCaptionModel
from .florence import FlorenceModel

# Registry of available models
_MODEL_REGISTRY: dict[str, type[BaseCaptionModel]] = {
    "florence": FlorenceModel,
}

# Model cache for reuse
_model_cache: dict[str, BaseCaptionModel] = {}


class ModelFactory:
    """
    Factory for creating and caching caption models.

    Usage:
        model = ModelFactory.get("florence")
        caption = model.generate(image)
        ModelFactory.release("florence")  # Optional: free memory
    """

    @staticmethod
    def get(name: str, model_id: Optional[str] = None) -> BaseCaptionModel:
        """
        Get a caption model by name.

        Args:
            name: Model name (e.g., "florence", "qwen", "joycaption")
            model_id: Optional custom model ID (HuggingFace repo)

        Returns:
            Caption model instance (cached)

        Raises:
            ValueError: If model name is not registered
        """
        cache_key = f"{name}:{model_id or 'default'}"

        if cache_key in _model_cache:
            return _model_cache[cache_key]

        if name not in _MODEL_REGISTRY:
            available = ", ".join(_MODEL_REGISTRY.keys())
            raise ValueError(f"Unknown model: {name}. Available: {available}")

        model_class = _MODEL_REGISTRY[name]
        model = model_class(model_id) if model_id else model_class()

        _model_cache[cache_key] = model
        return model

    @staticmethod
    def release(name: str, model_id: Optional[str] = None) -> None:
        """
        Release a model from cache and free memory.

        Args:
            name: Model name
            model_id: Optional custom model ID
        """
        cache_key = f"{name}:{model_id or 'default'}"

        if cache_key in _model_cache:
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
    def register(name: str, model_class: type[BaseCaptionModel]) -> None:
        """
        Register a new model type.

        Args:
            name: Model name for lookup
            model_class: Model class (must extend BaseCaptionModel)
        """
        _MODEL_REGISTRY[name] = model_class
