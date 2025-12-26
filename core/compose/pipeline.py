"""Unified composition pipeline."""

from typing import Dict, Optional, Any

from .base import (
    ComposeConfig, ComposeResult, ComposeMode, StyleConfig, ComposeAnalytics
)
from .standard import StandardGenerator
from .llm import LLMGenerator
from ..log import log, log_start, log_end


class ComposePipeline:
    """
    Unified pipeline for prompt composition.

    Manages generator selection and provides a clean API for the node.
    """

    def __init__(
        self,
        style: str = "z-image",
        mode: str = "standard",
        caption_source: str = "Auto",
        provider: str = "local",
        model: str = None,
        api_key: str = None,
        hf_token: str = None,
        config: Optional[ComposeConfig] = None,
    ):
        """
        Initialize composition pipeline.

        Args:
            style: Output style (z-image, etc.)
            mode: Generation mode (standard, enhance_ai)
            caption_source: Caption base source (Auto, VLM, Florence, Tags Only)
            provider: LLM provider for Enhance with AI mode
            model: Model name for Enhance with AI mode
            api_key: API key for cloud providers
            hf_token: HuggingFace token for gated local models
            config: Optional compose configuration
        """
        self.style = style
        self.mode = mode.lower().replace(" ", "_").replace("with", "")
        # Normalize mode names
        if self.mode in ["enhance_ai", "enhanceai", "enhance_with_ai", "llm"]:
            self.mode = "enhance_ai"
        self.caption_source = caption_source
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.hf_token = hf_token
        self.config = config or ComposeConfig(caption_source=caption_source)

        # Lazy-loaded generators
        self._standard = None
        self._enhance_ai = None

    def _get_generator(self):
        """Get the appropriate generator for current mode."""
        if self.mode == "standard":
            if self._standard is None:
                self._standard = StandardGenerator(
                    style=self.style,
                    config=self.config,
                    caption_source=self.caption_source,
                )
            return self._standard

        elif self.mode == "enhance_ai":
            if self._enhance_ai is None:
                self._enhance_ai = LLMGenerator(
                    style=self.style,
                    config=self.config,
                    caption_source=self.caption_source,
                    provider=self.provider,
                    model=self.model,
                    api_key=self.api_key,
                    hf_token=self.hf_token,
                )
            return self._enhance_ai

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def generate(self, metadata: Dict) -> ComposeResult:
        """
        Generate prompt from metadata.

        Args:
            metadata: Image analysis metadata (from SID_ImageAnalysis)

        Returns:
            ComposeResult with prompt and analytics
        """
        # Parse metadata if it's a string
        if isinstance(metadata, str):
            import json
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                metadata = {}

        mode_display = "Enhance with AI" if self.mode == "enhance_ai" else self.mode.title()
        start = log_start("Compose", f"Generating prompt ({mode_display} mode, {self.caption_source})")

        generator = self._get_generator()
        result = generator.generate(metadata)

        log_end("Compose", f"Prompt generated", start, f"{result.word_count} words")

        return result

    def unload(self):
        """Unload any loaded models."""
        if self._enhance_ai is not None:
            self._enhance_ai.unload()
            self._enhance_ai = None

    @staticmethod
    def get_available_styles() -> list:
        """Get list of available output styles."""
        return StyleConfig.available_styles()

    @staticmethod
    def get_available_modes() -> list:
        """Get list of available generation modes."""
        return ["Standard", "Enhance with AI"]
