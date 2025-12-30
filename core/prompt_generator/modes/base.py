"""
Base mode interface for prompt generation.

All prompt generation modes inherit from BaseMode and implement
the execute() method.
"""

from typing import Any, Optional
from ..types import GeneratorResult


class BaseMode:
    """
    Base class for prompt generation modes.

    Each mode implements a different strategy for generating prompts
    from images, with varying levels of complexity and LLM usage.

    Note: We don't use ABC because it can conflict with ComfyUI's
    internal class handling.
    """

    @property
    def name(self) -> str:
        """
        Return the mode name for metadata.
        Override in subclasses.
        """
        raise NotImplementedError("Subclasses must implement name property")

    @property
    def requires_llm(self) -> bool:
        """
        Whether this mode requires an LLM model.
        Override in subclasses if needed.
        """
        return True

    @property
    def requires_taggers(self) -> bool:
        """
        Whether this mode executes taggers.
        Override in subclasses if needed.
        """
        return False

    @property
    def requires_reasoning(self) -> bool:
        """
        Whether this mode requires LLM reasoning capability.
        Override in subclasses if needed.
        """
        return False

    def execute(
        self,
        image: Any = None,
        llm_model: Any = None,
        tagger_results: Any = None,
        decisions: Any = None,
        **kwargs
    ) -> GeneratorResult:
        """
        Execute prompt generation for this mode.

        Args:
            image: Image tensor from ComfyUI
            llm_model: LLMModelConfig if LLM is available
            tagger_results: TaggerResults if taggers were executed
            decisions: Decisions derived from tagger results
            **kwargs: Additional mode-specific parameters

        Returns:
            GeneratorResult with prompt and metadata
        """
        raise NotImplementedError("Subclasses must implement execute()")

    def validate(
        self,
        llm_model: Any = None,
        **kwargs
    ) -> tuple[bool, Optional[str]]:
        """
        Validate that this mode can be executed with given inputs.

        Args:
            llm_model: LLMModelConfig if provided
            **kwargs: Additional parameters

        Returns:
            Tuple of (is_valid, error_message)
        """
        if self.requires_llm and llm_model is None:
            return False, f"{self.name} mode requires an LLM model"

        if self.requires_reasoning:
            if llm_model is None:
                return False, f"{self.name} mode requires an LLM with reasoning capability"
            if not getattr(llm_model, 'supports_reasoning', False):
                return False, f"{self.name} mode requires an LLM with reasoning capability (current model doesn't support it)"

        return True, None
