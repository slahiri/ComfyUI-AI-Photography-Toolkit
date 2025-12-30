"""
Override mode - Pass-through user prompts.

When user_prompt is provided, this mode is activated and simply
passes the prompt through without any processing.
"""

from typing import Any
from .base import BaseMode
from ..types import GeneratorResult


class OverrideMode(BaseMode):
    """
    Override mode implementation.

    Simply passes through the user-provided prompt without
    any image analysis or LLM calls.
    """

    @property
    def name(self) -> str:
        return "override"

    @property
    def requires_llm(self) -> bool:
        return False

    @property
    def requires_taggers(self) -> bool:
        return False

    def execute(
        self,
        user_prompt: str = "",
        **kwargs
    ) -> GeneratorResult:
        """
        Execute override mode.

        Args:
            user_prompt: The user-provided prompt to pass through
            **kwargs: Ignored

        Returns:
            GeneratorResult with the user prompt passed through
        """
        prompt = user_prompt.strip() if user_prompt else ""

        return GeneratorResult(
            prompt=prompt,
            metadata={
                "mode": "override",
                "source": "user_prompt",
                "processing": {
                    "taggers_executed": False,
                    "florence_executed": False,
                    "llm_executed": False,
                }
            }
        )
