"""
Base Prompt Template

Abstract base class defining the interface for all prompt templates.
Follows Interface Segregation Principle - minimal required methods.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional


class BasePromptTemplate(ABC):
    """
    Abstract base class for prompt templates.

    Each template implementation can optimize prompts for specific LLM providers.
    This follows:
    - Single Responsibility: Each template handles one provider type
    - Open/Closed: New templates can be added without modifying existing code
    - Liskov Substitution: All templates are interchangeable
    - Interface Segregation: Minimal interface - only what's needed
    - Dependency Inversion: Code depends on this abstraction, not concretions
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the template name for logging."""
        pass

    @property
    @abstractmethod
    def max_system_prompt_tokens(self) -> int:
        """
        Maximum tokens to use for system prompts.
        Local models may need shorter prompts to save context.
        """
        pass

    @abstractmethod
    def build_classification_prompt(
        self,
        shot_framings: Dict[str, Any],
        genres: Dict[str, Any],
        focus_override: Optional[str] = None,
    ) -> tuple[str, str]:
        """
        Build the classification prompt (Stage 1).

        Args:
            shot_framings: Available shot framing codes and names
            genres: Available photography genre codes and names
            focus_override: Optional hint for classification focus

        Returns:
            Tuple of (system_prompt, user_message)
        """
        pass

    @abstractmethod
    def build_analysis_prompt(
        self,
        classification: Dict[str, Any],
        attribute_schema: Dict[str, Any],
        content_detail: str,
    ) -> tuple[str, str]:
        """
        Build the detailed analysis prompt (Stage 4).

        Args:
            classification: Results from Stage 1
            attribute_schema: Schema of attributes to extract
            content_detail: Detail level (minimal, standard, detailed, explicit)

        Returns:
            Tuple of (system_prompt, user_message)
        """
        pass

    @abstractmethod
    def build_composition_prompt(
        self,
        classification: Dict[str, Any],
        attributes: Dict[str, Any],
        user_prompt: str,
        prompt_mode: str,
        focus_options: Dict[str, bool],
        include_text_quotes: bool,
        min_words: int,
        max_words: int,
    ) -> tuple[str, str]:
        """
        Build the prompt composition prompt (Stage 5).

        Args:
            classification: Results from Stage 1
            attributes: Results from Stage 4
            user_prompt: User's guidance text
            prompt_mode: How to combine user prompt with analysis
            focus_options: Dict of focus toggles (subject, environment, etc.)
            include_text_quotes: Whether to quote visible text
            min_words: Minimum word count target
            max_words: Maximum word count target

        Returns:
            Tuple of (system_prompt, user_message)
        """
        pass

    @abstractmethod
    def build_refinement_prompt(
        self,
        draft_prompt: str,
        min_words: int,
        max_words: int,
    ) -> tuple[str, str]:
        """
        Build the refinement prompt (Stage 5 Deep mode).

        Args:
            draft_prompt: Initial generated prompt
            min_words: Minimum word count target
            max_words: Maximum word count target

        Returns:
            Tuple of (system_prompt, user_message)
        """
        pass

    def get_classification_max_tokens(self) -> int:
        """Max tokens for classification response."""
        return 500

    def get_analysis_max_tokens(self) -> int:
        """Max tokens for analysis response."""
        return 1500

    def get_temperature_modifier(self, stage: str) -> float:
        """
        Get temperature modifier for a stage.

        Args:
            stage: One of 'classification', 'analysis', 'composition', 'refinement'

        Returns:
            Multiplier to apply to base temperature (e.g., 0.5 = half)
        """
        modifiers = {
            "classification": 0.5,  # Lower for accurate classification
            "analysis": 1.0,        # Normal for attribute extraction
            "composition": 1.0,     # Normal for creative composition
            "refinement": 0.5,      # Lower for focused refinement
        }
        return modifiers.get(stage, 1.0)
