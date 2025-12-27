"""Base classes for prompt assembly (Phase 4).

The assembler takes classified tokens from Phase 3 and combines them
into natural language prompts using templates and grammar rules.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any

from ..classifier.base import ClassifiedImage, TokenClassification
from ..classifier.categories import (
    CanonicalCategory,
    SubjectDetailType,
    CATEGORY_ORDER,
    SUBJECT_DETAIL_ORDER,
    CATEGORY_DISPLAY_NAMES,
)


class PromptStyle(Enum):
    """Output prompt style."""
    NATURAL = "natural"  # Full sentences, flowing prose
    TAGS = "tags"  # Comma-separated tags (booru style)
    HYBRID = "hybrid"  # Sentence intro + tag details
    STRUCTURED = "structured"  # Category-segregated output with labels


@dataclass
class AssemblerConfig:
    """Configuration for prompt assembly."""
    style: PromptStyle = PromptStyle.NATURAL
    max_tokens_per_category: int = 50  # High default to preserve information
    include_quality_boosters: bool = True
    include_technical: bool = True
    min_confidence: float = 0.3
    deduplicate: bool = True
    use_pronouns: bool = True  # Use pronouns after first subject mention


@dataclass
class CategorySection:
    """A section of the assembled prompt for one category."""
    category: CanonicalCategory
    tokens: List[str]  # The token texts included
    text: str  # The assembled text for this section
    confidence: float  # Average confidence of tokens


@dataclass
class AssembledPrompt:
    """Result from prompt assembly."""
    prompt: str
    style: PromptStyle
    sections: List[CategorySection] = field(default_factory=list)
    word_count: int = 0
    token_count: int = 0
    categories_used: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.word_count = len(self.prompt.split())
        self.token_count = sum(len(s.tokens) for s in self.sections)
        self.categories_used = [s.category.value for s in self.sections if s.tokens]

    def get_section(self, category: CanonicalCategory) -> Optional[CategorySection]:
        """Get section for a specific category."""
        for section in self.sections:
            if section.category == category:
                return section
        return None


class BaseAssembler(ABC):
    """Base class for prompt assemblers."""

    def __init__(self, config: Optional[AssemblerConfig] = None):
        self.config = config or AssemblerConfig()

    @abstractmethod
    def assemble(self, classified: ClassifiedImage) -> AssembledPrompt:
        """Assemble classified tokens into a prompt.

        Args:
            classified: ClassifiedImage from the classifier

        Returns:
            AssembledPrompt with the generated prompt
        """
        pass

    def _get_tokens_for_category(
        self,
        classified: ClassifiedImage,
        category: CanonicalCategory,
        max_tokens: Optional[int] = None
    ) -> List[TokenClassification]:
        """Get tokens for a category, sorted by confidence.

        Args:
            classified: ClassifiedImage to get tokens from
            category: Category to get tokens for
            max_tokens: Maximum tokens to return (None = use config)

        Returns:
            List of TokenClassification sorted by confidence
        """
        tokens = classified.get_category(category)
        if not tokens:
            return []

        # Filter by minimum confidence
        tokens = [t for t in tokens if t.token.confidence >= self.config.min_confidence]

        # Sort by confidence
        tokens.sort(key=lambda t: t.token.confidence, reverse=True)

        # Limit count
        limit = max_tokens or self.config.max_tokens_per_category
        return tokens[:limit]

    def _deduplicate_tokens(
        self,
        tokens: List[TokenClassification]
    ) -> List[TokenClassification]:
        """Remove duplicate token texts."""
        seen = set()
        unique = []
        for t in tokens:
            text_lower = t.token.text.lower().strip()
            if text_lower not in seen:
                seen.add(text_lower)
                unique.append(t)
        return unique

    def _join_natural(self, items: List[str]) -> str:
        """Join items in natural language style."""
        if not items:
            return ""
        if len(items) == 1:
            return items[0]
        if len(items) == 2:
            return f"{items[0]} and {items[1]}"
        return ', '.join(items[:-1]) + f", and {items[-1]}"

    def _capitalize_first(self, text: str) -> str:
        """Capitalize first letter only."""
        if not text:
            return text
        return text[0].upper() + text[1:]
