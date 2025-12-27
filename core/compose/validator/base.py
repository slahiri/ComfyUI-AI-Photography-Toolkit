"""Base classes for prompt validation (Phase 5).

The validator checks assembled prompts for quality issues:
- Length constraints
- Duplicate detection
- Conflict detection
- Missing required content
- Quality scoring
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple

from ..assembler.base import AssembledPrompt


class ValidationSeverity(Enum):
    """Severity level for validation issues."""
    INFO = "info"  # Informational, no action needed
    WARNING = "warning"  # Potential issue, consider fixing
    ERROR = "error"  # Must be fixed before use


class ValidationCategory(Enum):
    """Category of validation issue."""
    LENGTH = "length"  # Prompt too long/short
    DUPLICATE = "duplicate"  # Repeated content
    CONFLICT = "conflict"  # Contradictory terms
    MISSING = "missing"  # Required content missing
    QUALITY = "quality"  # General quality issue
    GRAMMAR = "grammar"  # Grammar/syntax issue


@dataclass
class ValidationIssue:
    """A single validation issue found in the prompt."""
    category: ValidationCategory
    severity: ValidationSeverity
    message: str
    location: Optional[str] = None  # Where in prompt (section, position)
    suggestion: Optional[str] = None  # How to fix
    tokens: List[str] = field(default_factory=list)  # Related tokens


@dataclass
class ValidationResult:
    """Result from prompt validation."""
    is_valid: bool  # No errors (warnings OK)
    prompt: str  # The validated (possibly corrected) prompt
    original_prompt: str  # Original before corrections
    issues: List[ValidationIssue] = field(default_factory=list)
    corrections_made: int = 0
    quality_score: float = 1.0  # 0.0 to 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == ValidationSeverity.ERROR)

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == ValidationSeverity.WARNING)

    @property
    def info_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == ValidationSeverity.INFO)

    def get_issues_by_category(self, category: ValidationCategory) -> List[ValidationIssue]:
        return [i for i in self.issues if i.category == category]

    def get_summary(self) -> str:
        """Get human-readable validation summary."""
        if self.is_valid and not self.issues:
            return "Valid prompt, no issues found"

        parts = []
        if self.error_count:
            parts.append(f"{self.error_count} error(s)")
        if self.warning_count:
            parts.append(f"{self.warning_count} warning(s)")
        if self.info_count:
            parts.append(f"{self.info_count} info")
        if self.corrections_made:
            parts.append(f"{self.corrections_made} auto-corrected")

        status = "Invalid" if not self.is_valid else "Valid with issues"
        return f"{status}: {', '.join(parts)}"


@dataclass
class ValidatorConfig:
    """Configuration for prompt validation."""
    # Length constraints
    max_tokens: int = 225  # CLIP token limit
    max_words: int = 200  # Approximate word limit
    min_words: int = 10  # Minimum for meaningful prompt

    # Duplicate detection
    detect_duplicates: bool = True
    remove_duplicates: bool = True

    # Conflict detection
    detect_conflicts: bool = True

    # Required categories (warning if missing)
    required_categories: List[str] = field(default_factory=lambda: ["subject"])

    # Auto-correction
    auto_correct: bool = True
    truncate_if_too_long: bool = True


class BaseValidator(ABC):
    """Base class for prompt validators."""

    def __init__(self, config: Optional[ValidatorConfig] = None):
        self.config = config or ValidatorConfig()

    @abstractmethod
    def validate(self, prompt: AssembledPrompt) -> ValidationResult:
        """Validate an assembled prompt.

        Args:
            prompt: AssembledPrompt from the assembler

        Returns:
            ValidationResult with issues and corrected prompt
        """
        pass

    def validate_text(self, text: str) -> ValidationResult:
        """Validate a raw text prompt.

        Args:
            text: Raw prompt text

        Returns:
            ValidationResult with issues
        """
        # Create a minimal AssembledPrompt for validation
        from ..assembler.base import AssembledPrompt, PromptStyle
        prompt = AssembledPrompt(
            prompt=text,
            style=PromptStyle.NATURAL,
        )
        return self.validate(prompt)
