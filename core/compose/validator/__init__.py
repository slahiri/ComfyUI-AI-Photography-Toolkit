"""Validator module for prompt quality analysis (Phase 5).

This module handles validation of assembled prompts:
- Length constraint checking
- Duplicate detection and removal
- Conflict detection
- Missing category warnings
- Quality issue detection and auto-correction
"""

from .base import (
    BaseValidator,
    ValidatorConfig,
    ValidationResult,
    ValidationIssue,
    ValidationSeverity,
    ValidationCategory,
)

from .rules import (
    find_conflicts,
    find_duplicates,
    find_phrase_duplicates,
    remove_duplicate_words,
    remove_runaway_repetitions,
    is_garbage_output,
    check_quality_issues,
    fix_quality_issues,
    estimate_clip_tokens,
    CONFLICT_GROUPS,
)

from .standard import (
    StandardValidator,
    validate_prompt,
    validate_text,
)


__all__ = [
    # Base classes
    "BaseValidator",
    "ValidatorConfig",
    "ValidationResult",
    "ValidationIssue",
    "ValidationSeverity",
    "ValidationCategory",
    # Rules
    "find_conflicts",
    "find_duplicates",
    "find_phrase_duplicates",
    "remove_duplicate_words",
    "remove_runaway_repetitions",
    "is_garbage_output",
    "check_quality_issues",
    "fix_quality_issues",
    "estimate_clip_tokens",
    "CONFLICT_GROUPS",
    # Standard validator
    "StandardValidator",
    "validate_prompt",
    "validate_text",
]
