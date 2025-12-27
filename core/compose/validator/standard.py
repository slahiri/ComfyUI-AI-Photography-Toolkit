"""Standard prompt validator.

Validates assembled prompts for quality issues and optionally
auto-corrects problems.
"""

from typing import List, Optional

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
    check_quality_issues,
    fix_quality_issues,
    estimate_clip_tokens,
)
from ..assembler.base import AssembledPrompt


class StandardValidator(BaseValidator):
    """Standard prompt validator with auto-correction.

    Checks for:
    - Length constraints (token/word limits)
    - Duplicate words and phrases
    - Conflicting terms
    - Missing required categories
    - Quality issues (punctuation, spacing)
    """

    def validate(self, prompt: AssembledPrompt) -> ValidationResult:
        """Validate an assembled prompt.

        Args:
            prompt: AssembledPrompt from the assembler

        Returns:
            ValidationResult with issues and corrected prompt
        """
        issues: List[ValidationIssue] = []
        text = prompt.prompt
        original = text
        corrections = 0

        # 1. Check length constraints
        length_issues, text, length_corrections = self._check_length(text)
        issues.extend(length_issues)
        corrections += length_corrections

        # 2. Check for duplicates
        dup_issues, text, dup_corrections = self._check_duplicates(text)
        issues.extend(dup_issues)
        corrections += dup_corrections

        # 3. Check for conflicts
        conflict_issues = self._check_conflicts(prompt)
        issues.extend(conflict_issues)

        # 4. Check for missing required categories
        missing_issues = self._check_required_categories(prompt)
        issues.extend(missing_issues)

        # 5. Check quality issues
        quality_issues, text, quality_corrections = self._check_quality(text)
        issues.extend(quality_issues)
        corrections += quality_corrections

        # Calculate quality score
        quality_score = self._calculate_quality_score(issues, prompt)

        # Determine if valid (no errors)
        is_valid = all(i.severity != ValidationSeverity.ERROR for i in issues)

        return ValidationResult(
            is_valid=is_valid,
            prompt=text,
            original_prompt=original,
            issues=issues,
            corrections_made=corrections,
            quality_score=quality_score,
            metadata={
                "word_count": len(text.split()),
                "estimated_tokens": estimate_clip_tokens(text),
                "categories_used": prompt.categories_used,
            }
        )

    def _check_length(self, text: str) -> tuple:
        """Check length constraints."""
        issues = []
        corrections = 0

        word_count = len(text.split())
        token_count = estimate_clip_tokens(text)

        # Check minimum length
        if word_count < self.config.min_words:
            issues.append(ValidationIssue(
                category=ValidationCategory.LENGTH,
                severity=ValidationSeverity.WARNING,
                message=f"Prompt too short ({word_count} words, minimum {self.config.min_words})",
                suggestion="Add more descriptive content",
            ))

        # Check maximum word count
        if word_count > self.config.max_words:
            issues.append(ValidationIssue(
                category=ValidationCategory.LENGTH,
                severity=ValidationSeverity.WARNING,
                message=f"Prompt may be too long ({word_count} words, recommended max {self.config.max_words})",
                suggestion="Consider removing less important details",
            ))

            if self.config.auto_correct and self.config.truncate_if_too_long:
                # Truncate while preserving newlines for STRUCTURED style
                lines = text.split('\n')
                new_lines = []
                words_remaining = self.config.max_words
                for line in lines:
                    line_words = line.split()
                    if len(line_words) <= words_remaining:
                        new_lines.append(line)
                        words_remaining -= len(line_words)
                    else:
                        # Truncate this line
                        new_lines.append(' '.join(line_words[:words_remaining]))
                        break
                text = '\n'.join(new_lines)
                corrections += 1

        # Check token limit
        if token_count > self.config.max_tokens:
            issues.append(ValidationIssue(
                category=ValidationCategory.LENGTH,
                severity=ValidationSeverity.ERROR,
                message=f"Prompt exceeds token limit (~{token_count} tokens, max {self.config.max_tokens})",
                suggestion="Reduce prompt length to fit within CLIP token limit",
            ))

            if self.config.auto_correct and self.config.truncate_if_too_long:
                # Truncate while preserving newlines
                ratio = self.config.max_tokens / token_count
                target_words = int(word_count * ratio * 0.9)  # 90% to be safe
                lines = text.split('\n')
                new_lines = []
                words_remaining = target_words
                for line in lines:
                    line_words = line.split()
                    if len(line_words) <= words_remaining:
                        new_lines.append(line)
                        words_remaining -= len(line_words)
                    else:
                        new_lines.append(' '.join(line_words[:words_remaining]))
                        break
                text = '\n'.join(new_lines)
                corrections += 1

        return issues, text, corrections

    def _check_duplicates(self, text: str) -> tuple:
        """Check for duplicate words and phrases."""
        issues = []
        corrections = 0

        if not self.config.detect_duplicates:
            return issues, text, corrections

        # Check word duplicates
        word_dups = find_duplicates(text)
        significant_dups = [(w, c) for w, c in word_dups if c >= 3]

        if significant_dups:
            dup_list = ", ".join(f"'{w}' ({c}x)" for w, c in significant_dups[:5])
            issues.append(ValidationIssue(
                category=ValidationCategory.DUPLICATE,
                severity=ValidationSeverity.WARNING,
                message=f"Repeated words: {dup_list}",
                suggestion="Remove redundant repetitions",
                tokens=[w for w, c in significant_dups],
            ))

        # Check phrase duplicates
        phrase_dups = find_phrase_duplicates(text)
        if phrase_dups:
            dup_list = ", ".join(f"'{p}' ({c}x)" for p, c in phrase_dups[:3])
            issues.append(ValidationIssue(
                category=ValidationCategory.DUPLICATE,
                severity=ValidationSeverity.WARNING,
                message=f"Repeated phrases: {dup_list}",
                suggestion="Remove duplicate phrases",
                tokens=[p for p, c in phrase_dups],
            ))

        # Auto-correct duplicates
        if self.config.auto_correct and self.config.remove_duplicates and (significant_dups or phrase_dups):
            text = remove_duplicate_words(text)
            corrections += 1

        return issues, text, corrections

    def _check_conflicts(self, prompt: AssembledPrompt) -> List[ValidationIssue]:
        """Check for conflicting terms."""
        issues = []

        if not self.config.detect_conflicts:
            return issues

        # Collect all tokens from sections
        all_tokens = []
        for section in prompt.sections:
            all_tokens.extend(section.tokens)

        conflicts = find_conflicts(all_tokens)

        for token1, token2, conflict_type in conflicts:
            issues.append(ValidationIssue(
                category=ValidationCategory.CONFLICT,
                severity=ValidationSeverity.WARNING,
                message=f"Conflicting terms: '{token1}' vs '{token2}'",
                suggestion=f"Choose one from the {conflict_type} group",
                tokens=[token1, token2],
            ))

        return issues

    def _check_required_categories(self, prompt: AssembledPrompt) -> List[ValidationIssue]:
        """Check for missing required categories."""
        issues = []

        for required in self.config.required_categories:
            if required not in prompt.categories_used:
                issues.append(ValidationIssue(
                    category=ValidationCategory.MISSING,
                    severity=ValidationSeverity.WARNING,
                    message=f"Missing recommended category: {required}",
                    suggestion=f"Consider adding {required} information",
                ))

        return issues

    def _check_quality(self, text: str) -> tuple:
        """Check for quality issues."""
        issues = []
        corrections = 0

        quality_problems = check_quality_issues(text)

        for matched, message, suggestion in quality_problems:
            issues.append(ValidationIssue(
                category=ValidationCategory.QUALITY,
                severity=ValidationSeverity.INFO,
                message=f"{message}: '{matched}'",
                suggestion=suggestion,
            ))

        # Auto-fix quality issues
        if self.config.auto_correct and quality_problems:
            text = fix_quality_issues(text)
            corrections += 1

        return issues, text, corrections

    def _calculate_quality_score(
        self, issues: List[ValidationIssue], prompt: AssembledPrompt
    ) -> float:
        """Calculate overall quality score (0.0 to 1.0)."""
        score = 1.0

        # Deduct for issues
        for issue in issues:
            if issue.severity == ValidationSeverity.ERROR:
                score -= 0.2
            elif issue.severity == ValidationSeverity.WARNING:
                score -= 0.05
            # INFO doesn't affect score

        # Bonus for good coverage
        category_count = len(prompt.categories_used)
        if category_count >= 5:
            score += 0.1
        elif category_count >= 3:
            score += 0.05

        # Bonus for appropriate length
        word_count = prompt.word_count
        if 50 <= word_count <= 150:
            score += 0.05

        return max(0.0, min(1.0, score))


# Convenience function
def validate_prompt(
    prompt: AssembledPrompt,
    config: Optional[ValidatorConfig] = None
) -> ValidationResult:
    """Validate an assembled prompt.

    Args:
        prompt: AssembledPrompt from the assembler
        config: Optional validator configuration

    Returns:
        ValidationResult with issues and corrected prompt
    """
    validator = StandardValidator(config)
    return validator.validate(prompt)


def validate_text(
    text: str,
    config: Optional[ValidatorConfig] = None
) -> ValidationResult:
    """Validate raw prompt text.

    Args:
        text: Raw prompt text
        config: Optional validator configuration

    Returns:
        ValidationResult with issues
    """
    validator = StandardValidator(config)
    return validator.validate_text(text)
