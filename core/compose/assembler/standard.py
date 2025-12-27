"""Standard rule-based prompt assembler.

Assembles classified tokens into natural language prompts using
templates and grammar rules.
"""

from typing import Dict, List, Optional
from collections import defaultdict

from .base import (
    BaseAssembler,
    AssemblerConfig,
    AssembledPrompt,
    CategorySection,
    PromptStyle,
)
from .rules import (
    Gender,
    detect_gender,
    get_pronouns,
    clean_token_for_output,
    join_tokens_natural,
    build_subject_phrase,
    CATEGORY_TEMPLATES,
    SUBJECT_DETAIL_TEMPLATES,
)
from ..classifier.base import ClassifiedImage, TokenClassification
from ..classifier.categories import (
    CanonicalCategory,
    SubjectDetailType,
    CATEGORY_ORDER,
    SUBJECT_DETAIL_ORDER,
)


class StandardAssembler(BaseAssembler):
    """Standard rule-based prompt assembler.

    Converts classified tokens into natural language prompts following
    the canonical category order.
    """

    def assemble(self, classified: ClassifiedImage) -> AssembledPrompt:
        """Assemble classified tokens into a prompt.

        Args:
            classified: ClassifiedImage from the classifier

        Returns:
            AssembledPrompt with the generated prompt
        """
        sections: List[CategorySection] = []

        # Detect gender from subject/detail tokens for pronoun usage
        all_tokens = [c.token.text for c in classified.all_classifications]
        gender = detect_gender(all_tokens)
        subj, poss, obj = get_pronouns(gender)

        # Collect subject details by subcategory
        detail_by_subcat = self._collect_subject_details(classified)

        # Process each category in order
        for category in CATEGORY_ORDER:
            if category == CanonicalCategory.SUBJECT_DETAILS:
                # Handle subject details specially with subcategories
                section = self._build_subject_details_section(
                    classified, detail_by_subcat, gender
                )
            else:
                section = self._build_category_section(classified, category, gender)

            if section and section.tokens:
                sections.append(section)

        # Build final prompt based on style
        if self.config.style == PromptStyle.TAGS:
            prompt = self._build_tags_prompt(sections)
        elif self.config.style == PromptStyle.HYBRID:
            prompt = self._build_hybrid_prompt(sections, gender)
        elif self.config.style == PromptStyle.STRUCTURED:
            prompt = self._build_structured_prompt(sections)
        else:  # NATURAL
            prompt = self._build_natural_prompt(sections, gender)

        return AssembledPrompt(
            prompt=prompt,
            style=self.config.style,
            sections=sections,
            metadata={
                "gender": gender.value,
                "image_info": classified.image_info,
            }
        )

    def _collect_subject_details(
        self, classified: ClassifiedImage
    ) -> Dict[SubjectDetailType, List[TokenClassification]]:
        """Collect subject detail tokens by subcategory."""
        result = defaultdict(list)

        tokens = classified.get_category(CanonicalCategory.SUBJECT_DETAILS)
        if not tokens:
            return result

        for tc in tokens:
            if tc.token.confidence < self.config.min_confidence:
                continue
            subcat = tc.subcategory or SubjectDetailType.BODY  # Default
            result[subcat].append(tc)

        # Sort each subcategory by confidence and apply limits
        limit = self.config.get_limit_for_category(CanonicalCategory.SUBJECT_DETAILS)
        for subcat in result:
            result[subcat].sort(key=lambda t: t.token.confidence, reverse=True)
            # Limit per subcategory (use category limit, not max_tokens_per_category)
            # Divide limit across subcategories to prevent any single subcategory from dominating
            per_subcat_limit = max(5, limit // len(result)) if result else limit
            result[subcat] = result[subcat][:per_subcat_limit]

        return result

    def _build_category_section(
        self,
        classified: ClassifiedImage,
        category: CanonicalCategory,
        gender: Gender,
    ) -> Optional[CategorySection]:
        """Build a section for a standard category."""
        tokens = self._get_tokens_for_category(classified, category)
        if not tokens:
            return CategorySection(
                category=category,
                tokens=[],
                text="",
                confidence=0.0
            )

        if self.config.deduplicate:
            tokens = self._deduplicate_tokens(tokens)

        # Clean token texts
        token_texts = [clean_token_for_output(t.token.text) for t in tokens]
        token_texts = [t for t in token_texts if t]  # Remove empty

        if not token_texts:
            return CategorySection(
                category=category,
                tokens=[],
                text="",
                confidence=0.0
            )

        # Build phrase using template
        template = CATEGORY_TEMPLATES.get(category, {})
        join_style = template.get("join", ", ")

        # Determine join style from separator
        if join_style == " and ":
            joined = join_tokens_natural(token_texts, "and")
        elif join_style == " ":
            joined = join_tokens_natural(token_texts, "space")
        else:
            joined = join_tokens_natural(token_texts, "comma")

        # Apply template
        prefix = template.get("prefix", "")
        suffix = template.get("suffix", "")

        if len(token_texts) == 1:
            text = template.get("single", "{token}").format(token=joined)
        else:
            text = template.get("multiple", "{tokens}").format(tokens=joined)

        text = prefix + text + suffix

        # Calculate average confidence
        avg_conf = sum(t.token.confidence for t in tokens) / len(tokens)

        return CategorySection(
            category=category,
            tokens=token_texts,
            text=text.strip(),
            confidence=avg_conf
        )

    def _build_subject_details_section(
        self,
        classified: ClassifiedImage,
        detail_by_subcat: Dict[SubjectDetailType, List[TokenClassification]],
        gender: Gender,
    ) -> Optional[CategorySection]:
        """Build subject details section with subcategory organization."""
        all_tokens = []
        parts = []

        for subcat in SUBJECT_DETAIL_ORDER:
            if subcat not in detail_by_subcat:
                continue

            tokens = detail_by_subcat[subcat]
            if not tokens:
                continue

            # Clean token texts
            token_texts = [clean_token_for_output(t.token.text) for t in tokens]
            token_texts = [t for t in token_texts if t]

            if not token_texts:
                continue

            all_tokens.extend(token_texts)

            # Build phrase for this subcategory
            template = SUBJECT_DETAIL_TEMPLATES.get(subcat, {})

            # Determine join style
            join_sep = template.get("join", ", ")
            if join_sep == " ":
                joined = join_tokens_natural(token_texts, "space")
            elif "and" in join_sep:
                joined = join_tokens_natural(token_texts, "and")
            else:
                joined = join_tokens_natural(token_texts, "comma")

            if len(token_texts) == 1:
                phrase = template.get("single", "{token}").format(token=joined)
            else:
                phrase = template.get("multiple", "{tokens}").format(tokens=joined)

            parts.append(phrase)

        if not all_tokens:
            return CategorySection(
                category=CanonicalCategory.SUBJECT_DETAILS,
                tokens=[],
                text="",
                confidence=0.0
            )

        # Join all parts
        text = ", ".join(parts)

        # Calculate average confidence
        all_tcs = [t for tokens in detail_by_subcat.values() for t in tokens]
        avg_conf = sum(t.token.confidence for t in all_tcs) / len(all_tcs) if all_tcs else 0

        return CategorySection(
            category=CanonicalCategory.SUBJECT_DETAILS,
            tokens=all_tokens,
            text=text,
            confidence=avg_conf
        )

    def _build_natural_prompt(
        self, sections: List[CategorySection], gender: Gender
    ) -> str:
        """Build a natural language prompt from sections."""
        parts = []
        subj, poss, obj = get_pronouns(gender)

        # Quality boosters first (comma-separated)
        quality = next((s for s in sections if s.category == CanonicalCategory.QUALITY_BOOSTERS), None)
        if quality and quality.text:
            parts.append(quality.text)

        # Style/Medium
        style = next((s for s in sections if s.category == CanonicalCategory.STYLE_MEDIUM), None)
        if style and style.text:
            parts.append(style.text)

        # Subject - start of sentence
        subject = next((s for s in sections if s.category == CanonicalCategory.SUBJECT), None)
        subject_text = ""
        if subject and subject.text:
            subject_text = subject.text
            # Capitalize if not already
            if subject_text and subject_text[0].islower():
                subject_text = subject_text[0].upper() + subject_text[1:]

        # Subject details - append to subject
        details = next((s for s in sections if s.category == CanonicalCategory.SUBJECT_DETAILS), None)
        if subject_text:
            if details and details.text:
                parts.append(f"{subject_text}, {details.text}")
            else:
                parts.append(subject_text)
        elif details and details.text:
            parts.append(details.text)

        # Action/Pose
        action = next((s for s in sections if s.category == CanonicalCategory.ACTION_POSE), None)
        if action and action.text:
            parts.append(action.text)

        # Environment
        env = next((s for s in sections if s.category == CanonicalCategory.ENVIRONMENT), None)
        if env and env.text:
            parts.append(env.text)

        # Lighting
        lighting = next((s for s in sections if s.category == CanonicalCategory.LIGHTING), None)
        if lighting and lighting.text:
            parts.append(lighting.text)

        # Composition
        comp = next((s for s in sections if s.category == CanonicalCategory.COMPOSITION), None)
        if comp and comp.text:
            parts.append(comp.text)

        # Technical
        tech = next((s for s in sections if s.category == CanonicalCategory.TECHNICAL), None)
        if tech and tech.text:
            parts.append(tech.text)

        # Join with commas for natural flow
        prompt = ", ".join(p for p in parts if p)

        return prompt

    def _build_tags_prompt(self, sections: List[CategorySection]) -> str:
        """Build a comma-separated tags prompt."""
        all_tokens = []
        for section in sections:
            all_tokens.extend(section.tokens)
        return ", ".join(all_tokens)

    def _build_hybrid_prompt(
        self, sections: List[CategorySection], gender: Gender
    ) -> str:
        """Build a hybrid prompt: sentence intro + tag details."""
        subj, poss, obj = get_pronouns(gender)

        # Build intro sentence from subject/details/action
        intro_parts = []

        subject = next((s for s in sections if s.category == CanonicalCategory.SUBJECT), None)
        if subject and subject.text:
            intro_parts.append(subject.text)

        details = next((s for s in sections if s.category == CanonicalCategory.SUBJECT_DETAILS), None)
        if details and details.text:
            intro_parts.append(details.text)

        action = next((s for s in sections if s.category == CanonicalCategory.ACTION_POSE), None)
        if action and action.text:
            intro_parts.append(action.text)

        intro = ", ".join(intro_parts) if intro_parts else ""

        # Collect remaining tokens as tags
        tag_categories = [
            CanonicalCategory.QUALITY_BOOSTERS,
            CanonicalCategory.STYLE_MEDIUM,
            CanonicalCategory.ENVIRONMENT,
            CanonicalCategory.LIGHTING,
            CanonicalCategory.COMPOSITION,
            CanonicalCategory.TECHNICAL,
        ]
        tag_tokens = []
        for section in sections:
            if section.category in tag_categories:
                tag_tokens.extend(section.tokens)

        tags = ", ".join(tag_tokens) if tag_tokens else ""

        # Combine
        if intro and tags:
            return f"{intro}. {tags}"
        return intro or tags

    def _build_structured_prompt(self, sections: List[CategorySection]) -> str:
        """Build a category-segregated prompt with labels.

        Each category is on its own line with a label.
        Example:
            [SUBJECT] woman, nun
            [SUBJECT_DETAILS] black habit, white wimple, religious attire
            [ENVIRONMENT] church interior, stained glass window
        """
        # Category display names for cleaner output
        CATEGORY_LABELS = {
            CanonicalCategory.QUALITY_BOOSTERS: "QUALITY",
            CanonicalCategory.STYLE_MEDIUM: "STYLE",
            CanonicalCategory.SUBJECT: "SUBJECT",
            CanonicalCategory.SUBJECT_DETAILS: "DETAILS",
            CanonicalCategory.ACTION_POSE: "POSE",
            CanonicalCategory.ENVIRONMENT: "ENVIRONMENT",
            CanonicalCategory.LIGHTING: "LIGHTING",
            CanonicalCategory.COMPOSITION: "COMPOSITION",
            CanonicalCategory.TECHNICAL: "TECHNICAL",
        }

        lines = []
        for section in sections:
            if not section.tokens:
                continue

            label = CATEGORY_LABELS.get(section.category, section.category.value.upper())
            tokens_str = ", ".join(section.tokens)
            lines.append(f"[{label}] {tokens_str}")

        return "\n".join(lines)


# Convenience function
def assemble_prompt(
    classified: ClassifiedImage,
    config: Optional[AssemblerConfig] = None
) -> AssembledPrompt:
    """Assemble a prompt from classified tokens.

    Args:
        classified: ClassifiedImage from the classifier
        config: Optional assembler configuration

    Returns:
        AssembledPrompt with the generated prompt
    """
    assembler = StandardAssembler(config)
    return assembler.assemble(classified)
