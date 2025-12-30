"""
Prompt builder from templates.

Builds LLM prompts from templates and active sections.
"""

from typing import Dict, List, Any, Optional, Tuple
from ..types import Decisions, TaggerResults
from .tag_injector import get_unmapped_tags, format_tags_for_prompt
from .scene_prompts import format_scene_prompt, get_scene_group
from .detail_prompts import enhance_section_prompt, get_all_enhancements


class PromptBuilder:
    """
    Builds LLM prompts from templates.

    Combines base prompt with active section prompts to create
    a comprehensive LLM prompt for image analysis.
    """

    def __init__(self):
        """Initialize the prompt builder."""
        self._last_enhancements = []

    def get_last_enhancements(self) -> List[Dict[str, Any]]:
        """Get the enhancements used in the last build_llm_prompt call."""
        return self._last_enhancements

    def build_llm_prompt(
        self,
        template: Dict[str, Any],
        active_sections: List[str],
        decisions: Optional[Decisions] = None,
        tagger_results: Optional[TaggerResults] = None,
        tag_threshold: float = 0.5,
    ) -> str:
        """
        Build the LLM prompt from template and active sections.

        Args:
            template: Template configuration dict
            active_sections: List of section names to include
            decisions: Optional decisions for context
            tagger_results: Optional tagger results for tag injection
            tag_threshold: Confidence threshold for tag injection

        Returns:
            Complete LLM prompt string
        """
        parts = []

        # Add base prompt
        base_prompt = template.get("base_prompt", "Describe this image in detail.")
        parts.append(base_prompt.strip())

        # Add detected tags section if tagger_results provided
        if tagger_results is not None:
            detected_tags = get_unmapped_tags(
                tagger_results,
                threshold=tag_threshold,
                max_tags=15,
            )
            if detected_tags:
                tags_str = format_tags_for_prompt(detected_tags, show_confidence=True)
                parts.append(f"\nDetected elements: {tags_str}")
                parts.append("Incorporate these elements naturally into your description where relevant.")

        # Add section prompts
        sections = template.get("sections", {})
        section_prompts = []
        enhancements_used = []

        # Get scene-specific prompt if available
        scene_type = decisions.scene_type if decisions else None
        scene_group = get_scene_group(scene_type) if scene_type else None

        for section_name in active_sections:
            if section_name in sections:
                section = sections[section_name]
                prompt = section.get("prompt", "")

                # Replace environment section with scene-specific prompt
                if section_name == "environment" and scene_type:
                    prompt = format_scene_prompt(scene_type)

                # Apply detail enhancements (animal/clothing/tag-based)
                if prompt and tagger_results is not None and decisions is not None:
                    enhanced_prompt, enhancement_info = enhance_section_prompt(
                        section_name,
                        prompt,
                        decisions,
                        tagger_results,
                        tag_threshold,
                    )
                    if enhancement_info:
                        prompt = enhanced_prompt
                        enhancements_used.append({
                            "section": section_name,
                            **enhancement_info
                        })

                if prompt:
                    section_prompts.append(prompt.strip())

        # Store enhancements for metadata access
        self._last_enhancements = enhancements_used

        if section_prompts:
            parts.append("\nFocus on these aspects:")
            for i, prompt in enumerate(section_prompts, 1):
                # Clean and format section prompt
                lines = prompt.strip().split('\n')
                formatted = ' '.join(line.strip() for line in lines if line.strip())
                parts.append(f"{i}. {formatted}")

        # Add output instructions
        parts.append("\nWrite as a single flowing paragraph suitable for AI image generation.")
        parts.append("Be specific and detailed. Do not include meta-commentary.")

        return "\n".join(parts)

    def build_section_prompt(
        self,
        template: Dict[str, Any],
        section_name: str,
        context: Optional[str] = None,
    ) -> str:
        """
        Build prompt for a single section (used in multi-pass modes).

        Args:
            template: Template configuration dict
            section_name: Name of section to build prompt for
            context: Optional context from previous passes

        Returns:
            Section-specific LLM prompt
        """
        sections = template.get("sections", {})
        section = sections.get(section_name, {})

        prompt = section.get("prompt", f"Describe the {section_name}.")

        parts = [prompt.strip()]

        if context:
            parts.insert(0, f"Context from previous analysis:\n{context}\n\n")

        parts.append("\nBe specific and detailed. Write in natural flowing prose.")

        return "\n".join(parts)

    def assemble_final_prompt(
        self,
        section_outputs: Dict[str, str],
        assembly_order: List[str],
        separator: str = ". ",
    ) -> str:
        """
        Assemble final prompt from section outputs.

        Args:
            section_outputs: Dict mapping section name to generated text
            assembly_order: Order to assemble sections
            separator: String to join sections with

        Returns:
            Assembled final prompt
        """
        parts = []

        for section_name in assembly_order:
            if section_name in section_outputs:
                output = section_outputs[section_name]
                if output and output.strip():
                    # Clean the output
                    cleaned = self._clean_section_output(output)
                    if cleaned:
                        parts.append(cleaned)

        # Join with separator, handling punctuation
        result = []
        for i, part in enumerate(parts):
            if i > 0:
                # Check if previous part ended with punctuation
                if result and result[-1][-1] in '.!?':
                    result.append(" ")
                else:
                    result.append(separator)
            result.append(part)

        return "".join(result)

    def _clean_section_output(self, text: str) -> str:
        """
        Clean section output text.

        Args:
            text: Raw section output

        Returns:
            Cleaned text
        """
        import re

        if not text:
            return ""

        # Remove common preambles
        preambles = [
            r'^(?:Here is|Here\'s|This is)[:\s]*',
            r'^(?:The|This|She|He|It)[:\s]*',
        ]
        for pattern in preambles:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)

        # Remove markdown
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'\*([^*]+)\*', r'\1', text)

        # Clean whitespace
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\s+', ' ', text)

        return text.strip()


# Global singleton
_builder: Optional[PromptBuilder] = None


def get_builder() -> PromptBuilder:
    """Get the global PromptBuilder instance."""
    global _builder
    if _builder is None:
        _builder = PromptBuilder()
    return _builder
