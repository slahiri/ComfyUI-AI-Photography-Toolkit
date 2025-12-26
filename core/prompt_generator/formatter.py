"""
Prompt Formatter - Formats tags into prompt output.

Output format: Markdown with provider headers
Example:
    # WD14
    masterpiece, 1girl, solo, red hair

    # JoyTag
    woman, portrait, urban
"""

from typing import Dict, List
from ..taggers.base import TagItem, TaggerResult


# Display names for providers
PROVIDER_NAMES = {
    "wd14": "WD14",
    "pixai": "PixAI",
    "joytag": "JoyTag",
    "nudenet": "NudeNet",
    "iqa": "IQA",
    "composition": "Composition",
    "saliency": "Saliency",
    "photography": "Photography",
    "fashion": "Fashion",
    "fashion_yolov8": "YOLOv8",
    "fashion_yolos": "YOLOS",
    "fashion_clip": "FashionCLIP",
    "fashion_segformer": "SegFormer",
    "fashion_wargon": "Wargon",
    "pose": "Pose",
}


class PromptFormatter:
    """
    Formats tagger results into markdown prompt string.

    Output format:
        # Provider
        tag1, tag2, tag3

        # Provider2
        tag1, tag2, tag3
    """

    def __init__(self):
        """Initialize formatter."""
        pass

    def format_by_provider(
        self,
        tagger_results: Dict[str, TaggerResult],
        threshold: float = 0.0,
    ) -> str:
        """
        Format all tags grouped by provider in markdown format.

        Args:
            tagger_results: Dict of provider_name -> TaggerResult
            threshold: Minimum confidence to include tag

        Returns:
            Markdown formatted string with # headers per provider
        """
        if not tagger_results:
            return ""

        sections = []

        for provider, result in tagger_results.items():
            if not result or not result.tags:
                continue

            # Filter by threshold
            filtered_tags = [t for t in result.tags if t.confidence >= threshold]

            if not filtered_tags:
                continue

            # Get display name
            display_name = PROVIDER_NAMES.get(provider, provider.title())

            # Format tags as comma-separated
            tag_texts = [tag.text for tag in filtered_tags]
            tags_str = ", ".join(tag_texts)

            # Build section
            section = f"# {display_name}\n{tags_str}"
            sections.append(section)

        return "\n\n".join(sections)

    def format_flat(self, tags: List[TagItem], separator: str = ", ") -> str:
        """
        Format tags as flat comma-separated string (no grouping).

        Args:
            tags: List of TagItem objects
            separator: Separator between tags

        Returns:
            Formatted string: tag1, tag2, tag3
        """
        if not tags:
            return ""

        return separator.join(tag.text for tag in tags)

    def format_with_confidence(
        self,
        tagger_results: Dict[str, TaggerResult],
        threshold: float = 0.0,
    ) -> str:
        """
        Format tags with confidence scores, grouped by provider.

        Args:
            tagger_results: Dict of provider_name -> TaggerResult
            threshold: Minimum confidence to include tag

        Returns:
            Markdown formatted string with (tag:confidence) format
        """
        if not tagger_results:
            return ""

        sections = []

        for provider, result in tagger_results.items():
            if not result or not result.tags:
                continue

            # Filter by threshold
            filtered_tags = [t for t in result.tags if t.confidence >= threshold]

            if not filtered_tags:
                continue

            # Get display name
            display_name = PROVIDER_NAMES.get(provider, provider.title())

            # Format tags with confidence
            tag_parts = [f"({tag.text}:{tag.confidence:.2f})" for tag in filtered_tags]
            tags_str = ", ".join(tag_parts)

            # Build section
            section = f"# {display_name}\n{tags_str}"
            sections.append(section)

        return "\n\n".join(sections)
