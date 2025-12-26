"""Response combiner for merging multi-pass VLM outputs."""

import re
from typing import Optional

from .modules import REFINE_CONFIG


class ResponseCombiner:
    """
    Combines responses from multiple VLM passes into a final prompt.

    Handles:
    - Ordering sections logically
    - Removing redundancy
    - Formatting for target model (z-image, sdxl, flux, pony, etc.)
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or REFINE_CONFIG
        self.combiner_config = self.config.get("combiner", {})

    def combine(
        self,
        responses: dict[str, str],
        target_format: str = "z-image",
        verbose: bool = False,
    ) -> str:
        """
        Combine module responses into final prompt.

        Args:
            responses: Dict of module_name -> VLM response
            target_format: Target model format
            verbose: Enable logging

        Returns:
            Combined and formatted prompt
        """
        # Get section order
        section_order = self.combiner_config.get(
            "section_order",
            ["subject", "clothing", "body_detail", "pose", "framing", "environment", "lighting", "style"]
        )

        # Collect and clean responses in order
        parts = []
        for section in section_order:
            if section in responses:
                cleaned = self._clean_response(responses[section])
                if cleaned:
                    parts.append(cleaned)
                    if verbose:
                        print(f"[ResponseCombiner] {section}: {len(cleaned)} chars")

        # Add any remaining sections not in order
        for section, response in responses.items():
            if section not in section_order:
                cleaned = self._clean_response(response)
                if cleaned:
                    parts.append(cleaned)

        if not parts:
            return "Error: No valid responses from modules"

        # Join with separator
        separator = self.combiner_config.get("separator", ", ")
        combined = separator.join(parts)

        # Remove redundant phrases
        combined = self._remove_redundancy(combined)

        # Apply format-specific prefix/suffix
        combined = self._apply_format(combined, target_format)

        if verbose:
            print(f"[ResponseCombiner] Final: {len(combined)} chars")

        return combined

    def _clean_response(self, response: str) -> str:
        """Clean a single VLM response."""
        if not response:
            return ""

        text = response.strip()

        # Remove common VLM artifacts
        artifacts = [
            "Here is", "Here's", "I can see", "I see", "The image shows",
            "In this image", "This image", "Looking at the image",
            "Based on the image", "From the image", "The photo shows",
            "Descriptive phrases:", "Description:", "Answer:",
        ]

        for artifact in artifacts:
            if text.lower().startswith(artifact.lower()):
                text = text[len(artifact):].lstrip(":").strip()

        # Remove numbered list formatting (1. 2. 3. etc)
        text = re.sub(r'^\d+\.\s*', '', text, flags=re.MULTILINE)

        # Remove bullet points
        text = re.sub(r'^[-•*]\s*', '', text, flags=re.MULTILINE)

        # Remove section headers in CAPS or with colons
        text = re.sub(r'^[A-Z][A-Z\s]+:\s*', '', text, flags=re.MULTILINE)

        # Convert newlines to commas
        text = re.sub(r'\n+', ', ', text)

        # Clean up multiple commas/spaces
        text = re.sub(r',\s*,', ',', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r',\s*$', '', text)
        text = re.sub(r'^\s*,', '', text)

        return text.strip()

    def _remove_redundancy(self, text: str) -> str:
        """Remove redundant/duplicate phrases."""
        # Split into phrases
        phrases = [p.strip() for p in text.split(',')]

        # Track seen phrases (normalized)
        seen = set()
        unique = []

        for phrase in phrases:
            if not phrase:
                continue

            # Normalize for comparison
            normalized = phrase.lower().strip()

            # Skip if too similar to something we've seen
            is_duplicate = False
            for seen_phrase in seen:
                if self._is_similar(normalized, seen_phrase):
                    is_duplicate = True
                    break

            if not is_duplicate:
                seen.add(normalized)
                unique.append(phrase)

        return ", ".join(unique)

    def _is_similar(self, a: str, b: str) -> bool:
        """Check if two phrases are too similar."""
        # Exact match
        if a == b:
            return True

        # One contains the other
        if a in b or b in a:
            return True

        # Word overlap check
        words_a = set(a.split())
        words_b = set(b.split())

        if len(words_a) < 2 or len(words_b) < 2:
            return False

        overlap = len(words_a & words_b)
        min_len = min(len(words_a), len(words_b))

        # If >80% of smaller phrase overlaps, consider similar
        if overlap / min_len > 0.8:
            return True

        return False

    def _apply_format(self, text: str, target_format: str) -> str:
        """Apply format-specific prefix and suffix."""
        prefixes = self.combiner_config.get("format_prefixes", {})
        suffixes = self.combiner_config.get("format_suffixes", {})

        prefix = prefixes.get(target_format, "")
        suffix = suffixes.get(target_format, "")

        # Don't add prefix if text already starts with it
        if prefix and not text.lower().startswith(prefix.lower()[:20]):
            text = prefix + text

        if suffix:
            text = text + suffix

        return text

    def to_booru_tags(self, text: str) -> str:
        """
        Convert natural language to booru-style tags.

        Used for pony/illustrious formats.
        """
        # Split into phrases
        phrases = [p.strip() for p in text.split(',')]

        tags = []
        for phrase in phrases:
            if not phrase:
                continue

            # Convert to tag format
            tag = phrase.lower()

            # Replace spaces with underscores
            tag = re.sub(r'\s+', '_', tag)

            # Remove articles
            tag = re.sub(r'^(a|an|the)_', '', tag)

            # Clean special chars
            tag = re.sub(r'[^\w_]', '', tag)

            # Skip empty or too short
            if len(tag) > 1:
                tags.append(tag)

        return ", ".join(tags)
