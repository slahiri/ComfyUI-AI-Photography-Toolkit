"""Z-Image Prompt Assembler - Build natural language prompts from canonical structure.

Takes tagger metadata organized in canonical categories and VLM description,
assembles them into Z-Image styled natural language prompts.

Key principle: Z-Image prefers VERBOSE NATURAL LANGUAGE over tag soup.
The VLM description is the PRIMARY source - use full sentences and paragraphs.
"""

from typing import Dict, List, Optional, Tuple
import re


# Category display order (Z-Image optimized flow)
CATEGORY_ORDER = [
    "subject",
    "appearance",
    "clothing",
    "pose_action",
    "environment",
    "lighting",
    "style_quality",
    "composition",
    "scene",
    "constraints",
]

# Category headers for structured output
CATEGORY_HEADERS = {
    "subject": "SUBJECT",
    "appearance": "APPEARANCE",
    "clothing": "CLOTHING",
    "pose_action": "POSE & ACTION",
    "environment": "ENVIRONMENT",
    "lighting": "LIGHTING",
    "style_quality": "STYLE & QUALITY",
    "composition": "COMPOSITION",
    "scene": "SCENE",
    "constraints": "CONSTRAINTS",
}

# VLM section markers to parse
VLM_SECTION_MARKERS = {
    "subject": [r"SUBJECT:", r"MAIN SUBJECT:", r"^A\s+(young|adult|elderly)?\s*(woman|man|girl|boy|person)"],
    "appearance": [r"APPEARANCE:", r"PHYSICAL:", r"hair", r"eyes", r"skin"],
    "clothing": [r"CLOTHING:", r"OUTFIT:", r"wearing", r"dressed in"],
    "pose_action": [r"POSE", r"ACTION:", r"EXPRESSION:", r"standing", r"sitting", r"looking"],
    "environment": [r"ENVIRONMENT:", r"SETTING:", r"LOCATION:", r"indoor", r"outdoor"],
    "lighting": [r"LIGHTING:", r"LIGHT:", r"lit by", r"sunlight", r"natural light"],
    "style_quality": [r"STYLE", r"QUALITY:", r"TECHNICAL:", r"photorealistic", r"professional"],
    "composition": [r"COMPOSITION:", r"FRAMING:", r"shot", r"angle", r"depth of field"],
}


class ZImageAssembler:
    """Assembles verbose Z-Image prompts from canonical structure + VLM description."""

    def __init__(self, include_headers: bool = True, verbose: bool = True):
        """Initialize assembler.

        Args:
            include_headers: If True, include [CATEGORY] headers in output
            verbose: If True, use full VLM sentences (recommended for Z-Image)
        """
        self.include_headers = include_headers
        self.verbose = verbose

    def assemble(
        self,
        canonical_structure: Dict,
        vlm_description: str = "",
        scene_type: str = "portrait",
    ) -> Tuple[str, Dict[str, str]]:
        """Assemble Z-Image prompt from canonical structure + VLM description.

        The VLM description is the PRIMARY source for verbose content.
        Tagger metadata supplements where VLM is sparse.

        Args:
            canonical_structure: Dict with category -> {tag: confidence} mappings
            vlm_description: VLM description text (PRIMARY source)
            scene_type: Detected scene type for template selection

        Returns:
            Tuple of (combined_prompt, category_dict)
        """
        category_content: Dict[str, str] = {}

        # Parse VLM description into sections first (this is the primary content)
        vlm_sections = self._parse_vlm_sections(vlm_description) if vlm_description else {}

        # Process each category
        for category in CATEGORY_ORDER:
            content = self._process_category_verbose(
                category,
                canonical_structure.get(category, {}),
                vlm_sections.get(category, ""),
                vlm_description,
            )
            if content:
                category_content[category] = content

        # Build combined prompt
        combined = self._build_combined_prompt(category_content, scene_type)

        return combined, category_content

    def _parse_vlm_sections(self, vlm_description: str) -> Dict[str, str]:
        """Parse VLM description into category sections.

        Looks for section headers like SUBJECT:, APPEARANCE:, etc.
        Falls back to content extraction if no headers found.

        Args:
            vlm_description: Full VLM description text

        Returns:
            Dict of category -> section content
        """
        sections = {}

        # First try to find explicit section headers
        section_pattern = r"(SUBJECT|APPEARANCE|CLOTHING|POSE|ACTION|ENVIRONMENT|SETTING|LIGHTING|STYLE|QUALITY|COMPOSITION)[:\s]+"
        matches = list(re.finditer(section_pattern, vlm_description, re.IGNORECASE))

        if matches:
            # Extract content between section headers
            for i, match in enumerate(matches):
                section_name = match.group(1).upper()
                start = match.end()
                end = matches[i + 1].start() if i + 1 < len(matches) else len(vlm_description)
                content = vlm_description[start:end].strip()

                # Map to canonical category names
                category = self._map_section_to_category(section_name)
                if category:
                    if category in sections:
                        sections[category] += " " + content
                    else:
                        sections[category] = content
        else:
            # No explicit headers - extract by content analysis
            sections = self._extract_sections_by_content(vlm_description)

        return sections

    def _map_section_to_category(self, section_name: str) -> Optional[str]:
        """Map VLM section name to canonical category."""
        mapping = {
            "SUBJECT": "subject",
            "APPEARANCE": "appearance",
            "CLOTHING": "clothing",
            "POSE": "pose_action",
            "ACTION": "pose_action",
            "ENVIRONMENT": "environment",
            "SETTING": "environment",
            "LIGHTING": "lighting",
            "STYLE": "style_quality",
            "QUALITY": "style_quality",
            "COMPOSITION": "composition",
        }
        return mapping.get(section_name.upper())

    def _extract_sections_by_content(self, vlm_description: str) -> Dict[str, str]:
        """Extract category content from unstructured VLM description."""
        sections = {}

        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', vlm_description)

        for sentence in sentences:
            sentence_lower = sentence.lower()

            # Categorize each sentence based on content
            if any(kw in sentence_lower for kw in ["woman", "man", "girl", "boy", "person", "young", "adult", "alone", "solo"]):
                sections.setdefault("subject", []).append(sentence)

            if any(kw in sentence_lower for kw in ["hair", "eyes", "skin", "face", "lips", "nose", "tall", "slim", "athletic", "build"]):
                sections.setdefault("appearance", []).append(sentence)

            if any(kw in sentence_lower for kw in ["wearing", "dressed", "outfit", "dress", "shirt", "pants", "jewelry", "necklace", "earring"]):
                sections.setdefault("clothing", []).append(sentence)

            if any(kw in sentence_lower for kw in ["standing", "sitting", "lying", "pose", "looking", "gaze", "expression", "smile", "hands", "arms"]):
                sections.setdefault("pose_action", []).append(sentence)

            if any(kw in sentence_lower for kw in ["indoor", "outdoor", "room", "studio", "background", "setting", "location", "pool", "beach", "garden"]):
                sections.setdefault("environment", []).append(sentence)

            if any(kw in sentence_lower for kw in ["light", "lighting", "shadow", "sunlight", "golden hour", "soft", "warm", "cool", "rim"]):
                sections.setdefault("lighting", []).append(sentence)

            if any(kw in sentence_lower for kw in ["professional", "editorial", "artistic", "photorealistic", "quality", "resolution", "detailed"]):
                sections.setdefault("style_quality", []).append(sentence)

            if any(kw in sentence_lower for kw in ["close-up", "medium shot", "full body", "wide", "angle", "depth of field", "bokeh", "framing"]):
                sections.setdefault("composition", []).append(sentence)

        # Convert lists to joined strings
        return {cat: " ".join(sents) for cat, sents in sections.items()}

    def _process_category_verbose(
        self,
        category: str,
        tags: Dict,
        vlm_section: str,
        full_vlm: str,
    ) -> str:
        """Process a category into verbose natural language.

        Priority:
        1. VLM section content (full sentences)
        2. Tags converted to natural phrases (supplement only)

        Args:
            category: Category name
            tags: Dict of {tag: confidence} from taggers
            vlm_section: Extracted VLM section for this category
            full_vlm: Full VLM description for additional extraction

        Returns:
            Verbose natural language description
        """
        parts = []

        # PRIMARY: Use VLM section content (full sentences)
        if vlm_section:
            # Clean up and use the full VLM content
            cleaned = vlm_section.strip()
            if cleaned:
                parts.append(cleaned)

        # SUPPLEMENT: Add high-confidence tags not already mentioned
        if tags and isinstance(tags, dict):
            tag_phrases = self._extract_tag_phrases(tags, vlm_section)
            if tag_phrases:
                # Only add tags that aren't already in the VLM content
                new_phrases = [p for p in tag_phrases if p.lower() not in vlm_section.lower()]
                if new_phrases and len(new_phrases) <= 5:
                    # Add as supplementary detail
                    supplement = ", ".join(new_phrases)
                    if parts:
                        parts.append(f"Additional details: {supplement}.")
                    else:
                        parts.append(self._tags_to_sentence(new_phrases, category))

        # If still empty, try to extract from full VLM
        if not parts and full_vlm:
            extracted = self._extract_category_from_full_vlm(category, full_vlm)
            if extracted:
                parts.append(extracted)

        return " ".join(parts)

    def _extract_tag_phrases(self, tags: Dict, existing_content: str) -> List[str]:
        """Extract phrases from tags, filtering out what's already mentioned."""
        phrases = []
        existing_lower = existing_content.lower() if existing_content else ""

        # Handle nested or flat structure
        if tags:
            first_value = next(iter(tags.values())) if tags else None

            if isinstance(first_value, dict):
                # Nested: {subcategory: {tag: conf}}
                for subcat, subtags in tags.items():
                    if isinstance(subtags, dict):
                        for tag, conf in sorted(subtags.items(), key=lambda x: -x[1])[:5]:
                            if isinstance(conf, (int, float)) and conf >= 0.4:
                                phrase = self._clean_tag(tag)
                                if phrase.lower() not in existing_lower:
                                    phrases.append(phrase)
            elif isinstance(first_value, (int, float)):
                # Flat: {tag: confidence}
                for tag, conf in sorted(tags.items(), key=lambda x: -x[1])[:8]:
                    if conf >= 0.4:
                        phrase = self._clean_tag(tag)
                        if phrase.lower() not in existing_lower:
                            phrases.append(phrase)

        return phrases[:5]

    def _clean_tag(self, tag: str) -> str:
        """Clean a tag into readable phrase."""
        # Remove underscores, handle counts
        phrase = tag.replace("_", " ").strip()

        # Handle booru count tags
        count_match = re.match(r"(\d+)(girl|boy|woman|man|person)", phrase)
        if count_match:
            count, noun = count_match.groups()
            if count == "1":
                return f"a {noun}"
            return f"{count} {noun}s"

        if phrase == "solo":
            return "alone"

        return phrase

    def _tags_to_sentence(self, phrases: List[str], category: str) -> str:
        """Convert tag phrases into a natural sentence."""
        if not phrases:
            return ""

        # Category-specific sentence starters
        starters = {
            "subject": "The subject is ",
            "appearance": "Physical features include ",
            "clothing": "Wearing ",
            "pose_action": "The pose shows ",
            "environment": "The setting is ",
            "lighting": "Lighting consists of ",
            "style_quality": "The style is ",
            "composition": "Composition features ",
            "scene": "Scene type: ",
            "constraints": "Technical notes: ",
        }

        starter = starters.get(category, "")

        if len(phrases) == 1:
            return f"{starter}{phrases[0]}."
        elif len(phrases) == 2:
            return f"{starter}{phrases[0]} and {phrases[1]}."
        else:
            return f"{starter}{', '.join(phrases[:-1])}, and {phrases[-1]}."

    def _extract_category_from_full_vlm(self, category: str, vlm_description: str) -> str:
        """Last resort: extract category-relevant content from full VLM."""
        patterns = {
            "subject": r"(A\s+(?:young|adult|middle-aged|elderly)?\s*(?:woman|man|girl|boy|person)[^.]*\.)",
            "appearance": r"([^.]*(?:hair|eyes|skin|face|features)[^.]*\.)",
            "clothing": r"([^.]*(?:wearing|dressed in|outfit|dress|shirt)[^.]*\.)",
            "pose_action": r"([^.]*(?:standing|sitting|pose|looking|expression)[^.]*\.)",
            "environment": r"([^.]*(?:indoor|outdoor|background|setting|room|studio)[^.]*\.)",
            "lighting": r"([^.]*(?:light|lighting|shadow|sunlight|illuminat)[^.]*\.)",
        }

        if category in patterns:
            matches = re.findall(patterns[category], vlm_description, re.IGNORECASE)
            if matches:
                return " ".join(matches[:2])

        return ""

    def _build_combined_prompt(
        self,
        category_content: Dict[str, str],
        scene_type: str,
    ) -> str:
        """Build combined prompt from categories."""
        if self.include_headers:
            # Structured format with [CATEGORY] headers
            lines = []
            for category in CATEGORY_ORDER:
                if category in category_content and category_content[category]:
                    header = CATEGORY_HEADERS.get(category, category.upper())
                    content = category_content[category]
                    lines.append(f"[{header}]\n{content}")
            return "\n\n".join(lines)
        else:
            # Flowing prose format
            parts = []
            for category in CATEGORY_ORDER:
                if category in category_content and category_content[category]:
                    parts.append(category_content[category])
            return " ".join(parts)

    def format_structured(self, category_content: Dict[str, str]) -> str:
        """Format as structured output with headers."""
        lines = []
        for category in CATEGORY_ORDER:
            if category in category_content and category_content[category]:
                header = CATEGORY_HEADERS.get(category, category.upper())
                content = category_content[category]
                lines.append(f"[{header}]\n{content}")
        return "\n\n".join(lines)

    def format_flowing(self, category_content: Dict[str, str]) -> str:
        """Format as flowing prose."""
        parts = []
        for category in CATEGORY_ORDER:
            if category in category_content and category_content[category]:
                parts.append(category_content[category])
        return " ".join(parts)


# Convenience function
def assemble_zimage_prompt(
    canonical_structure: Dict,
    vlm_description: str = "",
    scene_type: str = "portrait",
    include_headers: bool = True,
) -> Tuple[str, Dict[str, str]]:
    """Assemble verbose Z-Image prompt from canonical structure + VLM description.

    Args:
        canonical_structure: Dict with category -> {tag: confidence}
        vlm_description: VLM description (PRIMARY source for content)
        scene_type: Scene type for template selection
        include_headers: Include [CATEGORY] headers in output

    Returns:
        Tuple of (prompt_string, category_dict)
    """
    assembler = ZImageAssembler(include_headers=include_headers, verbose=True)
    return assembler.assemble(canonical_structure, vlm_description, scene_type)
