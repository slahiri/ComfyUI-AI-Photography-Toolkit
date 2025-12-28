"""Florence Description Parser - Extract structured tags from VLM prose.

Parses Florence-2 descriptions to extract:
- Arm/hand poses (arms raised, hands covering face, etc.)
- Body positions (standing, sitting, leaning, etc.)
- Gaze direction (looking at camera, eyes closed, etc.)
- Body details (tattoos, clothing state, etc.)
"""

import re
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass

from .base import BaseTagger, TagItem, TaggerResult


# =============================================================================
# POSE EXTRACTION PATTERNS
# =============================================================================

ARM_PATTERNS: List[Tuple[str, str, float]] = [
    # Arms raised
    (r"arms?\s+raised\s+above\s+(her|his|their)?\s*head", "arms raised above head", 0.9),
    (r"arms?\s+raised\s+overhead", "arms raised above head", 0.9),
    (r"arms?\s+stretched\s+overhead", "arms stretched overhead", 0.9),
    (r"arms?\s+up", "arms up", 0.85),
    (r"one\s+arm\s+raised", "one arm raised", 0.85),
    (r"both\s+arms?\s+raised", "both arms raised", 0.9),
    # Arms behind head
    (r"arms?\s+behind\s+(her|his|their)?\s*head", "arms behind head", 0.9),
    (r"hands?\s+behind\s+(her|his|their)?\s*head", "hands behind head", 0.9),
    # Arms at sides
    (r"arms?\s+(at|by)\s+(her|his|their)?\s*side", "arms at sides", 0.85),
    (r"arms?\s+relaxed", "arms relaxed at sides", 0.8),
    # Arms crossed
    (r"arms?\s+crossed", "arms crossed", 0.9),
    (r"arms?\s+folded", "arms folded", 0.9),
    # Hand positions
    (r"hands?\s+on\s+(her|his|their)?\s*hips?", "hands on hips", 0.9),
    (r"hands?\s+in\s+(her|his|their)?\s*hair", "hands in hair", 0.85),
    (r"hand\s+running\s+through\s+(her|his|their)?\s*hair", "hand running through hair", 0.9),
    (r"touching\s+(her|his|their)?\s*hair", "hands in hair", 0.8),
    # Covering face/eyes
    (r"covering\s+(her|his|their)?\s*eyes?\s+with\s+(her|his|their)?\s*(fingers?|hands?)", "hands covering eyes", 0.95),
    (r"hands?\s+covering\s+(her|his|their)?\s*face", "hands covering face", 0.9),
    (r"hands?\s+covering\s+(her|his|their)?\s*eyes?", "hands covering eyes", 0.9),
    (r"fingers?\s+(spread\s+)?over\s+(her|his|their)?\s*eyes?", "fingers over eyes", 0.9),
    (r"hand\s+over\s+(her|his|their)?\s*mouth", "hand over mouth", 0.9),
    (r"shielding\s+(her|his|their)?\s*eyes?", "hands covering eyes", 0.85),
    # Face touching
    (r"hand\s+on\s+(her|his|their)?\s*cheek", "hand on cheek", 0.9),
    (r"hand\s+on\s+(her|his|their)?\s*chin", "hand on chin", 0.9),
    (r"touching\s+(her|his|their)?\s*face", "hand touching face", 0.85),
    # Other
    (r"hands?\s+clasped", "hands clasped together", 0.85),
    (r"hands?\s+in\s+(her|his|their)?\s*pockets?", "hands in pockets", 0.9),
    (r"pointing", "pointing gesture", 0.8),
    (r"peace\s+sign", "peace sign", 0.9),
    (r"waving", "waving hand", 0.85),
]

BODY_POSE_PATTERNS: List[Tuple[str, str, float]] = [
    # Standing
    (r"standing\s+straight", "standing straight", 0.9),
    (r"standing\s+by", "standing", 0.85),
    (r"standing\s+near", "standing", 0.85),
    (r"standing\s+in", "standing", 0.85),
    (r"stands?\s+with", "standing", 0.85),
    (r"standing\s+relaxed", "standing relaxed", 0.9),
    (r"standing\s+tall", "standing straight", 0.85),
    # Sitting
    (r"sitting\s+(on|in)", "sitting", 0.85),
    (r"seated", "sitting", 0.85),
    (r"sitting\s+cross-?legged", "sitting cross-legged", 0.9),
    (r"sitting\s+upright", "sitting upright", 0.9),
    (r"perched\s+on", "sitting on edge", 0.8),
    # Leaning
    (r"leaning\s+(forward|back|against|on)", "leaning", 0.85),
    (r"leaning\s+forward", "leaning forward", 0.9),
    (r"leaning\s+back(ward)?", "leaning backward", 0.9),
    (r"leaning\s+to\s+(the\s+)?(left|right|side)", "leaning to one side", 0.9),
    # Lying
    (r"lying\s+(down\s+)?on\s+(her|his|their)?\s*back", "lying on back", 0.9),
    (r"lying\s+on\s+(her|his|their)?\s*side", "lying on side", 0.9),
    (r"lying\s+on\s+(her|his|their)?\s*stomach", "lying on stomach", 0.9),
    (r"reclining", "reclining pose", 0.85),
    (r"lounging", "lounging pose", 0.85),
    # Kneeling
    (r"kneeling", "kneeling", 0.9),
    (r"on\s+(her|his|their)?\s*knees?", "kneeling", 0.85),
    # Crouching
    (r"crouching", "crouching", 0.9),
    (r"squatting", "squatting", 0.9),
    # Other poses
    (r"arched\s+back", "arched back pose", 0.9),
    (r"twisted\s+torso", "twisted torso", 0.85),
    (r"contrapposto", "contrapposto pose", 0.9),
    (r"s-?curve", "s-curve pose", 0.85),
    (r"relaxed\s+pose", "relaxed pose", 0.85),
    (r"candid\s+pose", "candid pose", 0.8),
]

GAZE_PATTERNS: List[Tuple[str, str, float]] = [
    (r"looking\s+(directly\s+)?(at|into)\s+(the\s+)?camera", "looking at camera", 0.95),
    (r"looking\s+at\s+(the\s+)?viewer", "looking at viewer", 0.95),
    (r"gazing\s+(at|into)\s+(the\s+)?camera", "looking at camera", 0.9),
    (r"eyes?\s+closed", "eyes closed", 0.9),
    (r"looking\s+away", "looking away", 0.9),
    (r"looking\s+down", "looking down", 0.9),
    (r"looking\s+up", "looking up", 0.9),
    (r"looking\s+(to\s+the\s+)?(left|right|side)", "looking to the side", 0.9),
    (r"looking\s+over\s+(her|his|their)?\s*shoulder", "looking over shoulder", 0.9),
    (r"side\s+glance", "side glance", 0.85),
    (r"eyes?\s+hidden", "eyes hidden", 0.9),
    (r"covering\s+(her|his|their)?\s*eyes?", "eyes hidden", 0.85),
    (r"downcast\s+eyes?", "downcast eyes", 0.9),
    (r"distant\s+(gaze|look)", "distant gaze", 0.85),
]

EXPRESSION_PATTERNS: List[Tuple[str, str, float]] = [
    (r"smiling", "smiling", 0.9),
    (r"slight\s+smile", "slight smile", 0.9),
    (r"big\s+smile", "big smile", 0.9),
    (r"grinning", "grinning", 0.9),
    (r"laughing", "laughing", 0.9),
    (r"smirking", "smirking", 0.85),
    (r"pouting", "pouting", 0.85),
    (r"frowning", "frowning", 0.85),
    (r"serious\s+(expression|look|face)", "serious expression", 0.9),
    (r"neutral\s+expression", "neutral expression", 0.9),
    (r"sultry\s+(expression|look)", "sultry expression", 0.9),
    (r"seductive\s+(expression|look)", "seductive look", 0.9),
    (r"playful\s+(expression|look)", "playful expression", 0.9),
    (r"confident\s+(expression|look)", "confident expression", 0.9),
    (r"lips?\s+parted", "lips parted", 0.85),
    (r"mouth\s+open", "mouth open", 0.85),
    (r"biting\s+(her|his|their)?\s*lip", "biting lip", 0.9),
]

BODY_DETAIL_PATTERNS: List[Tuple[str, str, float]] = [
    # Tattoos
    (r"tattoo\s+on\s+(her|his|their)?\s*(left\s+)?shoulder", "shoulder tattoo", 0.9),
    (r"tattoo\s+on\s+(her|his|their)?\s*(right\s+)?shoulder", "shoulder tattoo", 0.9),
    (r"tattoo\s+on\s+(her|his|their)?\s*arm", "arm tattoo", 0.9),
    (r"tattoo\s+on\s+(her|his|their)?\s*back", "back tattoo", 0.9),
    (r"tattoo\s+on\s+(her|his|their)?\s*chest", "chest tattoo", 0.9),
    (r"tattoo\s+on\s+(her|his|their)?\s*leg", "leg tattoo", 0.9),
    (r"tattoo\s+on\s+(her|his|their)?\s*wrist", "wrist tattoo", 0.9),
    (r"tattooed", "has tattoos", 0.8),
    (r"(a|has)\s+tattoo", "has tattoo", 0.85),
    # Hair
    (r"long[,]?\s+(dark|black|brown|blonde|red|white)\s+hair", "long hair", 0.9),
    (r"(her|his|their)\s+hair\s+falls?\s+down", "hair falling down", 0.85),
    (r"hair\s+cascad(es?|ing)", "hair cascading", 0.85),
    (r"hair\s+blow(s|ing)", "hair blowing", 0.85),
    # Skin
    (r"dark\s+skin", "dark skin", 0.9),
    (r"tan(ned)?\s+skin", "tanned skin", 0.9),
    (r"fair\s+skin", "fair skin", 0.9),
    (r"pale\s+skin", "pale skin", 0.9),
]

CLOTHING_STATE_PATTERNS: List[Tuple[str, str, float]] = [
    (r"wearing\s+a\s+([a-z]+\s+)?(swimsuit|bikini)", "wearing swimsuit", 0.9),
    (r"in\s+a\s+([a-z]+\s+)?(swimsuit|bikini)", "wearing swimsuit", 0.9),
    (r"wearing\s+a\s+([a-z]+\s+)?dress", "wearing dress", 0.9),
    (r"([a-z]+\s+)?print\s+(swimsuit|bikini|dress)", "patterned clothing", 0.85),
    (r"leopard\s+print", "leopard print", 0.9),
    (r"animal\s+print", "animal print", 0.85),
    (r"revealing", "revealing clothing", 0.8),
    (r"form-?fitting", "form-fitting clothing", 0.85),
    (r"tight(-fitting)?", "tight clothing", 0.8),
]


# =============================================================================
# PARSER CLASS
# =============================================================================

class FlorenceParser(BaseTagger):
    """Parse Florence descriptions to extract structured pose/body tags."""

    def __init__(self, threshold: float = 0.7):
        """Initialize parser.

        Args:
            threshold: Minimum confidence for extracted tags
        """
        super().__init__()
        self.threshold = threshold
        self._is_loaded = True  # No model to load

        # Compile patterns for efficiency
        self._arm_patterns = [(re.compile(p, re.IGNORECASE), tag, conf) for p, tag, conf in ARM_PATTERNS]
        self._body_pose_patterns = [(re.compile(p, re.IGNORECASE), tag, conf) for p, tag, conf in BODY_POSE_PATTERNS]
        self._gaze_patterns = [(re.compile(p, re.IGNORECASE), tag, conf) for p, tag, conf in GAZE_PATTERNS]
        self._expression_patterns = [(re.compile(p, re.IGNORECASE), tag, conf) for p, tag, conf in EXPRESSION_PATTERNS]
        self._body_detail_patterns = [(re.compile(p, re.IGNORECASE), tag, conf) for p, tag, conf in BODY_DETAIL_PATTERNS]
        self._clothing_patterns = [(re.compile(p, re.IGNORECASE), tag, conf) for p, tag, conf in CLOTHING_STATE_PATTERNS]

    def load(self) -> None:
        """No model to load."""
        self._is_loaded = True

    def tag(self, image) -> TaggerResult:
        """Not used - use parse() instead."""
        return TaggerResult(tags=[], metadata={"error": "Use parse() method with text input"})

    def parse(self, text: str) -> TaggerResult:
        """Parse Florence description text to extract tags.

        Args:
            text: Florence description text

        Returns:
            TaggerResult with extracted tags
        """
        if not text:
            return TaggerResult(tags=[], metadata={"error": "No text provided"})

        tags: List[TagItem] = []
        found_tags: Set[str] = set()  # Avoid duplicates

        # Extract from each pattern category
        arm_tags = self._extract_patterns(text, self._arm_patterns, "arm_pose", found_tags)
        body_pose_tags = self._extract_patterns(text, self._body_pose_patterns, "body_pose", found_tags)
        gaze_tags = self._extract_patterns(text, self._gaze_patterns, "gaze", found_tags)
        expression_tags = self._extract_patterns(text, self._expression_patterns, "expression", found_tags)
        body_detail_tags = self._extract_patterns(text, self._body_detail_patterns, "body_detail", found_tags)
        clothing_tags = self._extract_patterns(text, self._clothing_patterns, "clothing", found_tags)

        tags.extend(arm_tags)
        tags.extend(body_pose_tags)
        tags.extend(gaze_tags)
        tags.extend(expression_tags)
        tags.extend(body_detail_tags)
        tags.extend(clothing_tags)

        # Sort by confidence
        tags.sort(key=lambda x: x.confidence, reverse=True)

        return TaggerResult(
            tags=tags,
            metadata={
                "source": "florence_parser",
                "input_length": len(text),
                "tags_extracted": len(tags),
                "categories": {
                    "arm_pose": len(arm_tags),
                    "body_pose": len(body_pose_tags),
                    "gaze": len(gaze_tags),
                    "expression": len(expression_tags),
                    "body_detail": len(body_detail_tags),
                    "clothing": len(clothing_tags),
                }
            }
        )

    def _extract_patterns(
        self,
        text: str,
        patterns: List[Tuple[re.Pattern, str, float]],
        category: str,
        found_tags: Set[str],
    ) -> List[TagItem]:
        """Extract tags matching patterns from text.

        Args:
            text: Text to search
            patterns: List of (compiled_pattern, tag_text, confidence)
            category: Category name for tags
            found_tags: Set of already found tags (to avoid duplicates)

        Returns:
            List of TagItem
        """
        tags = []
        for pattern, tag_text, confidence in patterns:
            if tag_text in found_tags:
                continue

            if pattern.search(text):
                if confidence >= self.threshold:
                    tags.append(TagItem(
                        text=tag_text,
                        confidence=confidence,
                        category=category,
                    ))
                    found_tags.add(tag_text)

        return tags

    def parse_metadata(self, metadata: Dict) -> TaggerResult:
        """Parse all Florence fields from metadata dict.

        Args:
            metadata: Full image metadata dict

        Returns:
            TaggerResult with all extracted tags
        """
        all_text_parts = []

        # Collect all Florence text fields
        florence_fields = [
            "florence_caption",
            "florence_description",
            "florence_mixed_caption",
        ]

        for field in florence_fields:
            if field in metadata and metadata[field]:
                all_text_parts.append(str(metadata[field]))

        # Also check florence_analyze if it has descriptive text
        if "florence_analyze" in metadata and isinstance(metadata["florence_analyze"], dict):
            for key, value in metadata["florence_analyze"].items():
                if isinstance(value, str) and len(value) > 10:
                    all_text_parts.append(value)

        combined_text = " ".join(all_text_parts)
        return self.parse(combined_text)

    def unload(self) -> None:
        """No model to unload."""
        pass


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_parser_instance: Optional[FlorenceParser] = None


def get_florence_parser() -> FlorenceParser:
    """Get or create the global parser instance."""
    global _parser_instance
    if _parser_instance is None:
        _parser_instance = FlorenceParser()
    return _parser_instance


def parse_florence_description(text: str) -> TaggerResult:
    """Parse Florence description text.

    Args:
        text: Florence description text

    Returns:
        TaggerResult with extracted tags
    """
    return get_florence_parser().parse(text)


def parse_florence_metadata(metadata: Dict) -> TaggerResult:
    """Parse Florence fields from metadata dict.

    Args:
        metadata: Full image metadata dict

    Returns:
        TaggerResult with extracted tags
    """
    return get_florence_parser().parse_metadata(metadata)
