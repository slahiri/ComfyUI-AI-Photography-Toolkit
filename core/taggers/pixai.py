"""PixAI Tagger v0.9 - Anime image tagging with newer Danbooru data.

Uses dghs-imgutils library for proper inference.
"""

from PIL import Image

from .base import BaseTagger, TagItem, TaggerResult


class PixAITagger(BaseTagger):
    """
    PixAI Tagger v0.9 for anime/illustration tagging.

    Uses EVA02 backbone (same as WD14 v3) with newer 2025-01 Danbooru data.
    Better character recognition than WD14.

    - 13,461 tags (9,741 general + 3,720 character)
    - 448x448 input
    - Requires: pip install dghs-imgutils>=0.19.0
    """

    TAGGER_NAME = "pixai"

    def __init__(
        self,
        general_threshold: float = 0.30,
        character_threshold: float = 0.85,
        replace_underscore: bool = True,
        exclude_tags: str = "",
        **kwargs,
    ):
        """
        Initialize PixAI tagger.

        Args:
            general_threshold: Confidence threshold for general tags (default 0.30)
            character_threshold: Confidence threshold for character tags (default 0.85)
            replace_underscore: Replace underscores with spaces
            exclude_tags: Comma-separated list of tags to exclude
        """
        super().__init__(**kwargs)

        self.general_threshold = general_threshold
        self.character_threshold = character_threshold
        self.replace_underscore = replace_underscore
        self.exclude_tags = set(
            s.strip().lower() for s in exclude_tags.split(",") if s.strip()
        )

    def load(self) -> None:
        """Check if imgutils is available."""
        if self.is_loaded:
            return

        try:
            from imgutils.tagging import get_pixai_tags
            self._get_pixai_tags = get_pixai_tags
        except ImportError:
            raise ImportError(
                "dghs-imgutils is required for PixAI tagger. "
                "Install with: pip install dghs-imgutils>=0.19.0"
            )

        self._is_loaded = True
        print("[SID-PixAI] PixAI Tagger v0.9 ready (using imgutils)")

    def tag(self, image: Image.Image) -> TaggerResult:
        """
        Process image and return tags.

        Args:
            image: PIL Image to analyze

        Returns:
            TaggerResult with detected tags
        """
        if not self.is_loaded:
            self.load()

        # Ensure RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Get tags using imgutils (returns all tags, we filter by threshold)
        general_tags, character_tags, _, _ = self._get_pixai_tags(
            image,
            model_name='v0.9',
            fmt=('general', 'character', 'ips', 'ips_mapping'),
        )

        # Convert to TagItem list
        tags = []

        # Process general tags (filter by threshold)
        for tag_name, confidence in general_tags.items():
            if confidence < self.general_threshold:
                continue
            if tag_name.lower() in self.exclude_tags:
                continue

            # Process tag name
            if self.replace_underscore:
                tag_name = tag_name.replace("_", " ")

            # Escape parentheses for prompt compatibility
            tag_text = tag_name.replace("(", "\\(").replace(")", "\\)")

            tags.append(TagItem(
                text=tag_text,
                confidence=float(confidence),
                category="general"
            ))

        # Process character tags (filter by threshold)
        for tag_name, confidence in character_tags.items():
            if confidence < self.character_threshold:
                continue
            if tag_name.lower() in self.exclude_tags:
                continue

            # Process tag name
            if self.replace_underscore:
                tag_name = tag_name.replace("_", " ")

            # Escape parentheses for prompt compatibility
            tag_text = tag_name.replace("(", "\\(").replace(")", "\\)")

            tags.append(TagItem(
                text=tag_text,
                confidence=float(confidence),
                category="character"
            ))

        # Sort by confidence (descending)
        tags.sort(key=lambda x: x.confidence, reverse=True)

        return TaggerResult(
            tags=tags,
            metadata={
                "model": "pixai-tagger-v0.9",
                "general_threshold": self.general_threshold,
                "character_threshold": self.character_threshold,
                "general_tags": len(general_tags),
                "character_tags": len(character_tags),
            }
        )

    def unload(self) -> None:
        """Release resources."""
        self._is_loaded = False
        super().unload()
