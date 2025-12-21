"""Composite fashion tagger with individual model toggles and params."""

from PIL import Image

from ..base import BaseTagger, TagItem, TaggerResult


class FashionTagger(BaseTagger):
    """
    Composite fashion tagger that runs multiple detection models.

    Each model can be individually enabled/disabled with its own parameters.
    """

    TAGGER_NAME = "fashion"

    def __init__(
        self,
        models: dict[str, bool] | None = None,
        params: dict[str, any] | None = None,
        **kwargs
    ):
        """
        Initialize fashion tagger.

        Args:
            models: Dict of model_name -> enabled
            params: Dict of parameter_name -> value for each model
        """
        super().__init__(**kwargs)
        self.models_config = models or {"yolov8": True}
        self.params = params or {}
        self._sub_taggers = {}

    def _load_tagger(self, name: str) -> BaseTagger | None:
        """Load a specific tagger by name with its params."""
        if name in self._sub_taggers:
            return self._sub_taggers[name]

        try:
            if name == "yolov8":
                from .yolov8_clothing import YOLOv8ClothingTagger
                threshold = self.params.get("yolov8_threshold", 0.4)
                tagger = YOLOv8ClothingTagger(threshold=threshold)

            elif name == "yolos":
                from .yolos_fashionpedia import YOLOSFashionpediaTagger
                threshold = self.params.get("yolos_threshold", 0.4)
                tagger = YOLOSFashionpediaTagger(threshold=threshold)

            elif name == "clip":
                from .fashion_clip import FashionCLIPTagger
                threshold = self.params.get("clip_threshold", 0.3)
                top_k = self.params.get("clip_top_k", 5)
                tagger = FashionCLIPTagger(threshold=threshold, top_k=top_k)

            elif name == "segformer":
                from .segformer_clothes import SegFormerClothesTagger
                min_area = self.params.get("segformer_min_area", 0.03)
                tagger = SegFormerClothesTagger(min_area_percent=min_area)

            elif name == "wargon":
                from .wargon_classifier import WargonClothingTagger
                threshold = self.params.get("wargon_threshold", 0.3)
                top_k = self.params.get("wargon_top_k", 3)
                tagger = WargonClothingTagger(threshold=threshold, top_k=top_k)

            else:
                print(f"[SID-Fashion] Unknown model: {name}")
                return None

            self._sub_taggers[name] = tagger
            return tagger

        except ImportError as e:
            print(f"[SID-Fashion] Failed to load {name}: {e}")
            return None

    def load(self) -> None:
        """Load all enabled sub-taggers."""
        if self._is_loaded:
            return

        for name, enabled in self.models_config.items():
            if enabled:
                tagger = self._load_tagger(name)
                if tagger:
                    tagger.load()

        self._is_loaded = True

    def tag(self, image: Image.Image) -> TaggerResult:
        """
        Detect fashion items using all enabled taggers.

        Args:
            image: PIL Image to analyze

        Returns:
            TaggerResult with merged fashion tags
        """
        all_tags = []
        metadata = {
            "models_used": [],
        }

        # Run each enabled tagger
        for name, enabled in self.models_config.items():
            if not enabled:
                continue

            tagger = self._load_tagger(name)
            if not tagger:
                continue

            try:
                result = tagger.tag(image)
                all_tags.extend(result.tags)
                metadata["models_used"].append(name)
            except Exception as e:
                print(f"[SID-Fashion] {name} failed: {e}")

        # Deduplicate tags (keep highest confidence)
        seen = {}
        unique_tags = []
        for tag in sorted(all_tags, key=lambda x: x.confidence, reverse=True):
            tag_lower = tag.text.lower()
            if tag_lower not in seen:
                seen[tag_lower] = True
                unique_tags.append(tag)

        metadata["num_tags"] = len(unique_tags)

        return TaggerResult(
            tags=unique_tags,
            metadata=metadata,
        )

    def unload(self) -> None:
        """Unload all sub-taggers."""
        for tagger in self._sub_taggers.values():
            tagger.unload()
        self._sub_taggers.clear()
        super().unload()
