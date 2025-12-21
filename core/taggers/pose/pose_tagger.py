"""Composite pose tagger with individual model toggles and params."""

from PIL import Image

from ..base import BaseTagger, TagItem, TaggerResult


class PoseTagger(BaseTagger):
    """
    Composite pose tagger that runs multiple pose estimation models.

    Each model can be individually enabled/disabled with its own parameters.
    """

    TAGGER_NAME = "pose"

    def __init__(
        self,
        models: dict[str, bool] | None = None,
        params: dict[str, any] | None = None,
        **kwargs
    ):
        """
        Initialize pose tagger.

        Args:
            models: Dict of model_name -> enabled
            params: Dict of parameter_name -> value for each model
        """
        super().__init__(**kwargs)
        self.models_config = models or {"mediapipe": True}
        self.params = params or {}
        self._sub_taggers = {}

    def _load_tagger(self, name: str) -> BaseTagger | None:
        """Load a specific tagger by name with its params."""
        if name in self._sub_taggers:
            return self._sub_taggers[name]

        try:
            if name == "mediapipe":
                from .mediapipe_pose import MediaPipePoseTagger
                confidence = self.params.get("mediapipe_confidence", 0.5)
                tagger = MediaPipePoseTagger(
                    min_detection_confidence=confidence,
                    min_tracking_confidence=confidence,
                )

            elif name == "motionbert":
                from .motionbert import MotionBERTTagger
                tagger = MotionBERTTagger()

            elif name == "sapiens":
                from .sapiens_pose import SapiensPoseTagger
                tagger = SapiensPoseTagger()

            else:
                print(f"[SID-Pose] Unknown model: {name}")
                return None

            self._sub_taggers[name] = tagger
            return tagger

        except ImportError as e:
            print(f"[SID-Pose] Failed to load {name}: {e}")
            return None
        except RuntimeError as e:
            # VRAM check failed for Sapiens
            print(f"[SID-Pose] {name} unavailable: {e}")
            return None

    def load(self) -> None:
        """Load all enabled sub-taggers."""
        if self._is_loaded:
            return

        for name, enabled in self.models_config.items():
            if enabled:
                tagger = self._load_tagger(name)
                if tagger:
                    try:
                        tagger.load()
                    except Exception as e:
                        print(f"[SID-Pose] Failed to load {name}: {e}")

        self._is_loaded = True

    def tag(self, image: Image.Image) -> TaggerResult:
        """
        Detect pose using all enabled taggers.

        Args:
            image: PIL Image to analyze

        Returns:
            TaggerResult with merged pose tags
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

                # Merge sub-metadata
                if result.metadata:
                    metadata[name] = result.metadata

            except Exception as e:
                print(f"[SID-Pose] {name} failed: {e}")

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
