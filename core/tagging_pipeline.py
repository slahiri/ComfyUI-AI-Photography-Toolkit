"""Tagging pipeline that orchestrates multiple taggers."""

from dataclasses import dataclass, field
from PIL import Image

import json

from .taggers import WD14Tagger, PhotographyTagger, TagItem, TaggerResult
from .taggers import RAMPlusTagger, JoyTagTagger
from .taggers import NudeNetTagger
from .taggers import IQATagger, CADBCompositionTagger, SaliencyTagger
from .taggers.wd14 import set_verbose as set_wd14_verbose
from .taggers.utils import get_human_detection_tags

# Taggers that require human presence
HUMAN_SPECIFIC_TAGGERS = {"fashion", "pose"}


@dataclass
class PipelineResult:
    """Result from the tagging pipeline."""
    tags_string: str = ""
    tags_list: list[TagItem] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    tagger_results: dict[str, TaggerResult] = field(default_factory=dict)
    skipped_taggers: list[str] = field(default_factory=list)
    failed_taggers: dict[str, str] = field(default_factory=dict)


class TaggingPipeline:
    """
    Orchestrates multiple taggers based on configuration.

    Usage:
        config = {
            "taggers": {
                "wd14": {
                    "enabled": True,
                    "model": "wd-swinv2-tagger-v3",
                    "general_threshold": 0.35,
                    ...
                }
            },
            "max_tags": 50,
            "verbose": False,
        }

        pipeline = TaggingPipeline(config)
        result = pipeline.run(pil_image)
        print(result.tags_string)
    """

    def __init__(self, config: dict):
        """
        Initialize pipeline with configuration.

        Args:
            config: Tag configuration dictionary from SID_TagConfigurator
        """
        self.config = config
        self.verbose = config.get("verbose", False)
        self.max_tags = config.get("max_tags", 50)
        self.taggers_config = config.get("taggers", {})

        # Active tagger instances (lazy loaded)
        self._taggers = {}

    def _log(self, message: str) -> None:
        """Print message if verbose is enabled."""
        if self.verbose:
            print(f"[SID-Pipeline] {message}")

    def _get_tagger(self, name: str):
        """Get or create a tagger instance."""
        if name in self._taggers:
            return self._taggers[name]

        tagger_config = self.taggers_config.get(name, {})

        if name == "wd14":
            set_wd14_verbose(self.verbose)
            tagger = WD14Tagger(
                model_name=tagger_config.get("model", "wd-swinv2-tagger-v3"),
                general_threshold=tagger_config.get("general_threshold", 0.35),
                character_threshold=tagger_config.get("character_threshold", 0.85),
                replace_underscore=tagger_config.get("replace_underscore", True),
                exclude_tags=tagger_config.get("exclude_tags", ""),
            )
            self._taggers[name] = tagger
            return tagger

        elif name == "ram_plus":
            tagger = RAMPlusTagger(
                threshold=tagger_config.get("threshold", 0.5),
            )
            self._taggers[name] = tagger
            return tagger

        elif name == "joytag":
            tagger = JoyTagTagger(
                threshold=tagger_config.get("threshold", 0.4),
            )
            self._taggers[name] = tagger
            return tagger

        elif name == "nudenet":
            tagger = NudeNetTagger(
                threshold=tagger_config.get("threshold", 0.5),
                include_covered=tagger_config.get("include_covered", True),
                include_faces=tagger_config.get("include_faces", True),
            )
            self._taggers[name] = tagger
            return tagger

        elif name == "photography":
            tagger = PhotographyTagger()
            self._taggers[name] = tagger
            return tagger

        elif name == "fashion":
            from .taggers.fashion import FashionTagger
            tagger = FashionTagger(
                models=tagger_config.get("models", {"yolov8": True}),
                params=tagger_config.get("params", {}),
            )
            self._taggers[name] = tagger
            return tagger

        elif name == "pose":
            from .taggers.pose import PoseTagger
            tagger = PoseTagger(
                models=tagger_config.get("models", {"mediapipe": True}),
                params=tagger_config.get("params", {}),
            )
            self._taggers[name] = tagger
            return tagger

        elif name == "iqa":
            tagger = IQATagger(
                use_nima=tagger_config.get("use_nima", True),
                use_musiq=tagger_config.get("use_musiq", True),
                use_brisque=tagger_config.get("use_brisque", True),
                use_clipiqa=tagger_config.get("use_clipiqa", False),
            )
            self._taggers[name] = tagger
            return tagger

        elif name == "composition":
            tagger = CADBCompositionTagger(
                threshold=tagger_config.get("threshold", 0.3),
                detect_elements=tagger_config.get("detect_elements", True),
            )
            self._taggers[name] = tagger
            return tagger

        elif name == "saliency":
            tagger = SaliencyTagger(
                use_dino=tagger_config.get("use_dino", True),
                fallback_to_cv=tagger_config.get("fallback_to_cv", True),
            )
            self._taggers[name] = tagger
            return tagger

        raise ValueError(f"Unknown tagger: {name}")

    def _detect_human(self, wd14_result: TaggerResult) -> bool:
        """
        Check if WD14 tags indicate a human is present.

        Args:
            wd14_result: TaggerResult from WD14 tagger

        Returns:
            True if human detected, False otherwise
        """
        # Load human indicator tags from config
        human_tags = get_human_detection_tags()

        for tag in wd14_result.tags:
            tag_lower = tag.text.lower().strip()
            if tag_lower in human_tags:
                self._log(f"  Human detected via tag: '{tag.text}'")
                return True
        return False

    def run(self, image: Image.Image) -> PipelineResult:
        """
        Run all enabled taggers on the image.

        WD14 runs first to detect if a human is present.
        Human-specific taggers (fashion, pose) are skipped if no human detected.

        Args:
            image: PIL Image to analyze

        Returns:
            PipelineResult with aggregated tags and metadata
        """
        self._log(f"Running tagging pipeline on image {image.size}")
        self._log(f"Enabled taggers: {list(self.taggers_config.keys())}")

        all_tags = []
        tagger_results = {}
        metadata = {}
        skipped_taggers = []
        failed_taggers = {}
        human_detected = False

        # Step 1: Run WD14 first (always enabled)
        if "wd14" in self.taggers_config:
            self._log("Running WD14 tagger first for human detection...")
            try:
                tagger = self._get_tagger("wd14")
                result = tagger.tag(image)
                tagger_results["wd14"] = result

                for tag in result.tags:
                    tag.category = f"wd14:{tag.category}"
                    all_tags.append(tag)

                if result.metadata:
                    metadata["wd14"] = result.metadata

                self._log(f"  wd14: {len(result.tags)} tags")

                # Check for human presence
                human_detected = self._detect_human(result)
                self._log(f"  Human detected: {human_detected}")

            except Exception as e:
                self._log(f"  wd14 failed: {e}")
                print(f"[SID-Pipeline] Warning: wd14 tagger failed: {e}")
                failed_taggers["wd14"] = str(e)

        # Step 2: Run remaining taggers
        for tagger_name, tagger_config in self.taggers_config.items():
            # Skip WD14 (already ran)
            if tagger_name == "wd14":
                continue

            if not tagger_config.get("enabled", True):
                continue

            # Skip human-specific taggers if no human detected
            if tagger_name in HUMAN_SPECIFIC_TAGGERS and not human_detected:
                self._log(f"  Skipping {tagger_name} (no human detected)")
                skipped_taggers.append(tagger_name)
                continue

            self._log(f"Running tagger: {tagger_name}")

            try:
                tagger = self._get_tagger(tagger_name)
                result = tagger.tag(image)

                # Store individual result
                tagger_results[tagger_name] = result

                # Collect tags with tagger info
                for tag in result.tags:
                    tag.category = f"{tagger_name}:{tag.category}"
                    all_tags.append(tag)

                # Collect metadata
                if result.metadata:
                    metadata[tagger_name] = result.metadata

                self._log(f"  {tagger_name}: {len(result.tags)} tags")

            except Exception as e:
                self._log(f"  {tagger_name} failed: {e}")
                print(f"[SID-Pipeline] Warning: {tagger_name} tagger failed: {e}")
                failed_taggers[tagger_name] = str(e)

        # Sort all tags by confidence
        all_tags.sort(key=lambda x: x.confidence, reverse=True)

        # Deduplicate by tag text (keep highest confidence)
        seen = {}
        unique_tags = []
        for tag in all_tags:
            if tag.text.lower() not in seen:
                seen[tag.text.lower()] = True
                unique_tags.append(tag)

        # Limit to max_tags
        if len(unique_tags) > self.max_tags:
            unique_tags = unique_tags[:self.max_tags]

        # Build JSON output grouped by source
        tags_string = self._build_json_output(tagger_results, skipped_taggers, failed_taggers)

        self._log(f"Pipeline complete: {len(unique_tags)} unique tags")
        if skipped_taggers:
            self._log(f"Skipped taggers (no human): {skipped_taggers}")
        if unique_tags:
            self._log(f"Top tags: {list(json.loads(tags_string).keys())}")

        return PipelineResult(
            tags_string=tags_string,
            tags_list=unique_tags,
            metadata=metadata,
            tagger_results=tagger_results,
            skipped_taggers=skipped_taggers,
            failed_taggers=failed_taggers,
        )

    def _build_json_output(
        self,
        tagger_results: dict[str, TaggerResult],
        skipped_taggers: list[str] = None,
        failed_taggers: dict[str, str] = None,
    ) -> str:
        """
        Build JSON-formatted output per tagger with confidence scores.

        Args:
            tagger_results: Dict of tagger name -> TaggerResult
            skipped_taggers: List of tagger names that were skipped
            failed_taggers: Dict of tagger name -> error message

        Returns:
            JSON string with { tagger: { tags: [{tag, confidence}], metadata } }
        """
        skipped_taggers = skipped_taggers or []
        failed_taggers = failed_taggers or {}

        output = {}

        # Add results from each tagger (even if no tags)
        for tagger_name, result in tagger_results.items():
            # Build tag list with confidence scores
            tag_list = []
            for tag in result.tags:
                tag_list.append({
                    "tag": tag.text,
                    "confidence": round(tag.confidence, 3),
                    "category": tag.category
                })

            output[tagger_name] = {
                "tags": tag_list,
                "count": len(tag_list),
                "metadata": result.metadata or {}
            }

        # Add skipped taggers
        for tagger_name in skipped_taggers:
            output[tagger_name] = {
                "skipped": True,
                "reason": "no human detected",
                "tags": [],
                "count": 0
            }

        # Add failed taggers
        for tagger_name, error_msg in failed_taggers.items():
            output[tagger_name] = {
                "failed": True,
                "error": error_msg,
                "tags": [],
                "count": 0
            }

        return json.dumps(output, indent=2, ensure_ascii=False)

    def unload_all(self) -> None:
        """Unload all tagger models to free memory."""
        self._log("Unloading all taggers...")
        for name, tagger in self._taggers.items():
            self._log(f"  Unloading {name}")
            tagger.unload()
        self._taggers.clear()
