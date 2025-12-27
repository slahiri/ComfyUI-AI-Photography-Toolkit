"""
Prompt Generator Engine - Stage 1A Tagger Orchestrator.

Reuses the existing TaggingPipeline from core/tagging_pipeline.py.
Automatically routes taggers based on image content (human/non-human).
"""

from dataclasses import dataclass, field
from pathlib import Path

import yaml
from PIL import Image

from ..tagging_pipeline import TaggingPipeline
from ..taggers.base import TaggerResult
from .formatter import PromptFormatter


# Config file path
CONFIG_PATH = Path(__file__).parent.parent / "config" / "prompt_generator.yaml"


@dataclass
class EngineResult:
    """Result from the prompt generator engine."""
    prompt: str = ""
    tagger_results: dict[str, TaggerResult] = field(default_factory=dict)
    human_detected: bool = False
    taggers_run: list[str] = field(default_factory=list)
    taggers_skipped: list[str] = field(default_factory=list)
    taggers_failed: dict[str, str] = field(default_factory=dict)


def _load_config() -> dict:
    """Load tagger configuration from YAML file."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config not found: {CONFIG_PATH}")

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_pipeline_config(
    verbose: bool,
    wd14_model: str = None,
    dinov2_model: str = None,
    wd14_threshold_mode: str = "Fixed",
) -> dict:
    """
    Build configuration dict for TaggingPipeline from YAML config.
    """
    config = _load_config()
    taggers = config.get("taggers", {})

    # Override WD14 model if specified
    if wd14_model and "wd14" in taggers:
        taggers["wd14"]["model"] = wd14_model

    # Override WD14 threshold mode
    if "wd14" in taggers:
        taggers["wd14"]["threshold_mode"] = wd14_threshold_mode

    # Override DINOv2 model if specified
    if dinov2_model and "saliency" in taggers:
        taggers["saliency"]["model"] = dinov2_model

    return {
        "verbose": verbose,
        "taggers": taggers,
    }


class PromptGeneratorEngine:
    """
    Stage 1A Engine - Orchestrates all taggers using existing TaggingPipeline.

    Usage:
        engine = PromptGeneratorEngine()
        result = engine.run(pil_image)
        print(result.prompt)
    """

    def __init__(
        self,
        verbose: bool = False,
        hf_token: str = None,
        wd14_model: str = None,
        dinov2_model: str = None,
        wd14_threshold_mode: str = "Fixed",
    ):
        """
        Initialize engine.

        Args:
            verbose: Enable debug logging
            hf_token: Optional HuggingFace token for model downloads
            wd14_model: WD14 model to use (overrides config)
            dinov2_model: DINOv2 model for saliency (overrides config)
            wd14_threshold_mode: Threshold mode for WD14 (Fixed/MCut/Detailed)
        """
        self.verbose = verbose
        self.hf_token = hf_token
        self.wd14_model = wd14_model
        self.dinov2_model = dinov2_model
        self.wd14_threshold_mode = wd14_threshold_mode

        # Set global HF token if provided
        if hf_token:
            self._set_hf_token(hf_token)

        # Build config and create pipeline
        self._config = _build_pipeline_config(verbose, wd14_model, dinov2_model, wd14_threshold_mode)
        self._pipeline = TaggingPipeline(self._config)

        # Formatter for output
        self._formatter = PromptFormatter()

    def _set_hf_token(self, token: str) -> None:
        """Set HuggingFace token for authenticated downloads."""
        try:
            from huggingface_hub import login
            login(token=token, add_to_git_credential=False)
            print("[ImageToPrompt] HuggingFace token configured")
        except Exception as e:
            print(f"[ImageToPrompt] Warning: Failed to set HF token: {e}")

    def run(self, image: Image.Image, show_confidence: bool = False, progress_callback=None) -> EngineResult:
        """
        Run all applicable taggers on the image.

        Uses TaggingPipeline which automatically:
        - Runs WD14 first for human detection
        - Skips fashion/pose if no human detected
        - Handles errors per-tagger

        Args:
            image: PIL Image to analyze
            show_confidence: Include confidence scores in output
            progress_callback: Optional callable(tagger_name, index, total) for progress updates

        Returns:
            EngineResult with formatted prompt and metadata
        """
        print(f"[ImageToPrompt] Processing image {image.size}")

        # Run the existing pipeline with progress callback
        pipeline_result = self._pipeline.run(image, progress_callback=progress_callback)

        # Format output - grouped by provider
        if show_confidence:
            prompt = self._formatter.format_with_confidence(
                pipeline_result.tagger_results,
                threshold=0.0
            )
        else:
            prompt = self._formatter.format_by_provider(
                pipeline_result.tagger_results,
                threshold=0.0
            )

        # Determine if human was detected by checking if fashion taggers ran
        human_detected = not any(
            t in pipeline_result.skipped_taggers
            for t in ["fashion", "fashion_yolov8", "fashion_clip"]
        )

        # Get list of taggers that actually ran
        taggers_run = list(pipeline_result.tagger_results.keys())

        # Print summary table
        self._print_summary(
            pipeline_result.tagger_results,
            pipeline_result.skipped_taggers,
            pipeline_result.failed_taggers,
            len(pipeline_result.tags_list),
        )

        return EngineResult(
            prompt=prompt,
            tagger_results=pipeline_result.tagger_results,
            human_detected=human_detected,
            taggers_run=taggers_run,
            taggers_skipped=pipeline_result.skipped_taggers,
            taggers_failed=pipeline_result.failed_taggers,
        )

    def _print_summary(
        self,
        tagger_results: dict,
        skipped: list,
        failed: dict,
        total_unique: int,
    ) -> None:
        """Print summary table of tagger results."""
        print("\n" + "=" * 50)
        print("ImageToPrompt Summary")
        print("=" * 50)
        print(f"{'Tagger':<20} {'Status':<10} {'Tags':>6}")
        print("-" * 50)

        # Passed taggers
        for name, result in tagger_results.items():
            tag_count = len(result.tags) if result and result.tags else 0
            print(f"{name:<20} {'PASS':<10} {tag_count:>6}")

        # Skipped taggers
        for name in skipped:
            print(f"{name:<20} {'SKIP':<10} {'-':>6}")

        # Failed taggers
        for name, error in failed.items():
            # Truncate error message
            short_err = error[:25] + "..." if len(error) > 25 else error
            print(f"{name:<20} {'FAIL':<10} {'-':>6}  ({short_err})")

        print("-" * 50)
        print(f"{'Total Unique Tags':<20} {'':<10} {total_unique:>6}")
        print("=" * 50 + "\n")

    def unload(self) -> None:
        """Unload all tagger models to free memory."""
        print("[ImageToPrompt] Unloading all taggers...")
        self._pipeline.unload_all()
