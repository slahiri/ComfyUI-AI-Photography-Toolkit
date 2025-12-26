"""
SID Image Analysis Node - Stage 1: Extract structured metadata from images.

Analyzes images using:
- Multi-model taggers (WD14, JoyTag, NudeNet, Fashion, Saliency, etc.)
- Florence-2 VLM runs multiple passes:
  - All models: detailed_caption, more_detailed_caption
  - PromptGen models add: generate_tags, mixed_caption, analyze, mixed_caption_plus

Outputs:
- image: Pass-through of input image
- metadata: Structured JSON metadata (SID_METADATA type)
- prompt: Human-readable text analysis (for display/preview)
"""

import json
import torch
import numpy as np
from PIL import Image

import comfy.utils
from ..core.platform import cleanup_memory
from ..core.log import log, log_start, log_end, log_error

# WD14 model choices
WD14_MODELS = [
    "wd-eva02-large-tagger-v3",   # Highest accuracy (larger)
    "wd-swinv2-tagger-v3",        # Best accuracy (default)
    "wd-vit-tagger-v3",           # Good balance
    "wd-convnext-tagger-v3",      # Fast inference
    "wd-v1-4-moat-tagger-v2",     # Legacy
    "wd-v1-4-swinv2-tagger-v2",   # Legacy
    "wd-v1-4-convnext-tagger-v2", # Legacy
    "wd-v1-4-vit-tagger-v2",      # Legacy
]

# WD14 threshold modes
WD14_THRESHOLD_MODES = [
    "Fixed",    # Use fixed threshold (0.35)
    "MCut",     # Auto-detect optimal threshold (more detailed tags)
    "Detailed", # Lower threshold (0.25) for maximum detail
]

# DINOv2 model choices for saliency
DINOV2_MODELS = [
    "dinov2-base",   # Default, good balance
    "dinov2-small",  # Fastest, least VRAM
    "dinov2-large",  # Better accuracy
    "dinov2-giant",  # Best accuracy, high VRAM
]

# Florence-2 model choices (always runs - provides essential context for synthesis)
FLORENCE_MODELS = [
    # PromptGen models (best for prompt generation)
    "MiaoshouAI/Florence-2-large-PromptGen-v2.0",
    "MiaoshouAI/Florence-2-base-PromptGen-v2.0",
    "MiaoshouAI/Florence-2-large-PromptGen-v1.5",
    "MiaoshouAI/Florence-2-base-PromptGen-v1.5",
    # CogFlorence (enhanced variants)
    "thwri/CogFlorence-2.2-Large",
    "thwri/CogFlorence-2.1-Large",
    # Microsoft base models
    "microsoft/Florence-2-large-ft",
    "microsoft/Florence-2-base-ft",
    "microsoft/Florence-2-large",
    "microsoft/Florence-2-base",
]

# Florence-2 task prompts
# Standard tasks (all models)
FLORENCE_TASK_PROMPTS = {
    "caption": "<CAPTION>",
    "detailed_caption": "<DETAILED_CAPTION>",
    "more_detailed_caption": "<MORE_DETAILED_CAPTION>",
}

# PromptGen-specific tasks (only for MiaoshouAI PromptGen models)
FLORENCE_PROMPTGEN_TASKS = {
    "generate_tags": "<GENERATE_TAGS>",
    "mixed_caption": "<MIXED_CAPTION>",
    "analyze": "<ANALYZE>",
    "mixed_caption_plus": "<MIXED_CAPTION_PLUS>",
}


class SID_ImageAnalysis:
    """
    Stage 1: Analyze image and extract structured metadata.

    Runs multiple taggers and Florence-2 VLM (3 passes) to extract
    comprehensive image information.
    """

    CATEGORY = "SID Nodes"
    RETURN_TYPES = ("IMAGE", "SID_METADATA", "STRING")
    RETURN_NAMES = ("image", "metadata", "prompt")
    FUNCTION = "analyze"
    OUTPUT_NODE = False

    # Color styling
    COLOR = "#4B5320"
    BGCOLOR = "#3a4219"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Input image to analyze"
                }),
                "wd14_model": (WD14_MODELS, {
                    "default": "wd-swinv2-tagger-v3",
                    "tooltip": "WD14 tagger model"
                }),
                "wd14_threshold_mode": (WD14_THRESHOLD_MODES, {
                    "default": "MCut",
                    "tooltip": "Fixed: use 0.35 threshold | MCut: auto-detect optimal | Detailed: lower threshold (0.25) for more tags"
                }),
                "dinov2_model": (DINOV2_MODELS, {
                    "default": "dinov2-base",
                    "tooltip": "DINOv2 saliency model"
                }),
                "florence_model": (FLORENCE_MODELS, {
                    "default": "MiaoshouAI/Florence-2-large-PromptGen-v2.0",
                    "tooltip": "Florence-2 VLM (runs 3x: caption, detailed description, tag extraction)"
                }),
            },
            "optional": {
                "release_vram": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Release VRAM after analysis"
                }),
                "verbose": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable detailed logging"
                }),
                "hf_token": ("STRING", {
                    "default": "",
                    "tooltip": "HuggingFace token (optional)"
                }),
            },
        }

    def __init__(self):
        self._engine = None
        self._current_wd14_model = None
        self._current_dinov2_model = None
        self._current_threshold_mode = None
        # Florence cache
        self._florence_model = None
        self._florence_processor = None
        self._current_florence_model = None
        self._florence_device = None
        self._florence_offload_device = None
        self._florence_dtype = None

    def _get_engine(self, verbose: bool, hf_token: str, wd14_model: str, dinov2_model: str, threshold_mode: str = "Fixed"):
        """Get or create tagging engine."""
        from ..core.prompt_generator import PromptGeneratorEngine

        if (self._engine is None or
            self._current_wd14_model != wd14_model or
            self._current_dinov2_model != dinov2_model or
            self._current_threshold_mode != threshold_mode):
            self._engine = PromptGeneratorEngine(
                verbose=verbose,
                hf_token=hf_token if hf_token else None,
                wd14_model=wd14_model,
                dinov2_model=dinov2_model,
                wd14_threshold_mode=threshold_mode,
            )
            self._current_wd14_model = wd14_model
            self._current_dinov2_model = dinov2_model
            self._current_threshold_mode = threshold_mode
        return self._engine

    def _load_florence(self, model_id: str, verbose: bool):
        """Load Florence-2 model."""
        try:
            from ..core.models.florence import (
                _load_florence_model,
                _register_processor_from_model,
                _load_kijai_florence_class,
                _get_kijai_path,
            )
            from ..core.download import download_model
            from transformers import AutoProcessor
            import comfy.model_management as mm

            model_config = {"model_id": model_id, "revision": "main", "trust_remote_code": True}
            local_path = download_model(model_id, model_config)

            device = mm.get_torch_device()
            offload_device = mm.unet_offload_device()
            dtype = torch.float16

            start = log_start("ImageAnalysis", f"Loading Florence-2: {model_id}")
            self._florence_model = _load_florence_model(local_path, dtype, trust_remote=True)
            self._florence_model = self._florence_model.to(offload_device)

            kijai_path = _get_kijai_path()
            if kijai_path.exists():
                try:
                    _load_kijai_florence_class()
                except Exception:
                    pass
                _register_processor_from_model(local_path)

            self._florence_processor = AutoProcessor.from_pretrained(local_path, trust_remote_code=True)
            self._florence_device = device
            self._florence_offload_device = offload_device
            self._florence_dtype = dtype
            self._current_florence_model = model_id

            log_end("ImageAnalysis", "Florence-2 loaded", start)
        except Exception as e:
            log_error("ImageAnalysis", f"Failed to load Florence-2: {e}")
            import traceback
            traceback.print_exc()
            self._florence_model = None

    def _is_promptgen_model(self, model_id: str) -> bool:
        """Check if model supports PromptGen-specific tasks."""
        return "PromptGen" in model_id

    def _run_florence_task(self, image: Image.Image, task: str, verbose: bool) -> str:
        """Run a single Florence-2 task with greedy decoding (max 1024 tokens)."""
        if self._florence_model is None:
            return ""

        # Check standard tasks first, then PromptGen tasks
        task_prompt = FLORENCE_TASK_PROMPTS.get(task)
        if task_prompt is None:
            task_prompt = FLORENCE_PROMPTGEN_TASKS.get(task, "<MORE_DETAILED_CAPTION>")

        if verbose:
            log("ImageAnalysis", f"Florence task: {task} -> {task_prompt}")

        try:
            self._florence_model.to(self._florence_device)

            inputs = self._florence_processor(
                text=task_prompt, images=image, return_tensors="pt", do_rescale=False,
            ).to(self._florence_dtype).to(self._florence_device)

            # Use greedy decoding for consistent, deterministic output
            with torch.no_grad():
                generated_ids = self._florence_model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=1024,
                    num_beams=1,
                    do_sample=False,
                    early_stopping=False,
                    use_cache=False,
                )

            self._florence_model.to(self._florence_offload_device)

            generated_text = self._florence_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            if verbose:
                log("ImageAnalysis", f"Florence raw: {generated_text[:100]}...")

            result = self._florence_processor.post_process_generation(
                generated_text, task=task_prompt, image_size=(image.width, image.height)
            )

            # Get result - try task_prompt key first, then any value
            caption = result.get(task_prompt, "")
            if not caption and result:
                caption = list(result.values())[0] if result.values() else ""

            return caption

        except Exception as e:
            log_error("ImageAnalysis", f"Florence {task} error: {e}")
            import traceback
            traceback.print_exc()
            return ""

    def _run_florence_custom(self, image: Image.Image, custom_prompt: str, verbose: bool) -> str:
        """Run Florence-2 with a custom prompt for tag extraction."""
        if self._florence_model is None:
            return ""

        if verbose:
            log("ImageAnalysis", f"Florence custom prompt: {custom_prompt[:50]}...")

        try:
            self._florence_model.to(self._florence_device)

            inputs = self._florence_processor(
                text=custom_prompt, images=image, return_tensors="pt", do_rescale=False,
            ).to(self._florence_dtype).to(self._florence_device)

            with torch.no_grad():
                generated_ids = self._florence_model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=1024,
                    num_beams=1,
                    do_sample=False,
                    early_stopping=False,
                    use_cache=False,
                )

            self._florence_model.to(self._florence_offload_device)

            generated_text = self._florence_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            if verbose:
                log("ImageAnalysis", f"Florence custom raw: {generated_text[:100]}...")

            return generated_text.strip()

        except Exception as e:
            log_error("ImageAnalysis", f"Florence custom error: {e}")
            import traceback
            traceback.print_exc()
            return ""

    def _tensor_to_pil(self, image_tensor: torch.Tensor) -> Image.Image:
        """Convert ComfyUI tensor to PIL Image."""
        if len(image_tensor.shape) == 4:
            image_tensor = image_tensor[0]
        img_np = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(img_np, mode="RGB")

    def _release_models(self):
        """Release models from VRAM."""
        if self._florence_model is not None:
            start = log_start("ImageAnalysis", "Releasing Florence model")
            del self._florence_model
            del self._florence_processor
            self._florence_model = None
            self._florence_processor = None
            self._current_florence_model = None
            log_end("ImageAnalysis", "Florence released", start)

        cleanup_memory(aggressive=True)

    def _build_analysis_text(self, metadata: dict) -> str:
        """Build human-readable analysis text from metadata."""
        lines = []

        # Florence outputs (most valuable for prompt generation)
        florence_outputs = [
            ("florence_mixed_caption_plus", "MIXED CAPTION PLUS"),
            ("florence_mixed_caption", "MIXED CAPTION"),
            ("florence_analyze", "ANALYZE"),
            ("florence_generate_tags", "GENERATED TAGS"),
            ("florence_description", "DETAILED DESCRIPTION"),
            ("florence_caption", "CAPTION"),
        ]

        for key, title in florence_outputs:
            if key in metadata and metadata[key]:
                lines.append(f"=== {title} ===")
                lines.append(metadata[key])
                lines.append("")

        # Tagger summaries
        tagger_names = {
            "wd14": "WD14 Tags",
            "joytag": "JoyTag",
            "fashion_yolov8": "Fashion Detection",
            "fashion_yolos": "Fashion Items",
            "fashion_segformer": "Garment Segments",
            "fashion_clip": "Fashion Style",
            "nudenet": "Body Areas",
            "pose": "Pose",
            "saliency": "Focus Areas",
            "composition": "Composition",
        }

        for key, display_name in tagger_names.items():
            if key in metadata and isinstance(metadata[key], dict):
                # Sort by confidence
                sorted_tags = sorted(
                    metadata[key].items(),
                    key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0,
                    reverse=True
                )
                if sorted_tags:
                    lines.append(f"=== {display_name.upper()} ===")
                    tag_strs = [f"{tag} ({conf:.0%})" if isinstance(conf, float) else f"{tag} ({conf})"
                                for tag, conf in sorted_tags[:20]]
                    lines.append(", ".join(tag_strs))
                    lines.append("")

        return "\n".join(lines)

    def analyze(
        self,
        image: torch.Tensor,
        wd14_model: str = "wd-swinv2-tagger-v3",
        wd14_threshold_mode: str = "MCut",
        dinov2_model: str = "dinov2-base",
        florence_model: str = "MiaoshouAI/Florence-2-large-PromptGen-v2.0",
        release_vram: bool = True,
        verbose: bool = False,
        hf_token: str = "",
        # Legacy parameters (ignored, for backwards compatibility)
        **kwargs,
    ) -> tuple[torch.Tensor, str, str]:
        """
        Analyze image and return structured metadata.

        Returns:
            - image: Pass-through input image
            - metadata: JSON metadata string (SID_METADATA type)
            - prompt: Human-readable analysis text (for preview/display)

        Runs:
        1. Multi-model taggers (WD14, JoyTag, etc.)
        2. Florence-2 passes:
           - All models: detailed_caption, more_detailed_caption
           - PromptGen models add: generate_tags, mixed_caption, analyze, mixed_caption_plus
        """
        pil_image = self._tensor_to_pil(image)

        # Calculate total steps for progress bar
        # Steps: 1. Taggers, 2. Load Florence, 3-8. Florence passes (up to 6 for PromptGen models)
        # PromptGen models: 1 + 1 + 6 = 8 passes
        # Other models: 1 + 1 + 2 = 4 passes (but we allocate 8 and skip)
        total_steps = 8  # Taggers + Florence load + up to 6 passes

        pbar = comfy.utils.ProgressBar(total_steps)

        # Initialize simple 2-layer metadata
        metadata = {
            "image_info": {
                "width": pil_image.width,
                "height": pil_image.height,
            },
        }

        # Run tagging engine with threshold mode
        log("ImageAnalysis", f"Analyzing image ({pil_image.width}x{pil_image.height})")
        log("ImageAnalysis", f"WD14 threshold mode: {wd14_threshold_mode}")
        taggers_start = log_start("ImageAnalysis", "Running taggers")
        try:
            engine = self._get_engine(verbose, hf_token, wd14_model, dinov2_model, wd14_threshold_mode)
            result = engine.run(pil_image, show_confidence=True)
            log_end("ImageAnalysis", "Taggers complete", taggers_start, f"ran: {getattr(result, 'taggers_run', [])}")
            if getattr(result, 'taggers_failed', {}):
                log("ImageAnalysis", f"Taggers failed: {getattr(result, 'taggers_failed', {})}")
        except Exception as e:
            log_error("ImageAnalysis", f"Engine error: {e}")
            import traceback
            traceback.print_exc()
            result = None
        pbar.update(1)  # Taggers complete

        # Parse tagger results - simple format: { "tagger_name": { "tag": confidence } }
        if result and hasattr(result, 'tagger_results') and result.tagger_results:
            log("ImageAnalysis", f"Parsing {len(result.tagger_results)} tagger results")
            for tagger_name, tagger_result in result.tagger_results.items():
                if tagger_result and tagger_result.tags:
                    tag_count = len(tagger_result.tags)
                    metadata[tagger_name] = {
                        tag.text: round(tag.confidence, 3)
                        for tag in tagger_result.tags
                    }
                    log("ImageAnalysis", f"  {tagger_name}: {tag_count} tags")
                else:
                    log("ImageAnalysis", f"  {tagger_name}: no tags")
        else:
            log("ImageAnalysis", "No tagger results to parse")

        # Run Florence-2 passes - always runs for best synthesis context
        # Load Florence once
        if self._florence_model is None or self._current_florence_model != florence_model:
            self._load_florence(florence_model, verbose)
        pbar.update(1)  # Florence loaded

        is_promptgen = self._is_promptgen_model(florence_model)

        if self._florence_model is not None:
            # Pass 1: Detailed caption (all models)
            p1_start = log_start("ImageAnalysis", "Florence Pass 1: Detailed Caption")
            caption_result = self._run_florence_task(pil_image, "detailed_caption", verbose)
            if caption_result:
                metadata["florence_caption"] = caption_result
                log_end("ImageAnalysis", "Florence Pass 1", p1_start, f"{len(caption_result)} chars")
            pbar.update(1)

            # Pass 2: More detailed description (all models)
            p2_start = log_start("ImageAnalysis", "Florence Pass 2: More Detailed Caption")
            desc_result = self._run_florence_task(pil_image, "more_detailed_caption", verbose)
            if desc_result:
                metadata["florence_description"] = desc_result
                log_end("ImageAnalysis", "Florence Pass 2", p2_start, f"{len(desc_result)} chars")
            pbar.update(1)

            # PromptGen-specific passes (only for MiaoshouAI PromptGen models)
            if is_promptgen:
                # Pass 3: Generate Tags
                p3_start = log_start("ImageAnalysis", "Florence Pass 3: Generate Tags")
                tags_result = self._run_florence_task(pil_image, "generate_tags", verbose)
                if tags_result:
                    metadata["florence_generate_tags"] = tags_result
                    log_end("ImageAnalysis", "Florence Pass 3", p3_start, f"{len(tags_result)} chars")
                pbar.update(1)

                # Pass 4: Mixed Caption
                p4_start = log_start("ImageAnalysis", "Florence Pass 4: Mixed Caption")
                mixed_result = self._run_florence_task(pil_image, "mixed_caption", verbose)
                if mixed_result:
                    metadata["florence_mixed_caption"] = mixed_result
                    log_end("ImageAnalysis", "Florence Pass 4", p4_start, f"{len(mixed_result)} chars")
                pbar.update(1)

                # Pass 5: Analyze
                p5_start = log_start("ImageAnalysis", "Florence Pass 5: Analyze")
                analyze_result = self._run_florence_task(pil_image, "analyze", verbose)
                if analyze_result:
                    metadata["florence_analyze"] = analyze_result
                    log_end("ImageAnalysis", "Florence Pass 5", p5_start, f"{len(analyze_result)} chars")
                pbar.update(1)

                # Pass 6: Mixed Caption Plus
                p6_start = log_start("ImageAnalysis", "Florence Pass 6: Mixed Caption Plus")
                mixed_plus_result = self._run_florence_task(pil_image, "mixed_caption_plus", verbose)
                if mixed_plus_result:
                    metadata["florence_mixed_caption_plus"] = mixed_plus_result
                    log_end("ImageAnalysis", "Florence Pass 6", p6_start, f"{len(mixed_plus_result)} chars")
                pbar.update(1)
            else:
                # Non-PromptGen model: skip PromptGen tasks
                log("ImageAnalysis", "Skipping PromptGen tasks (not a PromptGen model)")
                pbar.update(4)  # Skip 4 PromptGen passes
        else:
            # Florence failed to load, update remaining steps
            pbar.update(6 if is_promptgen else 2)

        # Release VRAM if requested
        if release_vram:
            self._release_models()

        # Build outputs
        # 1. Metadata JSON (for SID_METADATA type)
        metadata_json = json.dumps(metadata, indent=2, ensure_ascii=False)

        # 2. Human-readable prompt text (for preview/display)
        prompt_text = self._build_analysis_text(metadata)

        if verbose:
            log("ImageAnalysis", f"Metadata: {len(metadata_json)} chars")
            log("ImageAnalysis", f"Prompt: {len(prompt_text)} chars")

        # Cleanup
        del pil_image
        cleanup_memory()

        return (image, metadata_json, prompt_text)

    def unload(self) -> None:
        """Unload all models and free memory."""
        start = log_start("ImageAnalysis", "Unloading all models")

        if self._engine is not None:
            self._engine.unload()
            self._engine = None

        self._release_models()

        log_end("ImageAnalysis", "All models unloaded", start)
