"""
SID Image Analysis Node - Stage 1: Extract structured metadata from images.

Analyzes images using:
- Multi-model taggers (WD14, JoyTag, NudeNet, Fashion, Saliency, etc.)
- Florence-2 VLM runs up to 8 passes:
  - Pass 1-2: detailed_caption, more_detailed_caption (all models)
  - Pass 3-6: generate_tags, mixed_caption, analyze, mixed_caption_plus (PromptGen only)
  - Pass 7: object_detection - detect objects with bounding boxes
  - Pass 8: dense_region_caption - captions for each detected region
- BBox analyzer for geometric composition insights
- Canonical structurer for scene detection and category classification

Outputs:
- metadata: Structured JSON metadata (SID_METADATA type)
"""

import json
import torch
import numpy as np
import uuid
from PIL import Image

import comfy.utils
from ..core.platform import cleanup_memory
from ..core.log import log, log_start, log_end, log_error
from ..core.compose import CanonicalStructurer
from ..core.taggers import BBoxAnalyzer

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

# Florence pass options
FLORENCE_PASSES = [
    "1 Pass (Caption)",           # Just detailed caption
    "2 Pass (Caption+Describe)",  # Caption + More detailed description
    "3 Pass (Tags+Analyze)",      # PromptGen: generate_tags, mixed_caption, analyze, mixed_caption_plus
    "4 Pass (+Detection)",        # + Object detection
    "5 Pass (All)",               # + Dense region captions (maximum detail)
]

# Florence-2 task prompts
# Standard tasks (all models)
FLORENCE_TASK_PROMPTS = {
    "caption": "<CAPTION>",
    "detailed_caption": "<DETAILED_CAPTION>",
    "more_detailed_caption": "<MORE_DETAILED_CAPTION>",
    "object_detection": "<OD>",
    "dense_region_caption": "<DENSE_REGION_CAPTION>",
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
    RETURN_TYPES = ("SID_METADATA", "STRING")
    RETURN_NAMES = ("metadata", "prompt")
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
                "analysis_enabled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable full analysis. If disabled, only Florence-2 caption is generated."
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
                    "tooltip": "Florence-2 VLM model"
                }),
                "florence_passes": (FLORENCE_PASSES, {
                    "default": "5 Pass (All)",
                    "tooltip": "1: Caption | 2: +Description | 3: +Tags/Analyze | 4: +Object Detection | 5: +Region Captions"
                }),
                "tag_filter": ("FLOAT", {
                    "default": 0.25,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Minimum confidence threshold for canonical categorization. Tags below this go to below_threshold."
                }),
            },
            "optional": {
                "verbose": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable detailed logging with quality output"
                }),
                "release_vram": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Release VRAM after analysis"
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
        # Canonical structurer for scene detection
        self._structurer = CanonicalStructurer()
        # BBox analyzer for geometric insights
        self._bbox_analyzer = BBoxAnalyzer()

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

    def _parse_detection_result(self, result: dict, verbose: bool = False) -> dict:
        """Parse Florence-2 detection result into a structured format.

        Input format from Florence:
            {'<OD>': {'bboxes': [[x1,y1,x2,y2], ...], 'labels': ['label1', ...]}}
        or:
            {'<DENSE_REGION_CAPTION>': {'bboxes': [...], 'labels': [...]}}

        Output format:
            {'label1': {'count': 2, 'bboxes': [...]}, 'label2': {...}}
        """
        parsed = {}

        try:
            # Handle nested dict from post_process_generation
            if isinstance(result, dict):
                # Get the inner dict (keyed by task prompt)
                inner = None
                for key in result:
                    if isinstance(result[key], dict) and 'bboxes' in result[key]:
                        inner = result[key]
                        break

                if inner is None:
                    # Maybe it's already in the right format
                    if 'bboxes' in result and 'labels' in result:
                        inner = result

                if inner and 'bboxes' in inner and 'labels' in inner:
                    bboxes = inner['bboxes']
                    labels = inner['labels']

                    # Group by label
                    for i, label in enumerate(labels):
                        label_clean = label.strip().lower()
                        if label_clean not in parsed:
                            parsed[label_clean] = {'count': 0, 'bboxes': []}
                        parsed[label_clean]['count'] += 1
                        if i < len(bboxes):
                            parsed[label_clean]['bboxes'].append(bboxes[i])

                    if verbose:
                        log("ImageAnalysis", f"  Parsed {len(labels)} detections into {len(parsed)} unique labels")

        except Exception as e:
            if verbose:
                log_error("ImageAnalysis", f"Failed to parse detection result: {e}")

        return parsed

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

    def analyze(
        self,
        image: torch.Tensor,
        analysis_enabled: bool = True,
        wd14_model: str = "wd-swinv2-tagger-v3",
        wd14_threshold_mode: str = "MCut",
        dinov2_model: str = "dinov2-base",
        florence_model: str = "MiaoshouAI/Florence-2-large-PromptGen-v2.0",
        florence_passes: str = "3 Pass (All)",
        tag_filter: float = 0.25,
        verbose: bool = False,
        release_vram: bool = True,
        hf_token: str = "",
        # Legacy parameters (ignored, for backwards compatibility)
        **kwargs,
    ) -> tuple[str, str]:
        """
        Analyze image and return structured metadata.

        Returns:
            - metadata: JSON metadata string (SID_METADATA type)
            - prompt: Florence-2 description text

        Runs:
        1. Multi-model taggers (WD14, JoyTag, etc.) - if analysis_enabled
        2. Florence-2 passes (based on florence_passes setting):
           - Pass 1: detailed_caption
           - Pass 2: more_detailed_caption
           - Pass 3-6 (PromptGen only): generate_tags, mixed_caption, analyze, mixed_caption_plus
           - Pass 7: object_detection (OD) - counts objects, detects props
           - Pass 8: dense_region_caption - captions for background elements
        """
        pil_image = self._tensor_to_pil(image)

        # Generate session ID
        session_id = str(uuid.uuid4())[:8]

        # Initialize simple 2-layer metadata
        metadata = {
            "session_id": session_id,
            "tag_filter": tag_filter,
            "image_info": {
                "width": pil_image.width,
                "height": pil_image.height,
            },
        }

        # If analysis disabled, only run Florence-2 for description
        if not analysis_enabled:
            log("ImageAnalysis", f"Analysis disabled - running Florence-2 only")

            pbar = comfy.utils.ProgressBar(2)  # Load + 1 pass

            # Load Florence
            if self._florence_model is None or self._current_florence_model != florence_model:
                self._load_florence(florence_model, verbose)
            pbar.update(1)

            prompt_output = ""
            if self._florence_model is not None:
                # Run most detailed caption
                p1_start = log_start("ImageAnalysis", "Florence: More Detailed Caption")
                desc_result = self._run_florence_task(pil_image, "more_detailed_caption", verbose)
                if desc_result:
                    metadata["florence_description"] = desc_result
                    prompt_output = desc_result
                    log_end("ImageAnalysis", "Florence caption", p1_start, f"{len(desc_result)} chars")
            pbar.update(1)

            # Release VRAM if requested
            if release_vram:
                self._release_models()

            # Build metadata JSON output
            metadata["analysis_enabled"] = False
            metadata_json = json.dumps(metadata, indent=2, ensure_ascii=False)

            log("ImageAnalysis", f"Quick mode complete: {len(prompt_output)} chars prompt")

            # Cleanup
            del pil_image
            cleanup_memory()

            return (metadata_json, prompt_output)

        # Full analysis mode
        metadata["analysis_enabled"] = True

        # Calculate total steps for progress bar
        # Count enabled taggers from config + Florence load + up to 8 Florence passes
        from pathlib import Path
        import yaml
        try:
            config_path = Path(__file__).parent.parent / "core" / "config" / "prompt_generator.yaml"
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            tagger_count = len([k for k, v in config.get("taggers", {}).items() if v.get("enabled", True)])
        except:
            tagger_count = 22  # Default estimate
        florence_steps = 9  # Load + 8 passes
        total_steps = tagger_count + florence_steps

        pbar = comfy.utils.ProgressBar(total_steps)
        current_step = [0]  # Use list for mutable closure

        def tagger_progress(tagger_name, index, total):
            """Callback for tagger progress updates."""
            current_step[0] = index
            pbar.update_absolute(index, total_steps)
            if verbose:
                log("ImageAnalysis", f"  [{index}/{total}] Running {tagger_name}")

        # Run tagging engine with threshold mode
        log("ImageAnalysis", f"Analyzing image ({pil_image.width}x{pil_image.height})")
        log("ImageAnalysis", f"WD14 threshold mode: {wd14_threshold_mode}")
        taggers_start = log_start("ImageAnalysis", "Running taggers")
        try:
            engine = self._get_engine(verbose, hf_token, wd14_model, dinov2_model, wd14_threshold_mode)
            result = engine.run(pil_image, show_confidence=True, progress_callback=tagger_progress)
            log_end("ImageAnalysis", "Taggers complete", taggers_start, f"ran: {getattr(result, 'taggers_run', [])}")
            if getattr(result, 'taggers_failed', {}):
                log("ImageAnalysis", f"Taggers failed: {getattr(result, 'taggers_failed', {})}")
        except Exception as e:
            log_error("ImageAnalysis", f"Engine error: {e}")
            import traceback
            traceback.print_exc()
            result = None

        # Update progress to end of tagger phase
        tagger_end_step = tagger_count
        pbar.update_absolute(tagger_end_step, total_steps)

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

        # Determine number of passes from selection
        if "5 Pass" in florence_passes:
            num_passes = 5
        elif "4 Pass" in florence_passes:
            num_passes = 4
        elif "3 Pass" in florence_passes:
            num_passes = 3
        elif "2 Pass" in florence_passes:
            num_passes = 2
        else:
            num_passes = 1
        is_promptgen = self._is_promptgen_model(florence_model)

        if verbose:
            log("ImageAnalysis", f"Florence passes: {num_passes} (PromptGen: {is_promptgen})")

        # Run Florence-2 passes - always runs for best synthesis context
        # Load Florence once
        florence_step = tagger_end_step  # Start Florence steps after taggers
        if self._florence_model is None or self._current_florence_model != florence_model:
            self._load_florence(florence_model, verbose)
        florence_step += 1
        pbar.update_absolute(florence_step, total_steps)  # Florence loaded

        if self._florence_model is not None:
            # Pass 1: Detailed caption (always runs)
            p1_start = log_start("ImageAnalysis", "Florence Pass 1: Detailed Caption")
            caption_result = self._run_florence_task(pil_image, "detailed_caption", verbose)
            if caption_result:
                metadata["florence_caption"] = caption_result
                log_end("ImageAnalysis", "Florence Pass 1", p1_start, f"{len(caption_result)} chars")
                if verbose:
                    log("ImageAnalysis", f"  Caption preview: {caption_result[:100]}...")
            florence_step += 1
            pbar.update_absolute(florence_step, total_steps)

            # Pass 2: More detailed description (if num_passes >= 2)
            if num_passes >= 2:
                p2_start = log_start("ImageAnalysis", "Florence Pass 2: More Detailed Caption")
                desc_result = self._run_florence_task(pil_image, "more_detailed_caption", verbose)
                if desc_result:
                    metadata["florence_description"] = desc_result
                    log_end("ImageAnalysis", "Florence Pass 2", p2_start, f"{len(desc_result)} chars")
                    if verbose:
                        log("ImageAnalysis", f"  Description preview: {desc_result[:100]}...")
            florence_step += 1
            pbar.update_absolute(florence_step, total_steps)

            # Pass 3+: PromptGen-specific passes (if num_passes >= 3 and is PromptGen model)
            if num_passes >= 3 and is_promptgen:
                # Pass 3: Generate Tags
                p3_start = log_start("ImageAnalysis", "Florence Pass 3: Generate Tags")
                tags_result = self._run_florence_task(pil_image, "generate_tags", verbose)
                if tags_result:
                    metadata["florence_generate_tags"] = tags_result
                    log_end("ImageAnalysis", "Florence Pass 3", p3_start, f"{len(tags_result)} chars")
                florence_step += 1
                pbar.update_absolute(florence_step, total_steps)

                # Pass 4: Mixed Caption
                p4_start = log_start("ImageAnalysis", "Florence Pass 4: Mixed Caption")
                mixed_result = self._run_florence_task(pil_image, "mixed_caption", verbose)
                if mixed_result:
                    metadata["florence_mixed_caption"] = mixed_result
                    log_end("ImageAnalysis", "Florence Pass 4", p4_start, f"{len(mixed_result)} chars")
                florence_step += 1
                pbar.update_absolute(florence_step, total_steps)

                # Pass 5: Analyze
                p5_start = log_start("ImageAnalysis", "Florence Pass 5: Analyze")
                analyze_result = self._run_florence_task(pil_image, "analyze", verbose)
                if analyze_result:
                    metadata["florence_analyze"] = analyze_result
                    log_end("ImageAnalysis", "Florence Pass 5", p5_start, f"{len(analyze_result)} chars")
                    if verbose:
                        # Show parsed analyze data
                        if isinstance(analyze_result, dict):
                            log("ImageAnalysis", f"  Analyze keys: {list(analyze_result.keys())}")
                florence_step += 1
                pbar.update_absolute(florence_step, total_steps)

                # Pass 6: Mixed Caption Plus
                p6_start = log_start("ImageAnalysis", "Florence Pass 6: Mixed Caption Plus")
                mixed_plus_result = self._run_florence_task(pil_image, "mixed_caption_plus", verbose)
                if mixed_plus_result:
                    metadata["florence_mixed_caption_plus"] = mixed_plus_result
                    log_end("ImageAnalysis", "Florence Pass 6", p6_start, f"{len(mixed_plus_result)} chars")
                florence_step += 1
                pbar.update_absolute(florence_step, total_steps)
            elif num_passes >= 3:
                # Non-PromptGen model with 3 passes: skip PromptGen tasks
                log("ImageAnalysis", "Skipping PromptGen tasks (not a PromptGen model)")
                florence_step += 4  # Skip 4 PromptGen passes
                pbar.update_absolute(florence_step, total_steps)
            else:
                # Less than 3 passes: skip all PromptGen passes
                florence_step += 4
                pbar.update_absolute(florence_step, total_steps)

            # Pass 7: Object Detection (num_passes >= 4)
            if num_passes >= 4:
                p7_start = log_start("ImageAnalysis", "Florence Pass 7: Object Detection")
                od_result = self._run_florence_task(pil_image, "object_detection", verbose)
                if od_result:
                    # Parse OD result - format: {'bboxes': [...], 'labels': [...]}
                    metadata["florence_objects"] = self._parse_detection_result(od_result, verbose)
                    log_end("ImageAnalysis", "Florence Pass 7", p7_start, f"{len(metadata.get('florence_objects', {}))} objects")
            florence_step += 1
            pbar.update_absolute(florence_step, total_steps)

            # Pass 8: Dense Region Caption (num_passes >= 5)
            if num_passes >= 5:
                p8_start = log_start("ImageAnalysis", "Florence Pass 8: Dense Region Caption")
                drc_result = self._run_florence_task(pil_image, "dense_region_caption", verbose)
                if drc_result:
                    # Parse DRC result - format: {'bboxes': [...], 'labels': [...]}
                    metadata["florence_region_captions"] = self._parse_detection_result(drc_result, verbose)
                    log_end("ImageAnalysis", "Florence Pass 8", p8_start, f"{len(metadata.get('florence_region_captions', {}))} regions")
            florence_step += 1
            pbar.update_absolute(florence_step, total_steps)
        else:
            # Florence failed to load, update remaining steps
            florence_step += 8  # Skip all 8 Florence passes
            pbar.update_absolute(florence_step, total_steps)

        # Run bbox analyzer for geometric insights (if we have detection data)
        if "florence_objects" in metadata and metadata["florence_objects"]:
            try:
                bbox_start = log_start("ImageAnalysis", "Running bbox analysis")
                bbox_result = self._bbox_analyzer.analyze(
                    florence_objects=metadata["florence_objects"],
                    florence_regions=metadata.get("florence_region_captions"),
                    image_width=pil_image.width,
                    image_height=pil_image.height,
                )
                metadata["bbox_analysis"] = self._bbox_analyzer.to_dict(bbox_result)
                log_end("ImageAnalysis", "BBox analysis", bbox_start,
                       f"shot={bbox_result.shot_type}, subjects={bbox_result.subject_count}")

                # Print bbox summary line
                rot = f"{bbox_result.rule_of_thirds_score:.0%}" if bbox_result.rule_of_thirds_score else "-"
                print(f"{'bbox_analysis':<20} {'PASS':<10} shot={bbox_result.shot_type}, RoT={rot}, subjects={bbox_result.subject_count}")
            except Exception as e:
                log_error("ImageAnalysis", f"BBox analyzer failed: {e}")
                import traceback
                traceback.print_exc()
                metadata["bbox_analysis"] = {"error": str(e)}
                print(f"{'bbox_analysis':<20} {'FAIL':<10} {str(e)[:30]}")

        # Run canonical structurer for scene detection and category classification
        try:
            canonical_start = log_start("ImageAnalysis", "Running canonical structurer")
            canonical_structure = self._structurer.structure(metadata, threshold=tag_filter)
            metadata["canonical"] = self._structurer.to_json(canonical_structure)
            log_end("ImageAnalysis", "Canonical structurer", canonical_start,
                   f"scene={canonical_structure.scene_type.primary.value}")

            # Print canonical summary line
            primary = canonical_structure.scene_type.primary.value
            conf = f"{canonical_structure.scene_type.primary_confidence:.0%}"
            secondary = canonical_structure.scene_type.secondary.value if canonical_structure.scene_type.secondary else "-"
            print(f"{'canonical':<20} {'PASS':<10} scene={primary} ({conf}), secondary={secondary}, filter={tag_filter}")

            # Print categorization summary
            canonical_json = metadata["canonical"]
            print("-" * 50)
            print(f"Categorization Summary (threshold >= {tag_filter})")
            print("-" * 50)
            categories = ["subject", "scene", "composition", "lighting", "style", "constraints", "uncategorized", "below_threshold"]
            total_categorized = 0
            for cat in categories:
                if cat in canonical_json:
                    count = len(canonical_json[cat])
                    print(f"{cat:<20} {count:>6} tags")
                    total_categorized += count
            print("-" * 50)
            print(f"{'Total':<20} {total_categorized:>6} tags")
            print("=" * 50)
        except Exception as e:
            log_error("ImageAnalysis", f"Canonical structurer failed: {e}")
            import traceback
            traceback.print_exc()
            metadata["canonical"] = {"error": str(e)}
            print(f"{'canonical':<20} {'FAIL':<10} {str(e)[:30]}")

        # Verbose quality summary
        if verbose:
            log("ImageAnalysis", "=== Quality Summary ===")
            tag_count = sum(len(v) for k, v in metadata.items() if isinstance(v, dict) and k not in ["image_info", "canonical", "bbox_analysis"])
            florence_count = sum(1 for k in metadata.keys() if k.startswith("florence_"))
            log("ImageAnalysis", f"  Total tags: {tag_count}")
            log("ImageAnalysis", f"  Florence outputs: {florence_count}")
            if "bbox_analysis" in metadata and not metadata["bbox_analysis"].get("error"):
                bbox = metadata["bbox_analysis"]
                log("ImageAnalysis", f"  Shot type: {bbox.get('shot_type')} ({bbox.get('shot_type_confidence', 0):.0%})")
                if "composition" in bbox:
                    comp = bbox["composition"]
                    log("ImageAnalysis", f"  Composition: RoT={comp.get('rule_of_thirds', 0):.0%}, Golden={comp.get('golden_ratio', 0):.0%}")
            if "canonical" in metadata and "scene_detection" in metadata.get("canonical", {}):
                scene = metadata["canonical"]["scene_detection"]
                log("ImageAnalysis", f"  Scene: {scene.get('primary')} ({scene.get('primary_confidence', 0):.0%})")
            if "iqa" in metadata:
                log("ImageAnalysis", f"  IQA scores: {metadata['iqa']}")

        # Release VRAM if requested
        if release_vram:
            self._release_models()

        # Build metadata JSON output
        metadata_json = json.dumps(metadata, indent=2, ensure_ascii=False)

        # Get prompt output from florence_description
        prompt_output = metadata.get("florence_description", "")

        if verbose:
            log("ImageAnalysis", f"Metadata: {len(metadata_json)} chars")
            log("ImageAnalysis", f"Prompt: {len(prompt_output)} chars")

        # Cleanup
        del pil_image
        cleanup_memory()

        return (metadata_json, prompt_output)

    def unload(self) -> None:
        """Unload all models and free memory."""
        start = log_start("ImageAnalysis", "Unloading all models")

        if self._engine is not None:
            self._engine.unload()
            self._engine = None

        self._release_models()

        log_end("ImageAnalysis", "All models unloaded", start)
