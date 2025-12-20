# -*- coding: utf-8 -*-
"""
ComfyUI-AI-Photography-Toolkit - SID Z-Image Prompt Generator V2

ComfyUI wrapper for the standalone prompt generator.
All core logic is in prompt_generator_core.py for standalone usage.

Author: Siddhartha Lahiri
License: MIT
"""

import gc
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from PIL import Image
from typing import Dict, Any

from comfy_api.latest import io

# Custom types for connecting prompt generator to other nodes
GENERATION_METADATA_Type = io.Custom("GENERATION_METADATA")
SID_PROMPT_Type = io.Custom("SID_PROMPT")

# Results storage directory
GENERATION_RESULTS_DIR = Path(__file__).parent / "generation_results"

# Local imports
from .llm_providers import LLMModelConfig
from .llm_providers.sid_llm_api import LLM_MODEL_Type
from .prompt_generator_core import PromptGenerator, LLMConfig

# Template registry (new system)
from .template_registry_loader import (
    get_template_names as get_registry_template_names,
    get_template as get_registry_template,
    get_auto_template,
)


# =============================================================================
# Helper Functions
# =============================================================================

class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def tensor_to_pil(tensor) -> Image.Image:
    """Convert ComfyUI tensor to PIL Image."""
    if len(tensor.shape) == 4:
        tensor = tensor[0]
    np_image = (tensor.cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(np_image)


def pil_to_tensor(pil_image: Image.Image):
    """Convert PIL Image to ComfyUI tensor."""
    import torch
    np_image = np.array(pil_image).astype(np.float32) / 255.0
    return torch.from_numpy(np_image).unsqueeze(0)


def save_generation_result(
    pil_image: Image.Image,
    prompt: str,
    negative: str,
    caption: str,
    metadata: Dict[str, Any]
) -> str:
    """
    Save generation result to disk for review.

    Args:
        pil_image: Source image
        prompt: Generated prompt
        negative: Negative prompt
        caption: Caption
        metadata: Generation metadata (settings, timing, etc.)

    Returns:
        Session ID (folder name)
    """
    GENERATION_RESULTS_DIR.mkdir(exist_ok=True)

    # Create session folder with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_id = f"gen_{timestamp}"
    session_dir = GENERATION_RESULTS_DIR / session_id
    session_dir.mkdir(exist_ok=True)

    # Save source image
    try:
        image_path = session_dir / "source.jpg"
        pil_image.save(image_path, "JPEG", quality=90)
    except Exception as e:
        print(f"[SID Prompt Generator] Warning: Could not save image: {e}")

    # Save prompt
    prompt_path = session_dir / "prompt.txt"
    prompt_path.write_text(prompt, encoding="utf-8")

    # Save negative prompt if exists
    if negative:
        negative_path = session_dir / "negative.txt"
        negative_path.write_text(negative, encoding="utf-8")

    # Save caption if exists
    if caption:
        caption_path = session_dir / "caption.txt"
        caption_path.write_text(caption, encoding="utf-8")

    # Save metadata as JSON (use custom encoder for numpy types)
    metadata_path = session_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False, cls=NumpyJSONEncoder)

    # Save human-readable summary
    summary_path = session_dir / "summary.txt"
    summary_lines = [
        "=" * 70,
        "SID PROMPT GENERATION SUMMARY",
        "=" * 70,
        "",
        f"Timestamp: {metadata.get('timestamp', 'N/A')}",
        f"Version: {metadata.get('version', 'N/A')}",
        "",
        "-" * 70,
        "MODEL CONFIGURATION",
        "-" * 70,
    ]

    model_config = metadata.get("model_config", {})
    summary_lines.extend([
        f"Provider: {model_config.get('provider', 'N/A')}",
        f"Model: {model_config.get('model', 'N/A')}",
        f"Text Model: {model_config.get('text_model', 'N/A')}",
        f"Temperature: {model_config.get('temperature', 'N/A')}",
        f"Max Tokens: {model_config.get('max_tokens', 'N/A')}",
        f"Analysis Mode: {model_config.get('analysis_mode', 'N/A')}",
        f"Supports Vision: {model_config.get('supports_vision', 'N/A')}",
        f"Supports Reasoning: {model_config.get('supports_reasoning', 'N/A')}",
    ])

    if model_config.get("api_url"):
        summary_lines.append(f"API URL: {model_config.get('api_url')}")

    if model_config.get("extra_params"):
        summary_lines.append(f"Extra Params: {model_config.get('extra_params')}")

    summary_lines.extend([
        "",
        "-" * 70,
        "GENERATION SETTINGS",
        "-" * 70,
    ])

    gen_settings = metadata.get("generation_settings", {})
    summary_lines.extend([
        f"Seed: {gen_settings.get('seed', 'N/A')}",
        f"Analysis Mode: {gen_settings.get('analysis_mode', 'N/A')}",
        f"Template: {gen_settings.get('template', 'N/A')}",
        f"Target Length: {gen_settings.get('prompt_length', 'N/A')} words",
        f"Generate Negative: {gen_settings.get('generate_negative', 'N/A')}",
        f"Generate Caption: {gen_settings.get('generate_caption', 'N/A')}",
        f"NSFW Mode: {gen_settings.get('nsfw_mode', 'N/A')}",
    ])

    if gen_settings.get("prompt_enhance"):
        summary_lines.append(f"Prompt Enhance: {gen_settings.get('prompt_enhance')}")

    # CV Analysis
    cv_analysis = metadata.get("cv_analysis", {})
    if cv_analysis:
        summary_lines.extend([
            "",
            "-" * 70,
            "CV ANALYSIS",
            "-" * 70,
            f"Has Human: {cv_analysis.get('has_human', 'N/A')}",
            f"Human Count: {cv_analysis.get('human_count', 'N/A')}",
            f"Shot Type: {cv_analysis.get('shot_type', 'N/A')}",
        ])
        if cv_analysis.get("composition"):
            comp = cv_analysis.get("composition", {})
            summary_lines.append(f"Composition: {comp.get('type', 'N/A')}")

    # Image metadata
    img_meta = metadata.get("image_metadata", {})
    if img_meta:
        summary_lines.extend([
            "",
            "-" * 70,
            "IMAGE METADATA",
            "-" * 70,
        ])
        if img_meta.get("original_size"):
            summary_lines.append(f"Original Size: {img_meta.get('original_size')}")
        if img_meta.get("compressed_size"):
            summary_lines.append(f"Compressed Size: {img_meta.get('compressed_size')}")
        if img_meta.get("file_size_kb"):
            summary_lines.append(f"File Size: {img_meta.get('file_size_kb'):.1f} KB")

    # Output stats
    output_stats = metadata.get("output_stats", {})
    if output_stats:
        summary_lines.extend([
            "",
            "-" * 70,
            "OUTPUT STATISTICS",
            "-" * 70,
            f"Prompt Length: {output_stats.get('prompt_length', 0)} chars",
            f"Prompt Word Count: {output_stats.get('prompt_word_count', 0)} words",
            f"Negative Length: {output_stats.get('negative_length', 0)} chars",
            f"Caption Length: {output_stats.get('caption_length', 0)} chars",
        ])

    # Timing
    timing = metadata.get("timing", {})
    if timing:
        summary_lines.extend([
            "",
            "-" * 70,
            "TIMING",
            "-" * 70,
            f"Total Time: {timing.get('total_seconds', 0):.2f}s",
        ])
        for key, value in timing.items():
            if key != "total_seconds" and isinstance(value, (int, float)):
                summary_lines.append(f"  {key}: {value:.2f}s")

    # Generated prompt
    summary_lines.extend([
        "",
        "=" * 70,
        "GENERATED PROMPT",
        "=" * 70,
        "",
        prompt,
    ])

    if negative:
        summary_lines.extend([
            "",
            "=" * 70,
            "NEGATIVE PROMPT",
            "=" * 70,
            "",
            negative,
        ])

    if caption:
        summary_lines.extend([
            "",
            "=" * 70,
            "CAPTION",
            "=" * 70,
            "",
            caption,
        ])

    summary_lines.append("")
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"[SID Prompt Generator] Results saved to: {session_dir}")
    return session_id


# =============================================================================
# Main Node Class
# =============================================================================

class SID_ZImagePromptGeneratorV2(io.ComfyNode):
    """
    SID Z-Image Prompt Generator V2

    ComfyUI node wrapper for fast, reliable prompt generation.
    - CV detection (YOLO + MediaPipe) for human detection
    - Vision LLM for detailed descriptions
    - Z-Image optimization always on
    - Optional negative and caption generation

    For standalone usage (batch processing, training data):
        from prompt_generator_core import PromptGenerator
        generator = PromptGenerator(provider="ollama", model="llava")
        result = generator.process_image("image.jpg")
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        # Get template names from registry with robust error handling
        try:
            template_names = get_registry_template_names() or ["Portrait", "General"]
        except Exception as e:
            print(f"[SID_ZImagePromptGenerator] Error loading templates: {e}")
            template_names = ["Portrait", "General"]

        return io.Schema(
            node_id="SID_ZImagePromptGeneratorV2",
            display_name="SID Z-Image Prompt Generator",
            category="SID Photography Toolkit",
            description="Fast, reliable prompt generation with CV detection + Vision LLM",
            is_output_node=True,
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip="Input image to analyze"
                ),
                LLM_MODEL_Type.Input(
                    "llm_model",
                    tooltip="Connect SID_LLM_API or SID_LLM_Local node"
                ),
                io.Combo.Input(
                    "analysis_mode",
                    options=["Template", "Simple", "Detailed"],
                    default="Simple",
                    tooltip="Template: Select from registry. Simple: Auto-detect portrait/general (1 call). Detailed: Multi-aspect analysis (3-4 calls)"
                ),
                io.Combo.Input(
                    "template",
                    options=template_names,
                    default=template_names[0] if template_names else "Portrait",
                    tooltip="Select template (Template mode only). Create custom templates via /sid/template-editor"
                ),
                io.Int.Input(
                    "prompt_length",
                    default=400,
                    min=50,
                    max=1000,
                    tooltip="Target word count (50-1000)"
                ),
                io.Boolean.Input(
                    "generate_negative",
                    default=False,
                    display_name="Generate Negative",
                    tooltip="Generate negative prompt"
                ),
                io.Boolean.Input(
                    "generate_caption",
                    default=False,
                    display_name="Generate Caption",
                    tooltip="Generate image caption"
                ),
                io.Boolean.Input(
                    "nsfw_mode",
                    default=False,
                    display_name="NSFW Mode",
                    tooltip="Enable detailed NSFW feature description (sideboob, underboob, cleavage, exposed skin, etc.)"
                ),
                io.Boolean.Input(
                    "enable_reasoning",
                    default=False,
                    display_name="Enable Reasoning",
                    tooltip="Enable extended thinking/reasoning mode. Requires a reasoning-capable model (e.g., Qwen3-VL-*-Thinking, Claude with thinking, o1/o3). Error if model doesn't support reasoning."
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    tooltip="Seed for reproducibility (0 = random)"
                ),
                io.String.Input(
                    "prompt_enhance",
                    default="",
                    multiline=True,
                    optional=True,
                    tooltip="Optional: Keywords/guidance to integrate into the prompt (e.g., 'red hair, blue eyes'). LLM will naturally incorporate these."
                ),
                io.Boolean.Input(
                    "prompt_override",
                    default=False,
                    display_name="Prompt Override",
                    tooltip="Enable custom prompt override. When enabled, uses the System Prompt, User Prompt, and Example fields below instead of template selection."
                ),
                io.Boolean.Input(
                    "raw_output",
                    default=False,
                    display_name="Raw Output",
                    tooltip="Skip all post-processing cleanup. Use this for structured output formats (JSON, tags, classification). The output will be exactly what the LLM returns."
                ),
                io.String.Input(
                    "override_system_prompt",
                    default="",
                    multiline=True,
                    optional=True,
                    display_name="System Prompt",
                    tooltip="Custom system prompt for the Vision LLM. Defines how the model should analyze and describe images."
                ),
                io.String.Input(
                    "override_user_prompt",
                    default="Describe this image.",
                    multiline=True,
                    optional=True,
                    display_name="User Prompt",
                    tooltip="Custom user prompt sent with the image. This is the actual request/question to the LLM."
                ),
                io.String.Input(
                    "override_example",
                    default="",
                    multiline=True,
                    optional=True,
                    display_name="Example",
                    tooltip="One-shot example of desired output. Appended to system prompt to guide the model's response format and style."
                ),
                io.Boolean.Input(
                    "store_results",
                    default=True,
                    display_name="Store Results",
                    tooltip="Save prompt, image, and metadata locally. View at /sid/generation-results"
                ),
            ],
            outputs=[
                SID_PROMPT_Type.Output("prompt", display_name="prompt", tooltip="Generated prompt. Connect to Debug Agent or other nodes."),
                io.String.Output("negative", display_name="negative"),
                io.String.Output("caption", display_name="caption"),
                GENERATION_METADATA_Type.Output("metadata", display_name="metadata", tooltip="Generation metadata (settings, model config, CV analysis, timing). Connect to Debug Agent."),
            ],
        )

    @classmethod
    def execute(
        cls,
        image,
        llm_model: LLMModelConfig,
        analysis_mode: str,
        template: str,
        prompt_length: int,
        generate_negative: bool,
        generate_caption: bool,
        nsfw_mode: bool,
        enable_reasoning: bool,
        seed: int,
        prompt_enhance: str = "",
        prompt_override: bool = False,
        raw_output: bool = False,
        override_system_prompt: str = "",
        override_user_prompt: str = "Describe this image.",
        override_example: str = "",
        store_results: bool = True,
    ):
        """Execute prompt generation using core module."""
        import random
        import time

        def cleanup_memory():
            """Clean up GPU and system memory."""
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    # Note: Don't call synchronize() here - it creates a barrier
                    # that can slow down subsequent CUDA operations like KSampler
            except Exception:
                pass

        start_time = time.time()

        # Get model's reasoning capability
        supports_reasoning = getattr(llm_model, 'supports_reasoning', False)
        model_name = getattr(llm_model, 'model', 'Unknown')

        # Validate: reasoning toggle must match model capability
        # (No cleanup needed here - nothing allocated yet)
        if enable_reasoning and not supports_reasoning:
            raise ValueError(
                f"REASONING MISMATCH: Enable Reasoning is ON but model '{model_name}' does not support reasoning.\n\n"
                f"To use reasoning, select a reasoning-capable model:\n"
                f"  - Local: Qwen3-VL-*-Thinking variants\n"
                f"  - Anthropic: Claude models with extended thinking\n"
                f"  - OpenAI: o1, o3 series\n"
                f"  - Google: Gemini 2.0 Flash Thinking\n\n"
                f"Or disable 'Enable Reasoning' in the prompt generator."
            )

        if not enable_reasoning and supports_reasoning:
            raise ValueError(
                f"REASONING MISMATCH: Enable Reasoning is OFF but model '{model_name}' is a reasoning/thinking model.\n\n"
                f"Thinking models are slower due to chain-of-thought processing.\n"
                f"For fast execution without reasoning, use a non-thinking model:\n"
                f"  - Local: Qwen3-VL-*-Instruct (not Thinking)\n"
                f"  - Anthropic: Standard Claude models\n"
                f"  - OpenAI: GPT-4o, GPT-4.1 (not o1/o3)\n"
                f"  - Google: Gemini Pro/Flash (not Thinking)\n\n"
                f"Or enable 'Enable Reasoning' to use this model's thinking capability."
            )

        # Set seed for reproducibility
        if seed == 0:
            seed = random.randint(1, 2147483647)
        random.seed(seed)

        # Clear VRAM before processing
        cleanup_memory()
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception:
            pass

        # Main processing wrapped in try/finally for guaranteed cleanup
        try:
            # Convert ComfyUI tensor to PIL Image
            pil_image = tensor_to_pil(image)

            # Get temperature from LLM config
            temperature = getattr(llm_model, 'temperature', 0.7)

            # For Simple mode, run CV detection first to determine portrait vs general
            cv_detection = None
            if analysis_mode.lower() == "simple":
                try:
                    from .cv_analyzer import CVAnalyzer
                    cv_detection = CVAnalyzer.analyze(np.array(pil_image))
                except Exception as e:
                    print(f"[SID Prompt Generator] CV detection failed: {e}, defaulting to general")
                    cv_detection = {"has_human": False}

            # Map analysis_mode to internal style and get template
            # Template -> uses selected template from registry
            # Simple -> auto-detect portrait/general via CV (single-pass)
            # Detailed -> multi-aspect analysis (3-4 LLM calls)
            template_prompt = None
            template_name = None
            template_info = None  # Full template info for debug agent

            # Get model size from LLM config (set by LLM provider node based on model metadata)
            internal_model_size = getattr(llm_model, 'model_size', 'large')
            print(f"[SID Prompt Generator] Model size: {internal_model_size}")

            # Custom override variables for PromptGenerator
            custom_user_prompt = None

            if prompt_override:
                # Custom prompt override takes priority - RAW CONTROL
                # System prompt is sent exactly as typed
                # Example is appended with just a newline (no special markers)
                system_parts = []
                if override_system_prompt and override_system_prompt.strip():
                    system_parts.append(override_system_prompt.strip())
                if override_example and override_example.strip():
                    # Append example directly - user controls the format
                    system_parts.append(override_example.strip())

                template_prompt = "\n\n".join(system_parts) if system_parts else None
                custom_user_prompt = override_user_prompt.strip() if override_user_prompt else "Describe this image."

                internal_style = "template"
                prompt_style = "template"
                core_analysis_mode = "standard"
                template_name = "Custom Override"
                template_info = {
                    "name": "Custom Override",
                    "path": None,
                    "category": "override",
                    "readonly": False,
                    "model_size_used": internal_model_size,
                    "system_prompt": template_prompt,
                    "user_prompt": custom_user_prompt,
                    "example": override_example.strip() if override_example else None,
                }
                print(f"[SID Prompt Generator] ════════════════════════════════════════")
                print(f"[SID Prompt Generator] CUSTOM OVERRIDE MODE ACTIVE")
                print(f"[SID Prompt Generator] Raw Output: {'ON - No cleanup' if raw_output else 'OFF - Z-Image cleanup applied'}")
                print(f"[SID Prompt Generator] System prompt: {len(template_prompt) if template_prompt else 0} chars")
                print(f"[SID Prompt Generator] User prompt: {custom_user_prompt[:80] if custom_user_prompt else 'None'}...")
                print(f"[SID Prompt Generator] ════════════════════════════════════════")

            elif analysis_mode.lower() == "template":
                # Template mode: use selected template from registry
                template_data = get_registry_template(template, internal_model_size)
                if template_data:
                    template_prompt = template_data.get("system", "")
                    template_name = template_data.get("name", template)
                    template_info = {
                        "name": template_data.get("name", template),
                        "path": template_data.get("path", ""),
                        "category": template_data.get("category", "unknown"),
                        "readonly": template_data.get("readonly", False),
                        "model_size_used": internal_model_size,
                        "system_prompt": template_data.get("system", ""),
                        "user_prompt": template_data.get("user", ""),
                    }
                    print(f"[SID Prompt Generator] Using template: {template_name} (size: {internal_model_size})")
                else:
                    print(f"[SID Prompt Generator] Template not found: {template}, falling back to Simple mode")
                    analysis_mode = "Simple"
                    # Fall through to Simple mode handling

                internal_style = "template"
                prompt_style = "template"
                core_analysis_mode = "standard"

            elif analysis_mode.lower() == "simple":
                # Simple mode: auto-detect portrait vs general using CV
                auto_template = get_auto_template(cv_detection or {"has_human": False}, internal_model_size)
                if auto_template:
                    template_prompt = auto_template.get("system", "")
                    template_name = auto_template.get("name", "Auto")
                    has_human = cv_detection.get("has_human", False) if cv_detection else False
                    template_info = {
                        "name": auto_template.get("name", "Auto"),
                        "path": auto_template.get("path", ""),
                        "category": auto_template.get("category", "core"),
                        "readonly": auto_template.get("readonly", True),
                        "model_size_used": internal_model_size,
                        "system_prompt": auto_template.get("system", ""),
                        "user_prompt": auto_template.get("user", ""),
                        "auto_detected": "portrait" if has_human else "general",
                    }
                    print(f"[SID Prompt Generator] Simple mode: {'Portrait' if has_human else 'General'} template (size: {internal_model_size})")
                internal_style = "template"
                prompt_style = "template"
                core_analysis_mode = "standard"

            elif analysis_mode.lower() == "detailed":
                # Detailed mode: multi-aspect analysis (unchanged)
                internal_style = "verbose"
                prompt_style = "inbuilt-prompt"
                core_analysis_mode = "detailed"
                template_name = "Detailed Analysis"
                template_info = {
                    "name": "Detailed Analysis",
                    "path": None,
                    "category": "detailed",
                    "readonly": True,
                    "model_size_used": internal_model_size,
                    "system_prompt": None,  # Uses built-in multi-pass prompts
                    "user_prompt": None,
                }

            # Create generator with LLM config from node connection
            generator = PromptGenerator(
                provider=llm_model.provider,
                model=llm_model.model,
                text_model=llm_model.text_model or "",
                api_key=llm_model.api_key or "",
                api_url=llm_model.api_url or "",
                temperature=temperature,
                analysis_mode=core_analysis_mode,
                enable_reasoning=enable_reasoning,
                prompt_style=internal_style,
                prompt_length=prompt_length,
                generate_negative=generate_negative,
                generate_caption=generate_caption,
                nsfw_mode=nsfw_mode,
                verbose=True,
                extra_params=llm_model.extra_params,
                template_prompt=template_prompt,
                template_user_prompt=custom_user_prompt,
                raw_output=raw_output
            )

            # Process image using core module
            enhance_text = prompt_enhance.strip() if prompt_enhance else ""
            result = generator.process_image(pil_image, user_guidance=enhance_text)

            # Print metadata and debug log to console
            reasoning_str = "ON" if enable_reasoning else "OFF"
            nsfw_str = "ON" if nsfw_mode else "OFF"
            print(f"\n[Seed: {seed}, Temperature: {temperature}, Mode: {analysis_mode}, Reasoning: {reasoning_str}, NSFW: {nsfw_str}]")
            print(result.get_metadata_str())

            # Build metadata (always, for output and optional storage)
            generation_time = time.time() - start_time
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "version": "4.3.0",

                # Generation settings
                "generation_settings": {
                    "seed": seed,
                    "analysis_mode": analysis_mode,
                    "model_size": internal_model_size,
                    "template_info": template_info,  # Full template data for debug agent
                    "prompt_length": prompt_length,
                    "generate_negative": generate_negative,
                    "generate_caption": generate_caption,
                    "nsfw_mode": nsfw_mode,
                    "enable_reasoning": enable_reasoning,
                    "prompt_enhance": enhance_text if enhance_text else None,
                },

                # Full model configuration
                "model_config": {
                    "provider": llm_model.provider,
                    "model": llm_model.model,
                    "text_model": llm_model.text_model or None,
                    "api_url": llm_model.api_url or None,
                    "temperature": temperature,
                    "max_tokens": llm_model.max_tokens,
                    "analysis_mode": analysis_mode,
                    "supports_vision": llm_model.supports_vision,
                    "supports_reasoning": supports_reasoning,
                    "reasoning_enabled": enable_reasoning,
                    "extra_params": llm_model.extra_params if llm_model.extra_params else None,
                },

                # CV analysis results
                "cv_analysis": result.cv_analysis if result.cv_analysis else None,

                # Image metadata from processing
                "image_metadata": result.metadata if result.metadata else None,

                # Timing breakdown
                "timing": {
                    "total_seconds": round(generation_time, 2),
                    **result.timing  # Include detailed timing from generator
                },

                # Output statistics
                "output_stats": {
                    "prompt_length": len(result.prompt),
                    "prompt_word_count": len(result.prompt.split()) if result.prompt else 0,
                    "negative_length": len(result.negative) if result.negative else 0,
                    "caption_length": len(result.caption) if result.caption else 0,
                },

                # Debug logs (if any)
                "debug_logs": result.logs if result.logs else None,
            }

            # Remove None values for cleaner output
            metadata = {k: v for k, v in metadata.items() if v is not None}
            for key in ["generation_settings", "model_config", "output_stats"]:
                if key in metadata and isinstance(metadata[key], dict):
                    metadata[key] = {k: v for k, v in metadata[key].items() if v is not None}

            # Store results if enabled
            if store_results:
                save_generation_result(pil_image, result.prompt, result.negative, result.caption, metadata)

            # Convert numpy types to native Python types for the metadata output
            metadata_clean = json.loads(json.dumps(metadata, cls=NumpyJSONEncoder))

            return io.NodeOutput(result.prompt, result.negative, result.caption, metadata_clean)

        finally:
            # Always clean up memory after processing (success or failure)
            cleanup_memory()


# =============================================================================
# Prompt Template Node
# =============================================================================

class SID_PromptTemplate(io.ComfyNode):
    """
    SID Prompt Template Node

    Select a template from the dropdown to populate the prompt field.
    Edit the prompt as needed, then connect to other nodes.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        # Get template names with robust error handling
        try:
            template_names = get_registry_template_names() or ["Default"]
        except Exception as e:
            print(f"[SID_PromptTemplate] Error loading templates: {e}")
            template_names = ["Default"]

        return io.Schema(
            node_id="SID_PromptTemplate",
            display_name="SID Prompt Template",
            category="SID Photography Toolkit",
            description="Select and edit prompt templates",
            inputs=[
                io.Combo.Input(
                    "template",
                    options=template_names,
                    default=template_names[0] if template_names else "Default",
                    tooltip="Select a prompt template to load"
                ),
                io.String.Input(
                    "prompt_text",
                    default="",
                    multiline=True,
                    display_name="prompt",
                    tooltip="Edit the prompt template here. Changes are used as output."
                ),
            ],
            outputs=[
                io.String.Output("prompt", display_name="prompt"),
            ],
        )

    @classmethod
    def execute(cls, template: str, prompt_text: str):
        """Return the prompt text."""
        # If prompt_text is empty, load from template
        if not prompt_text or not prompt_text.strip():
            template_data = get_registry_template(template)
            if template_data:
                prompt_text = template_data.get("system", "")

        return io.NodeOutput(prompt_text)
