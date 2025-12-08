"""
SID_ZImagePromptGenerator Node
Agentic multi-stage image analysis for Z-Image prompt generation.

This node analyzes an input image using Claude's vision capabilities and generates
a Z-Image compatible narrative prompt through a multi-stage agentic pipeline.
"""

import base64
import io
import json
import random
import time
from datetime import datetime
from typing import Any, Optional

import numpy as np
from PIL import Image
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io as comfy_io

from .utils.zimage_utils import (
    load_zimage_config,
    get_shot_framings,
    get_photography_genres,
    get_content_detail_schema,
    get_prompt_template,
    get_focus_override_config,
    get_detail_level_config,
    get_zimage_settings,
    clean_zimage_output,
    hash_image_tensor,
    get_cache_key,
    get_cached_output,
    set_cached_output,
    get_cache_stats,
    get_image_metadata,
    get_zimage_recommendations,
    build_attribute_schema_for_scene,
    format_attribute_schema_for_prompt,
)


class SID_ZImagePromptGenerator(comfy_io.ComfyNode):
    """
    Agentic multi-stage image analyzer for Z-Image prompt generation.

    Uses a 6-stage pipeline:
    1. Classification - Detect shot framing and photography genre
    2. Metadata - Extract image dimensions and properties
    3. Attribute Mapping - Select relevant attributes for the scene
    4. Detailed Analysis - LLM extraction of structured attributes
    5. Prompt Composition - Generate flowing narrative prompt
    6. Z-Image Optimization - Provide recommendations for Z-Image

    Supports NSFW content through content_detail levels (Z-Image compatible).
    """

    # Track seed state for increment/decrement modes
    _last_seed: int = 0

    @classmethod
    def define_schema(cls) -> comfy_io.Schema:
        """Define the node schema with all inputs and outputs."""

        # Load config for options
        try:
            config = load_zimage_config()
            genres = get_photography_genres()
            genre_options = ["Auto-detect"] + list(genres.keys())
        except Exception:
            genre_options = ["Auto-detect"]

        return comfy_io.Schema(
            node_id="SID_ZImagePromptGenerator",
            display_name="SID Z-Image Prompt Generator",
            category="SID Photography Toolkit/Z-Image",
            description="Agentic image analyzer that generates Z-Image compatible narrative prompts",
            inputs=[
                # Image input
                comfy_io.Image.Input(
                    "image",
                    tooltip="Input image to analyze"
                ),

                # API Settings
                comfy_io.Combo.Input(
                    "ai_provider",
                    options=["Anthropic"],
                    default="Anthropic",
                    tooltip="AI provider for image analysis (more providers coming soon)"
                ),
                comfy_io.String.Input(
                    "api_key",
                    default="",
                    multiline=False,
                    tooltip="Anthropic API key (get from https://console.anthropic.com/)"
                ),
                comfy_io.Combo.Input(
                    "model",
                    options=[
                        "claude-sonnet-4-5-20250929",
                        "claude-haiku-4-5-20251001",
                        "claude-opus-4-1-20250805",
                        "claude-3-5-haiku-20241022",
                    ],
                    default="claude-sonnet-4-5-20250929",
                    tooltip="Claude model to use for analysis"
                ),
                comfy_io.String.Input(
                    "api_url",
                    default="https://api.anthropic.com",
                    multiline=False,
                    tooltip="API endpoint URL (not used for Anthropic, for future providers)"
                ),

                # Analysis Options
                comfy_io.Combo.Input(
                    "detail_level",
                    options=["Quick", "Standard", "Deep"],
                    default="Standard",
                    tooltip="Analysis depth: Quick (1 call), Standard (2 calls), Deep (3 calls)"
                ),
                comfy_io.Combo.Input(
                    "focus_override",
                    options=[
                        "Auto-detect",
                        "Portrait/People",
                        "Full Body/Fashion",
                        "Landscape/Environment",
                        "Product/Object",
                        "Food/Beverage",
                        "Architecture/Interior",
                    ],
                    default="Auto-detect",
                    tooltip="Force a specific genre focus instead of auto-detection"
                ),
                comfy_io.Combo.Input(
                    "content_detail",
                    options=["minimal", "standard", "detailed", "explicit"],
                    default="standard",
                    tooltip="Body/clothing detail level. 'explicit' enables full NSFW attributes"
                ),

                # Prompt Direction
                comfy_io.String.Input(
                    "user_prompt",
                    default="",
                    multiline=True,
                    tooltip="Optional: Guide the analysis (e.g., 'focus on the dress') or provide prompt to enhance"
                ),
                comfy_io.Combo.Input(
                    "prompt_mode",
                    options=[
                        "Image Only (ignore prompt)",
                        "Prompt Guides Analysis",
                        "Prompt First, Image Fills Gaps",
                        "Prompt Dominates",
                    ],
                    default="Prompt Guides Analysis",
                    tooltip="How user_prompt interacts with image analysis"
                ),

                # Focus Area Toggles
                comfy_io.Boolean.Input(
                    "focus_subject",
                    default=True,
                    tooltip="Include detailed subject description"
                ),
                comfy_io.Boolean.Input(
                    "focus_environment",
                    default=True,
                    tooltip="Include background/environment description"
                ),
                comfy_io.Boolean.Input(
                    "focus_lighting",
                    default=True,
                    tooltip="Include lighting description"
                ),
                comfy_io.Boolean.Input(
                    "focus_colors",
                    default=True,
                    tooltip="Include colors and materials"
                ),
                comfy_io.Boolean.Input(
                    "focus_mood",
                    default=False,
                    tooltip="Include mood/atmosphere description"
                ),
                comfy_io.Boolean.Input(
                    "include_text_quotes",
                    default=True,
                    tooltip="Quote visible text with \"quotes\" for Z-Image text rendering"
                ),

                # Output Settings
                comfy_io.Int.Input(
                    "max_tokens",
                    default=300,
                    min=50,
                    max=500,
                    step=25,
                    display_mode=comfy_io.NumberDisplay.slider,
                    tooltip="Target output tokens for the prompt"
                ),

                # Generation Settings
                comfy_io.Float.Input(
                    "temperature",
                    default=0.7,
                    min=0.0,
                    max=1.0,
                    step=0.1,
                    round=0.1,
                    display_mode=comfy_io.NumberDisplay.slider,
                    tooltip="Creativity level (0=focused, 1=creative)"
                ),
                comfy_io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    control_after_generate=True,
                    tooltip="Random seed for reproducibility"
                ),
                comfy_io.Combo.Input(
                    "seed_mode",
                    options=["fixed", "randomize", "increment", "decrement"],
                    default="fixed",
                    tooltip="Seed behavior: fixed (deterministic), randomize (new each run), increment/decrement"
                ),
                comfy_io.Boolean.Input(
                    "cache_prompt",
                    default=True,
                    tooltip="Cache results locally. Same image + settings = instant cached result (saves API calls)"
                ),
            ],
            outputs=[
                comfy_io.Image.Output(
                    "output_image",
                    display_name="image",
                    tooltip="Pass-through of input image"
                ),
                comfy_io.String.Output(
                    "prompt",
                    display_name="zimage_prompt",
                    tooltip="Z-Image compatible narrative prompt ready for generation"
                ),
                comfy_io.Int.Output(
                    "width",
                    display_name="width",
                    tooltip="Image width in pixels"
                ),
                comfy_io.Int.Output(
                    "height",
                    display_name="height",
                    tooltip="Image height in pixels"
                ),
                comfy_io.String.Output(
                    "structured_data",
                    display_name="structured_data",
                    tooltip="JSON with classification and extracted attributes"
                ),
                comfy_io.String.Output(
                    "metadata",
                    display_name="image_metadata",
                    tooltip="JSON with image info and Z-Image recommendations"
                ),
                comfy_io.String.Output(
                    "debug_log",
                    display_name="debug_log",
                    tooltip="Stage-by-stage processing details"
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        image,
        ai_provider: str,
        api_key: str,
        model: str,
        api_url: str,
        detail_level: str,
        focus_override: str,
        content_detail: str,
        user_prompt: str,
        prompt_mode: str,
        focus_subject: bool,
        focus_environment: bool,
        focus_lighting: bool,
        focus_colors: bool,
        focus_mood: bool,
        include_text_quotes: bool,
        max_tokens: int,
        temperature: float,
        seed: int,
        seed_mode: str,
        cache_prompt: bool,
    ) -> comfy_io.NodeOutput:
        """Execute the agentic pipeline to generate Z-Image prompt."""

        debug_lines = []
        start_time = time.time()

        def log(message: str):
            debug_lines.append(message)
            print(message)

        log("=" * 60)
        log("SID Z-Image Prompt Generator - Debug Log")
        log("=" * 60)
        log(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log(f"Mode: {detail_level} ({get_detail_level_config(detail_level).get('llm_calls', 2)} LLM calls)")
        log("")

        # Handle seed mode
        actual_seed = cls._process_seed(seed, seed_mode)
        log(f"Seed: {actual_seed} (mode: {seed_mode})")
        log(f"Cache: {'enabled' if cache_prompt else 'disabled'}")
        log(f"Provider: {ai_provider}")

        # Get image dimensions early for all return paths
        if len(image.shape) == 4:
            img_height, img_width = image.shape[1], image.shape[2]
        else:
            img_height, img_width = image.shape[0], image.shape[1]

        # Validate API key
        if not api_key or api_key.strip() == "":
            error_msg = "ERROR: Anthropic API key is required. Get one at https://console.anthropic.com/"
            return comfy_io.NodeOutput(image, error_msg, img_width, img_height, "{}", "{}", error_msg)

        try:
            import anthropic
        except ImportError:
            error_msg = "ERROR: anthropic library not installed. Run: pip install anthropic"
            return comfy_io.NodeOutput(image, error_msg, img_width, img_height, "{}", "{}", error_msg)

        # Check cache if caching is enabled
        if cache_prompt:
            image_hash = hash_image_tensor(image)
            cache_key = get_cache_key(
                image_hash, actual_seed,
                model=model,
                detail_level=detail_level,
                focus_override=focus_override,
                content_detail=content_detail,
                user_prompt=user_prompt,
                prompt_mode=prompt_mode,
                focus_subject=focus_subject,
                focus_environment=focus_environment,
                focus_lighting=focus_lighting,
                focus_colors=focus_colors,
                focus_mood=focus_mood,
                include_text_quotes=include_text_quotes,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            cached = get_cached_output(cache_key)
            if cached:
                cache_stats = get_cache_stats()
                log("[CACHE HIT] Returning cached output (persistent disk cache)")
                log(f"  Cache: {cache_stats['disk_entries']} entries, {cache_stats['disk_size_mb']} MB")
                log("=" * 60)
                return comfy_io.NodeOutput(
                    image,
                    cached["prompt"],
                    img_width,
                    img_height,
                    cached["structured_data"],
                    cached["metadata"],
                    cached["debug_log"] + "\n\n[CACHE HIT - Loaded from persistent disk cache]"
                )
        else:
            cache_key = None

        try:
            # Initialize Anthropic client
            client = anthropic.Anthropic(api_key=api_key.strip())

            # Convert image to base64
            base64_image = cls._image_to_base64(image)

            # ===== STAGE 1: CLASSIFICATION =====
            log("[STAGE 1] Classification (LLM Call #1)")
            stage1_start = time.time()

            classification = cls._stage1_classification(
                client, model, base64_image, focus_override, temperature, log
            )

            log(f"  Shot Framing: {classification['shot_framing']} ({classification.get('shot_label', '')}) - {int(classification.get('confidence', 0) * 100)}% confidence")
            log(f"  Genre: {classification['genre']} ({classification.get('genre_label', '')})")
            log(f"  Secondary: {', '.join(classification.get('secondary_tags', []))}")
            log(f"  Subject Count: {classification.get('subject_count', 0)}")
            log(f"  Has Text: {classification.get('has_text', False)}")
            log(f"  Duration: {time.time() - stage1_start:.1f}s")
            log("")

            # ===== STAGE 2: METADATA =====
            log("[STAGE 2] Metadata Extraction")

            image_meta = get_image_metadata(image)
            log(f"  Dimensions: {image_meta['width']} x {image_meta['height']} px")
            log(f"  Aspect Ratio: {image_meta['aspect_ratio']} ({image_meta['aspect_decimal']})")
            log(f"  Orientation: {image_meta['orientation']}")
            log("")

            # ===== STAGE 3: ATTRIBUTE MAPPING =====
            log("[STAGE 3] Attribute Mapping")

            attribute_schema = build_attribute_schema_for_scene(
                classification["shot_framing"],
                classification["genre"],
                content_detail
            )
            log(f"  Scene: {classification['shot_framing']} + {classification['genre']}")
            log(f"  Selected categories: {', '.join(attribute_schema.keys())}")
            log("")

            # ===== STAGE 4: DETAILED ANALYSIS =====
            if detail_level in ["Standard", "Deep"]:
                log("[STAGE 4] Detailed Analysis (LLM Call #2)")
                stage4_start = time.time()

                attributes = cls._stage4_detailed_analysis(
                    client, model, base64_image,
                    classification, attribute_schema,
                    content_detail, temperature, log
                )

                attr_count = sum(len(v) if isinstance(v, dict) else 1 for v in attributes.values())
                log(f"  Attributes extracted: {len(attributes)} categories, {attr_count} properties")
                log(f"  Duration: {time.time() - stage4_start:.1f}s")
                log("")
            else:
                # Quick mode - use classification only
                attributes = {"classification": classification}
                log("[STAGE 4] Skipped (Quick mode)")
                log("")

            # ===== STAGE 5: PROMPT COMPOSITION =====
            log("[STAGE 5] Prompt Composition")
            stage5_start = time.time()

            if detail_level == "Deep":
                # Additional LLM call for refined prompt
                log("  (LLM Call #3 for refined composition)")
                prompt = cls._stage5_prompt_composition_llm(
                    client, model, base64_image,
                    classification, attributes,
                    user_prompt, prompt_mode,
                    focus_subject, focus_environment, focus_lighting,
                    focus_colors, focus_mood, include_text_quotes,
                    max_tokens, temperature, log
                )
            else:
                # Template-based composition
                prompt = cls._stage5_prompt_composition_template(
                    client, model, base64_image,
                    classification, attributes,
                    user_prompt, prompt_mode,
                    focus_subject, focus_environment, focus_lighting,
                    focus_colors, focus_mood, include_text_quotes,
                    max_tokens, temperature, log
                )

            # Clean the output
            prompt = clean_zimage_output(prompt, max_tokens * 4)  # ~4 chars per token

            word_count = len(prompt.split())
            estimated_tokens = int(word_count * 1.3)
            log(f"  Word count: {word_count}")
            log(f"  Estimated tokens: {estimated_tokens}")
            log(f"  Duration: {time.time() - stage5_start:.1f}s")
            log("")

            # ===== STAGE 6: Z-IMAGE RECOMMENDATIONS =====
            log("[STAGE 6] Z-Image Recommendations")

            zimage_recs = get_zimage_recommendations(image_meta)
            if zimage_recs["resize_needed"]:
                log(f"  Resize: {image_meta['width']}x{image_meta['height']} -> {zimage_recs['optimal_resolution'][0]}x{zimage_recs['optimal_resolution'][1]} ({zimage_recs['resize_method']} {zimage_recs['resize_direction']})")
            else:
                log(f"  Resize: Not needed (already optimal)")
            log(f"  Quality estimate: {zimage_recs['quality_estimate'].upper()}")
            log("")

            # Build structured data output
            structured_data = {
                "classification": classification,
                "attributes": attributes,
                "prompt_stats": {
                    "word_count": word_count,
                    "estimated_tokens": estimated_tokens,
                }
            }

            # Build metadata output
            metadata_output = {
                "image_info": image_meta,
                "z_image_recommendations": zimage_recs,
                "content_flags": {
                    "has_text": classification.get("has_text", False),
                    "has_multiple_subjects": classification.get("subject_count", 1) != 1,
                    "complexity": "high" if len(attributes) > 10 else "medium" if len(attributes) > 5 else "low",
                    "content_detail_level": content_detail,
                }
            }

            # Finalize debug log
            total_time = time.time() - start_time
            log("=" * 60)
            log(f"Total duration: {total_time:.1f}s")

            # Cache the result for fixed seed mode
            if cache_key:
                set_cached_output(cache_key, {
                    "prompt": prompt,
                    "structured_data": json.dumps(structured_data, indent=2),
                    "metadata": json.dumps(metadata_output, indent=2),
                    "debug_log": "\n".join(debug_lines),  # Log before cache info
                })
                cache_stats = get_cache_stats()
                log(f"[CACHED] Result saved to persistent disk cache")
                log(f"  Cache: {cache_stats['disk_entries']} entries, {cache_stats['disk_size_mb']} MB")

            log("=" * 60)
            debug_log = "\n".join(debug_lines)

            return comfy_io.NodeOutput(
                image,
                prompt,
                img_width,
                img_height,
                json.dumps(structured_data, indent=2),
                json.dumps(metadata_output, indent=2),
                debug_log,
            )

        except anthropic.APIError as e:
            error_msg = f"Anthropic API Error: {str(e)}"
            log(f"ERROR: {error_msg}")
            return comfy_io.NodeOutput(image, error_msg, img_width, img_height, "{}", "{}", "\n".join(debug_lines))
        except Exception as e:
            error_msg = f"Error generating prompt: {str(e)}"
            log(f"ERROR: {error_msg}")
            import traceback
            log(traceback.format_exc())
            return comfy_io.NodeOutput(image, error_msg, img_width, img_height, "{}", "{}", "\n".join(debug_lines))

    @classmethod
    def _process_seed(cls, seed: int, seed_mode: str) -> int:
        """Process seed based on mode and return actual seed to use."""
        if seed_mode == "randomize":
            return random.randint(0, 2147483647)
        elif seed_mode == "increment":
            cls._last_seed = seed + 1
            return seed
        elif seed_mode == "decrement":
            cls._last_seed = max(0, seed - 1)
            return seed
        else:  # fixed
            return seed

    @staticmethod
    def _image_to_base64(image_tensor) -> str:
        """Convert ComfyUI image tensor to base64 string."""
        if len(image_tensor.shape) == 4:
            image_np = image_tensor[0].cpu().numpy()
        else:
            image_np = image_tensor.cpu().numpy()

        image_np = (image_np * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_np)

        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()
        return base64.b64encode(img_bytes).decode("utf-8")

    @classmethod
    def _stage1_classification(
        cls,
        client,
        model: str,
        base64_image: str,
        focus_override: str,
        temperature: float,
        log
    ) -> dict:
        """Stage 1: Classify the image."""

        # Get shot framings and genres for reference
        shot_framings = get_shot_framings()
        genres = get_photography_genres()

        shot_codes = list(shot_framings.keys())
        genre_codes = list(genres.keys())

        system_prompt = f"""You are an expert image classifier for photography analysis.

Classify this image precisely into shot framing and photography genre.

SHOT FRAMINGS (distance to subject):
{json.dumps({k: v['name'] for k, v in shot_framings.items()}, indent=2)}

PHOTOGRAPHY GENRES:
{json.dumps({k: v['name'] for k, v in genres.items()}, indent=2)}

Output ONLY valid JSON (no markdown, no explanation):
{{
  "shot_framing": "<code from: {', '.join(shot_codes)}>",
  "shot_label": "<full name>",
  "genre": "<code from: {', '.join(genre_codes)}>",
  "genre_label": "<full name>",
  "genre_category": "<people|event|nature|commercial|artistic|lifestyle>",
  "secondary_tags": ["<tag1>", "<tag2>"],
  "subject_count": <0|1|2|"group">,
  "has_text": <true|false>,
  "confidence": <0.0-1.0>
}}"""

        # Apply focus override if specified
        user_message = "Classify this image."
        if focus_override != "Auto-detect":
            override_config = get_focus_override_config(focus_override)
            if override_config:
                user_message = f"Classify this image. Hint: Focus on {focus_override} characteristics."

        message = client.messages.create(
            model=model,
            max_tokens=500,
            temperature=temperature * 0.5,  # Lower temp for classification
            system=system_prompt,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": base64_image}},
                    {"type": "text", "text": user_message}
                ],
            }],
        )

        response_text = message.content[0].text

        # Parse JSON response
        try:
            # Clean up any markdown
            response_text = response_text.strip()
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
            response_text = response_text.strip()

            classification = json.loads(response_text)
        except json.JSONDecodeError:
            log(f"  Warning: Failed to parse classification JSON, using defaults")
            classification = {
                "shot_framing": "MS",
                "shot_label": "Medium Shot",
                "genre": "PRT",
                "genre_label": "Portrait",
                "genre_category": "people",
                "secondary_tags": [],
                "subject_count": 1,
                "has_text": False,
                "confidence": 0.5
            }

        return classification

    @classmethod
    def _stage4_detailed_analysis(
        cls,
        client,
        model: str,
        base64_image: str,
        classification: dict,
        attribute_schema: dict,
        content_detail: str,
        temperature: float,
        log
    ) -> dict:
        """Stage 4: Extract detailed attributes based on schema."""

        schema_text = format_attribute_schema_for_prompt(attribute_schema)

        # Build content detail guidance
        if content_detail == "explicit":
            detail_guidance = """CONTENT DETAIL: EXPLICIT (NSFW)
Include full body details: bust size/shape, cleavage visibility, body exposure areas,
clothing coverage and cutouts, sensuality indicators. Be specific and accurate."""
        elif content_detail == "detailed":
            detail_guidance = """CONTENT DETAIL: DETAILED
Include body proportions, clothing coverage details, exposure areas, and fit descriptions."""
        elif content_detail == "minimal":
            detail_guidance = """CONTENT DETAIL: MINIMAL
Include only basic build and clothing type/color."""
        else:
            detail_guidance = """CONTENT DETAIL: STANDARD
Include build, posture, clothing type, fit, color, and material."""

        system_prompt = f"""You are an expert visual analyst for Z-Image prompt generation.

Analyze this image and extract structured attributes.

CLASSIFICATION:
- Shot: {classification['shot_framing']} ({classification.get('shot_label', '')})
- Genre: {classification['genre']} ({classification.get('genre_label', '')})

{detail_guidance}

{schema_text}

RULES:
- Only describe VISIBLE elements
- Use concrete, objective language
- Be specific about colors, materials, textures
- No abstract adjectives (beautiful, mysterious)

Output ONLY valid JSON with the extracted attributes. Use the category names as keys."""

        message = client.messages.create(
            model=model,
            max_tokens=1500,
            temperature=temperature,
            system=system_prompt,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": base64_image}},
                    {"type": "text", "text": "Extract all visible attributes from this image according to the schema."}
                ],
            }],
        )

        response_text = message.content[0].text

        try:
            # Clean up any markdown
            response_text = response_text.strip()
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
            response_text = response_text.strip()
            if response_text.endswith("```"):
                response_text = response_text[:-3].strip()

            attributes = json.loads(response_text)
        except json.JSONDecodeError:
            log(f"  Warning: Failed to parse attributes JSON")
            attributes = {}

        return attributes

    @classmethod
    def _stage5_prompt_composition_template(
        cls,
        client,
        model: str,
        base64_image: str,
        classification: dict,
        attributes: dict,
        user_prompt: str,
        prompt_mode: str,
        focus_subject: bool,
        focus_environment: bool,
        focus_lighting: bool,
        focus_colors: bool,
        focus_mood: bool,
        include_text_quotes: bool,
        max_tokens: int,
        temperature: float,
        log
    ) -> str:
        """Stage 5: Compose prompt using LLM with template guidance."""

        # Calculate word targets based on max_tokens
        min_words = max(50, int(max_tokens * 0.6))
        max_words = int(max_tokens * 1.2)

        # Build focus instructions
        focus_parts = []
        if focus_subject:
            focus_parts.append("subject description (features, clothing, pose)")
        if focus_environment:
            focus_parts.append("environment/background")
        if focus_lighting:
            focus_parts.append("lighting (direction, quality, color)")
        if focus_colors:
            focus_parts.append("colors and materials")
        if focus_mood:
            focus_parts.append("mood/atmosphere")

        focus_instruction = "Focus on: " + ", ".join(focus_parts) if focus_parts else "Provide a balanced description."

        # Build prompt mode instruction
        if prompt_mode == "Image Only (ignore prompt)":
            mode_instruction = "Analyze the image directly. Ignore any user prompt."
            user_context = ""
        elif prompt_mode == "Prompt Guides Analysis":
            mode_instruction = "Analyze the image. Use the user prompt as guidance for emphasis."
            user_context = f"\nUser guidance: {user_prompt}" if user_prompt else ""
        elif prompt_mode == "Prompt First, Image Fills Gaps":
            mode_instruction = "Use the user prompt as primary structure. Fill gaps with image analysis."
            user_context = f"\nUser prompt (primary): {user_prompt}" if user_prompt else ""
        else:  # Prompt Dominates
            mode_instruction = "Use the user prompt as foundation. Add minimal visual details from image."
            user_context = f"\nUser prompt (override): {user_prompt}" if user_prompt else ""

        # Build text quote instruction
        text_instruction = 'Quote any visible text with "double quotes" for Z-Image text rendering.' if include_text_quotes else ""

        # Build attributes context
        attrs_text = json.dumps(attributes, indent=2) if attributes else "{}"

        system_prompt = f"""You are an expert prompt composer for Z-Image-Turbo.

Generate a flowing narrative prompt from the analysis data.

CLASSIFICATION:
- Shot: {classification['shot_framing']} ({classification.get('shot_label', '')})
- Genre: {classification['genre']} ({classification.get('genre_label', '')})

EXTRACTED ATTRIBUTES:
{attrs_text}

INSTRUCTIONS:
- {mode_instruction}
- {focus_instruction}
- {text_instruction}

OUTPUT FORMAT:
- Single flowing paragraph, {min_words}-{max_words} words
- Natural language narrative, NOT keyword lists
- NO meta-tags (8K, masterpiece, best quality)
- NO negative prompts or exclusions
- Every word should describe something VISIBLE

STRUCTURE:
1. Shot type and composition
2. Subject description (visible features only)
3. Clothing/objects (colors, materials, textures)
4. Environment/background (if applicable)
5. Lighting (direction, quality, color temperature)
6. Style hints (photography style)
{user_context}"""

        message = client.messages.create(
            model=model,
            max_tokens=max_tokens * 2,
            temperature=temperature,
            system=system_prompt,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": base64_image}},
                    {"type": "text", "text": "Generate the Z-Image narrative prompt based on this image and the analysis."}
                ],
            }],
        )

        return message.content[0].text

    @classmethod
    def _stage5_prompt_composition_llm(
        cls,
        client,
        model: str,
        base64_image: str,
        classification: dict,
        attributes: dict,
        user_prompt: str,
        prompt_mode: str,
        focus_subject: bool,
        focus_environment: bool,
        focus_lighting: bool,
        focus_colors: bool,
        focus_mood: bool,
        include_text_quotes: bool,
        max_tokens: int,
        temperature: float,
        log
    ) -> str:
        """Stage 5 (Deep mode): Refined prompt composition with additional LLM call."""
        # For deep mode, we do a more refined composition
        # First generate a draft, then refine it

        # Generate initial draft
        draft = cls._stage5_prompt_composition_template(
            client, model, base64_image,
            classification, attributes,
            user_prompt, prompt_mode,
            focus_subject, focus_environment, focus_lighting,
            focus_colors, focus_mood, include_text_quotes,
            max_tokens, temperature * 0.8, log
        )

        # Refine the draft
        refine_prompt = f"""Review and refine this Z-Image prompt for optimal quality.

DRAFT:
{draft}

REFINEMENT RULES:
1. Ensure natural flowing language (not keyword lists)
2. Remove any meta-tags (8K, masterpiece, best quality)
3. Ensure every word describes something visible
4. Check for contradictions or redundancy
5. Optimize word choice for Z-Image's natural language understanding
6. Keep within {int(max_tokens * 0.8)}-{max_tokens} words

Output ONLY the refined prompt, nothing else."""

        message = client.messages.create(
            model=model,
            max_tokens=max_tokens * 2,
            temperature=temperature * 0.5,
            system="You are a prompt refinement expert for Z-Image-Turbo.",
            messages=[{
                "role": "user",
                "content": refine_prompt
            }],
        )

        return message.content[0].text
