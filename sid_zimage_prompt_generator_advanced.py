"""
SID_ZImagePromptGenerator_Advanced Node
Agentic multi-stage image analysis for Z-Image prompt generation.

Advanced version that requires an external LLM provider node connection.
Use SID_Anthropic_LLM, SID_OpenAI_LLM, or other provider nodes.

This node analyzes an input image using vision LLM capabilities and generates
a Z-Image compatible narrative prompt through a multi-stage agentic pipeline.
"""

import base64
import io
import json
import random
import time
from datetime import datetime
from typing import Any

import numpy as np
from PIL import Image
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io as comfy_io
import comfy.utils

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
from .llm_providers.llm_model_type import LLMModelConfig
from .prompt_templates import get_prompt_template_for_provider, BasePromptTemplate

# Create custom LLM_MODEL type for ComfyUI (must match provider nodes)
LLM_MODEL_Type = comfy_io.Custom("LLM_MODEL")


def detect_and_clean_repetition(text: str, max_repeat_count: int = 3) -> tuple[str, bool]:
    """
    Detect and clean repetitive patterns in LLM output.

    Returns:
        tuple: (cleaned_text, had_repetition)
    """
    if not text:
        return text, False

    had_repetition = False

    # Pattern 1: Detect repeated key-value pairs like "shape": 0.99, "shape": 0.99
    import re

    # Find repeated JSON-like patterns (key: value repeated)
    repeated_json_pattern = r'("[^"]+"\s*:\s*[^,\n]+,?\s*)\1{2,}'
    if re.search(repeated_json_pattern, text):
        had_repetition = True
        # Keep only one instance
        text = re.sub(repeated_json_pattern, r'\1', text)

    # Pattern 2: Detect repeated phrases/sentences (composition stage issue)
    # Split into sentences and detect repetition
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if len(sentences) > 3:
        seen = set()
        unique_sentences = []
        for sentence in sentences:
            # Normalize for comparison
            normalized = sentence.strip().lower()[:50]  # First 50 chars
            if normalized and normalized not in seen:
                seen.add(normalized)
                unique_sentences.append(sentence)
            elif normalized:
                had_repetition = True
        if had_repetition:
            text = ' '.join(unique_sentences)

    # Pattern 3: Detect word-level repetition loops
    words = text.split()
    if len(words) > 20:
        # Check if last 10 words repeat
        window_size = 10
        for i in range(len(words) - window_size * 2):
            window1 = ' '.join(words[i:i+window_size])
            window2 = ' '.join(words[i+window_size:i+window_size*2])
            if window1 == window2:
                had_repetition = True
                # Truncate at repetition point
                text = ' '.join(words[:i+window_size])
                break

    # Pattern 4: Detect "style": "white" repeated (specific pattern from tests)
    if text.count('"style"') > 5:
        had_repetition = True
        # Extract first meaningful part before repetition
        first_style_pos = text.find('"style"')
        if first_style_pos > 0:
            # Try to find valid JSON end before repetition
            bracket_pos = text.rfind('}', 0, first_style_pos + 50)
            if bracket_pos > 0:
                text = text[:bracket_pos + 1]

    return text.strip(), had_repetition


def extract_json_from_text(text: str, include_raw_fallback: bool = True) -> dict:
    """
    Extract JSON from text that may contain markdown code blocks or other formatting.
    Handles truncated JSON by attempting repair.
    Falls back to extracting key-value pairs from prose if JSON parsing fails.

    Args:
        text: The raw LLM response text
        include_raw_fallback: If True, includes _raw_response in result when JSON parsing fails

    Returns:
        dict: Extracted JSON or fallback dictionary with _raw_response
    """
    import re

    if not text:
        return {}

    # Try to find JSON in markdown code blocks
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
    if json_match:
        json_text = json_match.group(1)
        result = _try_parse_json(json_text)
        if result and len(result) > 0:
            return result

    # Try to find raw JSON (starts with { ends with } or truncated)
    json_match = re.search(r'\{[\s\S]*', text)
    if json_match:
        json_text = json_match.group(0)
        result = _try_parse_json(json_text)
        if result and len(result) > 0:
            return result

    # Try aggressive key-value extraction from partial JSON
    partial_result = _extract_key_values_from_partial_json(text)
    if partial_result and len(partial_result) > 0:
        return partial_result

    # Fallback: Extract key information from prose
    prose_result = _extract_from_prose(text)

    # Always include raw response as ultimate fallback
    if include_raw_fallback and (not prose_result or len(prose_result) == 0):
        return {"_raw_response": text[:4000], "_parse_failed": True}

    if include_raw_fallback and prose_result:
        prose_result["_raw_response"] = text[:2000]  # Include truncated raw for reference

    return prose_result


def _try_parse_json(json_text: str) -> dict:
    """Try to parse JSON, with repair attempts for truncated responses."""
    import re

    # Clean up escaped underscores from some models
    json_text = json_text.replace('\\_', '_')

    # First try: direct parse
    try:
        return json.loads(json_text)
    except json.JSONDecodeError:
        pass

    # Second try: close truncated JSON
    # Count open braces and brackets
    open_braces = json_text.count('{') - json_text.count('}')
    open_brackets = json_text.count('[') - json_text.count(']')

    # Remove any trailing incomplete values (like 'appears to"')
    # Find last complete key-value pair
    repaired = json_text.rstrip()
    if repaired.endswith(','):
        repaired = repaired[:-1]

    # Remove incomplete string at end
    if repaired.count('"') % 2 == 1:
        # Odd number of quotes - find and remove last incomplete string
        last_quote = repaired.rfind('"')
        if last_quote > 0:
            # Look for the second-to-last quote
            second_last = repaired.rfind('"', 0, last_quote)
            if second_last > 0:
                # Check if this looks like a truncated value
                between = repaired[second_last:last_quote+1]
                if ':' not in between:
                    repaired = repaired[:second_last]

    # Remove trailing incomplete key-value
    repaired = re.sub(r',\s*"[^"]*"\s*:\s*"?[^",}]*$', '', repaired)
    repaired = re.sub(r',\s*"[^"]*"\s*$', '', repaired)

    # Close brackets and braces
    repaired = repaired.rstrip()
    if repaired.endswith(','):
        repaired = repaired[:-1]

    for _ in range(open_brackets):
        repaired += ']'
    for _ in range(open_braces):
        repaired += '}'

    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    # Third try: extract nested objects that are complete
    complete_objects = re.findall(r'"(\w+)"\s*:\s*(\{[^{}]*\})', json_text)
    if complete_objects:
        result = {}
        for key, value in complete_objects:
            try:
                result[key] = json.loads(value)
            except json.JSONDecodeError:
                pass
        if result:
            return result

    return {}


def _extract_key_values_from_partial_json(text: str) -> dict:
    """
    Aggressively extract key-value pairs from partial/truncated JSON.
    Works even when JSON is badly malformed or truncated mid-value.
    """
    import re

    result = {}

    # Pattern 1: Extract complete "key": "value" pairs
    string_pairs = re.findall(r'"([^"]+)"\s*:\s*"([^"]*)"', text)
    for key, value in string_pairs:
        if key and value and not key.startswith('_'):
            # Skip internal keys and empty values
            clean_key = key.lower().replace(' ', '_')
            result[clean_key] = value

    # Pattern 2: Extract "key": number pairs
    number_pairs = re.findall(r'"([^"]+)"\s*:\s*(-?\d+\.?\d*)', text)
    for key, value in number_pairs:
        if key and not key.startswith('_'):
            clean_key = key.lower().replace(' ', '_')
            try:
                result[clean_key] = float(value) if '.' in value else int(value)
            except ValueError:
                pass

    # Pattern 3: Extract "key": true/false/null pairs
    bool_pairs = re.findall(r'"([^"]+)"\s*:\s*(true|false|null)', text, re.IGNORECASE)
    for key, value in bool_pairs:
        if key and not key.startswith('_'):
            clean_key = key.lower().replace(' ', '_')
            result[clean_key] = value.lower() == 'true' if value.lower() != 'null' else None

    # Pattern 4: Extract nested objects that are complete
    nested_objects = re.findall(r'"(\w+)"\s*:\s*(\{[^{}]*\})', text)
    for key, value in nested_objects:
        try:
            result[key] = json.loads(value)
        except json.JSONDecodeError:
            pass

    # Pattern 5: Look for common analysis fields with specific patterns
    # Eye color
    eye_match = re.search(r'(?:eye_color|eyes?)["\s:]+([^",}\n]+)', text, re.IGNORECASE)
    if eye_match and 'eye_color' not in result:
        result['eye_color'] = eye_match.group(1).strip().strip('"')

    # Hair color
    hair_match = re.search(r'(?:hair_color|hair)["\s:]+([^",}\n]+)', text, re.IGNORECASE)
    if hair_match and 'hair_color' not in result:
        result['hair_color'] = hair_match.group(1).strip().strip('"')

    # Skin tone
    skin_match = re.search(r'(?:skin_tone|skin)["\s:]+([^",}\n]+)', text, re.IGNORECASE)
    if skin_match and 'skin_tone' not in result:
        result['skin_tone'] = skin_match.group(1).strip().strip('"')

    # Ethnicity
    ethnicity_match = re.search(r'(?:ethnicity|heritage|background)["\s:]+([^",}\n]+)', text, re.IGNORECASE)
    if ethnicity_match and 'ethnicity' not in result:
        result['ethnicity'] = ethnicity_match.group(1).strip().strip('"')

    return result


def _extract_from_prose(text: str) -> dict:
    """Extract key information from prose text when JSON parsing fails."""
    import re

    result = {}

    # Common patterns to extract
    patterns = {
        'gender': r'\b(female|male|woman|man)\b',
        'age_range': r'\b(young adult|adult|teen|child|middle-aged|elderly)\b',
        'hair_color': r'(?:hair[^.]*?)(black|dark brown|brown|chestnut|auburn|blonde|red|gray|white)',
        'eye_color': r'(?:eyes?[^.]*?)(dark brown|brown|blue|green|hazel|gray|black)',
        'ethnicity': r'\b(East Asian|South Asian|Asian|Caucasian|African|Hispanic|Latino|Middle Eastern|Mediterranean|European)\b',
        'skin_tone': r'(?:skin[^.]*?)(fair|light|medium|olive|tan|dark|brown|warm|cool)',
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            result[key] = match.group(1).lower() if len(match.groups()) == 1 else match.group(1).lower()

    return result


class SID_ZImagePromptGenerator_Advanced(comfy_io.ComfyNode):
    """
    Advanced Z-Image prompt generator with external LLM provider.

    Requires connection to an LLM provider node (e.g., SID_Anthropic_LLM).
    Use this for flexible provider selection and centralized LLM configuration.

    Uses a 6-stage pipeline:
    1. Classification - Detect shot framing and photography genre
    2. Metadata - Extract image dimensions and properties
    3. Attribute Mapping - Select relevant attributes for the scene
    4. Detailed Analysis - LLM extraction of structured attributes
    5. Prompt Composition - Generate flowing narrative prompt
    6. Z-Image Optimization - Provide recommendations for Z-Image
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
            node_id="SID_ZImagePromptGenerator_Advanced",
            display_name="SID Z-Image Prompt Generator (Advanced)",
            category="SID Photography Toolkit/Z-Image",
            description="Advanced Z-Image prompt generator with external LLM provider connection",
            inputs=[
                # Image input
                comfy_io.Image.Input(
                    "image",
                    tooltip="Input image to analyze"
                ),

                # Required external LLM model (from provider nodes)
                LLM_MODEL_Type.Input(
                    "llm_model",
                    tooltip="Connect LLM provider node (e.g., SID_Anthropic_LLM)"
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

                # Generation Settings (max_tokens and temperature come from llm_model)
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
        llm_model: LLMModelConfig,  # Required external LLM config
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
        log("SID Z-Image Prompt Generator (Advanced) - Debug Log")
        log("=" * 60)
        log(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log(f"Mode: {detail_level} ({get_detail_level_config(detail_level).get('llm_calls', 2)} LLM calls)")
        log("")

        # Get image dimensions early for all return paths
        if len(image.shape) == 4:
            img_height, img_width = image.shape[1], image.shape[2]
        else:
            img_height, img_width = image.shape[0], image.shape[1]

        # Use external LLM configuration from connected provider node
        actual_provider = llm_model.provider.capitalize()
        model = llm_model.model
        api_key = llm_model.api_key
        api_url = llm_model.api_url
        max_tokens = llm_model.max_tokens
        temperature = llm_model.temperature
        log(f"[LLM Provider] {actual_provider}")
        log(f"  Model: {model}")
        log(f"  Max Tokens: {max_tokens}")
        log(f"  Temperature: {temperature}")

        # Handle seed mode
        actual_seed = cls._process_seed(seed, seed_mode)
        log(f"Seed: {actual_seed} (mode: {seed_mode})")
        log(f"Cache: {'enabled' if cache_prompt else 'disabled'}")
        log(f"Provider: {actual_provider}")
        log(f"Model: {model}")

        # Validate API key (not required for Ollama or local endpoints)
        is_local = "localhost" in api_url or "127.0.0.1" in api_url
        if actual_provider not in ["Ollama", "Gguf"] and not is_local and (not api_key or api_key.strip() == ""):
            if actual_provider == "Anthropic":
                error_msg = "ERROR: Anthropic API key is required. Get one at https://console.anthropic.com/"
            elif actual_provider == "Openai":
                error_msg = "ERROR: API key is required for remote endpoints."
            else:
                error_msg = "ERROR: Grok API key is required. Get one at https://console.x.ai/"
            return comfy_io.NodeOutput(image, error_msg, img_width, img_height, "{}", "{}", error_msg)

        # Import provider libraries
        if actual_provider == "Anthropic":
            try:
                import anthropic
            except ImportError:
                error_msg = "ERROR: anthropic library not installed. Run: pip install anthropic"
                return comfy_io.NodeOutput(image, error_msg, img_width, img_height, "{}", "{}", error_msg)
        elif actual_provider in ["Openai", "Grok"]:
            try:
                import openai
            except ImportError:
                error_msg = "ERROR: openai library not installed. Run: pip install openai"
                return comfy_io.NodeOutput(image, error_msg, img_width, img_height, "{}", "{}", error_msg)
        elif actual_provider == "Gguf":
            try:
                from .llm_providers.sid_gguf_llm import LocalGGUFClient
            except ImportError:
                error_msg = "ERROR: llama-cpp-python not installed. Run: pip install llama-cpp-python"
                return comfy_io.NodeOutput(image, error_msg, img_width, img_height, "{}", "{}", error_msg)
        else:  # Ollama
            try:
                import requests
            except ImportError:
                error_msg = "ERROR: requests library not installed. Run: pip install requests"
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
            # Initialize client based on provider
            if actual_provider == "Anthropic":
                import anthropic
                client = anthropic.Anthropic(api_key=api_key.strip())
            elif actual_provider == "Openai":
                import openai
                openai_url = api_url.strip() if api_url.strip() else "https://api.openai.com/v1"
                client = openai.OpenAI(api_key=api_key.strip(), base_url=openai_url)
            elif actual_provider == "Grok":
                import openai
                grok_url = api_url.strip() if api_url.strip() else "https://api.x.ai/v1"
                client = openai.OpenAI(api_key=api_key.strip(), base_url=grok_url)
            elif actual_provider == "Gguf":
                # Local GGUF model - create LocalGGUFClient
                from .llm_providers.sid_gguf_llm import LocalGGUFClient
                extra = llm_model.extra_params
                log(f"[GGUF] Loading local model...")
                client = LocalGGUFClient(
                    model_path=extra.get("model_path"),
                    mmproj_path=extra.get("mmproj_path"),
                    chat_format=extra.get("chat_format", "llava-1-5"),
                    n_ctx=extra.get("n_ctx", 2048),
                    n_gpu_layers=extra.get("n_gpu_layers", -1),
                )
                log(f"[GGUF] Model loaded successfully")
            else:  # Ollama
                import requests
                ollama_url = api_url.strip() if api_url.strip() else "http://localhost:11434"
                client = {"url": ollama_url, "session": requests.Session()}

            # Convert image to base64
            base64_image = cls._image_to_base64(image)

            # Initialize progress bar (6 stages)
            pbar = comfy.utils.ProgressBar(6)

            # Get the appropriate prompt template for this provider
            prompt_template = get_prompt_template_for_provider(actual_provider, model)
            log(f"[TEMPLATE] Using: {prompt_template.name}")
            log("")

            # Track all LLM interactions for structured output
            llm_interactions = []

            # ===== STAGE 1: CLASSIFICATION =====
            log("[STAGE 1] Classification (LLM Call #1)")
            stage1_start = time.time()

            classification = cls._stage1_classification(
                client, model, base64_image, focus_override, temperature, log, prompt_template,
                interactions=llm_interactions
            )

            log(f"  Shot Framing: {classification['shot_framing']} ({classification.get('shot_label', '')}) - {int(classification.get('confidence', 0) * 100)}% confidence")
            log(f"  Genre: {classification['genre']} ({classification.get('genre_label', '')})")
            log(f"  Secondary: {', '.join(classification.get('secondary_tags', []))}")
            log(f"  Subject Count: {classification.get('subject_count', 0)}")
            log(f"  Has Text: {classification.get('has_text', False)}")
            log(f"  Duration: {time.time() - stage1_start:.1f}s")
            log("")
            pbar.update(1)

            # ===== STAGE 2: METADATA =====
            log("[STAGE 2] Metadata Extraction")

            image_meta = get_image_metadata(image)
            log(f"  Dimensions: {image_meta['width']} x {image_meta['height']} px")
            log(f"  Aspect Ratio: {image_meta['aspect_ratio']} ({image_meta['aspect_decimal']})")
            log(f"  Orientation: {image_meta['orientation']}")
            log("")
            pbar.update(1)

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
            pbar.update(1)

            # ===== STAGE 4: DETAILED ANALYSIS =====
            if detail_level in ["Standard", "Deep"]:
                log("[STAGE 4] Detailed Analysis (LLM Call #2)")
                stage4_start = time.time()

                attributes = cls._stage4_detailed_analysis(
                    client, model, base64_image,
                    classification, attribute_schema,
                    content_detail, temperature, log, prompt_template,
                    detail_level=detail_level,
                    interactions=llm_interactions
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
            pbar.update(1)

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
                    max_tokens, temperature, log, prompt_template,
                    interactions=llm_interactions
                )
            else:
                # Template-based composition
                prompt = cls._stage5_prompt_composition_template(
                    client, model, base64_image,
                    classification, attributes,
                    user_prompt, prompt_mode,
                    focus_subject, focus_environment, focus_lighting,
                    focus_colors, focus_mood, include_text_quotes,
                    max_tokens, temperature, log, prompt_template,
                    interactions=llm_interactions
                )

            # Clean the output
            prompt = clean_zimage_output(prompt, max_tokens * 4)  # ~4 chars per token

            word_count = len(prompt.split())
            estimated_tokens = int(word_count * 1.3)
            log(f"  Word count: {word_count}")
            log(f"  Estimated tokens: {estimated_tokens}")
            log(f"  Duration: {time.time() - stage5_start:.1f}s")
            log("")
            pbar.update(1)

            # ===== STAGE 6: Z-IMAGE RECOMMENDATIONS =====
            log("[STAGE 6] Z-Image Recommendations")

            zimage_recs = get_zimage_recommendations(image_meta)
            if zimage_recs["resize_needed"]:
                log(f"  Resize: {image_meta['width']}x{image_meta['height']} -> {zimage_recs['optimal_resolution'][0]}x{zimage_recs['optimal_resolution'][1]} ({zimage_recs['resize_method']} {zimage_recs['resize_direction']})")
            else:
                log(f"  Resize: Not needed (already optimal)")
            log(f"  Quality estimate: {zimage_recs['quality_estimate'].upper()}")
            log("")
            pbar.update(1)

            # Build structured data output with organized categories
            structured_data = cls._build_structured_output(
                classification=classification,
                attributes=attributes,
                prompt=prompt,
                word_count=word_count,
                estimated_tokens=estimated_tokens,
                # Model info
                provider=actual_provider,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                template_name=prompt_template.name,
                # Image info
                image_meta=image_meta,
                zimage_recs=zimage_recs,
                # Settings
                detail_level=detail_level,
                content_detail=content_detail,
                focus_override=focus_override,
                prompt_mode=prompt_mode,
                focus_options={
                    "subject": focus_subject,
                    "environment": focus_environment,
                    "lighting": focus_lighting,
                    "colors": focus_colors,
                    "mood": focus_mood,
                },
                seed=actual_seed,
                total_time=time.time() - start_time,
                # LLM interactions
                llm_interactions=llm_interactions,
            )

            # Build metadata output (simplified - main data now in structured_data)
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

        except Exception as e:
            error_type = type(e).__name__
            error_msg = f"API Error ({error_type}): {str(e)}"
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
    def _call_vision_llm(
        cls,
        client,
        model: str,
        base64_image: str,
        system_prompt: str,
        user_message: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        """
        Unified vision LLM call that handles Anthropic, OpenAI, Grok, and Ollama providers.
        Returns the text response from the model.
        """
        # Check if client is Ollama (dict with url/session)
        if isinstance(client, dict) and "url" in client:
            # Ollama API call
            ollama_model = model.replace("ollama/", "") if model.startswith("ollama/") else model
            url = f"{client['url']}/api/chat"

            payload = {
                "model": ollama_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": user_message,
                        "images": [base64_image]
                    }
                ],
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                }
            }

            response = client["session"].post(url, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            return result.get("message", {}).get("content", "")

        elif hasattr(client, 'chat') and hasattr(client.chat, 'completions'):
            # OpenAI-compatible API call (OpenAI, Grok, Together AI, etc.)
            response = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                            },
                            {"type": "text", "text": user_message}
                        ]
                    }
                ]
            )
            return response.choices[0].message.content

        else:
            # Anthropic API call
            message = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": base64_image}},
                        {"type": "text", "text": user_message}
                    ],
                }],
            )
            return message.content[0].text

    @classmethod
    def _call_text_llm(
        cls,
        client,
        model: str,
        system_prompt: str,
        user_message: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        """
        Unified text-only LLM call (no image) for refinement stages.
        """
        # Check if client is Ollama (dict with url/session)
        if isinstance(client, dict) and "url" in client:
            ollama_model = model.replace("ollama/", "") if model.startswith("ollama/") else model
            url = f"{client['url']}/api/chat"

            payload = {
                "model": ollama_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                }
            }

            response = client["session"].post(url, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            return result.get("message", {}).get("content", "")

        elif hasattr(client, 'chat') and hasattr(client.chat, 'completions'):
            # OpenAI-compatible API call (OpenAI, Grok, Together AI, etc.)
            response = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ]
            )
            return response.choices[0].message.content

        else:
            # Anthropic API call
            message = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )
            return message.content[0].text

    @classmethod
    def _stage1_classification(
        cls,
        client,
        model: str,
        base64_image: str,
        focus_override: str,
        temperature: float,
        log,
        prompt_template: BasePromptTemplate,
        interactions: list = None
    ) -> dict:
        """Stage 1: Classify the image using template-based prompts."""

        # Get shot framings and genres for reference
        shot_framings = get_shot_framings()
        genres = get_photography_genres()

        # Use template to build prompts (SOLID: template handles formatting)
        system_prompt, user_message = prompt_template.build_classification_prompt(
            shot_framings, genres, focus_override
        )

        # Get template-specific settings
        max_tokens = prompt_template.get_classification_max_tokens()
        temp_modifier = prompt_template.get_temperature_modifier("classification")
        actual_temp = temperature * temp_modifier

        response_text = cls._call_vision_llm(
            client, model, base64_image,
            system_prompt, user_message,
            max_tokens=max_tokens,
            temperature=actual_temp
        )

        # Track interaction
        if interactions is not None:
            interactions.append({
                "stage": "classification",
                "stage_number": 1,
                "type": "vision",
                "request": {
                    "system_prompt": system_prompt,
                    "user_message": user_message,
                    "max_tokens": max_tokens,
                    "temperature": actual_temp,
                },
                "response": {
                    "raw_text": response_text,
                },
            })

        # Clean repetition if detected
        response_text, had_repetition = detect_and_clean_repetition(response_text)
        if had_repetition:
            log("  Warning: Detected and cleaned repetitive output from LLM")

        # Parse JSON response using improved extraction
        classification = extract_json_from_text(response_text)

        # Validate and fill defaults
        defaults = {
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

        if not classification or "shot_framing" not in classification:
            log("  Warning: Failed to parse classification JSON, using defaults")
            classification = defaults
        else:
            # Fill in any missing fields
            for key, value in defaults.items():
                if key not in classification:
                    classification[key] = value

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
        log,
        prompt_template: BasePromptTemplate,
        detail_level: str = "Standard",
        interactions: list = None
    ) -> dict:
        """Stage 4: Extract detailed attributes using template-based prompts.

        For Deep mode with local GGUF models (multi-iteration templates):
        - Runs multiple focused analysis stages (subject, environment, lighting, style)
        - Each stage analyzes ONE aspect thoroughly
        - Results are consolidated into a single attribute dict

        This approach achieves similar quality to cloud models but with more iterations.
        """

        # Check if we should use multi-iteration deep mode
        uses_multi_iteration = (
            detail_level == "Deep"
            and hasattr(prompt_template, 'uses_multi_iteration_deep_mode')
            and prompt_template.uses_multi_iteration_deep_mode
        )

        if uses_multi_iteration:
            # Multi-iteration Deep mode for local models
            log(f"  [Multi-iteration mode] Running focused analysis stages...")

            stages = prompt_template.get_deep_mode_stages()
            stage_results = {}
            call_number = 2  # Stage 4 starts at call #2

            for i, stage in enumerate(stages, 1):
                log(f"    Stage {i}/{len(stages)}: {stage.capitalize()}")

                # Build focused prompt for this stage
                system_prompt, user_message = prompt_template.build_focused_analysis_prompt(
                    stage, classification, content_detail, stage_results
                )

                # Get stage-specific settings
                max_tokens = prompt_template.get_focused_analysis_max_tokens()
                temp_modifier = prompt_template.get_temperature_modifier("focused_analysis")
                actual_temp = temperature * temp_modifier

                response_text = cls._call_vision_llm(
                    client, model, base64_image,
                    system_prompt, user_message,
                    max_tokens=max_tokens,
                    temperature=actual_temp
                )

                # Track interaction
                if interactions is not None:
                    interactions.append({
                        "stage": f"analysis_focused_{stage}",
                        "stage_number": 4,
                        "substage": f"{i}/{len(stages)} - {stage}",
                        "type": "vision",
                        "request": {
                            "system_prompt": system_prompt,
                            "user_message": user_message,
                            "max_tokens": max_tokens,
                            "temperature": actual_temp,
                        },
                        "response": {
                            "raw_text": response_text,
                        },
                    })

                # Clean repetition if detected
                response_text, had_repetition = detect_and_clean_repetition(response_text)
                if had_repetition:
                    log(f"      Warning: Detected and cleaned repetitive output")

                # Parse JSON response
                stage_data = extract_json_from_text(response_text)
                if stage_data:
                    stage_results[stage] = stage_data
                    log(f"      Extracted: {list(stage_data.keys())}")
                else:
                    log(f"      Warning: Failed to parse {stage} JSON")
                    stage_results[stage] = {}

            # Consolidate results from all stages
            log(f"    Consolidating {len(stages)} analysis stages...")
            system_prompt, user_message = prompt_template.build_consolidation_prompt(
                classification, stage_results
            )

            temp_modifier = prompt_template.get_temperature_modifier("consolidation")
            actual_temp = temperature * temp_modifier
            max_tokens = prompt_template.get_analysis_max_tokens()

            response_text = cls._call_text_llm(
                client, model,
                system_prompt, user_message,
                max_tokens=max_tokens,
                temperature=actual_temp
            )

            # Track consolidation interaction
            if interactions is not None:
                interactions.append({
                    "stage": "analysis_consolidation",
                    "stage_number": 4,
                    "substage": "consolidation",
                    "type": "text",
                    "request": {
                        "system_prompt": system_prompt,
                        "user_message": user_message,
                        "max_tokens": max_tokens,
                        "temperature": actual_temp,
                    },
                    "response": {
                        "raw_text": response_text,
                    },
                })

            # Clean repetition if detected
            response_text, had_repetition = detect_and_clean_repetition(response_text)
            if had_repetition:
                log(f"    Warning: Detected and cleaned repetitive output in consolidation")

            log(f"    Consolidation response length: {len(response_text)} chars")

            attributes = extract_json_from_text(response_text, include_raw_fallback=True)
            if not attributes or len(attributes) == 0:
                log(f"    Warning: Failed to parse consolidated JSON, using raw stages")
                # Flatten stage results as fallback
                attributes = {}
                for stage_data in stage_results.values():
                    if isinstance(stage_data, dict):
                        attributes.update(stage_data)
                # Include raw response as additional context
                if not attributes:
                    attributes = {"_raw_response": response_text[:4000], "_parse_failed": True}
            else:
                real_attrs = {k: v for k, v in attributes.items() if not k.startswith('_')}
                log(f"    Consolidated {len(real_attrs)} attribute fields")

            return attributes

        else:
            # Standard single-call analysis (Claude, OpenAI, Grok, or non-Deep mode)
            system_prompt, user_message = prompt_template.build_analysis_prompt(
                classification, attribute_schema, content_detail
            )

            max_tokens = prompt_template.get_analysis_max_tokens()
            temp_modifier = prompt_template.get_temperature_modifier("analysis")
            actual_temp = temperature * temp_modifier

            response_text = cls._call_vision_llm(
                client, model, base64_image,
                system_prompt, user_message,
                max_tokens=max_tokens,
                temperature=actual_temp
            )

            # Track interaction
            if interactions is not None:
                interactions.append({
                    "stage": "analysis",
                    "stage_number": 4,
                    "type": "vision",
                    "request": {
                        "system_prompt": system_prompt,
                        "user_message": user_message,
                        "max_tokens": max_tokens,
                        "temperature": actual_temp,
                    },
                    "response": {
                        "raw_text": response_text,
                    },
                })

            # Clean repetition if detected
            response_text, had_repetition = detect_and_clean_repetition(response_text)
            if had_repetition:
                log(f"  Warning: Detected and cleaned repetitive output in analysis")

            # Log response length for debugging truncation issues
            log(f"  Analysis response length: {len(response_text)} chars")

            attributes = extract_json_from_text(response_text, include_raw_fallback=True)

            # Detailed logging for debugging
            if not attributes or len(attributes) == 0:
                log(f"  ERROR: Failed to extract ANY attributes from analysis")
                log(f"  Response preview: {response_text[:200]}...")
                attributes = {"_raw_response": response_text[:4000], "_parse_failed": True}
            elif attributes.get('_parse_failed'):
                log(f"  Warning: JSON parsing failed, using raw text fallback")
            elif attributes.get('_raw_response') and len(attributes) < 5:
                log(f"  Warning: Sparse attributes ({len(attributes)-1} fields), raw text included as fallback")
            else:
                # Filter out internal keys for counting
                real_attrs = {k: v for k, v in attributes.items() if not k.startswith('_')}
                log(f"  Extracted {len(real_attrs)} attribute fields successfully")

            return attributes

    @staticmethod
    def _clean_json_response(response_text: str) -> str:
        """Clean up JSON response from LLM (remove markdown formatting)."""
        response_text = response_text.strip()
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
        response_text = response_text.strip()
        if response_text.endswith("```"):
            response_text = response_text[:-3].strip()
        return response_text

    @classmethod
    def _build_structured_output(
        cls,
        classification: dict,
        attributes: dict,
        prompt: str,
        word_count: int,
        estimated_tokens: int,
        # Model info
        provider: str = "",
        model: str = "",
        max_tokens: int = 0,
        temperature: float = 0.0,
        template_name: str = "",
        # Image info
        image_meta: dict = None,
        zimage_recs: dict = None,
        # Settings
        detail_level: str = "",
        content_detail: str = "",
        focus_override: str = "",
        prompt_mode: str = "",
        focus_options: dict = None,
        seed: int = 0,
        total_time: float = 0.0,
        llm_interactions: list = None,
    ) -> dict:
        """
        Build a well-organized, categorized structured output JSON.

        Organizes all extracted data into clear categories for easy consumption.
        Includes model data, image metadata, settings, and all gathered information.
        """
        image_meta = image_meta or {}
        zimage_recs = zimage_recs or {}
        focus_options = focus_options or {}
        llm_interactions = llm_interactions or []

        # Helper to safely get nested values
        def get_nested(d: dict, *keys, default=None):
            for key in keys:
                if isinstance(d, dict):
                    d = d.get(key, default)
                else:
                    return default
            return d if d is not None else default

        # Extract subject data from attributes
        subject_data = attributes.get("subject", {})
        pose_data = attributes.get("pose", {})
        clothing_data = attributes.get("clothing", {})
        environment_data = attributes.get("environment", {})
        lighting_data = attributes.get("lighting", {})
        colors_data = attributes.get("colors", {})
        style_data = attributes.get("style", {})

        # Build organized structure
        structured = {
            # ===== METADATA SECTION =====
            "metadata": {
                "generator": "SID Z-Image Prompt Generator Advanced",
                "version": "1.0.0",
                "timestamp": datetime.now().isoformat(),
                "processing_time_seconds": round(total_time, 2),
            },

            # ===== MODEL CONFIGURATION =====
            "model": {
                "provider": provider,
                "model_name": model,
                "template": template_name,
                "settings": {
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "seed": seed,
                },
            },

            # ===== ANALYSIS SETTINGS =====
            "settings": {
                "detail_level": detail_level,
                "content_detail": content_detail,
                "focus_override": focus_override,
                "prompt_mode": prompt_mode,
                "focus_areas": focus_options,
            },

            # ===== IMAGE INFORMATION =====
            "image": {
                "dimensions": {
                    "width": image_meta.get("width", 0),
                    "height": image_meta.get("height", 0),
                    "megapixels": image_meta.get("megapixels", 0),
                },
                "aspect_ratio": {
                    "ratio": image_meta.get("aspect_ratio", ""),
                    "decimal": image_meta.get("aspect_decimal", 0),
                    "orientation": image_meta.get("orientation", ""),
                },
                "color_info": {
                    "mode": image_meta.get("mode", ""),
                    "has_alpha": image_meta.get("has_alpha", False),
                    "bit_depth": image_meta.get("bit_depth", 0),
                },
            },

            # ===== Z-IMAGE RECOMMENDATIONS =====
            "zimage_recommendations": {
                "optimal_resolution": zimage_recs.get("optimal_resolution", []),
                "resize_needed": zimage_recs.get("resize_needed", False),
                "resize_method": zimage_recs.get("resize_method", ""),
                "resize_direction": zimage_recs.get("resize_direction", ""),
                "quality_estimate": zimage_recs.get("quality_estimate", ""),
                "recommended_settings": zimage_recs.get("recommended_settings", {}),
            },

            # ===== IMAGE CLASSIFICATION =====
            "classification": {
                "shot_framing": {
                    "code": classification.get("shot_framing", ""),
                    "label": classification.get("shot_label", ""),
                },
                "genre": {
                    "code": classification.get("genre", ""),
                    "label": classification.get("genre_label", ""),
                    "category": classification.get("genre_category", ""),
                },
                "secondary_tags": classification.get("secondary_tags", []),
                "subject_count": classification.get("subject_count", 1),
                "has_text": classification.get("has_text", False),
                "confidence": classification.get("confidence", 0.0),
                "visual_metrics": classification.get("visual_metrics", {}),
            },

            # ===== SUBJECT ANALYSIS =====
            "subject": {
                "ethnicity": {
                    "heritage": get_nested(subject_data, "ethnicity", "apparent_heritage", default=""),
                    "distinctive_features": get_nested(subject_data, "ethnicity", "distinctive_features", default=[]),
                },
                "skin": {
                    "tone": get_nested(subject_data, "skin", "tone", default=""),
                    "undertone": get_nested(subject_data, "skin", "undertone", default=""),
                    "surface": get_nested(subject_data, "skin", "surface", default=""),
                    "details": get_nested(subject_data, "skin", "details", default=""),
                },
                "face": {
                    "shape": subject_data.get("face_shape", ""),
                    "head_direction": subject_data.get("head_direction", {}),
                    "expression": subject_data.get("expression", ""),
                },
                "eyes": subject_data.get("eyes", {}),
                "eyebrows": subject_data.get("eyebrows", {}),
                "nose": subject_data.get("nose", {}),
                "lips": subject_data.get("lips", {}),
                "hair": subject_data.get("hair", {}),
                "build": subject_data.get("build", ""),
                "notable_features": subject_data.get("notable_features", []),
            },

            # ===== COSMETICS =====
            "cosmetics": subject_data.get("cosmetics", {}),

            # ===== POSE ANALYSIS =====
            "pose": {
                "head": pose_data.get("head", {}),
                "neck": pose_data.get("neck", {}),
                "shoulders": pose_data.get("shoulders", {}),
                "arms": {
                    "left": pose_data.get("left_arm", {}),
                    "right": pose_data.get("right_arm", {}),
                },
                "hands": {
                    "left": pose_data.get("left_hand", {}),
                    "right": pose_data.get("right_hand", {}),
                },
                "torso": pose_data.get("torso", {}),
                "hips": pose_data.get("hips", {}),
                "legs": {
                    "left": pose_data.get("left_leg", ""),
                    "right": pose_data.get("right_leg", ""),
                },
                "feet": pose_data.get("feet", ""),
                "overall_gesture": pose_data.get("overall_gesture", ""),
            },

            # ===== CLOTHING ANALYSIS =====
            "clothing": {
                "garments": clothing_data.get("garments", []),
                "coverage": clothing_data.get("coverage", {}),
                "fabric_behavior": clothing_data.get("fabric_behavior", {}),
                "accessories": clothing_data.get("accessories", []),
                "overall_style": clothing_data.get("overall_style", ""),
            },

            # ===== ENVIRONMENT ANALYSIS =====
            "environment": {
                "setting": environment_data.get("setting_type", ""),
                "background": environment_data.get("background", {}),
                "surfaces": environment_data.get("surfaces", {}),
                "props": environment_data.get("props", []),
                "spatial": environment_data.get("spatial", {}),
            },

            # ===== LIGHTING ANALYSIS =====
            "lighting": {
                "key_light": lighting_data.get("key_light", {}),
                "fill_light": lighting_data.get("fill_light", {}),
                "rim_light": lighting_data.get("rim_light", {}),
                "shadows": lighting_data.get("shadows", {}),
            },

            # ===== COLOR ANALYSIS =====
            "colors": {
                "temperature": colors_data.get("temperature", ""),
                "dominant_palette": colors_data.get("dominant_palette", []),
                "saturation": colors_data.get("saturation", ""),
                "mood": colors_data.get("mood", ""),
            },

            # ===== STYLE ANALYSIS =====
            "style": {
                "technical": style_data.get("technical", {}),
                "composition": style_data.get("composition", {}),
                "aesthetic": style_data.get("aesthetic", {}),
            },

            # ===== GENERATED PROMPT =====
            "generated_prompt": {
                "text": prompt,
                "statistics": {
                    "word_count": word_count,
                    "estimated_tokens": estimated_tokens,
                    "chars": len(prompt),
                },
            },

            # ===== LLM INTERACTIONS (all queries and responses) =====
            "llm_interactions": llm_interactions,

            # ===== RAW DATA (for debugging/advanced use) =====
            "_raw": {
                "classification": classification,
                "attributes": attributes,
            },
        }

        # Clean up empty sections
        structured = cls._clean_empty_values(structured)

        return structured

    @classmethod
    def _clean_empty_values(cls, d: dict) -> dict:
        """Recursively remove empty values from dict while preserving structure."""
        if not isinstance(d, dict):
            return d

        cleaned = {}
        for k, v in d.items():
            if isinstance(v, dict):
                cleaned_v = cls._clean_empty_values(v)
                if cleaned_v:  # Only add non-empty dicts
                    cleaned[k] = cleaned_v
            elif isinstance(v, list):
                if v:  # Only add non-empty lists
                    cleaned[k] = v
            elif v not in (None, "", 0, 0.0) or k in ("confidence", "word_count", "estimated_tokens", "has_text"):
                # Keep zeros for specific fields that can legitimately be 0
                cleaned[k] = v

        return cleaned

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
        log,
        prompt_template: BasePromptTemplate,
        interactions: list = None
    ) -> str:
        """Stage 5: Compose prompt using template-based prompts."""

        # Calculate word targets based on max_tokens
        # Ensure minimum word counts for quality prompts (Standard mode should be 150-250+)
        min_words = max(150, int(max_tokens * 0.8))
        max_words = max(300, int(max_tokens * 1.5))

        # Build focus options dict
        focus_options = {
            "subject": focus_subject,
            "environment": focus_environment,
            "lighting": focus_lighting,
            "colors": focus_colors,
            "mood": focus_mood,
        }

        # Use template to build prompts (SOLID: template handles formatting)
        system_prompt, user_message = prompt_template.build_composition_prompt(
            classification, attributes, user_prompt, prompt_mode,
            focus_options, include_text_quotes, min_words, max_words
        )

        # Get template-specific settings
        temp_modifier = prompt_template.get_temperature_modifier("composition")
        actual_temp = temperature * temp_modifier
        actual_max_tokens = max_tokens * 2

        response_text = cls._call_vision_llm(
            client, model, base64_image,
            system_prompt, user_message,
            max_tokens=actual_max_tokens,
            temperature=actual_temp
        )

        # Clean repetition if detected (common in smaller models)
        response_text, had_repetition = detect_and_clean_repetition(response_text)
        if had_repetition:
            log("  Warning: Detected and cleaned repetitive output in composition")

        # Track interaction
        if interactions is not None:
            interactions.append({
                "stage": "composition",
                "stage_number": 5,
                "type": "vision",
                "request": {
                    "system_prompt": system_prompt,
                    "user_message": user_message,
                    "max_tokens": actual_max_tokens,
                    "temperature": actual_temp,
                },
                "response": {
                    "raw_text": response_text,
                },
            })

        return response_text

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
        log,
        prompt_template: BasePromptTemplate,
        interactions: list = None
    ) -> str:
        """Stage 5 (Deep mode): Refined prompt composition with additional LLM call."""

        # Generate initial draft using template (this will track its own interaction)
        draft = cls._stage5_prompt_composition_template(
            client, model, base64_image,
            classification, attributes,
            user_prompt, prompt_mode,
            focus_subject, focus_environment, focus_lighting,
            focus_colors, focus_mood, include_text_quotes,
            max_tokens, temperature * 0.8, log, prompt_template,
            interactions=interactions
        )

        # Calculate word targets - Deep mode should have even higher targets
        min_words = max(200, int(max_tokens * 1.0))
        max_words = max(400, int(max_tokens * 2.0))

        # Use template for refinement prompt
        system_prompt, user_message = prompt_template.build_refinement_prompt(
            draft, min_words, max_words
        )

        # Get template-specific settings
        temp_modifier = prompt_template.get_temperature_modifier("refinement")
        actual_max_tokens = max_tokens * 2
        actual_temp = temperature * temp_modifier

        response_text = cls._call_text_llm(
            client, model,
            system_prompt, user_message,
            max_tokens=actual_max_tokens,
            temperature=actual_temp
        )

        # Track this refinement LLM call
        if interactions is not None:
            interactions.append({
                "stage": "prompt_refinement",
                "stage_number": 5,
                "sub_stage": "refinement",
                "type": "text",
                "request": {
                    "system_prompt": system_prompt,
                    "user_message": user_message,
                    "max_tokens": actual_max_tokens,
                    "temperature": actual_temp,
                    "input_draft_length": len(draft),
                },
                "response": {
                    "raw_text": response_text,
                },
            })

        return response_text
