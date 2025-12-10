"""
SID_ZImagePromptGenerator Node (Unified)

Single Z-Image prompt generator with automatic mode selection.
Auto-switches between Single-Shot and Agentic modes based on LLM capabilities.

Features:
- 5 simple inputs: image, llm_model, analysis_mode, preset_style, user_guidance, seed
- Auto-detects reasoning capability from llm_model
- Component analysis based on analysis_mode
- Preset styles for common use cases
"""

import base64
import io
import json
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image
from comfy_api.latest import io as comfy_io
import comfy.utils

from .llm_providers.llm_model_type import LLMModelConfig
from . import config_loader

# Create custom LLM_MODEL type for ComfyUI
LLM_MODEL_Type = comfy_io.Custom("LLM_MODEL")


# =============================================================================
# Analysis Mode Configurations
# =============================================================================

ANALYSIS_MODES = {
    "Quick": {
        "components": ["framing", "ethnicity", "clothing"],
        "description": "Fast single-call caption (~3s)",
        "include_lighting": False,
        "include_pose": False,
        "include_intimate": False,
        "include_tattoos": False,
    },
    "Standard": {
        "components": ["framing", "ethnicity", "hair", "face", "eyes", "body_pose", "clothing", "lighting"],
        "description": "Balanced analysis (~10s)",
        "include_lighting": True,
        "include_pose": True,
        "include_intimate": False,
        "include_tattoos": False,
    },
    "Detailed": {
        "components": ["framing", "ethnicity", "hair", "face", "eyes", "nose_lips", "body_pose", "clothing", "intimate_apparel", "tattoos", "accessories", "lighting"],
        "description": "Full analysis with intimate/tattoos (~20s)",
        "include_lighting": True,
        "include_pose": True,
        "include_intimate": True,
        "include_tattoos": True,
    },
    "Extreme": {
        "components": ["framing", "ethnicity", "hair", "face", "eyes", "nose_lips", "body_pose", "clothing", "intimate_apparel", "tattoos", "accessories", "lighting"],
        "description": "Maximum detail, raw output (~25s)",
        "include_lighting": True,
        "include_pose": True,
        "include_intimate": True,
        "include_tattoos": True,
        "raw_mode": True,
    },
}


# =============================================================================
# Preset Style Configurations
# =============================================================================

PRESET_STYLES = {
    "Auto-Detect": {
        "description": "Smart subject detection (Woman/Man/Vehicle/Landscape)",
        "system_addon": "",
        "user_addon": "",
    },
    "Portrait": {
        "description": "Face, expression, upper body focus",
        "system_addon": "Focus heavily on facial features, expression, skin quality, and makeup details.",
        "user_addon": "Emphasize the subject's face, expression, and upper body appearance.",
    },
    "Fashion & Outfit": {
        "description": "Clothing, accessories, styling focus",
        "system_addon": "Focus heavily on clothing details, materials, colors, fit, and accessories.",
        "user_addon": "Emphasize all clothing items, materials, textures, and fashion styling.",
    },
    "Artistic Style": {
        "description": "Composition, lighting, mood focus",
        "system_addon": "Focus on artistic and photographic qualities - composition, lighting, color palette, mood.",
        "user_addon": "Emphasize the artistic qualities: composition, lighting style, color tones, and overall mood.",
    },
    "NSFW/Detailed": {
        "description": "Comprehensive body and clothing analysis",
        "system_addon": "Include comprehensive body and intimate apparel details. Be specific about all visible features.",
        "user_addon": "Provide detailed analysis of body, intimate apparel, and all visible features.",
    },
}


# =============================================================================
# Component Definitions for Analysis
# =============================================================================

COMPONENTS = {
    "framing": {
        "name": "Framing & Composition",
        "prompt": """Analyze ONLY the framing and composition:
1. COLOR MODE: Color or black & white?
2. SHOT TYPE: ECU/BCU/CU/MCU/MS/MFS/FS/LS
3. SUBJECT POSE: Standing/Seated/Lying/Kneeling/Leaning
4. CAMERA ANGLE: Eye level, above, below?
5. DEPTH OF FIELD: Shallow/medium/deep?

Output JSON:
{"shot_type": "<code>", "subject_pose": "<pose>", "frame_fill_percent": <number>, "camera_angle": "<angle>", "depth_of_field": "<dof>", "color_mode": "<color/bw>", "prompt_description": "<complete framing description>"}"""
    },

    "ethnicity": {
        "name": "Ethnicity & Demographics",
        "prompt": """Analyze the subject's demographics:
1. ETHNICITY: Be specific (East Asian, South Asian, European, etc.)
2. SKIN TONE: Base tone + undertone (warm/cool/olive)
3. AGE RANGE: young adult, adult, middle-aged, etc.
4. GENDER: female/male

Output JSON:
{"ethnicity": "<specific>", "skin_tone_base": "<tone>", "skin_tone_undertone": "<undertone>", "age_range": "<range>", "gender": "<gender>", "prompt_description": "<demographics description>"}"""
    },

    "hair": {
        "name": "Hair Analysis",
        "prompt": """Analyze the hair (ARRANGEMENT FIRST):
1. ARRANGEMENT: Updo/Ponytail/Braided/Half-up/Loose/Short
2. COLOR: Base color + highlights
3. TEXTURE: Straight/wavy/curly
4. LENGTH: Short/shoulder/mid-back/long

Output JSON:
{"arrangement": "<style>", "base_color": "<color>", "texture": "<texture>", "length": "<length>", "prompt_description": "<hair description starting with arrangement>"}"""
    },

    "face": {
        "name": "Face Shape & Angle",
        "prompt": """Analyze face shape and positioning:
1. FACE SHAPE: Oval/round/heart/square/diamond
2. FACE ANGLE: Frontal/3/4 left/3/4 right/profile
3. HEAD TILT: Level/tilted

Output JSON:
{"shape": "<shape>", "angle": "<angle>", "tilt": "<tilt>", "prompt_description": "<face description>"}"""
    },

    "eyes": {
        "name": "Eyes Analysis",
        "prompt": """Analyze the eyes:
1. COLOR: Specific color with variations
2. SHAPE: Almond/round/hooded/monolid
3. GAZE: Direction
4. EXPRESSION: Soft/intense/playful
5. MAKEUP: If any

Output JSON:
{"color": "<color>", "shape": "<shape>", "gaze": "<direction>", "expression": "<expression>", "makeup": "<if any>", "prompt_description": "<eyes description>"}"""
    },

    "nose_lips": {
        "name": "Nose & Lips",
        "prompt": """Analyze nose and lips:
NOSE: Shape, bridge height
LIPS: Shape, fullness, color/makeup, state (closed/parted/smiling)

Output JSON:
{"nose_shape": "<shape>", "lip_shape": "<shape>", "lip_color": "<color>", "lip_state": "<state>", "prompt_description": "<nose and lips description>"}"""
    },

    "body_pose": {
        "name": "Body & Pose",
        "prompt": """Analyze visible body and pose:
1. VISIBLE PARTS: What's visible
2. POSTURE: Upright/relaxed/leaning
3. ARM/HAND POSITIONS: If visible
4. BODY ANGLE: Facing camera or angled

Output JSON:
{"visible_parts": ["<list>"], "posture": "<posture>", "body_angle": "<angle>", "arms": "<position>", "prompt_description": "<pose description>"}"""
    },

    "clothing": {
        "name": "Clothing Analysis",
        "prompt": """Analyze ONLY VISIBLE clothing:
For each garment:
1. TYPE: Dress/top/pants/etc.
2. NECKLINE: If visible
3. COLOR: Specific
4. MATERIAL: Silk/satin/cotton/etc.
5. FIT: Fitted/loose

ONLY describe what you can SEE - do not invent details.

Output JSON:
{"garments": [{"type": "<type>", "color": "<color>", "material": "<material>"}], "overall_style": "<style>", "prompt_description": "<clothing description>"}"""
    },

    "intimate_apparel": {
        "name": "Intimate Apparel",
        "prompt": """Analyze visible intimate apparel/lingerie:
BRA: Style, coverage, straps, material, color
BOTTOMS: Style, coverage, material
GARTER/STOCKINGS: If visible

Output JSON:
{"bra": {"style": "<style>", "material": "<material>", "color": "<color>"}, "bottoms": {"style": "<style>", "color": "<color>"}, "stockings": {"present": true/false, "type": "<type>"}, "prompt_description": "<intimate apparel description>"}"""
    },

    "tattoos": {
        "name": "Tattoos",
        "prompt": """Analyze visible tattoos:
For each: Location, size, style, subject, colors

Output JSON:
{"has_tattoos": true/false, "tattoos": [{"location": "<location>", "style": "<style>", "subject": "<subject>"}], "prompt_description": "<tattoos description or 'no visible tattoos'>"}"""
    },

    "accessories": {
        "name": "Accessories",
        "prompt": """Analyze accessories/jewelry:
For each: Type, style, material, color, size

Output JSON:
{"accessories": [{"type": "<type>", "material": "<material>", "color": "<color>"}], "prompt_description": "<accessories description>"}"""
    },

    "lighting": {
        "name": "Lighting & Background",
        "prompt": """Analyze lighting and background:
LIGHTING: Type, direction, quality, color temperature
SHADOWS: Density, direction
BACKGROUND: Type, color, blur level

Output JSON:
{"light_type": "<type>", "light_direction": "<direction>", "light_quality": "<quality>", "background_type": "<type>", "background_color": "<color>", "background_blur": "<blur>", "prompt_description": "<lighting and background description>"}"""
    },

    "subject_detection": {
        "name": "Subject Detection",
        "prompt": """Identify the PRIMARY SUBJECT TYPE:
WOMAN, MAN, COUPLE, GROUP, VEHICLE, LANDSCAPE, ANIMAL, PRODUCT, OTHER

Output JSON:
{"subject_type": "<type>", "has_human": true/false, "human_gender": "<female/male/none>", "main_focus": "<description>"}"""
    },
}


# =============================================================================
# Main Node Class
# =============================================================================

class SID_ZImagePromptGenerator(comfy_io.ComfyNode):
    """
    Unified Z-Image Prompt Generator.

    Automatically switches between Single-Shot and Agentic modes
    based on LLM reasoning capability.

    Inputs:
    - image: Image to analyze
    - llm_model: Connect SID_LLM_API or SID_LLM_Local
    - analysis_mode: Quick/Standard/Detailed/Extreme
    - preset_style: Auto-Detect/Portrait/Fashion/Artistic/NSFW
    - user_guidance: Optional custom instructions
    - seed: Random seed
    """

    @classmethod
    def define_schema(cls) -> comfy_io.Schema:
        return comfy_io.Schema(
            node_id="SID_ZImagePromptGenerator",
            display_name="SID Z-Image Prompt Generator",
            category="SID Photography Toolkit",
            description="Generate Z-Image prompts. Auto-switches Single-Shot/Agentic based on LLM.",
            is_output_node=True,
            inputs=[
                comfy_io.Image.Input(
                    "image",
                    tooltip="Input image to analyze"
                ),
                LLM_MODEL_Type.Input(
                    "llm_model",
                    tooltip="Connect SID_LLM_API or SID_LLM_Local node"
                ),
                comfy_io.Combo.Input(
                    "analysis_mode",
                    options=["Quick", "Standard", "Detailed", "Extreme"],
                    default="Standard",
                    tooltip="Quick=fast, Standard=balanced, Detailed=full, Extreme=maximum"
                ),
                comfy_io.Combo.Input(
                    "preset_style",
                    options=list(PRESET_STYLES.keys()),
                    default="Auto-Detect",
                    tooltip="Focus area: Auto-Detect, Portrait, Fashion, Artistic, NSFW"
                ),
                comfy_io.String.Input(
                    "user_guidance",
                    default="",
                    multiline=True,
                    tooltip="Optional: Custom instructions (e.g., 'focus on the red dress')"
                ),
                comfy_io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    control_after_generate=True,
                    tooltip="Random seed for reproducibility"
                ),
            ],
            outputs=[
                comfy_io.String.Output("prompt", display_name="prompt"),
                comfy_io.Int.Output("width", display_name="width"),
                comfy_io.Int.Output("height", display_name="height"),
            ],
        )

    @classmethod
    def execute(
        cls,
        image,
        llm_model: LLMModelConfig,
        analysis_mode: str,
        preset_style: str,
        user_guidance: str,
        seed: int,
    ) -> comfy_io.NodeOutput:
        """Execute prompt generation with auto mode selection."""

        start_time = time.time()

        # Get image dimensions
        img_tensor = image[0]
        height, width = img_tensor.shape[0], img_tensor.shape[1]

        # Auto-detect pipeline based on LLM capabilities
        supports_reasoning = llm_model.supports_reasoning

        def log(msg: str):
            print(f"[SID-Prompt] {msg}")

        log("=" * 60)
        log("SID Z-Image Prompt Generator")
        log("=" * 60)
        log(f"Image: {width}x{height}")
        log(f"Provider: {llm_model.provider} | Model: {llm_model.model}")
        log(f"Analysis: {analysis_mode} | Style: {preset_style}")
        log(f"Pipeline: {'AGENTIC' if supports_reasoning else 'SINGLE-SHOT'}")

        try:
            if supports_reasoning:
                prompt = cls._execute_agentic(
                    image, llm_model, analysis_mode, preset_style, user_guidance
                )
            else:
                prompt = cls._execute_single_shot(
                    image, llm_model, analysis_mode, preset_style, user_guidance
                )

            # Stats
            total_time = time.time() - start_time
            word_count = len(prompt.split())
            log(f"Generated: {word_count} words in {total_time:.1f}s")
            log("=" * 60)

            return comfy_io.NodeOutput(prompt, width, height, ui={"text": (prompt,)})

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            log(error_msg)
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"[SID_ZImagePromptGenerator] {error_msg}") from e

    # =========================================================================
    # SINGLE-SHOT PIPELINE
    # =========================================================================

    @classmethod
    def _execute_single_shot(
        cls,
        image,
        llm_model: LLMModelConfig,
        analysis_mode: str,
        preset_style: str,
        user_guidance: str,
    ) -> str:
        """Fast single-call prompt generation."""

        print(f"[SID-Prompt] Single-shot mode...")

        # Convert image
        img_tensor = image[0]
        base64_image = cls._image_to_base64(img_tensor, llm_model)

        # Build prompts (tier-aware based on provider)
        system_prompt = cls._build_system_prompt(llm_model.provider, preset_style, user_guidance)
        user_prompt = cls._build_user_prompt(llm_model.provider, analysis_mode, preset_style)

        # Get client and call
        client = cls._get_client(llm_model)

        pbar = comfy.utils.ProgressBar(2)
        pbar.update(1)

        response = cls._call_llm(client, llm_model, base64_image, system_prompt, user_prompt)
        prompt = cls._clean_output(response, llm_model.provider)

        # Add user focus if provided
        if user_guidance and user_guidance.strip():
            prompt = f"[FOCUS: {user_guidance.strip()}] {prompt}"

        pbar.update(1)
        return prompt

    # =========================================================================
    # AGENTIC PIPELINE
    # =========================================================================

    @classmethod
    def _execute_agentic(
        cls,
        image,
        llm_model: LLMModelConfig,
        analysis_mode: str,
        preset_style: str,
        user_guidance: str,
    ) -> str:
        """Reasoning-enabled comprehensive analysis."""

        print(f"[SID-Prompt] Agentic mode (reasoning enabled)...")

        # Convert image
        img_tensor = image[0]
        base64_image = cls._image_to_base64(img_tensor, llm_model)

        # Get client
        client = cls._get_client(llm_model)

        pbar = comfy.utils.ProgressBar(3)

        # Step 1: Subject detection (if Auto-Detect)
        subject_type = "WOMAN"  # Default
        if preset_style == "Auto-Detect":
            print(f"[SID-Prompt] Detecting subject...")
            subject_result = cls._analyze_component(client, llm_model, base64_image, "subject_detection")
            subject_type = subject_result.get("subject_type", "WOMAN").upper()
            print(f"[SID-Prompt] Subject: {subject_type}")
        pbar.update(1)

        # Step 2: Get components for this mode
        mode_config = ANALYSIS_MODES.get(analysis_mode, ANALYSIS_MODES["Standard"])
        components = mode_config["components"].copy()

        print(f"[SID-Prompt] Analyzing {len(components)} components...")

        # Step 3: Build comprehensive agentic prompt
        agentic_prompt = cls._build_agentic_prompt(components, preset_style, user_guidance)
        pbar.update(1)

        # Step 4: Single reasoning call
        component_results = cls._call_reasoning_llm(client, llm_model, base64_image, agentic_prompt)

        # Step 5: Assemble prompt
        prompt = cls._assemble_prompt(component_results, mode_config, user_guidance)
        pbar.update(1)

        return prompt

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    @classmethod
    def _image_to_base64(cls, img_tensor, llm_model: LLMModelConfig) -> str:
        """Convert image tensor to base64."""
        img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
        pil_image = Image.fromarray(img_np)

        # Resize if needed
        height, width = img_tensor.shape[0], img_tensor.shape[1]
        max_size = llm_model.extra_params.get("max_image_size") if llm_model.extra_params else None
        if max_size and max(width, height) > max_size:
            if width > height:
                new_width, new_height = max_size, int(height * (max_size / width))
            else:
                new_height, new_width = max_size, int(width * (max_size / height))
            pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG", quality=95)
        return base64.standard_b64encode(buffer.getvalue()).decode("utf-8")

    @classmethod
    def _get_client(cls, llm_model: LLMModelConfig):
        """Get LLM client based on provider."""
        provider = llm_model.provider.lower()

        if provider == "anthropic":
            import anthropic
            import httpx
            timeout = httpx.Timeout(timeout=600.0, connect=30.0)
            return anthropic.Anthropic(api_key=llm_model.api_key, timeout=timeout)

        elif provider in ["openai", "openai_compatible", "grok", "gemini", "groq", "together", "openrouter", "fireworks", "cerebras", "huggingface", "mistral", "deepseek", "ollama", "lmstudio", "custom"]:
            from openai import OpenAI
            return OpenAI(
                api_key=llm_model.api_key or "not-needed",
                base_url=llm_model.api_url if llm_model.api_url else None
            )

        elif provider == "local":
            from .llm_providers.sid_llm_local import LocalModelClient
            extra = llm_model.extra_params or {}
            return LocalModelClient(
                model_name=llm_model.model,
                quantization=extra.get("quantization", "4-bit"),
                device=extra.get("device", "auto"),
                attention_mode=extra.get("attention_mode", "auto"),
                keep_model_loaded=extra.get("keep_model_loaded", True),
                top_p=extra.get("top_p", 0.9),
                repetition_penalty=extra.get("repetition_penalty", 1.2),
                use_torch_compile=extra.get("use_torch_compile", False),
            )

        else:
            raise ValueError(f"Unsupported provider: {provider}")

    @classmethod
    def _build_system_prompt(cls, provider: str, preset_style: str, user_guidance: str) -> str:
        """Build system prompt using TOML config based on provider tier."""
        return config_loader.build_system_prompt(provider, preset_style, user_guidance)

    @classmethod
    def _build_user_prompt(cls, provider: str, analysis_mode: str, preset_style: str) -> str:
        """Build user prompt using TOML config based on provider tier and mode."""
        return config_loader.build_user_prompt(provider, analysis_mode, preset_style)

    @classmethod
    def _get_stop_strings(cls, provider: str) -> List[str]:
        """Get provider-specific stop strings to prevent text leakage."""
        return config_loader.get_stop_strings(provider)

    @classmethod
    def _call_llm(cls, client, llm_model: LLMModelConfig, base64_image: str, system_prompt: str, user_prompt: str) -> str:
        """Make LLM call with provider-specific stop strings."""
        if hasattr(client, 'messages'):
            # Anthropic - doesn't need stop strings, handles well
            response = client.messages.create(
                model=llm_model.model,
                max_tokens=llm_model.max_tokens,
                temperature=llm_model.temperature,
                system=[{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}],
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": base64_image}},
                        {"type": "text", "text": user_prompt}
                    ]
                }]
            )
            return response.content[0].text
        else:
            # OpenAI-style - add stop strings for local providers
            stop_strings = cls._get_stop_strings(llm_model.provider)

            request_params = {
                "model": llm_model.model,
                "max_tokens": llm_model.max_tokens,
                "temperature": llm_model.temperature,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                        {"type": "text", "text": user_prompt}
                    ]}
                ]
            }

            # Add stop strings if provider needs them
            if stop_strings:
                request_params["stop"] = stop_strings

            response = client.chat.completions.create(**request_params)
            return response.choices[0].message.content

    @classmethod
    def _analyze_component(cls, client, llm_model: LLMModelConfig, base64_image: str, component_key: str) -> dict:
        """Analyze single component with stop strings."""
        comp = COMPONENTS.get(component_key, {})
        system = "You are an expert visual analyst. Analyze ONLY the specific aspect requested. Output valid JSON only."

        try:
            if hasattr(client, 'messages'):
                response = client.messages.create(
                    model=llm_model.model, max_tokens=1000, temperature=0.3,
                    system=system,
                    messages=[{"role": "user", "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": base64_image}},
                        {"type": "text", "text": comp.get("prompt", "")}
                    ]}]
                )
                text = response.content[0].text
            else:
                # OpenAI-style with stop strings for local providers
                stop_strings = cls._get_stop_strings(llm_model.provider)

                request_params = {
                    "model": llm_model.model,
                    "max_tokens": 1000,
                    "temperature": 0.3,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                            {"type": "text", "text": comp.get("prompt", "")}
                        ]}
                    ]
                }

                if stop_strings:
                    request_params["stop"] = stop_strings

                response = client.chat.completions.create(**request_params)
                text = response.choices[0].message.content

            return cls._parse_json(text)
        except Exception as e:
            print(f"[SID-Prompt] Error analyzing {component_key}: {e}")
            return {}

    @classmethod
    def _build_agentic_prompt(cls, components: List[str], preset_style: str, user_guidance: str) -> str:
        """Build comprehensive agentic prompt."""
        comp_specs = []
        for key in components:
            if key in COMPONENTS:
                comp = COMPONENTS[key]
                comp_specs.append(f"### {comp['name']} ({key})\n{comp['prompt']}")

        user_section = ""
        if user_guidance and user_guidance.strip():
            user_section = f"\n\n## USER REQUEST (HIGH PRIORITY)\n\"{user_guidance.strip()}\"\nEnsure your analysis captures details related to this request.\n"

        return f"""You are an expert visual analyst. Use reasoning to analyze this image comprehensively.
{user_section}
## INSTRUCTIONS
Analyze ALL components below and return a single JSON object with each component as a key.

## COMPONENTS TO ANALYZE

{chr(10).join(comp_specs)}

## OUTPUT FORMAT
Return a single JSON object:
```json
{{
    "framing": {{ ... }},
    "ethnicity": {{ ... }},
    ... etc for each component ...
}}
```"""

    @classmethod
    def _call_reasoning_llm(cls, client, llm_model: LLMModelConfig, base64_image: str, prompt: str) -> dict:
        """Call LLM with reasoning enabled and stop strings."""
        try:
            if llm_model.provider.lower() == "anthropic" and hasattr(client, 'messages'):
                response = client.messages.create(
                    model=llm_model.model,
                    max_tokens=16000,
                    thinking={"type": "enabled", "budget_tokens": 10000},
                    messages=[{"role": "user", "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": base64_image}},
                        {"type": "text", "text": prompt}
                    ]}]
                )
                text = ""
                for block in response.content:
                    if hasattr(block, 'text'):
                        text = block.text
                        break
            else:
                # OpenAI-style with stop strings for local providers
                stop_strings = cls._get_stop_strings(llm_model.provider)

                request_params = {
                    "model": llm_model.model,
                    "max_completion_tokens": 16000,
                    "messages": [{"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                        {"type": "text", "text": prompt}
                    ]}]
                }

                if stop_strings:
                    request_params["stop"] = stop_strings

                response = client.chat.completions.create(**request_params)
                text = response.choices[0].message.content

            return cls._parse_json(text)
        except Exception as e:
            print(f"[SID-Prompt] Reasoning call error: {e}")
            return {}

    @classmethod
    def _parse_json(cls, text: str) -> dict:
        """Parse JSON from response."""
        # Try markdown code block
        match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        if match:
            text = match.group(1)
        # Try raw JSON
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            try:
                return json.loads(match.group(0))
            except:
                pass
        return {}

    @classmethod
    def _assemble_prompt(cls, components: dict, mode_config: dict, user_guidance: str) -> str:
        """Assemble final prompt from components with example text filtering."""
        sections = []

        # Order: framing, ethnicity, face, hair, eyes, nose_lips, body_pose, clothing, intimate, tattoos, accessories, lighting
        order = ["framing", "ethnicity", "face", "hair", "eyes", "nose_lips", "body_pose", "clothing", "intimate_apparel", "tattoos", "accessories", "lighting"]

        for key in order:
            if key in components and components[key]:
                desc = components[key].get("prompt_description", "")
                if desc:
                    # Clean each component description to remove any leaked examples
                    cleaned_desc = cls._clean_component_description(desc)
                    if cleaned_desc:
                        sections.append(cleaned_desc)

        # Combine
        if mode_config.get("raw_mode"):
            prompt = ". ".join(s for s in sections if s)
        else:
            prompt = ", ".join(s for s in sections if s)
            prompt = prompt.replace(", ,", ",").replace("  ", " ")

        # Add user focus
        if user_guidance and user_guidance.strip():
            prompt = f"[FOCUS: {user_guidance.strip()}] {prompt}"

        return prompt.strip()

    @classmethod
    def _clean_output(cls, text: str, provider: str = "default") -> str:
        """Clean LLM output using TOML config patterns."""
        return config_loader.clean_output(text, provider)

    @classmethod
    def _clean_component_description(cls, desc: str, provider: str = "default") -> str:
        """Clean individual component description using TOML config patterns."""
        if not desc:
            return ""
        return config_loader.clean_output(desc, provider)
