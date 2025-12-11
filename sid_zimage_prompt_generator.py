"""
SID_ZImagePromptGenerator Node (Unified)

Single Z-Image prompt generator with automatic mode selection.
Auto-switches between Single-Shot and Agentic modes based on LLM capabilities.

Features:
- 5 simple inputs: image, llm_model, analysis_mode, preset_style, user_guidance, seed
- Auto-detects reasoning capability from llm_model
- Component analysis based on analysis_mode
- Preset styles for common use cases
- Image resize options for optimal model performance
"""

import base64
import io
import json
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from comfy_api.latest import io as comfy_io
import comfy.utils

from .llm_providers.llm_model_type import LLMModelConfig
from . import config_loader


# =============================================================================
# Model Resolution Configurations
# =============================================================================

# Optimal image sizes for different providers/models
# Format: (min_pixels, optimal_pixels, max_pixels)
# Pixels = width * height

MODEL_RESOLUTION_LIMITS = {
    # API providers - generally handle larger images well
    "anthropic": {
        "min_pixels": 256 * 256,      # ~65K pixels
        "optimal_pixels": 1024 * 1024, # ~1M pixels (good balance)
        "max_pixels": 2048 * 2048,    # ~4M pixels
    },
    "openai": {
        "min_pixels": 256 * 256,
        "optimal_pixels": 1024 * 1024,
        "max_pixels": 2048 * 2048,
    },
    "gemini": {
        "min_pixels": 256 * 256,
        "optimal_pixels": 1024 * 1024,
        "max_pixels": 3072 * 3072,
    },
    "grok": {
        "min_pixels": 256 * 256,
        "optimal_pixels": 1024 * 1024,
        "max_pixels": 2048 * 2048,
    },
    # Free/budget providers - smaller is faster
    "groq": {
        "min_pixels": 384 * 384,
        "optimal_pixels": 768 * 768,
        "max_pixels": 1024 * 1024,
    },
    "openrouter": {
        "min_pixels": 256 * 256,
        "optimal_pixels": 1024 * 1024,
        "max_pixels": 2048 * 2048,
    },
    "together": {
        "min_pixels": 384 * 384,
        "optimal_pixels": 768 * 768,
        "max_pixels": 1536 * 1536,
    },
    "fireworks": {
        "min_pixels": 384 * 384,
        "optimal_pixels": 768 * 768,
        "max_pixels": 1536 * 1536,
    },
    # Local providers - optimized for VRAM efficiency
    "local": {
        "min_pixels": 256 * 256,       # Fast processing
        "optimal_pixels": 672 * 672,   # Good balance for 4-8GB VRAM
        "max_pixels": 1280 * 1280,     # Maximum detail
    },
    "ollama": {
        "min_pixels": 256 * 256,
        "optimal_pixels": 672 * 672,
        "max_pixels": 1280 * 1280,
    },
    "lmstudio": {
        "min_pixels": 256 * 256,
        "optimal_pixels": 672 * 672,
        "max_pixels": 1280 * 1280,
    },
    # Default for unknown providers
    "default": {
        "min_pixels": 384 * 384,
        "optimal_pixels": 768 * 768,
        "max_pixels": 1536 * 1536,
    },
}

# Image resize modes
IMAGE_RESIZE_MODES = ["auto", "max", "min", "original"]


def calculate_resize_dimensions(
    width: int,
    height: int,
    target_pixels: int,
    maintain_aspect: bool = True
) -> Tuple[int, int]:
    """
    Calculate new dimensions to achieve target pixel count while maintaining aspect ratio.

    Args:
        width: Original width
        height: Original height
        target_pixels: Target total pixels (width * height)
        maintain_aspect: Whether to maintain aspect ratio

    Returns:
        Tuple of (new_width, new_height)
    """
    current_pixels = width * height

    if current_pixels <= target_pixels:
        return width, height

    # Calculate scale factor
    scale = (target_pixels / current_pixels) ** 0.5

    new_width = int(width * scale)
    new_height = int(height * scale)

    # Ensure dimensions are at least 64 pixels
    new_width = max(64, new_width)
    new_height = max(64, new_height)

    # Round to nearest multiple of 8 for better model compatibility
    new_width = (new_width // 8) * 8
    new_height = (new_height // 8) * 8

    return new_width, new_height

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
        "components": ["framing", "ethnicity", "hair", "face", "eyes", "body_pose", "body_figure", "clothing", "lighting"],
        "description": "Balanced analysis with body proportions (~12s)",
        "include_lighting": True,
        "include_pose": True,
        "include_intimate": False,
        "include_tattoos": False,
    },
    "Detailed": {
        "components": ["framing", "ethnicity", "hair", "face", "eyes", "nose_lips", "body_pose", "body_figure", "clothing", "intimate_apparel", "tattoos", "accessories", "lighting"],
        "description": "Full analysis with body figure/intimate/tattoos (~25s)",
        "include_lighting": True,
        "include_pose": True,
        "include_intimate": True,
        "include_tattoos": True,
    },
    "Extreme": {
        "components": ["framing", "ethnicity", "hair", "face", "eyes", "nose_lips", "body_pose", "body_figure", "clothing", "intimate_apparel", "tattoos", "accessories", "lighting"],
        "description": "Maximum detail with body figure, raw output (~30s)",
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
2. SHOT TYPE: Close-up (CU) / Medium (MS) / Wide (WS) / Extreme wide (EWS) / Detail shot
3. CAMERA ANGLE: Eye level / High angle / Low angle / Bird's eye / Worm's eye
4. DEPTH OF FIELD: Shallow (blurred background) / Medium / Deep (sharp throughout)
5. COMPOSITION: Rule of thirds / Centered / Leading lines / Symmetrical / Golden ratio

Output JSON:
{"shot_type": "<type>", "camera_angle": "<angle>", "depth_of_field": "<dof>", "composition": "<style>", "color_mode": "<color/bw>", "prompt_description": "<complete framing and composition description>"}"""
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

    "body_figure": {
        "name": "Body Figure & Proportions",
        "prompt": """Analyze BODY FIGURE, PROPORTIONS, and EXPOSURE LEVEL visible in this image.
CRITICAL: Only describe what is ACTUALLY VISIBLE.

1. NUDITY/EXPOSURE LEVEL:
   - Fully clothed / Partially clothed / Revealing / Swimwear-Lingerie / Topless / Nude / Implied nude

2. BREAST/CHEST EXPOSURE:
   - Covered / Cleavage only / Sideboob (left/right/both) / Underboob / Partial / Full
   - Nipple visibility: covered / through fabric / partial / full

3. BODY TYPE: Slim / Athletic / Average / Curvy-Voluptuous / Petite / Plus-size

4. BUST DETAILS (if visible):
   - Cup size estimate: AA/A (minimal) / B (small) / C (medium) / D (large) / DD-E (very large) / DDD-F (extra large) / G+ (extremely large)
   - Shape: natural / rounded / teardrop / projected / perky / full
   - Cleavage: none / minimal / moderate / prominent / deep
   - Cleavage shadow: none / subtle / defined / dramatic
   - Position: natural / lifted / pushed together / separated

5. WAIST (if visible):
   - Definition: undefined / slight / defined / very defined (hourglass)
   - Midriff visible: yes/no

6. HIPS & BUTTOCKS (if visible):
   - Hip width: narrow / medium / wide / very wide
   - Hip curve: subtle / moderate / pronounced / dramatic
   - Buttocks shape: flat / moderate / rounded / prominent / heart-shaped
   - Buttocks exposure: covered / partial / full

7. THIGHS (if visible):
   - Fullness: slim / toned / moderate / full / thick
   - Thigh gap: present / touching / close

8. BODY SHADOWS & CONTOURS:
   - Under bust shadow: none / subtle / defined / dramatic
   - Overall sculpting: soft / moderate / dramatic

Output JSON:
{"nudity_level": "<level>", "breast_exposure": {"level": "<level>", "sideboob": "<none/left/right/both>", "underboob": true/false, "nipple": "<visibility>"}, "body_type": "<type>", "bust": {"visible": true/false, "cup_size": "<AA-G+>", "shape": "<shape>", "cleavage": "<level>", "cleavage_shadow": "<level>", "position": "<position>"}, "waist": {"visible": true/false, "definition": "<level>", "midriff_exposed": true/false}, "hips": {"visible": true/false, "width": "<width>", "curve": "<curve>"}, "buttocks": {"visible": true/false, "shape": "<shape>", "exposure": "<level>"}, "thighs": {"visible": true/false, "fullness": "<level>", "gap": "<status>"}, "body_shadows": {"under_bust": "<level>", "sculpting": "<level>"}, "prompt_description": "<complete body figure description>"}"""
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

    # Landscape-specific components
    "landscape_terrain": {
        "name": "Landscape Terrain",
        "prompt": """Analyze the TERRAIN and NATURAL FEATURES:
1. TERRAIN TYPE: Mountains/Hills/Plains/Desert/Forest/Beach/Ocean/Lake/River/Urban/Rural
2. GROUND COVER: Grass/Sand/Rock/Snow/Water/Pavement/Dirt
3. VEGETATION: Dense forest/Sparse trees/Grassland/Tropical/None
4. WATER FEATURES: Ocean/Lake/River/Waterfall/Stream/Pond/None
5. ELEVATION: Flat/Rolling/Mountainous/Cliffs/Valleys

Output JSON:
{"terrain_type": "<type>", "ground_cover": "<cover>", "vegetation": "<type>", "water_features": "<type>", "elevation": "<type>", "prompt_description": "<terrain description>"}"""
    },

    "landscape_weather": {
        "name": "Weather & Atmosphere",
        "prompt": """Analyze the WEATHER and ATMOSPHERIC CONDITIONS:
1. SKY CONDITION: Clear/Partly cloudy/Overcast/Stormy/Sunset/Sunrise/Night
2. CLOUD TYPES: None/Cumulus/Stratus/Cirrus/Dramatic storm clouds
3. VISIBILITY: Clear/Hazy/Foggy/Misty/Smoky
4. WEATHER: Sunny/Rainy/Snowy/Windy/Calm
5. TIME OF DAY: Dawn/Morning/Midday/Afternoon/Golden hour/Dusk/Night

Output JSON:
{"sky_condition": "<condition>", "clouds": "<type>", "visibility": "<visibility>", "weather": "<weather>", "time_of_day": "<time>", "prompt_description": "<weather and atmosphere description>"}"""
    },

    "landscape_scenery": {
        "name": "Scenery & Elements",
        "prompt": """Analyze the SCENIC ELEMENTS and COMPOSITION:
1. FOCAL POINT: What draws the eye (mountain peak, building, tree, horizon, etc.)
2. FOREGROUND: What's in the nearest plane
3. MIDGROUND: Middle distance elements
4. BACKGROUND: Distant elements (horizon, sky, mountains)
5. MAN-MADE ELEMENTS: Buildings/Roads/Bridges/Fences/None
6. NATURAL LANDMARKS: Peaks/Rock formations/Trees/Rivers/None

Output JSON:
{"focal_point": "<element>", "foreground": "<description>", "midground": "<description>", "background": "<description>", "man_made": ["<elements>"], "natural_landmarks": ["<landmarks>"], "prompt_description": "<scenery description>"}"""
    },

    # Vehicle-specific components
    "vehicle_details": {
        "name": "Vehicle Details",
        "prompt": """Analyze the VEHICLE in detail:
1. VEHICLE TYPE: Car/Motorcycle/Truck/Bus/Bicycle/Boat/Aircraft/Train
2. MAKE/MODEL: If identifiable (e.g., "Ferrari", "Harley Davidson")
3. BODY STYLE: Sedan/Coupe/SUV/Convertible/Hatchback/Sports/Classic/etc.
4. COLOR: Primary color and any accents
5. CONDITION: New/Used/Vintage/Restored/Modified/Damaged
6. ANGLE: Front/Rear/Side/3-quarter/Top-down
7. NOTABLE FEATURES: Spoilers/Rims/Decals/Custom parts

Output JSON:
{"vehicle_type": "<type>", "make_model": "<make_model>", "body_style": "<style>", "color": "<color>", "condition": "<condition>", "angle": "<angle>", "features": ["<features>"], "prompt_description": "<vehicle description>"}"""
    },

    # Animal-specific component
    "animal_details": {
        "name": "Animal Details",
        "prompt": """Analyze the ANIMAL in detail:
1. SPECIES: Specific animal type (dog, cat, horse, bird, etc.)
2. BREED: If identifiable
3. COLORING: Fur/feather color and pattern
4. POSE: Standing/Sitting/Lying/Running/Flying/Swimming
5. EXPRESSION: Alert/Relaxed/Playful/Aggressive/Sleeping
6. SIZE: Small/Medium/Large (relative to frame)

Output JSON:
{"species": "<species>", "breed": "<breed>", "coloring": "<description>", "pose": "<pose>", "expression": "<expression>", "size": "<size>", "prompt_description": "<animal description>"}"""
    },

    # Product-specific component
    "product_details": {
        "name": "Product Details",
        "prompt": """Analyze the PRODUCT in detail:
1. PRODUCT TYPE: Electronics/Food/Cosmetics/Clothing/Furniture/Tool/etc.
2. BRAND: If visible
3. COLOR: Primary colors
4. MATERIAL: Metal/Plastic/Glass/Wood/Fabric/etc.
5. CONDITION: New/Used/Packaged/Displayed
6. STYLING: Product photography style (studio/lifestyle/flat-lay)

Output JSON:
{"product_type": "<type>", "brand": "<brand>", "color": "<color>", "material": "<material>", "condition": "<condition>", "styling": "<style>", "prompt_description": "<product description>"}"""
    },
}


# =============================================================================
# Subject Type to Components Mapping
# =============================================================================

# Components that are human-specific (only apply to WOMAN, MAN, COUPLE, GROUP)
HUMAN_COMPONENTS = {
    "ethnicity", "hair", "face", "eyes", "nose_lips", "body_pose", "body_figure",
    "clothing", "intimate_apparel", "tattoos", "accessories"
}

# Components for each subject type
SUBJECT_COMPONENTS = {
    "WOMAN": ["framing", "ethnicity", "hair", "face", "eyes", "nose_lips", "body_pose", "body_figure", "clothing", "intimate_apparel", "tattoos", "accessories", "lighting"],
    "MAN": ["framing", "ethnicity", "hair", "face", "eyes", "nose_lips", "body_pose", "body_figure", "clothing", "tattoos", "accessories", "lighting"],
    "COUPLE": ["framing", "ethnicity", "hair", "face", "eyes", "body_pose", "clothing", "accessories", "lighting"],
    "GROUP": ["framing", "ethnicity", "clothing", "lighting"],
    "LANDSCAPE": ["framing", "landscape_terrain", "landscape_weather", "landscape_scenery", "lighting"],
    "VEHICLE": ["framing", "vehicle_details", "lighting"],
    "ANIMAL": ["framing", "animal_details", "lighting"],
    "PRODUCT": ["framing", "product_details", "lighting"],
    "OTHER": ["framing", "lighting"],
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
                comfy_io.Combo.Input(
                    "image_resize",
                    options=IMAGE_RESIZE_MODES,
                    default="auto",
                    tooltip="auto=optimal for model, max=highest detail, min=fastest, original=no resize"
                ),
                comfy_io.Combo.Input(
                    "prompt_length",
                    options=["Short", "Medium", "Long", "Free"],
                    default="Free",
                    tooltip="Short=30-60 words, Medium=80-150 words, Long=150-250 words, Free=no constraint"
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
                comfy_io.String.Output("metadata", display_name="metadata", tooltip="JSON metadata about the analysis (subject type, components, timing)"),
                comfy_io.String.Output("debug", display_name="debug", tooltip="Debug information and raw component outputs"),
            ],
        )

    @classmethod
    def execute(
        cls,
        image,
        llm_model: LLMModelConfig,
        analysis_mode: str,
        preset_style: str,
        image_resize: str,
        prompt_length: str,
        user_guidance: str,
        seed: int,
    ) -> comfy_io.NodeOutput:
        """Execute prompt generation with auto mode selection."""

        start_time = time.time()
        debug_lines = []

        # Get image dimensions
        img_tensor = image[0]
        height, width = img_tensor.shape[0], img_tensor.shape[1]

        # Auto-detect pipeline based on LLM capabilities
        supports_reasoning = llm_model.supports_reasoning
        pipeline_type = "AGENTIC" if supports_reasoning else "SINGLE-SHOT"

        def log(msg: str):
            print(f"[SID-Prompt] {msg}")
            debug_lines.append(msg)

        log("=" * 60)
        log("SID Z-Image Prompt Generator")
        log("=" * 60)
        log(f"Image: {width}x{height}")
        log(f"Provider: {llm_model.provider} | Model: {llm_model.model}")
        log(f"Analysis: {analysis_mode} | Style: {preset_style} | Length: {prompt_length}")
        log(f"Resize: {image_resize} | Pipeline: {pipeline_type}")

        # Initialize metadata dict
        metadata_dict = {
            "image_width": width,
            "image_height": height,
            "provider": llm_model.provider,
            "model": llm_model.model,
            "analysis_mode": analysis_mode,
            "preset_style": preset_style,
            "prompt_length": prompt_length,
            "image_resize": image_resize,
            "pipeline": pipeline_type,
            "seed": seed,
        }

        try:
            if supports_reasoning:
                result = cls._execute_agentic(
                    image, llm_model, analysis_mode, preset_style, user_guidance, image_resize, prompt_length
                )
            else:
                result = cls._execute_single_shot(
                    image, llm_model, analysis_mode, preset_style, user_guidance, image_resize, prompt_length
                )

            # Handle result (can be tuple with metadata or just prompt string)
            if isinstance(result, tuple):
                prompt, extra_metadata, extra_debug = result
                metadata_dict.update(extra_metadata)
                debug_lines.extend(extra_debug)
            else:
                prompt = result

            # Stats
            total_time = time.time() - start_time
            word_count = len(prompt.split())
            log(f"Generated: {word_count} words in {total_time:.1f}s")
            log("=" * 60)

            # Finalize metadata
            metadata_dict["total_time_seconds"] = round(total_time, 2)
            metadata_dict["word_count"] = word_count

            # Convert to JSON strings
            metadata_str = json.dumps(metadata_dict, indent=2)
            debug_str = "\n".join(debug_lines)

            return comfy_io.NodeOutput(prompt, width, height, metadata_str, debug_str, ui={"text": (prompt,)})

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
        image_resize: str = "auto",
        prompt_length: str = "Free",
    ) -> tuple:
        """Fast single-call prompt generation. Returns (prompt, metadata, debug_lines)."""

        debug_lines = []
        debug_lines.append("Single-shot mode...")
        print(f"[SID-Prompt] Single-shot mode...")

        # Convert image with resize
        img_tensor = image[0]
        base64_image, resize_info = cls._image_to_base64(img_tensor, llm_model, image_resize)
        if resize_info:
            debug_lines.append(resize_info)
            print(f"[SID-Prompt] {resize_info}")

        # Build prompts (tier-aware based on provider, with length constraints)
        system_prompt = cls._build_system_prompt(llm_model.provider, preset_style, user_guidance, prompt_length)
        user_prompt = cls._build_user_prompt(llm_model.provider, analysis_mode, preset_style, prompt_length)
        debug_lines.append(f"System prompt length: {len(system_prompt)} chars")
        debug_lines.append(f"User prompt length: {len(user_prompt)} chars")

        # Get client and call
        client = cls._get_client(llm_model)

        pbar = comfy.utils.ProgressBar(2)
        pbar.update(1)

        response = cls._call_llm(client, llm_model, base64_image, system_prompt, user_prompt)
        debug_lines.append(f"Raw LLM response length: {len(response)} chars")
        prompt = cls._clean_output(response, llm_model.provider)

        # Second LLM call: Enhance prompt with user instructions (only if instructions provided)
        if user_guidance and user_guidance.strip():
            debug_lines.append(f"Making second LLM call to apply user instructions...")
            prompt, enhanced = cls._enhance_prompt_with_instructions(prompt, user_guidance, llm_model)
            if enhanced:
                debug_lines.append(f"Successfully enhanced prompt with user instructions")
            else:
                debug_lines.append(f"Enhancement call did not modify prompt")

        pbar.update(1)

        # Build metadata
        metadata = {
            "subject_type": "unknown",  # Single-shot doesn't do subject detection
            "components_analyzed": [],
            "component_count": 0,
        }

        return prompt, metadata, debug_lines

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
        image_resize: str = "auto",
        prompt_length: str = "Free",
    ) -> tuple:
        """Reasoning-enabled comprehensive analysis. Returns (prompt, metadata, debug_lines)."""

        debug_lines = []
        debug_lines.append("Agentic mode (reasoning enabled)...")
        print(f"[SID-Prompt] Agentic mode (reasoning enabled)...")

        # Convert image with resize
        img_tensor = image[0]
        base64_image, resize_info = cls._image_to_base64(img_tensor, llm_model, image_resize)
        if resize_info:
            debug_lines.append(resize_info)
            print(f"[SID-Prompt] {resize_info}")

        # Get client
        client = cls._get_client(llm_model)

        pbar = comfy.utils.ProgressBar(3)

        # Step 1: Subject detection (if Auto-Detect)
        subject_type = "WOMAN"  # Default
        subject_result = {}
        if preset_style == "Auto-Detect":
            debug_lines.append("Detecting subject...")
            print(f"[SID-Prompt] Detecting subject...")
            subject_result = cls._analyze_component(client, llm_model, base64_image, "subject_detection")
            subject_type = subject_result.get("subject_type", "WOMAN").upper()
            debug_lines.append(f"Subject detection result: {json.dumps(subject_result)}")
            print(f"[SID-Prompt] Subject: {subject_type}")
        pbar.update(1)

        # Step 2: Get components for this mode, filtered by subject type
        mode_config = ANALYSIS_MODES.get(analysis_mode, ANALYSIS_MODES["Standard"])
        mode_components = mode_config["components"].copy()

        # Get allowed components for this subject type
        allowed_components = set(SUBJECT_COMPONENTS.get(subject_type, SUBJECT_COMPONENTS["OTHER"]))

        # Filter mode components to only include those valid for this subject type
        # Also add any subject-specific components that aren't in the base mode
        if subject_type in ["LANDSCAPE", "VEHICLE", "ANIMAL", "PRODUCT"]:
            # For non-human subjects, replace human components with subject-specific ones
            components = [c for c in mode_components if c in allowed_components]
            # Add subject-specific components that might not be in mode_components
            subject_specific = SUBJECT_COMPONENTS.get(subject_type, [])
            for sc in subject_specific:
                if sc not in components and sc in COMPONENTS:
                    components.append(sc)
        else:
            # For human subjects, use mode components as-is
            components = mode_components

        debug_lines.append(f"Subject type: {subject_type}, analyzing {len(components)} components: {components}")
        print(f"[SID-Prompt] Subject type: {subject_type}, analyzing {len(components)} components: {components}")

        # Step 3: Build comprehensive agentic prompt
        agentic_prompt = cls._build_agentic_prompt(components, preset_style, user_guidance, prompt_length)
        pbar.update(1)

        # Step 4: Single reasoning call
        component_results = cls._call_reasoning_llm(client, llm_model, base64_image, agentic_prompt)
        debug_lines.append(f"Raw component results: {json.dumps(component_results, indent=2)}")

        # Step 5: Handle fallback for local Thinking models that output text instead of JSON
        if "_raw_fallback_text" in component_results:
            # JSON parsing failed, use the raw text as the prompt
            raw_text = component_results["_raw_fallback_text"]
            debug_lines.append(f"Using raw fallback text ({len(raw_text)} chars) - JSON parsing failed")
            print(f"[SID-Prompt] Using raw fallback text as prompt ({len(raw_text)} chars)")

            # Clean the raw text
            prompt = cls._clean_output(raw_text, llm_model.provider)

            # If user guidance, try to enhance with string-based approach
            if user_guidance and user_guidance.strip():
                prompt, enhanced = cls._enhance_prompt_with_instructions(prompt, user_guidance, llm_model)
                if enhanced:
                    debug_lines.append("Enhanced fallback prompt with user instructions")

            pbar.update(1)

            metadata = {
                "subject_type": subject_type,
                "subject_detection": subject_result,
                "components_analyzed": components,
                "component_count": len(components),
                "fallback_mode": True,
                "component_results": component_results,  # Contains _raw_fallback_text
            }
            return prompt, metadata, debug_lines

        # Step 6: Assemble prompt from components
        prompt = cls._assemble_prompt(component_results, mode_config, user_guidance)

        # Step 7: If user guidance was provided but user_focus wasn't properly extracted,
        # use second LLM call to apply the modification (same as single-shot pipeline)
        if user_guidance and user_guidance.strip():
            # Check if user_focus was successfully applied (prompt should start with the modification)
            user_focus_raw = component_results.get("user_focus", "")
            user_focus_valid = False
            if isinstance(user_focus_raw, dict):
                user_focus_valid = bool(user_focus_raw.get("user_focus", ""))
            elif isinstance(user_focus_raw, str):
                user_focus_valid = bool(user_focus_raw.strip())

            if not user_focus_valid:
                debug_lines.append("user_focus not found in agentic results, using enhancement call...")
                print("[SID-Prompt] user_focus not found, making enhancement call...")
                prompt, enhanced = cls._enhance_prompt_with_instructions(prompt, user_guidance, llm_model)
                if enhanced:
                    debug_lines.append("Successfully enhanced prompt with user instructions")
                else:
                    debug_lines.append("Enhancement call did not modify prompt")

        pbar.update(1)

        # Build metadata with component data from LLM
        metadata = {
            "subject_type": subject_type,
            "subject_detection": subject_result,
            "components_analyzed": components,
            "component_count": len(components),
            "component_results": component_results,  # Raw data from LLM for each component
        }

        return prompt, metadata, debug_lines

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    @classmethod
    def _image_to_base64(
        cls,
        img_tensor,
        llm_model: LLMModelConfig,
        resize_mode: str = "auto"
    ) -> Tuple[str, Optional[str]]:
        """
        Convert image tensor to base64 with optional resizing.

        Args:
            img_tensor: Image tensor from ComfyUI
            llm_model: LLM model configuration
            resize_mode: "auto", "max", "min", or "original"

        Returns:
            Tuple of (base64_string, resize_info_string or None)
        """
        img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
        pil_image = Image.fromarray(img_np)

        # Get original dimensions
        height, width = img_tensor.shape[0], img_tensor.shape[1]
        original_pixels = width * height
        resize_info = None

        # Get provider resolution limits
        provider = llm_model.provider.lower()
        limits = MODEL_RESOLUTION_LIMITS.get(provider, MODEL_RESOLUTION_LIMITS["default"])

        # Determine target pixels based on resize mode
        if resize_mode == "original":
            # No resizing
            target_pixels = original_pixels
        elif resize_mode == "min":
            # Use minimum pixels for fastest processing
            target_pixels = limits["min_pixels"]
        elif resize_mode == "max":
            # Use maximum pixels for best detail
            target_pixels = limits["max_pixels"]
        else:  # "auto" (default)
            # Use optimal pixels for balance of quality and speed
            target_pixels = limits["optimal_pixels"]

        # Calculate new dimensions if resizing is needed
        if resize_mode != "original" and original_pixels > target_pixels:
            new_width, new_height = calculate_resize_dimensions(width, height, target_pixels)

            # Only resize if there's a meaningful difference (at least 10% reduction)
            if (new_width * new_height) < (original_pixels * 0.9):
                pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                resize_info = f"Image resized: {width}x{height} -> {new_width}x{new_height} ({resize_mode} mode, {original_pixels//1000}K -> {(new_width*new_height)//1000}K pixels)"

        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG", quality=95)
        return base64.standard_b64encode(buffer.getvalue()).decode("utf-8"), resize_info

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
    def _build_system_prompt(cls, provider: str, preset_style: str, user_guidance: str, prompt_length: str = "Free") -> str:
        """Build system prompt using TOML config based on provider tier."""
        return config_loader.build_system_prompt(provider, preset_style, user_guidance, prompt_length)

    @classmethod
    def _build_user_prompt(cls, provider: str, analysis_mode: str, preset_style: str, prompt_length: str = "Free") -> str:
        """Build user prompt using TOML config based on provider tier and mode."""
        return config_loader.build_user_prompt(provider, analysis_mode, preset_style, prompt_length)

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

            # Handle empty or malformed response from LM Studio / local providers
            if response is None:
                raise ValueError("LLM returned None response - check if model is loaded")
            if not hasattr(response, 'choices') or response.choices is None:
                raise ValueError(f"LLM returned invalid response format: {response}")
            if len(response.choices) == 0:
                raise ValueError("LLM returned empty choices array")
            if response.choices[0].message is None or response.choices[0].message.content is None:
                raise ValueError("LLM returned empty message content - model may have failed to generate")

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

                # Validate response from OpenAI-style API
                if response is None or not hasattr(response, 'choices') or response.choices is None:
                    raise ValueError("LLM returned invalid response")
                if len(response.choices) == 0 or response.choices[0].message is None:
                    raise ValueError("LLM returned empty response")
                text = response.choices[0].message.content or ""

            return cls._parse_json(text)
        except Exception as e:
            print(f"[SID-Prompt] Error analyzing {component_key}: {e}")
            return {}

    @classmethod
    def _build_agentic_prompt(cls, components: List[str], preset_style: str, user_guidance: str, prompt_length: str = "Free") -> str:
        """Build comprehensive agentic prompt."""
        comp_specs = []
        for key in components:
            if key in COMPONENTS:
                comp = COMPONENTS[key]
                comp_specs.append(f"### {comp['name']} ({key})\n{comp['prompt']}")

        user_section = ""
        user_focus_instruction = ""
        if user_guidance and user_guidance.strip():
            user_section = f"""

## USER MODIFICATION REQUEST (CRITICAL - MUST APPLY)
"{user_guidance.strip()}"

IMPORTANT: This is a MODIFICATION directive, not just a focus area. You MUST:
1. Apply this modification to ALL relevant components in your analysis
2. Describe the subject AS IF this modification is true, overriding what's literally visible
3. For clothing/nudity changes: describe the MODIFIED state, not the original
4. For other changes: incorporate the modification into your descriptions

Example: If user says "make topless" and image shows bikini, describe as topless (no bikini top).
Example: If user says "wearing red dress" and image shows blue, describe as red dress.
"""
            user_focus_instruction = f"""
### User Modification Applied (user_focus)
The user requested: "{user_guidance.strip()}"
Create a description that APPLIES this modification:
1. Describe the subject AS IF the modification is already applied
2. Do NOT describe the original/visible state that contradicts the user request
3. Be specific and detailed about the modified appearance

Output JSON:
{{"user_focus": "<description with user modification APPLIED, not just acknowledged>"}}
"""

        # Get length constraint
        length_constraint = config_loader.get_length_constraint(prompt_length)
        length_section = f"\n\n## OUTPUT LENGTH\n{length_constraint}" if length_constraint else ""

        return f"""You are an expert visual analyst. Use reasoning to analyze this image comprehensively.
{user_section}
## INSTRUCTIONS
Analyze ALL components below and return a single JSON object with each component as a key.
{"REMEMBER: Apply the user's modification request to ALL relevant component descriptions!" if user_guidance else ""}
{length_section}

## COMPONENTS TO ANALYZE

{chr(10).join(comp_specs)}
{user_focus_instruction}
## OUTPUT FORMAT
Return a single JSON object:
```json
{{
    "framing": {{ ... }},
    "ethnicity": {{ ... }},
    ... etc for each component ...,
    "user_focus": "<description with modification applied>" // Only if user provided guidance
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

            # Try to parse JSON
            parsed = cls._parse_json(text)

            # If JSON parsing failed (empty dict) and we have text, store raw text as fallback
            # This helps local Thinking models that output reasoning text instead of JSON
            if not parsed and text and text.strip():
                raw_text = cls._clean_thinking_text(text)

                # Return special dict with raw text for fallback use
                print(f"[SID-Prompt] JSON parsing failed, storing {len(raw_text)} chars as fallback")
                return {"_raw_fallback_text": raw_text}

            return parsed
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
    def _clean_thinking_text(cls, text: str) -> str:
        """
        Clean output from Thinking models by extracting useful content.

        Handles:
        - <think>...</think> tags (extract content after or inside)
        - Chain-of-thought reasoning patterns
        - Ensures non-empty output
        """
        if not text:
            return ""

        original_text = text

        # Method 1: Try to extract content AFTER </think> tag
        think_end_match = re.search(r'</think>\s*(.+)', text, re.DOTALL | re.IGNORECASE)
        if think_end_match:
            after_think = think_end_match.group(1).strip()
            if after_think and len(after_think) > 30:
                text = after_think
            else:
                # Method 2: Extract from INSIDE <think> tags if nothing useful after
                think_content_match = re.search(r'<think>([\s\S]*?)</think>', original_text, re.IGNORECASE)
                if think_content_match:
                    text = think_content_match.group(1).strip()

        # Method 3: Remove reasoning markers and preambles
        reasoning_patterns = [
            r'^(?:Let me|I will|I\'ll|First,|Now,|Okay,|Alright,)[^.]*\.\s*',
            r'^(?:We are given|The user|Looking at|Based on|Given the)[^.]*\.\s*',
            r'^(?:Step \d+:|Steps?:)[^\n]*\n?',
            r'(?:Therefore|So|Thus|Hence|In conclusion)[,:]?\s*',
            r'^\s*\d+\.\s+[^\n]*\n',  # Numbered steps at start
        ]
        for pattern in reasoning_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)

        # Method 4: Look for final answer markers
        final_markers = [
            r'(?:Final (?:answer|prompt|description)[:\s]*)(.*)',
            r'(?:Here is the (?:prompt|description)[:\s]*)(.*)',
            r'(?:The (?:prompt|description) is[:\s]*)(.*)',
        ]
        for pattern in final_markers:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match and match.group(1).strip():
                candidate = match.group(1).strip()
                if len(candidate) > 30:
                    text = candidate
                    break

        # Clean up whitespace
        text = re.sub(r'\n{2,}', ' ', text)
        text = re.sub(r'\s{2,}', ' ', text)
        text = text.strip()

        # Never return empty - use original if all cleaning failed
        if not text or len(text) < 20:
            # Try to find the longest sentence-like segment
            sentences = re.split(r'[.!?]\s+', original_text)
            if sentences:
                longest = max(sentences, key=len)
                if len(longest) > 20:
                    text = longest.strip()
                else:
                    text = original_text.strip()

        return text

    @classmethod
    def _assemble_prompt(cls, components: dict, mode_config: dict, user_guidance: str) -> str:
        """Assemble final prompt from components with example text filtering and user modification handling."""
        sections = []

        # Check if user_focus contains modification (should override clothing/body_figure)
        user_focus = ""
        if user_guidance and user_guidance.strip():
            user_focus_raw = components.get("user_focus", "")
            if isinstance(user_focus_raw, dict):
                user_focus = user_focus_raw.get("user_focus", "")
            elif isinstance(user_focus_raw, str):
                user_focus = user_focus_raw

        # Determine which components might be overridden by user modification
        # Common modification targets: clothing, body_figure, intimate_apparel
        override_keywords = ["topless", "nude", "naked", "dress", "wear", "outfit", "clothing", "bikini", "lingerie"]
        user_guidance_lower = user_guidance.lower() if user_guidance else ""
        has_clothing_modification = any(kw in user_guidance_lower for kw in override_keywords)

        # Order: framing, ethnicity, face, hair, eyes, nose_lips, body_pose, body_figure, clothing, intimate, tattoos, accessories, lighting
        # Also include landscape/vehicle/animal/product components
        order = [
            "framing", "ethnicity", "face", "hair", "eyes", "nose_lips", "body_pose", "body_figure",
            "clothing", "intimate_apparel", "tattoos", "accessories", "lighting",
            "landscape_terrain", "landscape_weather", "landscape_scenery",
            "vehicle_details", "animal_details", "product_details"
        ]

        # Components that user_focus might override
        clothing_components = {"clothing", "intimate_apparel", "body_figure"}

        for key in order:
            if key in components and components[key]:
                desc = components[key].get("prompt_description", "")
                if desc:
                    # Skip clothing-related components if user guidance has clothing modification
                    # (user_focus will provide the modified description instead)
                    if has_clothing_modification and key in clothing_components and user_focus:
                        continue

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

        # Add user modification at the BEGINNING (highest priority)
        # Note: user_focus handling is done in execute methods where we have access to llm_model
        # This method now just returns the base prompt; user_focus/enhancement is handled by caller
        if user_guidance and user_guidance.strip():
            if user_focus and isinstance(user_focus, str) and user_focus.strip():
                # Use the LLM's modified description at the start
                prompt = f"{user_focus.strip()}. {prompt}"
            # No else fallback here - let the caller handle enhancement

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

    @classmethod
    def _enhance_prompt_with_instructions(
        cls,
        prompt: str,
        user_guidance: str,
        llm_model: LLMModelConfig,
    ) -> Tuple[str, bool]:
        """
        Enhance/modify the generated prompt based on user instructions.

        For API providers: Makes a second LLM call to intelligently rewrite the prompt.
        For local vision models: Uses string-based prepending since vision models
        don't work well with text-only input.

        Args:
            prompt: The generated prompt from the first LLM call
            user_guidance: User's modification/enhancement request
            llm_model: LLM model configuration

        Returns:
            Tuple of (enhanced_prompt, was_enhanced)
        """
        if not prompt or not user_guidance or not user_guidance.strip():
            return prompt, False

        print(f"[SID-Prompt] Enhancing prompt with instructions: {user_guidance[:50]}...")

        # For local vision models, use string-based enhancement
        # Vision models don't work well with text-only LLM calls
        provider_lower = llm_model.provider.lower()
        if provider_lower == "local":
            # Simple string-based approach: prepend the modification instruction
            # and let the image generation model interpret it
            enhanced = cls._apply_string_enhancement(prompt, user_guidance)
            print(f"[SID-Prompt] Applied string-based enhancement for local model")
            return enhanced, True

        try:
            client = cls._get_client(llm_model)

            enhancement_system = """You are an expert prompt modifier for AI image generation.
Your task is to modify an existing image description prompt based on user instructions.

RULES:
1. Apply the user's modification to the prompt
2. Keep all other details from the original prompt
3. Remove any descriptions that conflict with the user's modification
4. Output ONLY the modified prompt text - no explanations, no markdown, no quotes
5. Maintain the same descriptive style and flow as the original

Examples:
- If user says "make topless" and prompt mentions bikini top, remove bikini top description and add "topless, bare chest"
- If user says "change hair to blonde", find hair description and change the color
- If user says "add a red dress", replace/add clothing description accordingly"""

            enhancement_prompt = f"""Original prompt:
{prompt}

User modification request: {user_guidance.strip()}

Apply the user's modification to the prompt. Output ONLY the modified prompt:"""

            # Make text-only LLM call (no image needed)
            if hasattr(client, 'messages'):
                # Anthropic
                response = client.messages.create(
                    model=llm_model.model,
                    max_tokens=2000,
                    temperature=0.3,
                    system=enhancement_system,
                    messages=[{"role": "user", "content": enhancement_prompt}]
                )
                enhanced = response.content[0].text.strip()
            else:
                # OpenAI-style (OpenAI, Gemini, Groq, Together, etc.)
                response = client.chat.completions.create(
                    model=llm_model.model,
                    max_tokens=2000,
                    temperature=0.3,
                    messages=[
                        {"role": "system", "content": enhancement_system},
                        {"role": "user", "content": enhancement_prompt}
                    ]
                )
                enhanced = response.choices[0].message.content.strip()

            # Clean up the response
            enhanced = cls._clean_output(enhanced, llm_model.provider)

            # Verify we got a valid response
            if enhanced and len(enhanced) > 20:
                print(f"[SID-Prompt] Enhanced prompt: {len(enhanced)} chars")
                return enhanced, True
            else:
                print(f"[SID-Prompt] Enhancement response too short, using string-based fallback")
                return cls._apply_string_enhancement(prompt, user_guidance), True

        except Exception as e:
            print(f"[SID-Prompt] Enhancement failed: {e}, using string-based fallback")
            return cls._apply_string_enhancement(prompt, user_guidance), True

    @classmethod
    def _apply_string_enhancement(cls, prompt: str, user_guidance: str) -> str:
        """
        Apply user guidance to prompt using string-based approach.

        This is used for local vision models that can't do text-only generation,
        or as a fallback when LLM enhancement fails.

        The approach is to intelligently prepend or modify the prompt based on
        common modification patterns.
        """
        guidance_lower = user_guidance.lower().strip()

        # Common clothing/appearance modifications - add at the beginning
        clothing_keywords = ["wear", "dress", "outfit", "clothing", "topless", "nude",
                           "naked", "bikini", "lingerie", "shirt", "pants", "skirt"]
        color_keywords = ["yellow", "red", "blue", "green", "black", "white", "pink",
                         "purple", "orange", "color", "colour"]
        hair_keywords = ["hair", "blonde", "brunette", "redhead", "braid", "ponytail"]

        # Check if this is a clothing/color modification
        is_clothing_mod = any(kw in guidance_lower for kw in clothing_keywords)
        is_color_mod = any(kw in guidance_lower for kw in color_keywords)
        is_hair_mod = any(kw in guidance_lower for kw in hair_keywords)

        if is_clothing_mod or is_color_mod:
            # For clothing/color mods, extract the key modification
            mod_text = user_guidance.strip()

            # Convert instruction to descriptive form by removing common prefixes
            prefixes_to_remove = [
                "make ", "change to ", "convert to ",
                "make the ", "change the ", "convert the ",
                "make it ", "change it to ", "convert it to ",
            ]
            mod_lower = mod_text.lower()
            for prefix in prefixes_to_remove:
                if mod_lower.startswith(prefix):
                    mod_text = mod_text[len(prefix):]
                    mod_lower = mod_text.lower()
                    break

            # Handle "X into Y" patterns (e.g., "dress into yellow colour")
            if " into " in mod_lower:
                # Extract what's after "into" as the key descriptor
                parts = mod_text.split(" into ", 1)
                if len(parts) == 2:
                    # "dress into yellow colour" -> "yellow colour dress"
                    subject = parts[0].strip()
                    descriptor = parts[1].strip()
                    mod_text = f"{descriptor} {subject}"

            # Build modified prompt
            return f"{mod_text}, {prompt}"

        elif is_hair_mod:
            # For hair mods, extract the modification
            mod_text = user_guidance.strip()
            prefixes_to_remove = ["make ", "make the ", "change to ", "change the "]
            mod_lower = mod_text.lower()
            for prefix in prefixes_to_remove:
                if mod_lower.startswith(prefix):
                    mod_text = mod_text[len(prefix):]
                    break
            return f"with {mod_text}, {prompt}"

        else:
            # General case: prepend as a style/modification directive
            return f"{user_guidance.strip()}, {prompt}"
