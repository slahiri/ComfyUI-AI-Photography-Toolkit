# -*- coding: utf-8 -*-
"""
ComfyUI-AI-Photography-Toolkit - Preset Prompts

Predefined prompts for common image analysis tasks with local vision models,
following the style established by ComfyUI-QwenVL.

Author: Siddhartha Lahiri
Email: siddhartha.lahiri@gmail.com
License: MIT
"""

from typing import Dict, List


# Preset prompt names for UI display
PRESET_PROMPT_NAMES: List[str] = [
    "Custom (Use custom_prompt)",
    "Tags",
    "Simple Description",
    "Detailed Description",
    "Ultra Detailed Description",
    "Cinematic Description",
    "Detailed Analysis",
    "Prompt Refine & Expand",
]

# System prompts for each preset
PRESET_PROMPTS: Dict[str, str] = {
    "Tags": (
        "Your task is to generate a clean list of comma-separated tags for a text-to-image AI, "
        "based *only* on the visual information in the image. Limit the output to a maximum of "
        "50 unique tags. Strictly describe visual elements like subject, clothing, environment, "
        "colors, lighting, and composition. Do not include abstract concepts, interpretations, "
        "marketing terms, or technical jargon (e.g., no 'SEO', 'brand-aligned', 'viral potential'). "
        "The goal is a concise list of visual descriptors. Avoid repeating tags."
    ),

    "Simple Description": (
        "Analyze the image and write a single concise sentence that describes the main subject "
        "and setting. Keep it grounded in visible details only."
    ),

    "Detailed Description": (
        "Generate a detailed paragraph that combines the subject, actions, environment, lighting, "
        "and mood into 2-3 cohesive sentences. Focus on accurate visual details rather than speculation."
    ),

    "Ultra Detailed Description": (
        "Produce an extremely rich description touching on appearance, clothing textures, background "
        "elements, light quality, shadows, and atmosphere. Aim for an immersive depiction rooted in "
        "what the image shows."
    ),

    "Cinematic Description": (
        "Describe the scene as if capturing a cinematic shot. Cover subject, pose, environment, "
        "lighting, mood, and artistic style (photorealistic, painterly, etc.) in one vivid paragraph "
        "emphasizing visual impact."
    ),

    "Detailed Analysis": (
        "Describe this image in detail, breaking down the subject, attire, accessories, background, "
        "and composition into separate sections."
    ),

    "Prompt Refine & Expand": (
        "Refine and enhance the following user prompt for creative text-to-image generation. "
        "Keep the meaning and keywords, make it more expressive and visually rich. "
        "Output **only the improved prompt text itself**, without any reasoning steps, "
        "thinking process, or additional commentary."
    ),
}


def get_preset_prompt(preset_name: str, custom_prompt: str = "") -> str:
    """
    Get the prompt text for a given preset name.

    Args:
        preset_name: Name of the preset (from PRESET_PROMPT_NAMES)
        custom_prompt: Custom prompt text (used when preset is "Custom")

    Returns:
        The prompt text to use
    """
    if preset_name == "Custom (Use custom_prompt)" or preset_name.startswith("Custom"):
        return custom_prompt.strip() if custom_prompt else "Describe this image in detail."

    return PRESET_PROMPTS.get(preset_name, custom_prompt or "Describe this image in detail.")


def get_preset_names() -> List[str]:
    """Get list of available preset names."""
    return PRESET_PROMPT_NAMES.copy()
