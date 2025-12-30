"""
SID_UserPrompt - Custom system and user prompts for LLM modes.

Allows users to customize the prompts sent to the LLM for image analysis.
"""

from typing import Any, Dict, Tuple

# Default prompts (from Quick mode)
DEFAULT_SYSTEM_PROMPT = """You are an expert visual analyst specializing in creating detailed image descriptions for AI image generation.

Your task is to describe the image in comprehensive detail, suitable for recreating it with an AI image generator.

Guidelines:
- Be specific and detailed about all visual elements
- Use natural, flowing language
- Include subject, appearance, clothing, pose, environment, lighting, and composition
- Do NOT include meta-commentary about the image itself (no "the image shows" or "in the photo")
- Do NOT make assumptions about what you cannot see
- Write as a single flowing paragraph

Focus on factual visual description only."""

DEFAULT_USER_PROMPT = """Describe this image in detail for AI image generation.

Include:
1. SUBJECT: Who or what is the main subject
2. APPEARANCE: Physical features, ethnicity, age impression, skin tone
3. FACE: Features, expression, gaze direction
4. HAIR: Color, length, style, texture
5. CLOTHING: All visible garments, colors, materials, fit
6. POSE: Body position, gesture, stance
7. ENVIRONMENT: Setting, background, atmosphere
8. LIGHTING: Quality, direction, mood
9. COMPOSITION: Framing, camera angle, depth of field

Write as a single flowing paragraph suitable for image generation. Be specific and detailed."""


class SID_UserPrompt:
    """
    Custom prompt configuration for LLM-based image analysis.

    Provides customizable system and user prompts that override
    the default prompts in Quick and Standard modes.
    """

    CATEGORY = "SID Photography Toolkit"
    RETURN_TYPES = ("USER_PROMPT",)
    RETURN_NAMES = ("user_prompt_config",)
    FUNCTION = "create_config"

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "system_prompt": ("STRING", {
                    "default": DEFAULT_SYSTEM_PROMPT,
                    "multiline": True,
                    "tooltip": "System prompt that sets the LLM's behavior and guidelines"
                }),
                "user_prompt": ("STRING", {
                    "default": DEFAULT_USER_PROMPT,
                    "multiline": True,
                    "tooltip": "User prompt that tells the LLM what to describe"
                }),
            },
        }

    def create_config(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> Tuple[Dict[str, str]]:
        """
        Create prompt configuration.

        Args:
            system_prompt: Custom system prompt for LLM
            user_prompt: Custom user prompt for LLM

        Returns:
            Tuple containing prompt config dict
        """
        config = {
            "system_prompt": system_prompt.strip(),
            "user_prompt": user_prompt.strip(),
        }
        return (config,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "SID_UserPrompt": SID_UserPrompt,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SID_UserPrompt": "SID User Prompt",
}
