"""
SID_ZImagePromptGenerator Node (Basic)

Simple, user-friendly Z-Image prompt generator with preset options.
Uses LLM_MODEL input from provider nodes for flexibility.

Key Features:
- Preset prompt styles for common use cases
- Simple detail level selection
- Single-shot generation (fast)
- Optional custom guidance

For advanced options, use SID_ZImagePromptGenerator_Advanced_V2.
"""

import base64
import io
import re
import time
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image
from comfy_api.latest import io as comfy_io
import comfy.utils

from .llm_providers.llm_model_type import LLMModelConfig

# Create custom LLM_MODEL type for ComfyUI
LLM_MODEL_Type = comfy_io.Custom("LLM_MODEL")


# Preset prompt styles - user-friendly presets like QwenVL
PRESET_STYLES = {
    "Detailed Description": {
        "emoji": "📸",
        "description": "Comprehensive image description with all visible details",
        "system": """You are an expert visual analyst for text-to-image AI prompts.
Generate a detailed, flowing narrative prompt that captures all visible elements in the image.
Focus on: composition, subject features, clothing, colors, materials, lighting, and background.
Output ONLY the prompt text - no explanations or formatting.""",
        "user": """Analyze this image and generate a detailed prompt covering:
1. Shot type and composition
2. Subject description (features, pose, expression)
3. Clothing/attire (colors, materials, style)
4. Lighting and shadows
5. Background/environment

Generate a natural flowing paragraph combining all elements."""
    },

    "Portrait Focus": {
        "emoji": "👤",
        "description": "Emphasizes facial features, expression, and upper body",
        "system": """You are an expert portrait analyst for text-to-image AI prompts.
Generate a detailed prompt focused on the subject's appearance and expression.
Emphasize: face shape, features, skin tone, hair, expression, and visible clothing.
Output ONLY the prompt text - no explanations or formatting.""",
        "user": """Analyze this portrait and generate a prompt covering:
1. Face shape, skin tone, ethnicity
2. Facial features (eyes, nose, lips)
3. Hair style, color, and arrangement
4. Expression and gaze
5. Upper body and clothing
6. Lighting on the face

Generate a natural flowing paragraph."""
    },

    "Fashion & Outfit": {
        "emoji": "👗",
        "description": "Detailed clothing and styling analysis",
        "system": """You are an expert fashion analyst for text-to-image AI prompts.
Generate a detailed prompt focused on clothing, accessories, and styling.
Emphasize: garment types, colors, materials, fit, and accessories.
Output ONLY the prompt text - no explanations or formatting.""",
        "user": """Analyze this image focusing on fashion and generate a prompt covering:
1. Garment types (dress, top, pants, etc.)
2. Colors and patterns
3. Materials and textures
4. Fit and style
5. Accessories (jewelry, bags, etc.)
6. Overall fashion aesthetic

Generate a natural flowing paragraph."""
    },

    "Artistic Style": {
        "emoji": "🎨",
        "description": "Focuses on artistic and photographic qualities",
        "system": """You are an expert art and photography analyst for text-to-image AI prompts.
Generate a prompt that captures the artistic and photographic qualities of the image.
Emphasize: composition, lighting style, color palette, mood, and artistic techniques.
Output ONLY the prompt text - no explanations or formatting.""",
        "user": """Analyze this image's artistic qualities and generate a prompt covering:
1. Composition and framing
2. Lighting style and direction
3. Color palette and tones
4. Mood and atmosphere
5. Photographic style (studio, natural, editorial)
6. Any artistic effects

Generate a natural flowing paragraph."""
    },

    "Quick Caption": {
        "emoji": "⚡",
        "description": "Brief, concise description (faster)",
        "system": """You are a concise image captioner.
Generate a short, accurate prompt describing the main elements of the image.
Keep it under 100 words. Focus on the most important visual elements.
Output ONLY the prompt text.""",
        "user": "Generate a brief, accurate caption for this image in 50-100 words."
    },

    "NSFW/Detailed": {
        "emoji": "🔞",
        "description": "Comprehensive body and clothing analysis (adults only)",
        "system": """You are an expert visual analyst for detailed image prompts.
Generate a comprehensive prompt including body features and clothing details.
Be specific about: body proportions, skin, clothing coverage, and exposure.
This is for adult content generation. Be accurate and detailed.
Output ONLY the prompt text - no explanations or formatting.""",
        "user": """Analyze this image comprehensively and generate a detailed prompt covering:
1. Framing and composition
2. Subject's body (build, proportions, skin tone)
3. Facial features and expression
4. Hair style and color
5. All clothing/intimate apparel details
6. Body positioning and pose
7. Lighting and background

Generate a detailed, flowing paragraph. Include all visible details."""
    },
}


class SID_ZImagePromptGenerator(comfy_io.ComfyNode):
    """
    Simple Z-Image Prompt Generator.

    Easy-to-use node with preset styles for common use cases.
    Connect any LLM provider node (Anthropic, OpenAI, QwenVL, etc.).

    For advanced options, use SID_ZImagePromptGenerator_Advanced_V2.
    """

    @classmethod
    def define_schema(cls) -> comfy_io.Schema:
        """Define the node schema with simple, user-friendly options."""

        # Build preset options with emojis
        preset_options = [
            f"{v['emoji']} {k}" for k, v in PRESET_STYLES.items()
        ]

        return comfy_io.Schema(
            node_id="SID_ZImagePromptGenerator",
            display_name="SID Z-Image Prompt Generator",
            category="SID Photography Toolkit/Z-Image",
            description="Simple Z-Image prompt generator. Connect LLM provider, select style, generate!",
            is_output_node=True,
            inputs=[
                comfy_io.Image.Input(
                    "image",
                    tooltip="Input image to analyze"
                ),

                LLM_MODEL_Type.Input(
                    "llm_model",
                    tooltip="Connect LLM provider node (SID_Anthropic_LLM, SID_QwenVL_LLM, etc.)"
                ),

                # Simple preset selection like QwenVL
                comfy_io.Combo.Input(
                    "preset_style",
                    options=preset_options,
                    default=preset_options[0],  # Detailed Description
                    tooltip="Preset analysis style - determines what aspects to emphasize"
                ),

                # Optional custom guidance
                comfy_io.String.Input(
                    "custom_guidance",
                    default="",
                    multiline=True,
                    tooltip="Optional: Add specific instructions (e.g., 'focus on the red dress', 'emphasize the lighting')"
                ),

                # Simple seed for reproducibility
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
                comfy_io.String.Output(
                    "prompt",
                    display_name="prompt",
                    tooltip="Generated Z-Image prompt"
                ),
                comfy_io.Int.Output(
                    "width",
                    display_name="width",
                    tooltip="Image width"
                ),
                comfy_io.Int.Output(
                    "height",
                    display_name="height",
                    tooltip="Image height"
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        image,
        llm_model: LLMModelConfig,
        preset_style: str,
        custom_guidance: str,
        seed: int,
    ) -> comfy_io.NodeOutput:
        """Execute the prompt generation."""

        start_time = time.time()

        def log(msg: str):
            print(f"[SID-Basic] {msg}")

        # Get image dimensions
        img_tensor = image[0]
        height, width = img_tensor.shape[0], img_tensor.shape[1]

        log("=" * 50)
        log("SID Z-Image Prompt Generator (Basic)")
        log("=" * 50)
        log(f"Style: {preset_style}")
        log(f"Image: {width}x{height}")
        log(f"Provider: {llm_model.provider}")
        log(f"Model: {llm_model.model}")

        try:
            # Parse preset style (remove emoji)
            style_name = preset_style.split(" ", 1)[1] if " " in preset_style else preset_style
            preset = PRESET_STYLES.get(style_name, PRESET_STYLES["Detailed Description"])

            log(f"Using preset: {style_name}")

            # Convert image to base64
            img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
            pil_image = Image.fromarray(img_np)

            # Check for max_image_size optimization
            max_image_size = llm_model.extra_params.get("max_image_size") if llm_model.extra_params else None
            if max_image_size and max(width, height) > max_image_size:
                if width > height:
                    new_width = max_image_size
                    new_height = int(height * (max_image_size / width))
                else:
                    new_height = max_image_size
                    new_width = int(width * (max_image_size / height))
                pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                log(f"Image resized: {width}x{height} -> {new_width}x{new_height}")

            buffer = io.BytesIO()
            pil_image.save(buffer, format="JPEG", quality=95)
            base64_image = base64.standard_b64encode(buffer.getvalue()).decode("utf-8")

            # Build prompts
            system_prompt = preset["system"]
            user_prompt = preset["user"]

            # Add custom guidance if provided
            if custom_guidance and custom_guidance.strip():
                user_prompt += f"\n\nADDITIONAL GUIDANCE: {custom_guidance.strip()}"
                log(f"Custom guidance: {custom_guidance.strip()[:50]}...")

            # Get client and make LLM call
            client = cls._get_client(llm_model)

            # Progress bar
            pbar = comfy.utils.ProgressBar(2)
            log("Generating prompt...")
            pbar.update(1)

            prompt = cls._call_llm(
                client,
                llm_model,
                base64_image,
                system_prompt,
                user_prompt
            )

            # Clean output
            prompt = cls._clean_output(prompt)

            # Add custom guidance prefix if significant
            if custom_guidance and custom_guidance.strip() and len(custom_guidance.strip()) > 10:
                prompt = f"[FOCUS: {custom_guidance.strip()}] {prompt}"

            pbar.update(1)

            # Stats
            total_time = time.time() - start_time
            word_count = len(prompt.split())

            log(f"Generated: {word_count} words")
            log(f"Time: {total_time:.1f}s")
            log("=" * 50)

            return comfy_io.NodeOutput(prompt, width, height, ui={"text": (prompt,)})

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            log(error_msg)
            import traceback
            traceback.print_exc()
            return comfy_io.NodeOutput(error_msg, width, height, ui={"text": (error_msg,)})

    @classmethod
    def _get_client(cls, llm_model: LLMModelConfig):
        """Get the appropriate LLM client based on provider."""
        provider = llm_model.provider.lower()

        if provider == "anthropic":
            import anthropic
            return anthropic.Anthropic(api_key=llm_model.api_key)

        elif provider in ["openai", "openai_compatible"]:
            from openai import OpenAI
            return OpenAI(
                api_key=llm_model.api_key or "not-needed",
                base_url=llm_model.api_url if llm_model.api_url else None
            )

        elif provider == "grok":
            from openai import OpenAI
            return OpenAI(
                api_key=llm_model.api_key,
                base_url="https://api.x.ai/v1"
            )

        elif provider == "gguf":
            from .llm_providers.sid_gguf_llm import LocalGGUFClient
            extra = llm_model.extra_params or {}
            return LocalGGUFClient(
                model_path=extra.get("model_path", ""),
                mmproj_path=extra.get("mmproj_path"),
                chat_format=extra.get("chat_format", "llava-1-5"),
                n_ctx=extra.get("n_ctx", 4096),
                n_gpu_layers=extra.get("n_gpu_layers", -1),
                verbose=False,
            )

        elif provider == "qwenvl":
            from .llm_providers.sid_qwenvl_llm import QwenVLClient
            extra = llm_model.extra_params or {}
            return QwenVLClient(
                model_name=llm_model.model,
                quantization=extra.get("quantization", "4-bit"),
                device=extra.get("device", "auto"),
                attention_mode=extra.get("attention_mode", "auto"),
                keep_model_loaded=extra.get("keep_model_loaded", True),
                top_p=extra.get("top_p", 0.9),
                repetition_penalty=extra.get("repetition_penalty", 1.2),
                num_beams=extra.get("num_beams", 1),
            )

        else:
            raise ValueError(f"Unsupported provider: {provider}")

    @classmethod
    def _call_llm(
        cls,
        client,
        llm_model: LLMModelConfig,
        base64_image: str,
        system_prompt: str,
        user_prompt: str
    ) -> str:
        """Make LLM call with image."""

        model = llm_model.model
        max_tokens = llm_model.max_tokens
        temperature = llm_model.temperature

        if hasattr(client, 'messages'):
            # Anthropic
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": base64_image,
                            }
                        },
                        {
                            "type": "text",
                            "text": user_prompt
                        }
                    ]
                }]
            )
            return response.content[0].text

        else:
            # OpenAI-style (including QwenVL, GGUF)
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
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            },
                            {
                                "type": "text",
                                "text": user_prompt
                            }
                        ]
                    }
                ]
            )
            return response.choices[0].message.content

    @classmethod
    def _clean_output(cls, text: str) -> str:
        """Clean up LLM output."""

        # Remove common prefixes
        prefixes_to_remove = [
            r"^Here'?s? (?:the |a )?(?:detailed |comprehensive )?prompt[:\s]*",
            r"^(?:The )?prompt[:\s]*",
            r"^Output[:\s]*",
            r"^Description[:\s]*",
            r"^Caption[:\s]*",
        ]
        for pattern in prefixes_to_remove:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)

        # Remove markdown
        text = re.sub(r'```[a-z]*\n?', '', text)
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)

        # Clean whitespace
        text = re.sub(r'\n\s*\n', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()

        return text
