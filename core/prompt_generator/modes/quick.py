"""
Quick mode - Single-shot LLM prompt generation.

Fast single LLM call without any tagger execution.
Uses a detailed prompt to extract comprehensive image description.
"""

import base64
import io
import time
from typing import Any, Optional

import numpy as np
from PIL import Image

from .base import BaseMode
from ..types import GeneratorResult


class QuickMode(BaseMode):
    """
    Quick mode implementation.

    Makes a single LLM call with a comprehensive prompt to
    generate a detailed image description. No taggers are executed.
    """

    # System prompt optimized for Z-Image/Flux generation
    SYSTEM_PROMPT = """You are an expert visual analyst creating prompts for AI image generation (Z-Image/Flux).

CRITICAL RULES:
- Use CONCRETE, SPECIFIC, VISUAL descriptions only
- Describe what you SEE, not what you interpret or feel
- NO poetic, philosophical, abstract, or emotional language
- NO metaphors, symbolism, or artistic interpretation
- NO phrases like "captures the essence", "evokes a sense of", "speaks to"
- NO commentary about meaning, mood interpretation, or artistic intent
- NO meta-commentary (no "the image shows", "in the photo", "this photograph")
- Focus on: subject details, colors, textures, lighting, composition, materials

Write precise, literal descriptions of observable visual elements only."""

    # User prompt for image analysis
    USER_PROMPT = """Describe this image for AI image generation using ONLY concrete visual details.

REQUIRED (describe each):
1. SUBJECT: Specific identity (gender, age, ethnicity, build)
2. FACE: Shape, features, expression, eye color, skin tone
3. HAIR: Exact color, length, style, texture
4. BODY: Visible anatomy, proportions, skin details
5. CLOTHING: Each garment with color, material, fit, style
6. POSE: Exact body position, hand placement, stance
7. ENVIRONMENT: Specific location, objects, surfaces, textures
8. LIGHTING: Direction, quality, color temperature, shadows
9. CAMERA: Angle, framing, focus, depth of field

Write as a single flowing paragraph. NO poetic language. Be literal and specific."""

    @property
    def name(self) -> str:
        return "quick"

    @property
    def requires_llm(self) -> bool:
        return True

    @property
    def requires_taggers(self) -> bool:
        return False

    def execute(
        self,
        image: Any = None,
        llm_model: Any = None,
        prompt_config: Optional[dict] = None,
        **kwargs
    ) -> GeneratorResult:
        """
        Execute Quick mode.

        Args:
            image: Image tensor from ComfyUI
            llm_model: LLMModelConfig with provider settings
            prompt_config: Optional custom system/user prompts
            **kwargs: Ignored

        Returns:
            GeneratorResult with LLM-generated description
        """
        if image is None:
            return GeneratorResult(
                prompt="[No image provided]",
                metadata={"mode": "quick", "error": "No image provided"}
            )

        if llm_model is None:
            return GeneratorResult(
                prompt="[No LLM model provided]",
                metadata={"mode": "quick", "error": "No LLM model provided"}
            )

        start_time = time.time()

        try:
            # Convert image to base64
            base64_image = self._image_to_base64(image, llm_model)

            # Get LLM client
            client = self._get_client(llm_model)

            # Use custom prompts if provided, otherwise use defaults
            system_prompt = self.SYSTEM_PROMPT
            user_prompt = self.USER_PROMPT
            if prompt_config:
                system_prompt = prompt_config.get("system_prompt", system_prompt)
                user_prompt = prompt_config.get("user_prompt", user_prompt)

            # Make LLM call
            response = self._call_llm(
                client=client,
                llm_model=llm_model,
                base64_image=base64_image,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )

            # Clean response
            prompt = self._clean_response(response)

            inference_time = int((time.time() - start_time) * 1000)

            return GeneratorResult(
                prompt=prompt,
                metadata={
                    "mode": "quick",
                    "source": "llm_single_shot",
                    "llm_provider": llm_model.provider,
                    "llm_model": llm_model.model,
                    "processing": {
                        "taggers_executed": False,
                        "florence_executed": False,
                        "llm_executed": True,
                        "llm_passes": 1,
                    },
                    "timing": {
                        "llm_ms": inference_time,
                    }
                }
            )

        except Exception as e:
            error_msg = str(e)
            print(f"[Quick] Error: {error_msg}")
            import traceback
            traceback.print_exc()

            return GeneratorResult(
                prompt=f"[Quick mode error: {error_msg}]",
                metadata={
                    "mode": "quick",
                    "error": error_msg,
                    "llm_provider": getattr(llm_model, 'provider', 'unknown'),
                }
            )

    def _image_to_base64(self, image_tensor: Any, llm_model: Any) -> str:
        """Convert image tensor to base64 string."""
        # Get image as numpy array
        if hasattr(image_tensor, 'cpu'):
            img_np = image_tensor[0].cpu().numpy()
        else:
            img_np = np.array(image_tensor[0])

        # Convert to uint8
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
        else:
            img_np = img_np.astype(np.uint8)

        # Create PIL image
        pil_image = Image.fromarray(img_np)

        # Optionally resize based on provider
        pil_image = self._resize_for_provider(pil_image, llm_model.provider)

        # Convert to base64
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG", quality=95)
        return base64.standard_b64encode(buffer.getvalue()).decode("utf-8")

    def _resize_for_provider(self, image: Image.Image, provider: str) -> Image.Image:
        """Resize image to optimal size for provider."""
        # Provider-specific optimal resolutions
        optimal_pixels = {
            "anthropic": 1024 * 1024,
            "openai": 1024 * 1024,
            "gemini": 1024 * 1024,
            "local": 672 * 672,
            "ollama": 672 * 672,
            "lmstudio": 672 * 672,
        }

        target_pixels = optimal_pixels.get(provider.lower(), 768 * 768)
        current_pixels = image.width * image.height

        if current_pixels > target_pixels:
            scale = (target_pixels / current_pixels) ** 0.5
            new_width = max(64, int(image.width * scale))
            new_height = max(64, int(image.height * scale))
            # Round to nearest 8 for model compatibility
            new_width = (new_width // 8) * 8
            new_height = (new_height // 8) * 8
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        return image

    def _get_client(self, llm_model: Any) -> Any:
        """Get LLM client based on provider."""
        provider = llm_model.provider.lower()

        if provider == "anthropic":
            import anthropic
            import httpx
            timeout = httpx.Timeout(timeout=300.0, connect=30.0)
            return anthropic.Anthropic(api_key=llm_model.api_key, timeout=timeout)

        elif provider == "local":
            # Import local model client (4 dots: modes -> prompt_generator -> core -> package root)
            from ....llm_providers.sid_llm_local import LocalModelClient
            extra = llm_model.extra_params or {}
            return LocalModelClient(
                model_name=llm_model.model,
                quantization=extra.get("quantization", "4-bit"),
                device=extra.get("device", "auto"),
                attention_mode=extra.get("attention_mode", "auto"),
                keep_model_loaded=extra.get("keep_model_loaded", True),
                hf_token=extra.get("hf_token"),
            )

        else:
            # OpenAI-compatible providers
            from openai import OpenAI
            return OpenAI(
                api_key=llm_model.api_key or "not-needed",
                base_url=llm_model.api_url if llm_model.api_url else None,
            )

    def _call_llm(
        self,
        client: Any,
        llm_model: Any,
        base64_image: str,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """Make LLM call with image."""
        if hasattr(client, 'messages'):
            # Anthropic API
            response = client.messages.create(
                model=llm_model.model,
                max_tokens=llm_model.max_tokens,
                temperature=llm_model.temperature,
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
                        {"type": "text", "text": user_prompt}
                    ]
                }]
            )
            return response.content[0].text

        else:
            # OpenAI-compatible API
            response = client.chat.completions.create(
                model=llm_model.model,
                max_tokens=llm_model.max_tokens,
                temperature=llm_model.temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                            },
                            {"type": "text", "text": user_prompt}
                        ]
                    }
                ]
            )

            # Validate response
            if not response or not response.choices or not response.choices[0].message:
                raise ValueError("LLM returned empty response")

            return response.choices[0].message.content

    def _clean_response(self, text: str) -> str:
        """Clean LLM response text."""
        import re

        if not text:
            return ""

        # Remove common preambles
        preambles = [
            r'^(?:Here is|Here\'s|This is|The image shows|In this image|This image depicts)[:\s]*',
            r'^(?:I can see|I see|Looking at)[:\s]*',
        ]
        for pattern in preambles:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)

        # Remove markdown formatting
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'\*([^*]+)\*', r'\1', text)

        # Clean whitespace
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\s+', ' ', text)

        return text.strip()
