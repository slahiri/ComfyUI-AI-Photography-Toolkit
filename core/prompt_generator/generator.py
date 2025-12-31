"""
SID_PromptGenerator - Main generator node.

This is the primary ComfyUI node for prompt generation with automatic
mode selection based on inputs.

Inputs:
    - image (IMAGE): Required - image to analyze
    - llm_model (LLM_MODEL): Optional - LLM configuration
    - user_prompt (USER_PROMPT): Optional - custom system/user prompts from SID User Prompt

Outputs:
    - prompt (STRING): Generated prompt for image generation
    - prompt_metadata (STRING/JSON): Detailed metadata about generation

Mode Selection:
    1. No llm_model -> Florence mode (VLM description)
    2. llm_model + Quick -> Quick mode (single LLM, no tags)
    3. llm_model + Standard -> Standard mode (tags + template + LLM)
    4. llm_model + Detailed -> Detailed mode (tags + multi-pass LLM)
"""

import json
import time
from typing import Any, Dict, Optional, Tuple

import comfy.model_management

from .types import AnalysisMode, GeneratorResult, Decisions, TaggerResults
from .modes.base import BaseMode


def check_interrupt():
    """Check if execution was interrupted and raise exception if so."""
    comfy.model_management.throw_exception_if_processing_interrupted()

# Available Florence models for fallback VLM mode
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


class SID_PromptGenerator:
    """
    Unified prompt generator with automatic mode selection.

    Generates image prompts using multiple strategies based on
    available inputs and selected analysis mode.
    """

    # ComfyUI node configuration
    CATEGORY = "SID Photography Toolkit"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt", "caption")
    FUNCTION = "generate"

    def __init__(self):
        """Initialize the generator with mode implementations."""
        self._modes: Dict[str, BaseMode] = {}
        self._load_modes()

    def _load_modes(self):
        """Lazy-load mode implementations."""
        # Modes will be loaded on first use
        # This avoids import issues during ComfyUI startup
        pass

    def _get_mode(self, mode_name: str) -> Optional[BaseMode]:
        """Get a mode implementation by name."""
        if mode_name not in self._modes:
            self._load_mode(mode_name)
        return self._modes.get(mode_name)

    def _load_mode(self, mode_name: str):
        """Load a specific mode implementation."""
        try:
            if mode_name == "florence":
                from .modes.florence import FlorenceMode
                self._modes["florence"] = FlorenceMode()
            elif mode_name == "quick":
                from .modes.quick import QuickMode
                self._modes["quick"] = QuickMode()
            elif mode_name == "standard":
                from .modes.standard import StandardMode
                self._modes["standard"] = StandardMode()
            elif mode_name == "detailed":
                from .modes.detailed import DetailedMode
                self._modes["detailed"] = DetailedMode()
        except ImportError as e:
            print(f"[SID_PromptGenerator] Warning: Could not load {mode_name} mode: {e}")

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        """Define ComfyUI input types."""
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "llm_model": ("LLM_MODEL",),
                "user_prompt": ("USER_PROMPT", {"tooltip": "Custom system/user prompts from SID User Prompt node"}),
                "analysis_mode": (["Quick", "Standard", "Detailed"], {"default": "Standard"}),
                "tag_threshold": ("FLOAT", {"default": 0.65, "min": 0.1, "max": 1.0, "step": 0.05, "tooltip": "Minimum confidence threshold for tags (0.65 = 65%)"}),
                "max_injected_tags": ("INT", {"default": 30, "min": 5, "max": 100, "step": 5, "tooltip": "Maximum number of detected tags to inject into LLM prompt"}),
                "florence_model": (FLORENCE_MODELS, {"default": FLORENCE_MODELS[0], "tooltip": "Florence model for fallback VLM mode (when no LLM connected)"}),
                "hf_token": ("STRING", {"default": "", "tooltip": "HuggingFace token for gated models (Florence, etc.)"}),
                "release_vram": ("BOOLEAN", {"default": False, "tooltip": "Unload Florence model from VRAM after use"}),
                "generate_caption": ("BOOLEAN", {"default": False, "tooltip": "Generate Instagram captions (3 styles: Poetic, Technical, Personal)"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Random seed for reproducibility (change to force re-generation)"}),
            },
        }

    def generate(
        self,
        image: Any,
        llm_model: Any = None,
        user_prompt: Optional[Dict[str, str]] = None,
        analysis_mode: str = "Standard",
        tag_threshold: float = 0.65,
        max_injected_tags: int = 30,
        florence_model: str = "MiaoshouAI/Florence-2-large-PromptGen-v2.0",
        hf_token: str = "",
        release_vram: bool = False,
        generate_caption: bool = False,
        seed: int = 0,
    ) -> Tuple[str, str]:
        """
        Generate prompt from image with automatic mode selection.

        Args:
            image: Image tensor from ComfyUI
            llm_model: Optional LLMModelConfig from SID_LLM_API or SID_LLM_Local
            user_prompt: Optional custom system/user prompts from SID_UserPrompt node
            analysis_mode: Analysis depth (Quick/Standard/Detailed/Extreme)
            tag_threshold: Minimum confidence threshold for tags (0.65 = 65%)
            max_injected_tags: Maximum number of tags to inject into LLM prompt
            florence_model: Florence model to use for fallback VLM mode
            hf_token: HuggingFace token for gated models (Florence, etc.)
            release_vram: Unload Florence model from VRAM after use
            generate_caption: Generate Instagram captions (3 styles)
            seed: Random seed for reproducibility

        Returns:
            Tuple of (prompt, caption)
        """
        start_time = time.time()

        # Build base metadata
        metadata = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "analysis_mode_requested": analysis_mode,
            "tag_threshold": tag_threshold,
            "max_injected_tags": max_injected_tags,
        }

        try:
            # Check for interrupt before starting
            check_interrupt()

            result = self._execute_mode(
                image=image,
                llm_model=llm_model,
                user_prompt=user_prompt,
                analysis_mode=analysis_mode,
                tag_threshold=tag_threshold,
                max_injected_tags=max_injected_tags,
                florence_model=florence_model,
                hf_token=hf_token,
            )

            # Check for interrupt after mode execution
            check_interrupt()

            # Merge result metadata with base metadata
            metadata.update(result.metadata)

            # Generate caption if requested and LLM is available
            caption = ""
            if generate_caption and llm_model is not None:
                check_interrupt()  # Check before caption generation
                caption_start = time.time()
                caption = self._generate_caption(image, llm_model, result.prompt)
                metadata["caption_generated"] = True
                metadata["caption_timing_ms"] = int((time.time() - caption_start) * 1000)
            elif generate_caption and llm_model is None:
                caption = "[Caption generation requires LLM model connection]"
                metadata["caption_generated"] = False

            metadata["timing"] = {
                "total_ms": int((time.time() - start_time) * 1000)
            }

            # Release VRAM if requested
            if release_vram:
                self._release_vram()

            # Save metadata to logs folder
            self._save_metadata(metadata)

            return (result.prompt, caption)

        except Exception as e:
            error_msg = str(e)
            print(f"[SID_PromptGenerator] Error: {error_msg}")
            import traceback
            traceback.print_exc()

            metadata["error"] = error_msg
            metadata["mode"] = "error"

            # Save error metadata
            self._save_metadata(metadata)

            return ("", "")

    def _execute_mode(
        self,
        image: Any,
        llm_model: Any,
        user_prompt: Optional[Dict[str, str]],
        analysis_mode: str,
        tag_threshold: float = 0.65,
        max_injected_tags: int = 30,
        florence_model: str = "MiaoshouAI/Florence-2-large-PromptGen-v2.0",
        hf_token: str = "",
    ) -> GeneratorResult:
        """
        Determine and execute the appropriate mode.

        Mode selection priority:
        1. Florence (no llm_model)
        2. LLM mode based on analysis_mode
        """
        # Priority 1: Florence mode - no LLM model
        if llm_model is None:
            return self._execute_florence(image, florence_model=florence_model, hf_token=hf_token)

        # Priority 2: LLM modes based on analysis_mode
        mode_key = analysis_mode.lower()

        return self._execute_llm_mode(image, llm_model, mode_key, user_prompt, tag_threshold, max_injected_tags)

    def _execute_florence(self, image: Any, florence_model: str = "MiaoshouAI/Florence-2-large-PromptGen-v2.0", hf_token: str = "") -> GeneratorResult:
        """Execute Florence mode - VLM description."""
        mode = self._get_mode("florence")
        if mode:
            return mode.execute(image=image, model_id=florence_model, hf_token=hf_token)

        # Fallback if mode not loaded
        return GeneratorResult(
            prompt="[Florence mode not available - please install Florence model]",
            metadata={
                "mode": "florence",
                "source": "florence_description",
                "error": "Florence mode not loaded",
                "processing": {
                    "taggers_executed": False,
                    "florence_executed": False,
                    "llm_executed": False,
                }
            }
        )

    def _execute_llm_mode(
        self,
        image: Any,
        llm_model: Any,
        mode_key: str,
        user_prompt: Optional[Dict[str, str]] = None,
        tag_threshold: float = 0.65,
        max_injected_tags: int = 30,
    ) -> GeneratorResult:
        """Execute an LLM-based mode."""
        mode = self._get_mode(mode_key)

        if not mode:
            # Fallback to quick mode
            print(f"[SID_PromptGenerator] {mode_key} mode not available, using quick mode")
            mode = self._get_mode("quick")

        if not mode:
            return GeneratorResult(
                prompt="[LLM mode not available]",
                metadata={
                    "mode": mode_key,
                    "error": f"{mode_key} mode not loaded",
                }
            )

        # For Standard/Detailed/Extreme, run taggers first
        tagger_results = None
        decisions = None

        if mode.requires_taggers:
            tagger_results, decisions = self._run_taggers(image, tag_threshold)

        return mode.execute(
            image=image,
            llm_model=llm_model,
            tagger_results=tagger_results,
            decisions=decisions,
            prompt_config=user_prompt,
            tag_threshold=tag_threshold,
            max_injected_tags=max_injected_tags,
        )

    def _run_taggers(self, image: Any, tag_threshold: float = 0.65) -> Tuple[TaggerResults, Decisions]:
        """
        Run all taggers and make decisions based on results.

        Args:
            image: Image tensor from ComfyUI
            tag_threshold: Minimum confidence threshold for tags

        Returns:
            Tuple of (TaggerResults, Decisions)
        """
        from .taggers import get_runner
        from .decisions import get_engine

        # Run taggers with threshold
        runner = get_runner()
        tagger_results = runner.run_standard(image, threshold=tag_threshold)

        # Unload tagger models to free VRAM for VLM/LLM
        runner.unload_all()
        self._clear_vram()

        # Make decisions based on tagger results
        engine = get_engine()
        decisions = engine.make_decisions(tagger_results)

        return tagger_results, decisions

    def _clear_vram(self):
        """Clear VRAM by running garbage collection and emptying CUDA cache."""
        import gc
        gc.collect()

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                print("[SID_PromptGenerator] VRAM cleared for VLM/LLM execution")
        except ImportError:
            pass

    def _release_vram(self):
        """Release VRAM by unloading Florence and other cached models."""
        # Unload Florence models
        florence_mode = self._modes.get("florence")
        if florence_mode:
            from .modes.florence import FlorenceMode
            FlorenceMode.unload_model()
            print("[SID_PromptGenerator] Released Florence VRAM")

    def _save_metadata(self, metadata: Dict[str, Any]) -> None:
        """Save metadata to logs folder."""
        import os
        from datetime import datetime

        # Create logs folder in the package directory
        package_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        logs_dir = os.path.join(package_dir, "logs", "prompt_generator")
        os.makedirs(logs_dir, exist_ok=True)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode = metadata.get("mode", "unknown")
        filename = f"{timestamp}_{mode}.json"
        filepath = os.path.join(logs_dir, filename)

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            print(f"[SID_PromptGenerator] Metadata saved to: logs/prompt_generator/{filename}")
        except Exception as e:
            print(f"[SID_PromptGenerator] Failed to save metadata: {e}")

    def _generate_caption(self, image: Any, llm_model: Any, prompt: str) -> str:
        """
        Generate Instagram captions in 3 styles using the LLM.

        Args:
            image: Image tensor from ComfyUI
            llm_model: LLMModelConfig with provider settings
            prompt: Generated image prompt to use as context

        Returns:
            Markdown-formatted captions in 3 styles
        """
        import base64
        import io
        import numpy as np
        from PIL import Image

        # Convert image to base64
        if hasattr(image, 'cpu'):
            img_np = image[0].cpu().numpy()
        else:
            img_np = np.array(image[0])

        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
        else:
            img_np = img_np.astype(np.uint8)

        pil_image = Image.fromarray(img_np)

        # Resize for efficiency
        max_pixels = 672 * 672
        current_pixels = pil_image.width * pil_image.height
        if current_pixels > max_pixels:
            scale = (max_pixels / current_pixels) ** 0.5
            new_width = max(64, (int(pil_image.width * scale) // 8) * 8)
            new_height = max(64, (int(pil_image.height * scale) // 8) * 8)
            pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG", quality=90)
        base64_image = base64.standard_b64encode(buffer.getvalue()).decode("utf-8")

        # Caption generation system prompt
        system_prompt = """You are a social media expert creating Instagram captions for photographers.

Generate 3 different caption styles for this image. Each style should include:
- A caption (1-2 engaging sentences)
- A description (2-3 sentences expanding on the image)
- 10-15 relevant hashtags

FORMAT YOUR RESPONSE EXACTLY LIKE THIS:

## Poetic
**Caption:** [Evocative, emotional, artistic caption]
**Description:** [Lyrical description with metaphors and feeling]
**Hashtags:** #hashtag1 #hashtag2 #hashtag3...

## Technical
**Caption:** [Professional, precise, photography-focused caption]
**Description:** [Details about technique, equipment, settings, composition]
**Hashtags:** #hashtag1 #hashtag2 #hashtag3...

## Personal
**Caption:** [Casual, relatable, storytelling caption]
**Description:** [Behind-the-scenes, personal connection, authentic voice]
**Hashtags:** #hashtag1 #hashtag2 #hashtag3...

Keep each section concise. Use relevant photography and subject-specific hashtags."""

        user_prompt = f"""Create 3 Instagram caption styles for this image.

Context from image analysis:
{prompt}

Generate Poetic, Technical, and Personal caption styles following the exact format specified."""

        # Get LLM client and make call
        try:
            client = self._get_caption_client(llm_model)

            if hasattr(client, 'messages'):
                # Anthropic API
                response = client.messages.create(
                    model=llm_model.model,
                    max_tokens=1500,
                    temperature=0.7,
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
                    max_tokens=1500,
                    temperature=0.7,
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

                if response and response.choices and response.choices[0].message:
                    return response.choices[0].message.content
                return "[Caption generation failed - empty response]"

        except Exception as e:
            print(f"[SID_PromptGenerator] Caption generation error: {e}")
            return f"[Caption generation error: {str(e)}]"

    def _get_caption_client(self, llm_model: Any) -> Any:
        """Get LLM client for caption generation."""
        provider = llm_model.provider.lower()

        if provider == "anthropic":
            import anthropic
            import httpx
            timeout = httpx.Timeout(timeout=120.0, connect=30.0)
            return anthropic.Anthropic(api_key=llm_model.api_key, timeout=timeout)

        elif provider == "local":
            from .modes.standard import StandardMode
            mode = StandardMode()
            return mode._get_client(llm_model)

        else:
            from openai import OpenAI
            return OpenAI(
                api_key=llm_model.api_key or "not-needed",
                base_url=llm_model.api_url if llm_model.api_url else None,
            )


# ComfyUI node registration
NODE_CLASS_MAPPINGS = {
    "SID_PromptGenerator": SID_PromptGenerator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SID_PromptGenerator": "SID Prompt Generator",
}
