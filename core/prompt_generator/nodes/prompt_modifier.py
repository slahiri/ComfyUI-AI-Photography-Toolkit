"""
SID_PromptModifier - Semantically modify prompts using LLM.

Takes an input prompt and modification instructions for each semantic section,
then uses an LLM to intelligently modify the prompt while preserving structure.

Sections (Z-Image structure):
1. Subject - age, ethnicity, build, features
2. Clothing - materials, colors, fit, accessories
3. Pose - expression, body position, gesture
4. Environment - background, setting, location
5. Lighting - quality, direction, color temperature
6. Camera - angle, framing, depth of field
"""

import base64
import io
import json
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image


class SID_PromptModifier:
    """
    Semantically modify prompts using LLM with section-based instructions.
    """

    CATEGORY = "SID Photography Toolkit"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt", "caption")
    FUNCTION = "modify"

    # Section definitions for Z-Image prompt structure
    SECTIONS = {
        "subject": {
            "name": "Subject",
            "description": "Main subject attributes (age, ethnicity, build, features, hair, face)",
            "keywords": ["woman", "man", "person", "model", "face", "hair", "eyes", "skin", "features"],
        },
        "clothing": {
            "name": "Clothing & Accessories",
            "description": "Clothing items, materials, colors, fit, jewelry, accessories",
            "keywords": ["wearing", "dress", "shirt", "pants", "shoes", "jewelry", "accessories", "fabric"],
        },
        "pose": {
            "name": "Pose & Expression",
            "description": "Body position, stance, gesture, facial expression, mood",
            "keywords": ["standing", "sitting", "pose", "expression", "looking", "hands", "arms", "gesture"],
        },
        "environment": {
            "name": "Environment & Background",
            "description": "Setting, location, background elements, props, atmosphere",
            "keywords": ["background", "setting", "location", "room", "outdoor", "indoor", "scene"],
        },
        "lighting": {
            "name": "Lighting",
            "description": "Light quality, direction, color temperature, shadows, highlights",
            "keywords": ["lighting", "light", "shadows", "highlights", "golden hour", "soft", "dramatic"],
        },
        "camera": {
            "name": "Camera & Framing",
            "description": "Camera angle, shot type, framing, depth of field, focus",
            "keywords": ["camera", "angle", "shot", "close-up", "portrait", "wide", "depth of field", "bokeh"],
        },
    }

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        """Define ComfyUI input types."""
        return {
            "required": {
                "prompt": ("STRING", {"forceInput": True, "tooltip": "Input prompt to modify (connect from SID_PromptGenerator)"}),
                "llm_model": ("LLM_MODEL",),
            },
            "optional": {
                "processing_mode": (["Single Pass", "Section by Section"], {"default": "Single Pass", "tooltip": "Single Pass: one LLM call. Section by Section: separate call per section"}),
                "subject_instruction": ("STRING", {"multiline": True, "default": "", "tooltip": "How to modify subject (e.g., 'Make younger, add freckles')"}),
                "clothing_instruction": ("STRING", {"multiline": True, "default": "", "tooltip": "How to modify clothing (e.g., 'Change to elegant red dress')"}),
                "pose_instruction": ("STRING", {"multiline": True, "default": "", "tooltip": "How to modify pose (e.g., 'More confident stance, hands on hips')"}),
                "environment_instruction": ("STRING", {"multiline": True, "default": "", "tooltip": "How to modify environment (e.g., 'Move to beach at sunset')"}),
                "lighting_instruction": ("STRING", {"multiline": True, "default": "", "tooltip": "How to modify lighting (e.g., 'Golden hour, warm tones')"}),
                "camera_instruction": ("STRING", {"multiline": True, "default": "", "tooltip": "How to modify camera (e.g., 'Close-up portrait, shallow DOF')"}),
                "generate_caption": ("BOOLEAN", {"default": False, "tooltip": "Generate Instagram caption from modified prompt"}),
            },
        }

    def modify(
        self,
        prompt: str,
        llm_model: Any,
        processing_mode: str = "Single Pass",
        subject_instruction: str = "",
        clothing_instruction: str = "",
        pose_instruction: str = "",
        environment_instruction: str = "",
        lighting_instruction: str = "",
        camera_instruction: str = "",
        generate_caption: bool = False,
    ) -> Tuple[str, str]:
        """
        Modify prompt based on section instructions.

        Args:
            prompt: Input prompt to modify
            llm_model: LLM configuration (temperature and other params from LLM node)
            processing_mode: "Single Pass" or "Section by Section"
            subject_instruction: Modification for subject section
            clothing_instruction: Modification for clothing section
            pose_instruction: Modification for pose section
            environment_instruction: Modification for environment section
            lighting_instruction: Modification for lighting section
            camera_instruction: Modification for camera section
            generate_caption: Whether to generate Instagram caption

        Returns:
            Tuple of (modified_prompt, caption)
        """
        start_time = time.time()

        # Get temperature from LLM model config
        temperature = getattr(llm_model, 'temperature', 0.7)

        # Collect instructions
        instructions = {
            "subject": subject_instruction.strip(),
            "clothing": clothing_instruction.strip(),
            "pose": pose_instruction.strip(),
            "environment": environment_instruction.strip(),
            "lighting": lighting_instruction.strip(),
            "camera": camera_instruction.strip(),
        }

        # Filter out empty instructions
        active_instructions = {k: v for k, v in instructions.items() if v}

        # If no instructions, return original prompt
        if not active_instructions:
            print("[SID_PromptModifier] No instructions provided, returning original prompt")
            caption = ""
            if generate_caption:
                caption = self._generate_caption(llm_model, prompt, temperature)
            return (prompt, caption)

        # Modify prompt based on processing mode
        if processing_mode == "Single Pass":
            modified_prompt = self._single_pass_modify(prompt, llm_model, active_instructions, temperature)
        else:
            modified_prompt = self._section_by_section_modify(prompt, llm_model, active_instructions, temperature)

        # Generate caption if requested
        caption = ""
        if generate_caption:
            caption = self._generate_caption(llm_model, modified_prompt, temperature)

        elapsed = int((time.time() - start_time) * 1000)
        print(f"[SID_PromptModifier] Completed in {elapsed}ms (mode: {processing_mode}, temp: {temperature})")

        return (modified_prompt, caption)

    def _single_pass_modify(
        self,
        prompt: str,
        llm_model: Any,
        instructions: Dict[str, str],
        temperature: float,
    ) -> str:
        """Modify prompt in a single LLM call."""

        # Build instruction list
        instruction_text = ""
        for section_key, instruction in instructions.items():
            section_name = self.SECTIONS[section_key]["name"]
            instruction_text += f"- **{section_name}**: {instruction}\n"

        system_prompt = """You are an expert prompt engineer for AI image generation (Stable Diffusion, Flux, Z-Image).

Your task is to modify an existing image prompt based on specific instructions for each section.

RULES:
1. PRESERVE the overall structure and flow of the original prompt
2. ONLY modify the sections mentioned in the instructions
3. Keep unmentioned sections UNCHANGED
4. Maintain the same writing style (concrete, visual, descriptive)
5. Output ONLY the modified prompt - no explanations or commentary
6. Write as a single flowing paragraph"""

        user_prompt = f"""ORIGINAL PROMPT:
{prompt}

MODIFICATION INSTRUCTIONS:
{instruction_text}

Modify the prompt according to the instructions above. Output only the modified prompt."""

        try:
            client = self._get_client(llm_model)
            response = self._call_llm(client, llm_model, system_prompt, user_prompt, temperature)
            return self._clean_response(response)
        except Exception as e:
            print(f"[SID_PromptModifier] Error in single pass: {e}")
            return prompt

    def _section_by_section_modify(
        self,
        prompt: str,
        llm_model: Any,
        instructions: Dict[str, str],
        temperature: float,
    ) -> str:
        """Modify prompt section by section with separate LLM calls."""

        modified_prompt = prompt

        system_prompt = """You are an expert prompt engineer for AI image generation.

Your task is to modify ONLY ONE specific section of an image prompt based on the instruction.

RULES:
1. ONLY modify the specified section
2. Keep ALL other parts of the prompt EXACTLY the same
3. Maintain the same writing style
4. Output ONLY the complete modified prompt - no explanations
5. Write as a single flowing paragraph"""

        for section_key, instruction in instructions.items():
            section_info = self.SECTIONS[section_key]
            section_name = section_info["name"]
            section_desc = section_info["description"]

            user_prompt = f"""CURRENT PROMPT:
{modified_prompt}

SECTION TO MODIFY: {section_name}
SECTION DESCRIPTION: {section_desc}
INSTRUCTION: {instruction}

Modify ONLY the {section_name} section according to the instruction. Keep everything else unchanged. Output only the complete modified prompt."""

            try:
                client = self._get_client(llm_model)
                response = self._call_llm(client, llm_model, system_prompt, user_prompt, temperature)
                modified_prompt = self._clean_response(response)
                print(f"[SID_PromptModifier] Modified section: {section_name}")
            except Exception as e:
                print(f"[SID_PromptModifier] Error modifying {section_name}: {e}")
                # Continue with current prompt

        return modified_prompt

    def _generate_caption(self, llm_model: Any, prompt: str, temperature: float) -> str:
        """Generate Instagram caption from prompt."""

        system_prompt = """You are a social media expert creating Instagram captions for photographers.

Generate 3 different caption styles. Each style should include:
- A caption (1-2 engaging sentences)
- A description (2-3 sentences)
- 10-15 relevant hashtags

FORMAT YOUR RESPONSE EXACTLY LIKE THIS:

## Poetic
**Caption:** [Evocative, emotional, artistic caption]
**Description:** [Lyrical description with metaphors and feeling]
**Hashtags:** #hashtag1 #hashtag2 #hashtag3...

## Technical
**Caption:** [Professional, precise, photography-focused caption]
**Description:** [Details about technique, composition, style]
**Hashtags:** #hashtag1 #hashtag2 #hashtag3...

## Personal
**Caption:** [Casual, relatable, storytelling caption]
**Description:** [Behind-the-scenes, personal connection, authentic voice]
**Hashtags:** #hashtag1 #hashtag2 #hashtag3...

Keep each section concise."""

        user_prompt = f"""Create 3 Instagram caption styles based on this image description:

{prompt}

Generate Poetic, Technical, and Personal caption styles following the exact format specified."""

        try:
            client = self._get_client(llm_model)
            response = self._call_llm(client, llm_model, system_prompt, user_prompt, temperature)
            return response
        except Exception as e:
            print(f"[SID_PromptModifier] Caption generation error: {e}")
            return f"[Caption generation error: {str(e)}]"

    def _get_client(self, llm_model: Any) -> Any:
        """Get LLM client based on provider."""
        provider = llm_model.provider.lower()

        if provider == "anthropic":
            import anthropic
            import httpx
            timeout = httpx.Timeout(timeout=120.0, connect=30.0)
            return anthropic.Anthropic(api_key=llm_model.api_key, timeout=timeout)

        elif provider == "local_text":
            # Local text-only models (Qwen text)
            from ....llm_providers.sid_llm_local_text import LocalTextModelClient
            extra = llm_model.extra_params or {}
            return LocalTextModelClient(
                model_name=llm_model.model,
                quantization=extra.get("quantization", "4-bit"),
                device=extra.get("device", "auto"),
                keep_model_loaded=extra.get("keep_model_loaded", True),
                repo_id=extra.get("repo_id", ""),
                hf_token=extra.get("hf_token"),
            )

        elif provider == "local":
            # Local vision models - fallback to OpenAI-compatible
            from openai import OpenAI
            return OpenAI(
                api_key="not-needed",
                base_url=llm_model.api_url if llm_model.api_url else "http://localhost:11434/v1",
            )

        else:
            # OpenAI-compatible API
            from openai import OpenAI
            return OpenAI(
                api_key=llm_model.api_key or "not-needed",
                base_url=llm_model.api_url if llm_model.api_url else None,
            )

    def _call_llm(
        self,
        client: Any,
        llm_model: Any,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
    ) -> str:
        """Make LLM call (text-only, no image)."""

        if hasattr(client, 'messages'):
            # Anthropic API
            response = client.messages.create(
                model=llm_model.model,
                max_tokens=2000,
                temperature=temperature,
                system=system_prompt,
                messages=[{
                    "role": "user",
                    "content": user_prompt
                }]
            )
            return response.content[0].text
        else:
            # OpenAI-compatible API
            response = client.chat.completions.create(
                model=llm_model.model,
                max_tokens=2000,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )

            if response and response.choices and response.choices[0].message:
                return response.choices[0].message.content
            return ""

    def _clean_response(self, text: str) -> str:
        """Clean LLM response text."""
        import re

        if not text:
            return ""

        # Remove common preambles
        preambles = [
            r'^(?:Here is|Here\'s|This is|The modified prompt|Modified prompt)[:\s]*',
            r'^(?:Sure|Okay|Of course)[,.\s]*(?:here is|here\'s)?[:\s]*',
        ]
        for pattern in preambles:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)

        # Remove markdown code blocks if present
        text = re.sub(r'^```[\w]*\n?', '', text)
        text = re.sub(r'\n?```$', '', text)

        # Remove quotes if the entire response is quoted
        text = text.strip()
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]

        # Clean whitespace
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\s+', ' ', text)

        return text.strip()


# Node registration
NODE_CLASS_MAPPINGS = {
    "SID_PromptModifier": SID_PromptModifier,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SID_PromptModifier": "SID Prompt Modifier",
}
