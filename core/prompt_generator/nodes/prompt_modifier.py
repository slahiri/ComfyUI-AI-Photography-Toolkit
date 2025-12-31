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

    # Photography Templates - Pre-defined photography styles/setups
    PHOTOGRAPHY_TEMPLATES = {
        "None": "",
        "Studio Portrait": "Professional studio portrait with seamless background, controlled lighting setup with key light, fill light, and hair light, clean and polished look",
        "Fashion Editorial": "High-fashion editorial style, dramatic poses, bold styling, magazine-quality composition, striking visual impact",
        "Street Photography": "Candid street photography aesthetic, urban environment, natural moments, documentary style, authentic atmosphere",
        "Beauty/Cosmetic": "Close-up beauty photography, flawless skin, detailed makeup visibility, soft diffused lighting, commercial beauty aesthetic",
        "Lifestyle": "Natural lifestyle photography, relaxed and authentic moments, warm inviting atmosphere, relatable everyday settings",
        "Glamour": "Glamorous photography style, luxurious setting, elegant styling, soft flattering light, aspirational mood",
        "Boudoir": "Intimate boudoir photography, soft romantic lighting, elegant poses, private luxurious setting, tasteful sensuality",
        "Fine Art": "Fine art photography aesthetic, artistic composition, conceptual elements, gallery-worthy presentation",
        "Commercial": "Clean commercial photography, product-focused when applicable, professional lighting, advertising quality",
        "Vintage/Retro": "Vintage photography aesthetic, period-appropriate styling, nostalgic color grading, classic film look",
        "Cinematic": "Cinematic photography style, movie-like framing, dramatic lighting, widescreen composition, narrative mood",
        "Minimalist": "Minimalist photography, clean simple backgrounds, negative space, focus on essential elements only",
        "Environmental Portrait": "Environmental portrait showing subject in their natural setting, context-rich background, storytelling composition",
        "High Key": "High key photography, bright white background, minimal shadows, clean ethereal look, overexposed aesthetic",
        "Low Key": "Low key photography, dark moody background, dramatic shadows, rim lighting, mysterious atmosphere",
    }

    # Photography Effects - Visual effects to apply
    PHOTOGRAPHY_EFFECTS = {
        "None": "",
        "Film Grain": "Add subtle film grain texture, analog film aesthetic, organic noise pattern",
        "Soft Focus": "Dreamy soft focus effect, gentle blur, romantic ethereal quality",
        "High Contrast": "High contrast look, deep blacks and bright highlights, bold tonal separation",
        "Desaturated": "Muted desaturated colors, subtle pastel tones, understated color palette",
        "Rich Colors": "Vibrant rich saturated colors, punchy color palette, vivid tones",
        "Golden Hour": "Warm golden hour color grading, orange and amber tones, sunset warmth",
        "Blue Hour": "Cool blue hour tones, twilight color palette, serene blue cast",
        "Cross Process": "Cross-processed film look, shifted colors, experimental color cast",
        "Faded Film": "Faded vintage film effect, lifted blacks, reduced contrast, nostalgic fade",
        "HDR": "HDR-style processing, enhanced details in shadows and highlights, surreal clarity",
        "Matte Finish": "Matte film finish, lifted shadows, reduced contrast, modern film look",
        "Teal and Orange": "Teal and orange color grading, cinematic color contrast, Hollywood look",
        "Black and White": "Classic black and white, rich grayscale tones, timeless monochrome",
        "Sepia": "Warm sepia toning, antique photograph aesthetic, brown monochrome",
        "Split Toning": "Split toning effect, different colors in highlights and shadows, artistic color separation",
        "Lens Flare": "Natural lens flare, sun rays, organic light artifacts, atmospheric glow",
        "Bokeh": "Pronounced bokeh effect, creamy out-of-focus areas, shallow depth of field aesthetic",
        "Vignette": "Subtle vignette, darkened edges, focus drawn to center, classic finish",
    }

    # Famous Photographer Styles
    PHOTOGRAPHER_STYLES = {
        "None": "",
        "Annie Leibovitz": "Annie Leibovitz style - dramatic theatrical portraits, conceptual storytelling, bold artistic vision, celebrity portrait mastery, elaborate production value",
        "Peter Lindbergh": "Peter Lindbergh style - raw natural beauty, black and white mastery, minimal retouching, emotional authenticity, supermodel portraits, timeless elegance",
        "Helmut Newton": "Helmut Newton style - provocative glamour, strong powerful women, high contrast black and white, bold sexuality, cinematic noir aesthetic",
        "Richard Avedon": "Richard Avedon style - stark white backgrounds, dynamic movement, psychological intensity, fashion innovation, raw emotional portraits",
        "Irving Penn": "Irving Penn style - elegant simplicity, meticulous composition, neutral backgrounds, still life precision, timeless sophistication",
        "Mario Testino": "Mario Testino style - vibrant glamorous energy, sun-kissed skin, joyful spontaneity, luxury fashion, warm golden tones",
        "Steven Meisel": "Steven Meisel style - chameleon versatility, trendsetting fashion, narrative editorial stories, transformative character studies",
        "Patrick Demarchelier": "Patrick Demarchelier style - classic French elegance, natural beauty, soft romantic lighting, effortless sophistication",
        "David LaChapelle": "David LaChapelle style - hyper-saturated colors, surreal fantasy worlds, pop culture commentary, extravagant maximalism",
        "Tim Walker": "Tim Walker style - whimsical fairytale aesthetic, elaborate fantasy sets, dreamlike storytelling, magical romanticism",
        "Ellen von Unwerth": "Ellen von Unwerth style - playful feminine energy, retro pin-up glamour, flirtatious spontaneity, empowered sensuality",
        "Herb Ritts": "Herb Ritts style - sculptural body photography, graphic black and white, classical beauty, athletic forms, sun-drenched California aesthetic",
        "Guy Bourdin": "Guy Bourdin style - surrealist fashion, bold graphic colors, mysterious narratives, provocative compositions, artistic tension",
        "Paolo Roversi": "Paolo Roversi style - ethereal Polaroid aesthetic, soft diffused light, intimate portraits, romantic timelessness, painterly quality",
        "Juergen Teller": "Juergen Teller style - raw unpolished aesthetic, flash photography, anti-glamour authenticity, candid imperfection",
        "Terry Richardson": "Terry Richardson style - harsh flash photography, snapshot aesthetic, provocative directness, high contrast pop",
        "Nick Knight": "Nick Knight style - experimental digital innovation, avant-garde fashion, futuristic aesthetic, boundary-pushing imagery",
        "Rankin": "Rankin style - bold pop portraits, vibrant colors, direct eye contact, confident subjects, commercial edge",
        "Bruce Weber": "Bruce Weber style - all-American athleticism, outdoor naturalism, youthful energy, golden hour warmth, Americana nostalgia",
        "Nadav Kander": "Nadav Kander style - contemplative portraits, muted color palette, psychological depth, understated power, fine art approach",
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
                # Photography presets
                "photography_template": (list(cls.PHOTOGRAPHY_TEMPLATES.keys()), {"default": "None", "tooltip": "Apply a photography style template"}),
                "photography_effect": (list(cls.PHOTOGRAPHY_EFFECTS.keys()), {"default": "None", "tooltip": "Apply a visual effect"}),
                "photographer_style": (list(cls.PHOTOGRAPHER_STYLES.keys()), {"default": "None", "tooltip": "Emulate a famous photographer's style"}),
                # Section instructions
                "subject_instruction": ("STRING", {"multiline": True, "default": "", "tooltip": "How to modify subject (e.g., 'Make younger, add freckles')"}),
                "clothing_instruction": ("STRING", {"multiline": True, "default": "", "tooltip": "How to modify clothing (e.g., 'Change to elegant red dress')"}),
                "pose_instruction": ("STRING", {"multiline": True, "default": "", "tooltip": "How to modify pose (e.g., 'More confident stance, hands on hips')"}),
                "environment_instruction": ("STRING", {"multiline": True, "default": "", "tooltip": "How to modify environment (e.g., 'Move to beach at sunset')"}),
                "lighting_instruction": ("STRING", {"multiline": True, "default": "", "tooltip": "How to modify lighting (e.g., 'Golden hour, warm tones')"}),
                "camera_instruction": ("STRING", {"multiline": True, "default": "", "tooltip": "How to modify camera (e.g., 'Close-up portrait, shallow DOF')"}),
                "generate_caption": ("BOOLEAN", {"default": False, "tooltip": "Generate Instagram caption from modified prompt"}),
                "release_vram": ("BOOLEAN", {"default": True, "tooltip": "Release VRAM after execution (recommended)"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Random seed for reproducibility (change to force re-generation)"}),
            },
        }

    def modify(
        self,
        prompt: str,
        llm_model: Any,
        processing_mode: str = "Single Pass",
        photography_template: str = "None",
        photography_effect: str = "None",
        photographer_style: str = "None",
        subject_instruction: str = "",
        clothing_instruction: str = "",
        pose_instruction: str = "",
        environment_instruction: str = "",
        lighting_instruction: str = "",
        camera_instruction: str = "",
        generate_caption: bool = False,
        release_vram: bool = True,
        seed: int = 0,
    ) -> Tuple[str, str]:
        """
        Modify prompt based on section instructions and photography presets.

        Args:
            prompt: Input prompt to modify
            llm_model: LLM configuration (temperature and other params from LLM node)
            processing_mode: "Single Pass" or "Section by Section"
            photography_template: Photography style template to apply
            photography_effect: Visual effect to apply
            photographer_style: Famous photographer style to emulate
            subject_instruction: Modification for subject section
            clothing_instruction: Modification for clothing section
            pose_instruction: Modification for pose section
            environment_instruction: Modification for environment section
            lighting_instruction: Modification for lighting section
            camera_instruction: Modification for camera section
            generate_caption: Whether to generate Instagram caption
            release_vram: Release VRAM after execution
            seed: Random seed for reproducibility (change to force re-generation)

        Returns:
            Tuple of (modified_prompt, caption)
        """
        start_time = time.time()

        # Get temperature from LLM model config
        temperature = getattr(llm_model, 'temperature', 0.7)

        # Collect section instructions
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

        # Collect photography presets
        presets = {}
        if photography_template != "None":
            presets["template"] = self.PHOTOGRAPHY_TEMPLATES.get(photography_template, "")
        if photography_effect != "None":
            presets["effect"] = self.PHOTOGRAPHY_EFFECTS.get(photography_effect, "")
        if photographer_style != "None":
            presets["photographer"] = self.PHOTOGRAPHER_STYLES.get(photographer_style, "")

        # If no instructions or presets, return original prompt
        if not active_instructions and not presets:
            print("[SID_PromptModifier] No modifications requested, returning original prompt")
            caption = ""
            if generate_caption:
                caption = self._generate_caption(llm_model, prompt, temperature)
            return (prompt, caption)

        # Log what we're applying
        if presets:
            preset_names = []
            if photography_template != "None":
                preset_names.append(f"Template: {photography_template}")
            if photography_effect != "None":
                preset_names.append(f"Effect: {photography_effect}")
            if photographer_style != "None":
                preset_names.append(f"Style: {photographer_style}")
            print(f"[SID_PromptModifier] Applying: {', '.join(preset_names)}")

        # Modify prompt based on processing mode
        if processing_mode == "Single Pass":
            modified_prompt = self._single_pass_modify(prompt, llm_model, active_instructions, temperature, presets)
        else:
            modified_prompt = self._section_by_section_modify(prompt, llm_model, active_instructions, temperature, presets)

        # Generate caption if requested
        caption = ""
        if generate_caption:
            caption = self._generate_caption(llm_model, modified_prompt, temperature)

        # Release VRAM if requested
        if release_vram:
            self._release_vram()

        elapsed = int((time.time() - start_time) * 1000)
        print(f"[SID_PromptModifier] Completed in {elapsed}ms (mode: {processing_mode}, temp: {temperature})")

        return (modified_prompt, caption)

    def _single_pass_modify(
        self,
        prompt: str,
        llm_model: Any,
        instructions: Dict[str, str],
        temperature: float,
        presets: Dict[str, str] = None,
    ) -> str:
        """Modify prompt in a single LLM call."""
        presets = presets or {}

        # Build instruction list for sections
        instruction_text = ""
        for section_key, instruction in instructions.items():
            section_name = self.SECTIONS[section_key]["name"]
            instruction_text += f"- **{section_name}**: {instruction}\n"

        # Build preset instructions
        preset_text = ""
        if presets.get("template"):
            preset_text += f"- **Photography Template**: {presets['template']}\n"
        if presets.get("effect"):
            preset_text += f"- **Visual Effect**: {presets['effect']}\n"
        if presets.get("photographer"):
            preset_text += f"- **Photographer Style**: {presets['photographer']}\n"

        system_prompt = """You are an expert prompt engineer for AI image generation (Stable Diffusion, Flux, Z-Image).

Your task is to modify an existing image prompt based on specific instructions and style presets.

RULES:
1. PRESERVE the overall structure and flow of the original prompt
2. Apply section modifications as specified
3. Integrate photography templates, effects, and photographer styles naturally
4. Keep unmentioned sections UNCHANGED
5. Maintain concrete, visual, descriptive writing style
6. Output ONLY the modified prompt - no explanations or commentary
7. Write as a single flowing paragraph"""

        # Build user prompt with all modifications
        user_prompt = f"""ORIGINAL PROMPT:
{prompt}

"""
        if preset_text:
            user_prompt += f"""STYLE PRESETS TO APPLY:
{preset_text}
"""
        if instruction_text:
            user_prompt += f"""SECTION MODIFICATIONS:
{instruction_text}
"""
        user_prompt += "Modify the prompt according to the instructions above. Integrate all style presets and modifications naturally. Output only the modified prompt."

        try:
            client = self._get_client(llm_model)
            response = self._call_llm(client, llm_model, system_prompt, user_prompt, temperature, prompt)
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
        presets: Dict[str, str] = None,
    ) -> str:
        """Modify prompt section by section with separate LLM calls."""
        presets = presets or {}

        modified_prompt = prompt

        # First, apply presets if any (as a single pass for style integration)
        if presets:
            preset_text = ""
            if presets.get("template"):
                preset_text += f"- **Photography Template**: {presets['template']}\n"
            if presets.get("effect"):
                preset_text += f"- **Visual Effect**: {presets['effect']}\n"
            if presets.get("photographer"):
                preset_text += f"- **Photographer Style**: {presets['photographer']}\n"

            preset_system = """You are an expert prompt engineer for AI image generation.

Your task is to integrate photography style presets into an existing prompt.

RULES:
1. Integrate the style presets naturally into the prompt
2. Preserve the subject and core content of the original prompt
3. Enhance lighting, camera, and mood to match the styles
4. Output ONLY the modified prompt - no explanations
5. Write as a single flowing paragraph"""

            preset_user = f"""CURRENT PROMPT:
{modified_prompt}

STYLE PRESETS TO APPLY:
{preset_text}

Integrate these style presets into the prompt. Output only the modified prompt."""

            try:
                client = self._get_client(llm_model)
                response = self._call_llm(client, llm_model, preset_system, preset_user, temperature, modified_prompt)
                modified_prompt = self._clean_response(response)
                print(f"[SID_PromptModifier] Applied style presets")
            except Exception as e:
                print(f"[SID_PromptModifier] Error applying presets: {e}")

        # Then apply section-by-section modifications
        section_system = """You are an expert prompt engineer for AI image generation.

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
                response = self._call_llm(client, llm_model, section_system, user_prompt, temperature, modified_prompt)
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
            response = self._call_llm(client, llm_model, system_prompt, user_prompt, temperature, prompt)
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
                repetition_penalty=extra.get("repetition_penalty", 1.3),
                top_p=extra.get("top_p", 0.9),
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

    def _calculate_max_tokens(self, input_prompt: str, llm_model: Any) -> int:
        """Calculate max tokens based on input size + 30% buffer."""
        # Estimate tokens from input (roughly 1 token per 4 characters)
        input_chars = len(input_prompt)
        estimated_input_tokens = input_chars // 4

        # Add 30% buffer for modifications
        buffer_multiplier = 1.3
        calculated_tokens = int(estimated_input_tokens * buffer_multiplier)

        # Set minimum and maximum bounds
        min_tokens = 500
        max_tokens = getattr(llm_model, 'max_tokens', 2000)

        # Clamp to bounds
        result = max(min_tokens, min(calculated_tokens, max_tokens))
        return result

    def _call_llm(
        self,
        client: Any,
        llm_model: Any,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        input_prompt: str = "",
    ) -> str:
        """Make LLM call (text-only, no image)."""
        # Calculate max tokens based on input size + 30%
        max_tokens = self._calculate_max_tokens(input_prompt or user_prompt, llm_model)

        if hasattr(client, 'messages'):
            # Anthropic API
            response = client.messages.create(
                model=llm_model.model,
                max_tokens=max_tokens,
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
                max_tokens=max_tokens,
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

    def _release_vram(self):
        """Release VRAM by clearing GPU memory and running garbage collection."""
        import gc

        gc.collect()

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                print("[SID_PromptModifier] VRAM released")
        except ImportError:
            pass


# Node registration
NODE_CLASS_MAPPINGS = {
    "SID_PromptModifier": SID_PromptModifier,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SID_PromptModifier": "SID Prompt Modifier",
}
