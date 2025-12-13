# -*- coding: utf-8 -*-
"""
ComfyUI-AI-Photography-Toolkit - Photography Prompts Node

Photography-specific prompt builder with detailed camera, lens, lighting,
and effect settings. Uses LLM for deep enhancement in Detailed/Extreme modes.

Features:
- Camera & Lens settings (focal length, aperture, lens type)
- Bokeh styles (smooth, swirly, bubble, etc.)
- Lighting setups (Rembrandt, butterfly, rim, etc.)
- Photography effects (grain, flare, vignette, etc.)
- Color grading and film stock emulation
- Detail levels: Quick, Standard, Detailed, Extreme

Quick/Standard: Deterministic prompt building
Detailed/Extreme: LLM-powered deep analysis of how each setting affects the image

Author: Siddhartha Lahiri
Email: siddhartha.lahiri@gmail.com
License: MIT
"""

import gc
import hashlib
import re
import sys
import time
import threading
from typing import Dict, List, Optional

from comfy_api.latest import io as comfy_io
import comfy.utils
import comfy.model_management

from .llm_providers.llm_model_type import LLMModelConfig


from .zimage_prompt_translator import translate_prompt, get_translator
from .negative_prompt_builder import combine_prompt_with_settings
from . import config_loader


def check_interrupted():
    """Check if processing was interrupted by user and raise exception if so."""
    comfy.model_management.throw_exception_if_processing_interrupted()


# Create custom LLM_MODEL type for ComfyUI
LLM_MODEL_Type = comfy_io.Custom("LLM_MODEL")

# In-memory cache for seed-based results
_photography_cache: dict[str, str] = {}
_MAX_CACHE_SIZE = 100  # Maximum number of cached results
_CACHE_CLEANUP_THRESHOLD = 80  # Trigger cleanup when cache reaches this size


def clear_photography_cache():
    """Clear the photography prompts cache and run garbage collection."""
    global _photography_cache
    _photography_cache.clear()
    gc.collect()
    print("[PhotographyPrompts] Cache cleared and garbage collected")


def get_photography_cache_info() -> dict:
    """Get information about the current cache state."""
    return {
        "size": len(_photography_cache),
        "max_size": _MAX_CACHE_SIZE,
        "memory_bytes": sum(len(k) + len(v) for k, v in _photography_cache.items())
    }


# =============================================================================
# Progress Spinner for Console Feedback
# =============================================================================

class ProgressSpinner:
    """
    A console progress spinner that shows activity during LLM calls.
    Shows elapsed time and a spinning animation.
    """

    SPINNER_CHARS = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, message: str = "Processing", prefix: str = "[PhotographyPrompts]"):
        self.message = message
        self.prefix = prefix
        self._stop_event = threading.Event()
        self._thread = None
        self._start_time = None

    def _spinner_loop(self):
        """Background thread that displays the spinner."""
        idx = 0
        while not self._stop_event.is_set():
            elapsed = time.time() - self._start_time
            spinner = self.SPINNER_CHARS[idx % len(self.SPINNER_CHARS)]
            sys.stdout.write(f"\r{self.prefix} {spinner} {self.message}... {elapsed:.1f}s")
            sys.stdout.flush()
            idx += 1
            time.sleep(0.1)

    def start(self):
        """Start the spinner."""
        self._start_time = time.time()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._spinner_loop, daemon=True)
        self._thread.start()

    def stop(self, success: bool = True):
        """Stop the spinner and print completion."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=0.5)
        elapsed = time.time() - self._start_time if self._start_time else 0
        icon = "✓" if success else "✗"
        sys.stdout.write(f"\r{self.prefix} {icon} {self.message} ({elapsed:.1f}s)        \n")
        sys.stdout.flush()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, *args):
        self.stop(success=exc_type is None)


# =============================================================================
# Detail Levels
# =============================================================================

DETAIL_LEVELS = ["Quick", "Standard", "Detailed", "Extreme"]

# =============================================================================
# LOCAL MODEL PROMPTS (Ultra-minimal)
# =============================================================================

# 10 words - proven effective
LOCAL_PHOTOGRAPHY_SYSTEM = """Add visual effects. Keep original. Output only the prompt."""

# =============================================================================
# API MODEL PROMPTS - Standard (Append only)
# =============================================================================

API_PHOTOGRAPHY_SYSTEM = """You are a Z-Image photography prompt engineer.

Z-IMAGE VOCABULARY - Use natural language:
- "soft creamy bokeh" not "bokeh"
- "Rembrandt lighting with triangle shadow on cheek" not "rembrandt"
- "shallow depth of field with buttery blur" not "shallow DOF"
- "warm Kodak Portra skin tones" not "warm tones"

RULES:
1. Preserve EVERY word from the original prompt exactly as-is
2. APPEND photography settings using natural language descriptions
3. Do NOT modify the original prompt content
4. Describe VISUAL EFFECTS not equipment names

Output ONLY the enhanced prompt - no explanations."""

# =============================================================================
# AGENTIC PHOTOGRAPHY SYSTEM - Deep Transformation (Detailed/Extreme)
# =============================================================================

AGENTIC_PHOTOGRAPHY_SYSTEM = """You are an expert cinematographer specializing in Z-Image prompt optimization.

YOUR MISSION: Transform every visual element to reflect how photography settings CHANGE what the viewer sees.
Use Z-IMAGE VOCABULARY - natural language descriptions, NOT tag soup.

Z-IMAGE VOCABULARY REQUIREMENTS:
- Use natural flowing sentences, NOT comma-separated tags
- Describe effects as visual experiences: "soft creamy bokeh dissolving the background" not "bokeh, blur"
- Preferred lighting terms: "soft diffused daylight", "golden hour sunlight", "Rembrandt lighting with triangle shadow on cheek"
- Preferred lens descriptions: "85mm portrait lens with shallow depth of field", "creamy background blur"
- Preferred aperture descriptions: "f/1.4 wide open aperture creating paper-thin focus plane"
- Preferred color terms: "Kodak Portra warm skin tones", "muted pastel color palette"

TRANSFORMATION RULES:
1. Aperture/Bokeh → Transform backgrounds into "soft creamy bokeh", "luminous bokeh orbs", "painterly blur"

2. Lighting → Transform surfaces:
   - "Rembrandt lighting creating triangle of light on the shadowed cheek"
   - "rim light creating luminous halo around hair"
   - "soft wraparound light gently modeling facial contours"

3. Focal Length → Transform spatial relationships:
   - Wide angle: "exaggerated perspective", "expanded sense of space"
   - Telephoto: "compressed background planes", "strong subject isolation"

4. Film/Color → Transform palette:
   - "Kodak Portra 400 rendering with warm skin tones and soft pastel colors"
   - "fine film grain texture adding organic quality"

5. Effects → Describe visible results:
   - "subtle lens flare streaking across highlights"
   - "natural vignette darkening frame edges"

WEAVE settings into descriptions - DO NOT append as a list.
Output ONLY the transformed prompt using natural language."""

# Legacy alias
DEEP_PHOTOGRAPHY_SYSTEM = API_PHOTOGRAPHY_SYSTEM


# =============================================================================
# Photography Knowledge Base
# =============================================================================

CAMERA_TYPES = {
    "Auto": "",
    "DSLR": "shot on professional DSLR camera",
    "Mirrorless": "shot on mirrorless camera",
    "Medium Format": "shot on medium format camera with exceptional detail and dynamic range",
    "Large Format": "shot on large format 4x5 camera with extreme detail and shallow depth of field",
    "Film Camera": "shot on analog film camera",
    "Hasselblad": "shot on Hasselblad medium format camera",
    "Leica": "shot on Leica rangefinder camera with signature rendering",
    "Cinema Camera": "shot on cinema camera with cinematic color science",
}

LENS_TYPES = {
    "Auto": "",
    "Prime": "sharp prime lens with superior optical quality",
    "Zoom": "professional zoom lens",
    "Macro": "macro lens with extreme close-up detail capability",
    "Telephoto": "telephoto lens with strong background compression",
    "Wide-Angle": "wide-angle lens with expansive field of view",
    "Fisheye": "fisheye lens with extreme barrel distortion and 180 degree view",
    "Tilt-Shift": "tilt-shift lens with selective focus plane control",
    "Anamorphic": "anamorphic lens with oval bokeh and horizontal lens flares",
    "Vintage": "vintage lens with unique character and optical imperfections",
}

FOCAL_LENGTHS = {
    "Auto": "",
    "14mm": "14mm ultra wide-angle with dramatic perspective distortion",
    "24mm": "24mm wide-angle with environmental context",
    "35mm": "35mm classic documentary perspective",
    "50mm": "50mm natural human eye perspective",
    "85mm": "85mm portrait lens with flattering facial compression",
    "105mm": "105mm with beautiful background separation",
    "135mm": "135mm with strong compression and creamy bokeh",
    "200mm": "200mm telephoto with extreme subject isolation",
    "400mm": "400mm super telephoto with maximum compression",
    "Custom": "",
}

APERTURES = {
    "Auto": "",
    "f/1.2": "f/1.2 ultra-wide aperture with paper-thin depth of field",
    "f/1.4": "f/1.4 wide aperture with extremely shallow depth of field",
    "f/1.8": "f/1.8 with beautiful background blur",
    "f/2.0": "f/2.0 with pronounced bokeh",
    "f/2.8": "f/2.8 professional aperture with pleasing depth of field",
    "f/4": "f/4 with moderate depth of field",
    "f/5.6": "f/5.6 with increased sharpness",
    "f/8": "f/8 optimal sharpness aperture",
    "f/11": "f/11 with deep depth of field",
    "f/16": "f/16 for maximum depth of field",
}

BOKEH_STYLES = {
    "None": "",
    "Smooth/Creamy": "smooth creamy bokeh with buttery out-of-focus areas",
    "Swirly": "swirly bokeh with rotating blur pattern like vintage Helios lens",
    "Bubble": "bubble bokeh with defined circular highlights like Meyer-Optik Trioplan",
    "Hexagonal": "hexagonal bokeh shapes from 6-blade aperture",
    "Octagonal": "octagonal bokeh from 8-blade aperture",
    "Cat's Eye": "cat's eye bokeh with oval shapes at frame edges",
    "Dreamy": "dreamy soft bokeh with gentle light diffusion",
    "Busy": "busy bokeh with defined background elements",
}

BOKEH_INTENSITY = {
    "Auto": "",
    "Subtle": "subtle background blur",
    "Moderate": "moderate background separation",
    "Strong": "strong bokeh with significant background blur",
    "Extreme": "extreme bokeh with completely dissolved background",
}

SHOT_TYPES = {
    "Auto": "",
    "Extreme Close-up": "extreme close-up shot focusing on specific detail",
    "Close-up": "close-up shot of face or subject",
    "Medium Close-up": "medium close-up from chest up",
    "Medium Shot": "medium shot from waist up",
    "Medium Full": "medium full shot from knees up",
    "Full Body": "full body shot showing entire subject",
    "Wide Shot": "wide shot with environmental context",
    "Extreme Wide": "extreme wide shot with expansive environment",
    "POV": "point-of-view shot from subject perspective",
    "Over-the-Shoulder": "over-the-shoulder shot",
    "Dutch Angle": "dutch angle with tilted camera",
    "Low Angle": "low angle shot looking up at subject",
    "High Angle": "high angle shot looking down at subject",
    "Bird's Eye": "bird's eye view from directly above",
    "Worm's Eye": "worm's eye view from ground level",
}

LIGHT_TYPES = {
    "Auto": "",
    "Natural": "natural available light",
    "Studio": "professional studio lighting",
    "Window": "soft window light",
    "Golden Hour": "warm golden hour sunlight",
    "Blue Hour": "cool blue hour ambient light",
    "Overcast": "soft diffused overcast lighting",
    "Hard Sun": "hard direct sunlight with strong shadows",
    "Neon": "colorful neon lighting",
    "Candlelight": "warm intimate candlelight",
    "Moonlight": "cool ethereal moonlight",
    "Mixed": "mixed color temperature lighting",
    "Practical": "practical on-set lighting sources",
    "Dramatic": "dramatic high-contrast lighting",
}

LIGHT_DIRECTIONS = {
    "Auto": "",
    "Front": "flat front lighting",
    "Rembrandt": "Rembrandt lighting with triangle highlight on cheek",
    "Loop": "loop lighting with small shadow from nose",
    "Butterfly": "butterfly lighting from above creating shadow under nose",
    "Split": "split lighting with half face in shadow",
    "Rim/Back": "rim lighting creating glowing edge outline",
    "Side": "side lighting with dramatic shadows",
    "Under": "under lighting with dramatic upward shadows",
    "Top": "top lighting with overhead illumination",
    "Broad": "broad lighting illuminating wider side of face",
    "Short": "short lighting illuminating narrow side of face",
}

LIGHT_QUALITY = {
    "Auto": "",
    "Soft": "soft diffused light with gentle shadows",
    "Hard": "hard light with sharp defined shadows",
    "Diffused": "heavily diffused ethereal light",
    "Specular": "specular highlights with shine",
    "Dappled": "dappled light filtering through leaves",
    "Volumetric": "volumetric light rays visible in atmosphere",
    "Wraparound": "wraparound soft light from multiple directions",
}

FILM_GRAIN = {
    "None": "",
    "Fine": "fine subtle film grain",
    "Medium": "medium film grain texture",
    "Heavy": "heavy pronounced film grain",
    "ISO 400": "natural ISO 400 film grain",
    "ISO 800": "visible ISO 800 grain",
    "ISO 1600": "noticeable ISO 1600 grain",
    "ISO 3200": "prominent ISO 3200 high-ISO grain",
    "ISO 6400": "heavy ISO 6400 noise grain",
}

LENS_FLARE = {
    "None": "",
    "Subtle": "subtle lens flare",
    "Cinematic": "cinematic lens flare with light streaks",
    "Anamorphic": "horizontal anamorphic lens flare streaks",
    "Vintage": "vintage multi-element lens flare",
    "Sun Star": "starburst sun flare",
}

VIGNETTE = {
    "None": "",
    "Light": "light natural vignette",
    "Medium": "medium vignette darkening edges",
    "Heavy": "heavy dramatic vignette",
    "Artistic": "artistic pronounced vignette",
}

COLOR_GRADES = {
    "None": "",
    "Teal & Orange": "teal and orange color grading",
    "Desaturated": "desaturated muted colors",
    "High Contrast": "high contrast punchy colors",
    "Muted": "muted pastel color palette",
    "Vibrant": "vibrant saturated colors",
    "Warm": "warm color temperature shift",
    "Cool": "cool color temperature shift",
    "Cross-Process": "cross-processed color shift",
    "Bleach Bypass": "bleach bypass desaturated high contrast",
    "Cinematic": "cinematic color grading with lifted blacks",
}

FILM_STOCKS = {
    "None": "",
    "Kodak Portra 400": "Kodak Portra 400 film with warm skin tones and pastel colors",
    "Kodak Portra 800": "Kodak Portra 800 film with visible grain and warm tones",
    "Kodak Ektar 100": "Kodak Ektar 100 with vivid saturated colors",
    "Kodak Gold 200": "Kodak Gold 200 consumer film aesthetic",
    "Fuji Pro 400H": "Fuji Pro 400H with green-shifted shadows and soft colors",
    "Fuji Superia": "Fuji Superia with green tint and nostalgic colors",
    "Fuji Velvia 50": "Fuji Velvia 50 slide film with punchy saturated colors",
    "Cinestill 800T": "Cinestill 800T tungsten film with halation and cinema colors",
    "Ilford HP5": "Ilford HP5 black and white film with rich tones",
    "Kodak Tri-X": "Kodak Tri-X black and white with classic contrast",
    "Polaroid": "Polaroid instant film aesthetic with soft colors",
}

CHROMATIC_ABERRATION = {
    "None": "",
    "Subtle": "subtle chromatic aberration on edges",
    "Moderate": "visible RGB fringing chromatic aberration",
    "Stylized": "stylized pronounced chromatic aberration",
}

MOTION_BLUR = {
    "None": "",
    "Subject Motion": "motion blur on moving subject",
    "Panning": "panning motion blur with sharp subject and blurred background",
    "Zoom Burst": "zoom burst radial motion blur",
    "Long Exposure": "long exposure motion blur trails",
}


# =============================================================================
# ComfyUI Node
# =============================================================================

class SID_ZImagePhotographyPrompts(comfy_io.ComfyNode):
    """
    Photography Prompt Builder for Z-Image workflows.

    Detail levels:
    - Quick: Basic combination of settings (no LLM)
    - Standard: LLM-enhanced with photography context
    - Detailed: Deep analysis of how each setting affects the image
    - Extreme: Two-pass deep analysis with synthesis

    Creates detailed photography-specific prompts including camera, lens,
    lighting, effects, and color grading.
    """

    @classmethod
    def define_schema(cls) -> comfy_io.Schema:
        return comfy_io.Schema(
            node_id="SID_ZImagePhotographyPrompts",
            display_name="SID Z-Image Photography Prompts",
            category="SID Photography Toolkit/Prompt",
            description="Photography prompts with camera, lens, lighting, effects - LLM-enhanced",
            inputs=[
                # LLM Model Input (optional for Quick mode)
                LLM_MODEL_Type.Input(
                    "llm_model",
                    optional=True,
                    tooltip="Connect SID_LLM_API or SID_LLM_Local (required for Standard/Detailed/Extreme)"
                ),

                # Base prompt
                comfy_io.String.Input(
                    "prompt",
                    default="",
                    tooltip="The base prompt to enhance with photography settings"
                ),

                # Analysis Mode (wire from Generator or type manually)
                comfy_io.String.Input(
                    "analysis_mode",
                    default="Standard",
                    tooltip="Quick/Standard/Detailed/Extreme - wire from Generator's analysis_mode output"
                ),

                # Prompt Length
                comfy_io.Int.Input(
                    "prompt_length",
                    default=150,
                    min=0,
                    max=500,
                    tooltip="Target word count (0=unlimited, 80-250 optimal for Z-Image). Wire from Generator for consistency."
                ),

                # Camera & Lens
                comfy_io.Combo.Input("camera_type", options=list(CAMERA_TYPES.keys()), default="Auto"),
                comfy_io.Combo.Input("lens_type", options=list(LENS_TYPES.keys()), default="Auto"),
                comfy_io.Combo.Input("focal_length", options=list(FOCAL_LENGTHS.keys()), default="85mm"),
                comfy_io.Int.Input("custom_focal_mm", default=50, min=8, max=800, tooltip="Custom focal length when 'Custom' selected"),
                comfy_io.Combo.Input("aperture", options=list(APERTURES.keys()), default="f/1.4"),

                # Bokeh
                comfy_io.Combo.Input("bokeh_style", options=list(BOKEH_STYLES.keys()), default="Smooth/Creamy"),
                comfy_io.Combo.Input("bokeh_intensity", options=list(BOKEH_INTENSITY.keys()), default="Strong"),

                # Shot Type
                comfy_io.Combo.Input("shot_type", options=list(SHOT_TYPES.keys()), default="Auto"),

                # Lighting
                comfy_io.Combo.Input("light_type", options=list(LIGHT_TYPES.keys()), default="Auto"),
                comfy_io.Combo.Input("light_direction", options=list(LIGHT_DIRECTIONS.keys()), default="Rembrandt"),
                comfy_io.Combo.Input("light_quality", options=list(LIGHT_QUALITY.keys()), default="Soft"),

                # Effects
                comfy_io.Combo.Input("film_grain", options=list(FILM_GRAIN.keys()), default="None"),
                comfy_io.Combo.Input("lens_flare", options=list(LENS_FLARE.keys()), default="None"),
                comfy_io.Combo.Input("vignette", options=list(VIGNETTE.keys()), default="None"),
                comfy_io.Combo.Input("chromatic_aberration", options=list(CHROMATIC_ABERRATION.keys()), default="None"),
                comfy_io.Combo.Input("motion_blur", options=list(MOTION_BLUR.keys()), default="None"),

                # Color
                comfy_io.Combo.Input("color_grade", options=list(COLOR_GRADES.keys()), default="None"),
                comfy_io.Combo.Input("film_stock", options=list(FILM_STOCKS.keys()), default="None"),

                # Seed Control
                comfy_io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xffffffffffffffff,
                    control_after_generate=True,
                    tooltip="Random seed for reproducibility"
                ),

                # Z-Image Vocabulary Optimization
                comfy_io.Boolean.Input(
                    "zimage_optimize",
                    default=True,
                    display_name="Z-Image Optimize",
                    tooltip="Apply Z-Image vocabulary optimization (converts tag soup, removes anti-patterns, injects lighting/composition)"
                ),
            ],
            outputs=[
                comfy_io.String.Output("enhanced_prompt", display_name="enhanced_prompt"),
                comfy_io.String.Output("original_prompt", display_name="original_prompt"),
                comfy_io.String.Output("photography_settings", display_name="photography_settings"),
            ],
        )

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
            return None

        else:
            raise ValueError(f"Unsupported provider: {provider}")

    @classmethod
    def _call_llm(cls, client, llm_model: LLMModelConfig, system_prompt: str, user_prompt: str, spinner_msg: str = "Calling LLM") -> str:
        """Make LLM call with progress spinner."""
        # Check for interrupt before LLM call
        check_interrupted()

        if llm_model.provider.lower() == "local":
            return cls._call_local_llm(llm_model, system_prompt, user_prompt, spinner_msg)

        with ProgressSpinner(spinner_msg, prefix="[PhotographyPrompts]"):
            if hasattr(client, 'messages'):
                response = client.messages.create(
                    model=llm_model.model,
                    max_tokens=llm_model.max_tokens,
                    temperature=llm_model.temperature,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}]
                )
                return response.content[0].text
            else:
                response = client.chat.completions.create(
                    model=llm_model.model,
                    max_tokens=llm_model.max_tokens,
                    temperature=llm_model.temperature,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                )
                return response.choices[0].message.content

    @classmethod
    def _call_local_llm(cls, llm_model: LLMModelConfig, system_prompt: str, user_prompt: str, spinner_msg: str = "Generating") -> str:
        """Call local text model (spinner handled by LocalModelClient)."""
        # Check for interrupt before local LLM call
        check_interrupted()

        from .llm_providers.sid_llm_local import LocalModelClient

        print(f"[PhotographyPrompts] {spinner_msg} (local model)...")
        extra = llm_model.extra_params or {}
        client = LocalModelClient(
            model_name=llm_model.model,
            quantization=extra.get("quantization", "4-bit"),
            device=extra.get("device", "auto"),
            attention_mode=extra.get("attention_mode", "auto"),
            keep_model_loaded=extra.get("keep_model_loaded", True),
            top_p=extra.get("top_p", 0.9),
            repetition_penalty=extra.get("repetition_penalty", 1.2),
            use_torch_compile=extra.get("use_torch_compile", False),
        )
        return client.generate_text(
            prompt=user_prompt,
            system_prompt=system_prompt,
            max_tokens=llm_model.max_tokens,
            temperature=llm_model.temperature,
        )

    @classmethod
    def _build_settings_dict(
        cls,
        camera_type: str, lens_type: str, focal_length: str, custom_focal_mm: int,
        aperture: str, bokeh_style: str, bokeh_intensity: str, shot_type: str,
        light_type: str, light_direction: str, light_quality: str,
        film_grain: str, lens_flare: str, vignette: str,
        chromatic_aberration: str, motion_blur: str,
        color_grade: str, film_stock: str
    ) -> Dict[str, str]:
        """Build a dictionary of active settings."""
        settings = {}

        if CAMERA_TYPES.get(camera_type):
            settings["Camera"] = CAMERA_TYPES[camera_type]
        if LENS_TYPES.get(lens_type):
            settings["Lens"] = LENS_TYPES[lens_type]
        if focal_length == "Custom":
            settings["Focal Length"] = f"{custom_focal_mm}mm focal length"
        elif FOCAL_LENGTHS.get(focal_length):
            settings["Focal Length"] = FOCAL_LENGTHS[focal_length]
        if APERTURES.get(aperture):
            settings["Aperture"] = APERTURES[aperture]
        if BOKEH_STYLES.get(bokeh_style):
            settings["Bokeh Style"] = BOKEH_STYLES[bokeh_style]
        if BOKEH_INTENSITY.get(bokeh_intensity):
            settings["Bokeh Intensity"] = BOKEH_INTENSITY[bokeh_intensity]
        if SHOT_TYPES.get(shot_type):
            settings["Shot Type"] = SHOT_TYPES[shot_type]
        if LIGHT_TYPES.get(light_type):
            settings["Light Type"] = LIGHT_TYPES[light_type]
        if LIGHT_DIRECTIONS.get(light_direction):
            settings["Light Direction"] = LIGHT_DIRECTIONS[light_direction]
        if LIGHT_QUALITY.get(light_quality):
            settings["Light Quality"] = LIGHT_QUALITY[light_quality]
        if FILM_GRAIN.get(film_grain):
            settings["Film Grain"] = FILM_GRAIN[film_grain]
        if LENS_FLARE.get(lens_flare):
            settings["Lens Flare"] = LENS_FLARE[lens_flare]
        if VIGNETTE.get(vignette):
            settings["Vignette"] = VIGNETTE[vignette]
        if CHROMATIC_ABERRATION.get(chromatic_aberration):
            settings["Chromatic Aberration"] = CHROMATIC_ABERRATION[chromatic_aberration]
        if MOTION_BLUR.get(motion_blur):
            settings["Motion Blur"] = MOTION_BLUR[motion_blur]
        if COLOR_GRADES.get(color_grade):
            settings["Color Grade"] = COLOR_GRADES[color_grade]
        if FILM_STOCKS.get(film_stock):
            settings["Film Stock"] = FILM_STOCKS[film_stock]

        return settings

    @classmethod
    def _generate_cache_key(cls, seed: int, prompt: str, analysis_mode: str, settings_str: str, model: str) -> str:
        """Generate a unique cache key based on inputs and seed."""
        key_data = f"{seed}|{prompt}|{analysis_mode}|{settings_str}|{model}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:32]

    @classmethod
    def execute(
        cls,
        llm_model: LLMModelConfig,
        prompt: str,
        analysis_mode: str,
        prompt_length: int,
        camera_type: str,
        lens_type: str,
        focal_length: str,
        custom_focal_mm: int,
        aperture: str,
        bokeh_style: str,
        bokeh_intensity: str,
        shot_type: str,
        light_type: str,
        light_direction: str,
        light_quality: str,
        film_grain: str,
        lens_flare: str,
        vignette: str,
        chromatic_aberration: str,
        motion_blur: str,
        color_grade: str,
        film_stock: str,
        seed: int,
        zimage_optimize: bool = True,
    ) -> comfy_io.NodeOutput:
        """Build and optionally enhance photography prompt."""
        global _photography_cache
        start_time = time.time()

        # Check for user interrupt at start
        check_interrupted()

        # Clear VRAM and run garbage collection before starting
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                free_vram = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
                print(f"[PhotographyPrompts] VRAM cleared. Available: {free_vram / 1024**3:.1f}GB")
        except Exception:
            pass

        # Validate analysis_mode, default to Standard if invalid
        if analysis_mode not in DETAIL_LEVELS:
            print(f"[PhotographyPrompts] Invalid analysis_mode '{analysis_mode}', using Standard")
            analysis_mode = "Standard"

        # Initialize progress bar (3 steps: build settings, enhance, z-image optimize)
        pbar = comfy.utils.ProgressBar(3)

        # Header
        print("")
        print("=" * 60)
        print("[PhotographyPrompts] Starting photography prompt enhancement")
        print("=" * 60)

        # Build settings dictionary
        settings = cls._build_settings_dict(
            camera_type, lens_type, focal_length, custom_focal_mm,
            aperture, bokeh_style, bokeh_intensity, shot_type,
            light_type, light_direction, light_quality,
            film_grain, lens_flare, vignette,
            chromatic_aberration, motion_blur,
            color_grade, film_stock
        )

        # Build photography settings string
        photography_settings = ", ".join(settings.values()) if settings else ""

        # Show input stats
        word_count = len(prompt.split()) if prompt.strip() else 0
        settings_count = len(settings)
        print(f"[PhotographyPrompts] Input: {word_count} words")
        length_info = f"{prompt_length} words" if prompt_length > 0 else "unlimited"
        print(f"[PhotographyPrompts] Mode: {analysis_mode} | Length: {length_info}")
        print(f"[PhotographyPrompts] Active settings: {settings_count}")
        print(f"[PhotographyPrompts] Seed: {seed}")
        if settings:
            for key, val in list(settings.items())[:5]:  # Show first 5 settings
                print(f"  - {key}: {val[:40]}..." if len(val) > 40 else f"  - {key}: {val}")
            if len(settings) > 5:
                print(f"  ... and {len(settings) - 5} more settings")

        # Update progress: settings built
        pbar.update(1)

        try:
            # Quick mode - just combine (no LLM needed, no caching needed)
            if analysis_mode == "Quick":
                print("-" * 60)
                print("[PhotographyPrompts] Quick Mode: Combining prompt with settings...")
                if prompt.strip() and photography_settings:
                    enhanced = f"{prompt.strip()}, {photography_settings}"
                elif prompt.strip():
                    enhanced = prompt.strip()
                else:
                    enhanced = photography_settings

                # Update progress: enhancement done
                pbar.update(1)

                # Apply Z-Image vocabulary optimization if enabled
                if zimage_optimize:
                    translator = get_translator()
                    result = translator.translate(enhanced)
                    enhanced = result.translated

                # Update progress: Z-Image optimization done
                pbar.update(1)

                elapsed = time.time() - start_time
                print(f"[PhotographyPrompts] ✓ Quick combine complete ({elapsed:.1f}s)")
                print("=" * 60)
                print("")
                return comfy_io.NodeOutput(enhanced, prompt, photography_settings)

            # LLM-powered modes require llm_model
            if llm_model is None:
                print("[PhotographyPrompts] WARNING: No LLM connected, falling back to Quick mode")
                if prompt.strip() and photography_settings:
                    enhanced = f"{prompt.strip()}, {photography_settings}"
                else:
                    enhanced = prompt.strip() or photography_settings
                # Update progress: enhancement done
                pbar.update(1)
                # Apply Z-Image vocabulary optimization if enabled
                if zimage_optimize:
                    translator = get_translator()
                    result = translator.translate(enhanced)
                    enhanced = result.translated
                # Update progress: Z-Image optimization done
                pbar.update(1)
                return comfy_io.NodeOutput(enhanced, prompt, photography_settings)

            # Generate cache key for LLM modes
            cache_key = cls._generate_cache_key(
                seed, prompt, analysis_mode, photography_settings, llm_model.model
            )

            # Check cache
            if cache_key in _photography_cache:
                cached_result = _photography_cache[cache_key]
                elapsed = time.time() - start_time
                print(f"[PhotographyPrompts] Cache HIT (seed: {seed})")
                print(f"[PhotographyPrompts] Returning cached result ({len(cached_result.split())} words)")
                print(f"[PhotographyPrompts] Total time: {elapsed:.3f}s")
                print("=" * 60)
                print("")
                # Update progress: all steps done (cache hit)
                pbar.update(2)
                return comfy_io.NodeOutput(cached_result, prompt, photography_settings)

            print(f"[PhotographyPrompts] Cache MISS (seed: {seed})")

            # Check if local model supports text
            if llm_model.provider.lower() == "local":
                if not llm_model.extra_params.get("supports_text", False):
                    raise ValueError(
                        f"Model '{llm_model.model}' is a vision-only model. "
                        "Please use a (Text) model."
                    )

            # Efficiency optimization: Local models use Quick mode (no LLM call) to avoid garbage output
            is_local = llm_model.provider.lower() == "local" or llm_model.extra_params.get("is_local", False)
            supports_reasoning = llm_model.supports_reasoning
            
            # Local models: Force Quick mode - LLM integration often produces garbage
            if is_local and analysis_mode != "Quick":
                print(f"[PhotographyPrompts] Local model detected: {analysis_mode} -> Quick (no LLM call)")
                # Return Quick mode result directly
                if prompt.strip() and photography_settings:
                    enhanced = f"{prompt.strip()}, {photography_settings}"
                elif prompt.strip():
                    enhanced = prompt.strip()
                else:
                    enhanced = photography_settings
                # Update progress: all steps done
                pbar.update(2)
                elapsed = time.time() - start_time
                print(f"[PhotographyPrompts] Quick combine complete ({elapsed:.1f}s)")
                print("=" * 60)
                return comfy_io.NodeOutput(enhanced, prompt, photography_settings)
            
            print(f"[PhotographyPrompts] Provider: {llm_model.provider}")
            print(f"[PhotographyPrompts] Model: {llm_model.model}")

            # Check if model supports reasoning for agentic mode
            supports_reasoning = llm_model.supports_reasoning
            print(f"[PhotographyPrompts] Reasoning: {'enabled' if supports_reasoning else 'disabled'}")

            print("-" * 60)

            # Standard mode: Python-only (NO LLM) - just combine and let Z-Image vocab optimize
            if analysis_mode == "Standard":
                print("[PhotographyPrompts] Standard Mode: Python-only integration (no LLM)...")
                enhanced = combine_prompt_with_settings(prompt, photography_settings, natural_language=True)
                print(f"[PhotographyPrompts] Combined: {len(enhanced.split())} words")
                client = None  # No client needed

            # Detailed/Extreme: Use LLM - AGENTIC transformation only with API + Reasoning
            elif analysis_mode in ["Detailed", "Extreme"]:
                print("[PhotographyPrompts] Initializing LLM client...")
                client = cls._get_client(llm_model)

                if supports_reasoning and not is_local:
                    # AGENTIC MODE: Deep transformation (API + Reasoning only)
                    if analysis_mode == "Detailed":
                        print("[PhotographyPrompts] Detailed Mode (Agentic): Deep photography TRANSFORMATION...")
                        enhanced = cls._deep_enhance(client, llm_model, prompt, settings, prompt_length)
                    else:  # Extreme
                        print("[PhotographyPrompts] Extreme Mode (Agentic): Two-pass deep TRANSFORMATION...")
                        enhanced = cls._extreme_enhance(client, llm_model, prompt, settings, prompt_length)
                else:
                    # SINGLE-SHOT MODE: LLM integrates settings while preserving original (non-reasoning or local)
                    mode_reason = "local model" if is_local else "non-reasoning model"
                    print(f"[PhotographyPrompts] {analysis_mode} Mode (Single-shot): {mode_reason} - integrating settings...")
                    enhanced = cls._singleshot_enhance(client, llm_model, prompt, settings, photography_settings, prompt_length)
                    print(f"[PhotographyPrompts] Note: For deep transformation, use API model with reasoning enabled")

            else:
                # Fallback
                enhanced = combine_prompt_with_settings(prompt, photography_settings, natural_language=True)
                client = None

            # Store in cache (with size limit and cleanup)
            if len(_photography_cache) >= _MAX_CACHE_SIZE:
                # Remove 20% of oldest entries to prevent constant eviction
                num_to_remove = max(1, _MAX_CACHE_SIZE // 5)
                keys_to_remove = list(_photography_cache.keys())[:num_to_remove]
                for key in keys_to_remove:
                    del _photography_cache[key]
                print(f"[PhotographyPrompts] Cache cleanup: removed {num_to_remove} old entries")
                gc.collect()
            _photography_cache[cache_key] = enhanced
            print(f"[PhotographyPrompts] Result cached (cache size: {len(_photography_cache)})")

            # Update progress: enhancement done
            pbar.update(1)

            # Apply Z-Image vocabulary optimization if enabled
            if zimage_optimize:
                print("[PhotographyPrompts] Applying Z-Image vocabulary optimization...")
                translator = get_translator()
                result = translator.translate(enhanced)
                enhanced = result.translated
                if result.changes_made:
                    print(f"[PhotographyPrompts] Z-Image changes: {len(result.changes_made)}")
                    for change in result.changes_made[:3]:
                        print(f"  - {change}")

            # Update progress: Z-Image optimization done
            pbar.update(1)

            elapsed = time.time() - start_time
            output_words = len(enhanced.split())
            word_diff = output_words - word_count
            diff_str = f"+{word_diff}" if word_diff >= 0 else str(word_diff)

            print("-" * 60)
            print(f"[PhotographyPrompts] Complete!")
            print(f"[PhotographyPrompts] Input: {word_count} words -> Output: {output_words} words ({diff_str})")
            print(f"[PhotographyPrompts] Total time: {elapsed:.1f}s")
            print("=" * 60)
            print("")

            # Unload local model to free VRAM for subsequent nodes
            if llm_model and llm_model.provider.lower() == "local":
                try:
                    from .llm_providers.sid_llm_local import LocalModelClient
                    LocalModelClient.unload_model()
                except Exception:
                    pass

            return comfy_io.NodeOutput(enhanced, prompt, photography_settings)

        except Exception as e:
            print(f"[PhotographyPrompts] ERROR: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to simple combination
            if prompt.strip() and photography_settings:
                return comfy_io.NodeOutput(f"{prompt.strip()}, {photography_settings}", prompt, photography_settings)
            return comfy_io.NodeOutput(prompt or photography_settings, prompt, photography_settings)

    @classmethod
    def _standard_enhance(cls, client, llm_model: LLMModelConfig, prompt: str, settings: Dict, photography_str: str) -> str:
        """Standard: Basic LLM enhancement with photography context - FAST MODE."""
        # Select prompt based on model type
        is_local = llm_model.provider.lower() == "local" or llm_model.extra_params.get("is_local", False)
        if is_local:
            system = """Add these photography settings to the prompt. Keep original content.
Output ONLY the combined prompt."""
        else:
            system = """Integrate photography settings into this prompt.
Keep all original content. Settings describe visual effects (blur, grain, tone).
Output ONLY the combined prompt - no explanations."""

        user_prompt = f"""Base prompt (PRESERVE ALL CONTENT):
{prompt if prompt.strip() else "(no base prompt)"}

Photography settings to integrate as VISUAL EFFECTS (not visible equipment):
{photography_str}

Create an enhanced prompt that keeps ALL original details and adds photography visual effects:"""

        result = cls._call_llm(client, llm_model, system, user_prompt, "Standard integration")
        return cls._clean_response(result, f"{prompt}, {photography_str}" if prompt else photography_str)

    @classmethod
    def _singleshot_enhance(cls, client, llm_model: LLMModelConfig, prompt: str, settings: Dict, photography_str: str, prompt_length: int = 150) -> str:
        """Single-shot: LLM integrates settings while preserving original (for non-reasoning/local models)."""
        # Build settings breakdown
        settings_breakdown = "\n".join([f"- {k}: {v}" for k, v in settings.items()])

        # Get length constraint
        length_constraint = config_loader.get_length_constraint(prompt_length)
        length_section = f"\n\nOUTPUT LENGTH:\n{length_constraint}" if length_constraint else ""

        # Select prompt based on model type
        is_local = llm_model.provider.lower() == "local" or llm_model.extra_params.get("is_local", False)

        if is_local:
            system = """Add photography visual effects to the prompt using natural language.
Keep original content intact. Use descriptive phrases not tags.
Output ONLY the enhanced prompt."""
        else:
            system = """You are a Z-Image prompt engineer. Add photography settings using NATURAL LANGUAGE.

Z-IMAGE VOCABULARY RULES:
- Use flowing sentences, NOT comma-separated tags
- Describe visual effects: "soft creamy bokeh" not just "bokeh"
- Preferred terms: "shallow depth of field with buttery background blur"
- Lighting as visual: "Rembrandt lighting with triangle shadow on cheek"
- Color as experience: "warm Kodak Portra skin tones"

RULES:
1. KEEP the original prompt content exactly as-is
2. ADD photography settings as natural language visual descriptions
3. Describe visual EFFECTS not equipment names
4. Append naturally at appropriate points or end

Output ONLY the enhanced prompt - no explanations."""

        user_prompt = f"""ORIGINAL PROMPT (preserve completely):
{prompt if prompt.strip() else "(general photography shot)"}

PHOTOGRAPHY SETTINGS TO ADD AS VISUAL EFFECTS:
{settings_breakdown}
{length_section}

Add these settings as natural language visual effect descriptions.
Use Z-Image vocabulary: "soft diffused lighting", "creamy bokeh blur", "warm film tones"
Keep the original prompt intact.

Enhanced prompt:"""

        result = cls._call_llm(client, llm_model, system, user_prompt, "Single-shot enhancement")
        return cls._clean_response(result, f"{prompt}, {photography_str}" if prompt else photography_str)

    @classmethod
    def _deep_enhance(cls, client, llm_model: LLMModelConfig, prompt: str, settings: Dict, prompt_length: int = 150) -> str:
        """Detailed: Deep transformation of how each setting affects every visual element."""
        # Build detailed settings breakdown with transformation hints
        settings_breakdown = "\n".join([f"- {k}: {v}" for k, v in settings.items()])

        # Get length constraint
        length_constraint = config_loader.get_length_constraint(prompt_length)
        length_section = f"\n\nOUTPUT LENGTH:\n{length_constraint}" if length_constraint else ""

        user_prompt = f"""TRANSFORM this prompt using Z-IMAGE VOCABULARY, applying photography settings to EVERY visual element.

=== ORIGINAL PROMPT ===
{prompt if prompt.strip() else "(general photography shot)"}
=== END ORIGINAL PROMPT ===

=== PHOTOGRAPHY SETTINGS TO APPLY ===
{settings_breakdown}
=== END SETTINGS ===
{length_section}

Z-IMAGE VOCABULARY - USE THESE NATURAL LANGUAGE PATTERNS:
- Aperture: "f/1.4 wide open aperture creating paper-thin depth of field with soft creamy bokeh"
- Lighting: "Rembrandt lighting creating signature triangle of light on the shadowed cheek"
- Background: "background dissolving into luminous bokeh orbs with painterly blur"
- Film: "Kodak Portra 400 rendering with warm peachy skin tones and muted pastel colors"
- Lens: "85mm portrait lens compressing background planes with flattering perspective"

TRANSFORMATION REQUIREMENTS:

1. **Backgrounds**: Transform to "soft creamy bokeh", "luminous out-of-focus orbs", "painterly blur dissolving details"

2. **Subject Lighting**: Apply as visual descriptions:
   - "Rembrandt lighting with triangle shadow on cheek"
   - "soft wraparound light gently modeling facial contours"
   - "rim light creating luminous halo around hair edges"

3. **Colors/Film**: Transform entire palette:
   - "warm Kodak Portra skin tones with peachy undertones"
   - "fine organic film grain adding tactile texture"

4. **Atmosphere**: Describe visual effects:
   - "subtle lens flare streaking across highlights"
   - "natural vignette drawing focus to center"

WEAVE settings naturally throughout - DO NOT append as a list.

Example Z-Image style:
"close-up portrait of woman with golden blonde hair catching warm rim light creating luminous halo effect, Rembrandt lighting sculpting her features with signature triangle of light on the shadowed cheek, soft diffused background dissolving into creamy bokeh orbs, shot with 85mm portrait lens at f/1.4 wide open aperture, Kodak Portra 400 film rendering warm peachy skin tones with fine organic grain texture"

Output the TRANSFORMED prompt:"""

        result = cls._call_llm(client, llm_model, AGENTIC_PHOTOGRAPHY_SYSTEM, user_prompt, "Deep photography transformation")
        return cls._clean_response(result, prompt)

    @classmethod
    def _extreme_enhance(cls, client, llm_model: LLMModelConfig, prompt: str, settings: Dict, prompt_length: int = 150) -> str:
        """Extreme: Two-pass deep transformation with micro-detail synthesis."""
        # First pass: Deep transformation
        settings_breakdown = "\n".join([f"- {k}: {v}" for k, v in settings.items()])

        # Get length constraint
        length_constraint = config_loader.get_length_constraint(prompt_length)
        length_section = f"\n\nOUTPUT LENGTH:\n{length_constraint}" if length_constraint else ""

        user_prompt = f"""PASS 1: DEEP TRANSFORMATION using Z-IMAGE VOCABULARY
Transform EVERY visual element using natural language photography descriptions.

=== ORIGINAL PROMPT ===
{prompt if prompt.strip() else "(general photography shot)"}
=== END ORIGINAL PROMPT ===

=== PHOTOGRAPHY SETTINGS TO APPLY ===
{settings_breakdown}
=== END SETTINGS ===
{length_section}

Z-IMAGE VOCABULARY PATTERNS TO USE:
- Bokeh: "soft creamy bokeh", "luminous bokeh orbs", "painterly background blur"
- DOF: "paper-thin depth of field", "razor-sharp focus with rapid falloff"
- Lighting: "Rembrandt lighting with triangle shadow", "soft wraparound light modeling contours"
- Film: "Kodak Portra warm skin tones", "fine organic film grain texture"
- Lens: "85mm portrait lens with flattering compression", "shallow depth of field with buttery blur"

TRANSFORMATION CHECKLIST:

□ BACKGROUNDS → "dissolving into soft creamy bokeh orbs", "painterly blur"
□ LIGHTING → "Rembrandt triangle of light on shadowed cheek", "luminous rim light halo"
□ SKIN → "warm Portra skin tones with peachy undertones", "soft light modeling facial contours"
□ HAIR → "rim light creating luminous edge glow", "individual strands catching warm highlights"
□ COLORS → "shifted to warm film stock palette", "muted pastel tones"
□ ATMOSPHERE → "fine organic film grain", "subtle natural vignette"

Example Z-Image transformation:
Original: "woman in red dress in garden"
→ "woman bathed in soft golden hour light wrapping around her form, the red dress rendered in deep coral-orange by the warm sunset and Kodak Portra's signature color science, background garden dissolving into soft creamy bokeh pools of green and gold, Rembrandt lighting creating gentle triangle shadow on her cheek, rim light adding luminous halo to her hair edges, fine organic film grain adding tactile texture, shot at f/1.4 with paper-thin depth of field"

Output the TRANSFORMED prompt using natural language:"""

        enhanced = cls._call_llm(client, llm_model, AGENTIC_PHOTOGRAPHY_SYSTEM, user_prompt, "Pass 1: Deep transformation")
        enhanced = cls._clean_response(enhanced, prompt)

        pass1_words = len(enhanced.split())
        print(f"[PhotographyPrompts] Pass 1 complete: {pass1_words} words")

        # Second pass: Micro-detail synthesis
        synthesis_prompt = f"""PASS 2: Z-IMAGE MICRO-DETAIL SYNTHESIS
Add the finest photographic micro-details using natural language descriptions.

=== TRANSFORMED PROMPT ===
{enhanced}
=== END TRANSFORMED PROMPT ===
{length_section}

ADD Z-IMAGE STYLE MICRO-DETAILS:

1. **Light Interaction** (natural language):
   - "soft subsurface scattering creating warm glow through ear edges"
   - "specular catchlights dancing in the eyes"
   - "light wrapping gently around facial contours"

2. **Texture Details** (descriptive):
   - "individual fabric threads catching directional light"
   - "skin texture rendered with pore-level detail in focus plane"
   - "hair strands separated by rim light with visible light transmission"

3. **Photographic Artifacts** (as visual effects):
   - "fine organic film grain distributed across midtones"
   - "subtle focus breathing in transition zones"
   - "natural optical vignette darkening frame corners"

4. **Depth Transitions**:
   - "razor-sharp focus transitioning to buttery soft blur"
   - "luminous bokeh orbs with subtle highlight clipping"

Maintain natural language flow - NO tag soup. ENRICH existing content.

Output the final Z-IMAGE OPTIMIZED prompt:"""

        final = cls._call_llm(client, llm_model, AGENTIC_PHOTOGRAPHY_SYSTEM, synthesis_prompt, "Pass 2: Micro-detail synthesis")
        return cls._clean_response(final, enhanced)

    @classmethod
    def _clean_response(cls, response: str, original: str) -> str:
        """Clean up the LLM response."""
        if not response:
            return original

        # Remove common prefixes
        prefixes = [
            r'^(?:Here\'?s?|This is|The|An?) (?:the |an? )?(?:enhanced|improved|final|refined) (?:prompt|version)[:\s]*\n*',
            r'^(?:Enhanced|Improved|Final|Refined) (?:prompt|version)[:\s]*\n*',
            r'^(?:Sure|Certainly|Of course)[,!]?[^:]*[:\s]*\n*',
        ]
        for p in prefixes:
            response = re.sub(p, '', response, flags=re.IGNORECASE | re.MULTILINE)

        # Remove markdown code blocks
        code_match = re.search(r'```(?:\w+)?\s*([\s\S]*?)```', response)
        if code_match:
            response = code_match.group(1)

        # Remove surrounding quotes
        if (response.startswith('"') and response.endswith('"')) or \
           (response.startswith("'") and response.endswith("'")):
            response = response[1:-1]

        # Remove trailing explanations
        explanations = [
            r'\n\n(?:This|The|I\'ve|Note:|Key|Changes)[\s\S]*$',
            r'\n\n(?:Let me|Feel free|I hope)[\s\S]*$',
        ]
        for p in explanations:
            response = re.sub(p, '', response, flags=re.IGNORECASE)

        return response.strip() or original
