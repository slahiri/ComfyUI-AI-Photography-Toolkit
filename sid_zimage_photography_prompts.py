"""
SID_ZImagePhotographyPrompts Node

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
"""

import gc
import hashlib
import re
import sys
import time
import threading
from typing import Dict, List, Optional

from comfy_api.latest import io as comfy_io

from .llm_providers.llm_model_type import LLMModelConfig


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

# Deep analysis system prompt - analyzes how each photography setting affects the image
DEEP_PHOTOGRAPHY_SYSTEM = """You are a master photographer and prompt engineer specializing in photographic imagery.

## CRITICAL PRESERVATION RULES - MUST FOLLOW:
1. **NEVER DELETE** - Every word, phrase, and detail from the original prompt MUST appear in your output
2. **NEVER REPLACE** - Do not substitute original descriptions with your own interpretations
3. **NEVER SUMMARIZE** - Do not condense or paraphrase existing content
4. **ONLY ADD** - Insert new details BETWEEN or AFTER existing content to enrich it
5. **PRESERVE SPECIFICS** - Keep exact measurements, colors, textures, positions, clothing details, body descriptions

## CRITICAL - NO EQUIPMENT IN SCENE:
- Photography settings describe HOW the image is captured, NOT what appears in the image
- NEVER add cameras, lenses, tripods, or photography equipment as visible objects
- Terms like "shot on Hasselblad" mean the IMAGE QUALITY, not a visible camera
- Describe the VISUAL EFFECTS of settings (blur, compression, grain) not the equipment itself

## Your Task:
Enhance the prompt by describing how each photography setting VISUALLY AFFECTS the scene:

**Aperture Effects** (describe the visual result):
- Wide aperture → "razor-thin plane of focus with subject emerging from creamy dissolved background"
- NOT "f/1.2 aperture" but "paper-thin depth of field isolating subject in crystalline sharpness"

**Focal Length Effects** (describe what the viewer sees):
- Telephoto → "compressed spatial planes, flattened perspective, intimate framing"
- NOT "200mm lens" but "compressed background appearing closer, flattering facial proportions"

**Lighting Effects** (describe how light falls):
- Rembrandt → "triangular highlight on cheek, dimensional shadows sculpting facial structure"
- NOT "Rembrandt lighting setup" but the actual visual effect on the subject

**Film/Color Effects** (describe the color/tone result):
- Portra → "warm peachy skin tones, lifted pastel shadows, nostalgic color palette"

## Enhancement Method:
For each existing description in the prompt:
1. Keep the original text EXACTLY as written
2. INSERT additional micro-details about how photography settings enhance that specific element
3. Add texture, light interaction, and atmosphere details

## What You CANNOT Do:
- Remove ANY existing description from the original prompt
- Replace specific details (e.g., "espresso brown" → "dark brown")
- Change any camera/lens specs the user already specified
- Introduce visible photography equipment into the scene
- Simplify or condense the original prompt

The output must be LONGER than the input, containing 100% of original content plus visual enhancements.

Output ONLY the enhanced prompt - no analysis, no markdown, no explanations."""


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

                # Detail Level
                comfy_io.Combo.Input(
                    "detail_level",
                    options=DETAIL_LEVELS,
                    default="Standard",
                    tooltip="Quick: combine only, Standard: LLM enhance, Detailed/Extreme: deep analysis"
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
                    max=2147483647,
                    control_after_generate=True,
                    tooltip="Random seed for reproducibility"
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
    def _generate_cache_key(cls, seed: int, prompt: str, detail_level: str, settings_str: str, model: str) -> str:
        """Generate a unique cache key based on inputs and seed."""
        key_data = f"{seed}|{prompt}|{detail_level}|{settings_str}|{model}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:32]

    @classmethod
    def execute(
        cls,
        llm_model: LLMModelConfig,
        prompt: str,
        detail_level: str,
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
    ) -> comfy_io.NodeOutput:
        """Build and optionally enhance photography prompt."""
        global _photography_cache
        start_time = time.time()

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
        print(f"[PhotographyPrompts] Mode: {detail_level}")
        print(f"[PhotographyPrompts] Active settings: {settings_count}")
        print(f"[PhotographyPrompts] Seed: {seed}")
        if settings:
            for key, val in list(settings.items())[:5]:  # Show first 5 settings
                print(f"  - {key}: {val[:40]}..." if len(val) > 40 else f"  - {key}: {val}")
            if len(settings) > 5:
                print(f"  ... and {len(settings) - 5} more settings")

        try:
            # Quick mode - just combine (no LLM needed, no caching needed)
            if detail_level == "Quick":
                print("-" * 60)
                print("[PhotographyPrompts] Quick Mode: Combining prompt with settings...")
                if prompt.strip() and photography_settings:
                    enhanced = f"{prompt.strip()}, {photography_settings}"
                elif prompt.strip():
                    enhanced = prompt.strip()
                else:
                    enhanced = photography_settings

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
                return comfy_io.NodeOutput(enhanced, prompt, photography_settings)

            # Generate cache key for LLM modes
            cache_key = cls._generate_cache_key(
                seed, prompt, detail_level, photography_settings, llm_model.model
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
                return comfy_io.NodeOutput(cached_result, prompt, photography_settings)

            print(f"[PhotographyPrompts] Cache MISS (seed: {seed})")

            # Check if local model supports text
            if llm_model.provider.lower() == "local":
                if not llm_model.extra_params.get("supports_text", False):
                    raise ValueError(
                        f"Model '{llm_model.model}' is a vision-only model. "
                        "Please use a (Text) model."
                    )

            print(f"[PhotographyPrompts] Provider: {llm_model.provider}")
            print(f"[PhotographyPrompts] Model: {llm_model.model}")

            # Get client
            print("[PhotographyPrompts] Initializing LLM client...")
            client = cls._get_client(llm_model)

            print("-" * 60)
            if detail_level == "Standard":
                print("[PhotographyPrompts] Standard Mode: Integrating photography settings...")
                enhanced = cls._standard_enhance(client, llm_model, prompt, settings, photography_settings)

            elif detail_level == "Detailed":
                print("[PhotographyPrompts] Detailed Mode: Deep photography analysis...")
                enhanced = cls._deep_enhance(client, llm_model, prompt, settings)

            elif detail_level == "Extreme":
                print("[PhotographyPrompts] Extreme Mode: Two-pass deep analysis...")
                enhanced = cls._extreme_enhance(client, llm_model, prompt, settings)

            else:
                enhanced = f"{prompt.strip()}, {photography_settings}" if prompt.strip() else photography_settings

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

            # Results
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
        """Standard: Basic LLM enhancement with photography context."""
        system = """You are a photography prompt engineer. Integrate photography settings into an existing prompt.

CRITICAL RULES:
1. PRESERVE every detail from the base prompt - do NOT remove or replace anything
2. Photography settings describe VISUAL EFFECTS, not visible equipment
3. Do NOT add cameras, lenses, or equipment as objects in the scene
4. Describe what the viewer SEES (blur, compression, grain) not the equipment
5. The output must contain 100% of the original prompt content

Output ONLY the enhanced prompt - no explanations."""

        user_prompt = f"""Base prompt (PRESERVE ALL CONTENT):
{prompt if prompt.strip() else "(no base prompt)"}

Photography settings to integrate as VISUAL EFFECTS (not visible equipment):
{photography_str}

Create an enhanced prompt that keeps ALL original details and adds photography visual effects:"""

        result = cls._call_llm(client, llm_model, system, user_prompt, "Standard integration")
        return cls._clean_response(result, f"{prompt}, {photography_str}" if prompt else photography_str)

    @classmethod
    def _deep_enhance(cls, client, llm_model: LLMModelConfig, prompt: str, settings: Dict) -> str:
        """Detailed: Deep analysis of how each setting affects the image."""
        # Build detailed settings breakdown
        settings_breakdown = "\n".join([f"- {k}: {v}" for k, v in settings.items()])

        user_prompt = f"""CRITICAL: You MUST preserve EVERY word and detail from the base prompt below. Do NOT remove, replace, or summarize anything.

=== BASE PROMPT (PRESERVE 100% OF THIS CONTENT) ===
{prompt if prompt.strip() else "(general photography shot)"}
=== END BASE PROMPT ===

Photography settings to describe as VISUAL EFFECTS (NOT visible equipment in scene):
{settings_breakdown}

YOUR TASK:
1. Copy the ENTIRE base prompt as your foundation
2. INSERT photography visual effects BETWEEN and AFTER existing descriptions
3. Describe what the VIEWER SEES (blur quality, light behavior, color shifts) - NOT cameras or equipment
4. The output MUST be longer than the input, containing ALL original content

For each setting, ADD descriptions of its VISUAL effect:
- Aperture → describe the blur quality, focus falloff, bokeh character
- Focal length → describe compression, perspective, spatial relationships
- Lighting → describe how light falls on surfaces, shadow shapes, highlight behavior
- Film/color → describe the color palette, tonal quality, mood

Output the enhanced prompt (must contain 100% of original):"""

        result = cls._call_llm(client, llm_model, DEEP_PHOTOGRAPHY_SYSTEM, user_prompt, "Deep photography analysis")
        return cls._clean_response(result, prompt)

    @classmethod
    def _extreme_enhance(cls, client, llm_model: LLMModelConfig, prompt: str, settings: Dict) -> str:
        """Extreme: Two-pass deep analysis with synthesis."""
        # First pass: Deep analysis
        settings_breakdown = "\n".join([f"- {k}: {v}" for k, v in settings.items()])

        user_prompt = f"""CRITICAL: You MUST preserve EVERY word and detail from the base prompt below. Do NOT remove, replace, or summarize anything.

=== BASE PROMPT (PRESERVE 100% OF THIS CONTENT) ===
{prompt if prompt.strip() else "(general photography shot)"}
=== END BASE PROMPT ===

Photography settings to describe as VISUAL EFFECTS (NOT visible equipment):
{settings_breakdown}

YOUR TASK - PASS 1 (Deep Enhancement):
1. Copy the ENTIRE base prompt as your foundation - every single word
2. INSERT rich visual descriptions of how each setting affects the scene
3. Describe VISUAL RESULTS only - never mention cameras/lenses as visible objects
4. Add micro-details: how light interacts with skin, fabric texture under specific lighting, bokeh character

For each existing description in the base prompt:
- Keep it EXACTLY as written
- ADD details about how photography settings enhance that specific element
- Describe the interplay between settings (e.g., wide aperture + rim light = luminous edge separation)

The output MUST contain 100% of original content plus visual enhancements.

Output the enhanced prompt:"""

        enhanced = cls._call_llm(client, llm_model, DEEP_PHOTOGRAPHY_SYSTEM, user_prompt, "Pass 1: Deep analysis")
        enhanced = cls._clean_response(enhanced, prompt)

        pass1_words = len(enhanced.split())
        print(f"[PhotographyPrompts] Pass 1 complete: {pass1_words} words")

        # Second pass: Synthesis
        synthesis_prompt = f"""CRITICAL: The prompt below has been carefully enhanced. You MUST preserve EVERY detail - do NOT remove or summarize anything.

=== ENHANCED PROMPT (PRESERVE 100% OF THIS CONTENT) ===
{enhanced}
=== END ENHANCED PROMPT ===

YOUR TASK - PASS 2 (Final Polish):
1. Keep ALL existing content exactly as written
2. ADD only micro-level refinements:
   - How light creates subtle subsurface scattering on skin
   - Micro-texture details (individual fabric threads, pore-level skin detail)
   - Atmospheric micro-particles (dust motes in light beams)
   - Edge definition and focus transitions
3. Do NOT summarize, condense, or rewrite existing descriptions
4. Do NOT add any visible photography equipment to the scene

The output must be AT LEAST as long as the input, containing 100% of original content.

Output the final refined prompt:"""

        final = cls._call_llm(client, llm_model, DEEP_PHOTOGRAPHY_SYSTEM, synthesis_prompt, "Pass 2: Synthesis")
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
