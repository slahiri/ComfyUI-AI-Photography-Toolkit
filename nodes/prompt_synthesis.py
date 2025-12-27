"""
SID Prompt Synthesis Node - Stage 2: Extract detailed image description using VLM.

Supports multiple VLM providers:
- Local Qwen models (use hf_token for gated models)
- Google Gemini, OpenAI, Anthropic (use api_key)

Single pass: Generates extremely detailed caption.

Outputs:
- prompt: Detailed description
- metadata: Updated with vlm_description
"""

import json
import torch
from PIL import Image
import numpy as np
import base64
import io

import comfy.utils
from ..core.platform import cleanup_memory
from ..core.log import log, log_start, log_end, log_error


# =============================================================================
# Model Configuration - Single Dropdown
# =============================================================================

VLM_MODELS = [
    # Local Qwen - Abliterated (uncensored)
    "[Local] Qwen3-VL-2B Abliterated",
    "[Local] Qwen3-VL-4B Abliterated",
    "[Local] Qwen3-VL-8B Abliterated",
    "[Local] Qwen2.5-VL-3B Abliterated",
    "[Local] Qwen2.5-VL-7B Abliterated",
    "[Local] Qwen2-VL-2B Abliterated",
    # Local Qwen - Instruct
    "[Local] Qwen3-VL-2B Instruct",
    "[Local] Qwen3-VL-4B Instruct",
    "[Local] Qwen3-VL-8B Instruct",
    "[Local] Qwen2.5-VL-3B Instruct",
    "[Local] Qwen2.5-VL-7B Instruct",
    "[Local] Qwen2-VL-2B Instruct",
    "[Local] Qwen2-VL-7B Instruct",
    # Google Gemini
    "[Gemini] gemini-2.0-flash-exp",
    "[Gemini] gemini-1.5-flash",
    "[Gemini] gemini-1.5-pro",
    # OpenAI
    "[OpenAI] gpt-4o",
    "[OpenAI] gpt-4o-mini",
    "[OpenAI] gpt-4-turbo",
    # Anthropic
    "[Anthropic] claude-sonnet-4-20250514",
    "[Anthropic] claude-3-5-sonnet-20241022",
    "[Anthropic] claude-3-5-haiku-20241022",
]

QWEN_MODEL_MAP = {
    "[Local] Qwen3-VL-2B Abliterated": "qwen3_2b_abliterated",
    "[Local] Qwen3-VL-4B Abliterated": "qwen3_4b_abliterated",
    "[Local] Qwen3-VL-8B Abliterated": "qwen3_8b_abliterated",
    "[Local] Qwen2.5-VL-3B Abliterated": "qwen25_3b_abliterated",
    "[Local] Qwen2.5-VL-7B Abliterated": "qwen25_7b_abliterated",
    "[Local] Qwen2-VL-2B Abliterated": "qwen2_2b_abliterated",
    "[Local] Qwen3-VL-2B Instruct": "qwen3_2b",
    "[Local] Qwen3-VL-4B Instruct": "qwen3_4b",
    "[Local] Qwen3-VL-8B Instruct": "qwen3_8b",
    "[Local] Qwen2.5-VL-3B Instruct": "qwen25_3b",
    "[Local] Qwen2.5-VL-7B Instruct": "qwen25_7b",
    "[Local] Qwen2-VL-2B Instruct": "qwen2_2b_instruct",
    "[Local] Qwen2-VL-7B Instruct": "qwen2_7b_instruct",
}

QWEN_PRECISION = ["4-Bit", "FP8", "FP16", "Auto"]


# =============================================================================
# Prompts for VLM passes
# =============================================================================

SYSTEM_PROMPT = """You are an expert image analyst with exceptional attention to detail.
Describe exactly what you see with maximum precision and completeness.
Be specific about every visible element: colors, shapes, textures, materials, positions, lighting, composition.
Do not guess, assume, or hallucinate - only describe what is clearly visible in the image."""

DESCRIPTION_PROMPT = """Provide an extremely detailed, high-resolution description of this image.
Capture EVERY visible detail as if creating a complete visual record.

Describe with maximum detail:

MAIN SUBJECT(S):
- What is the primary focus? Describe completely.
- If person: exact appearance, age range, gender, ethnicity, body type, skin tone, facial features
- If animal: species, breed, coloring, markings, size, posture
- If object: type, material, color, condition, size, brand if visible

FACE & EXPRESSION (if applicable):
- Eye color, shape, gaze direction
- Facial structure, distinctive features
- Expression, emotion conveyed
- Makeup, facial hair, glasses

HAIR/FUR (if applicable):
- Color (be specific: not just "brown" but "chestnut brown with golden highlights")
- Length, texture, style
- How it's arranged or styled

CLOTHING & ACCESSORIES (if applicable):
- Each garment: type, color, material, fit, condition, style
- Neckline, sleeves, hemline details
- Patterns, prints, textures
- Every accessory: jewelry, watches, bags, hats, scarves
- Materials and colors of accessories

POSE & BODY LANGUAGE:
- Exact body position and orientation
- Hand positions and gestures
- Stance, posture, weight distribution
- Movement or stillness

ENVIRONMENT & SETTING:
- Indoor/outdoor
- Specific location type
- Visible objects in scene
- Foreground, middle ground, background elements

LIGHTING:
- Light source direction(s)
- Quality: hard, soft, diffused, dappled
- Color temperature: warm, cool, neutral
- Shadows: direction, softness, depth
- Highlights and reflections

COMPOSITION & FRAMING:
- Shot type: close-up, medium, full body, wide
- Camera angle: eye level, high, low, dutch
- What's in focus, what's blurred
- Depth of field

COLORS & PALETTE:
- Dominant colors
- Color relationships and contrasts
- Saturation and tone

TEXTURES & MATERIALS:
- Surface qualities of all visible elements
- Material types: fabric, metal, wood, skin, etc.

MOOD & ATMOSPHERE:
- Overall feeling conveyed
- Visual atmosphere

TECHNICAL QUALITIES:
- Image sharpness and clarity
- Any visible grain, noise, or effects
- Photographic or artistic style

Be exhaustive. Every detail matters. Write in flowing, descriptive paragraphs."""

class SID_PromptSynthesis:
    """
    Stage 2: Extract detailed image description using VLM.

    Single pass: Very detailed, high-resolution description.
    """

    CATEGORY = "SID Nodes"
    RETURN_TYPES = ("IMAGE", "SID_METADATA", "STRING")
    RETURN_NAMES = ("image", "metadata", "prompt")
    FUNCTION = "synthesize"
    OUTPUT_NODE = False

    COLOR = "#324B53"
    BGCOLOR = "#283d44"

    _vlm_wrapper = None
    _current_model = None
    _current_precision = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input image"}),
                "vlm_model": (VLM_MODELS, {
                    "default": "[Local] Qwen2.5-VL-3B Abliterated",
                    "tooltip": "VLM model: [Local] runs on GPU, others need api_key"
                }),
            },
            "optional": {
                "metadata": ("SID_METADATA", {"tooltip": "Metadata from SID_ImageAnalysis"}),
                "api_key": ("STRING", {"default": "", "tooltip": "API key for Gemini/OpenAI/Anthropic"}),
                "precision": (QWEN_PRECISION, {"default": "4-Bit", "tooltip": "Local model precision"}),
                "verbose": ("BOOLEAN", {"default": False, "tooltip": "Enable detailed logging with quality output"}),
                "release_vram": ("BOOLEAN", {"default": True, "tooltip": "Release local VLM after use"}),
                "hf_token": ("STRING", {"default": "", "tooltip": "HuggingFace token for gated local models"}),
            },
        }

    def _get_provider(self, model_name: str) -> str:
        if model_name.startswith("[Local]"):
            return "local"
        elif model_name.startswith("[Gemini]"):
            return "gemini"
        elif model_name.startswith("[OpenAI]"):
            return "openai"
        elif model_name.startswith("[Anthropic]"):
            return "anthropic"
        return "unknown"

    def _get_model_id(self, model_name: str) -> str:
        if model_name.startswith("[Gemini]"):
            return model_name.replace("[Gemini] ", "")
        elif model_name.startswith("[OpenAI]"):
            return model_name.replace("[OpenAI] ", "")
        elif model_name.startswith("[Anthropic]"):
            return model_name.replace("[Anthropic] ", "")
        return QWEN_MODEL_MAP.get(model_name, "qwen25_3b_abliterated")

    def _tensor_to_pil(self, image_tensor: torch.Tensor) -> Image.Image:
        if len(image_tensor.shape) == 4:
            image_tensor = image_tensor[0]
        img_np = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(img_np, mode="RGB")

    def _pil_to_base64(self, image: Image.Image, max_size: int = 1024) -> str:
        if image.width > max_size or image.height > max_size:
            ratio = min(max_size / image.width, max_size / image.height)
            new_size = (int(image.width * ratio), int(image.height * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        if image.mode != "RGB":
            image = image.convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        buffer.seek(0)
        return base64.standard_b64encode(buffer.read()).decode("utf-8")

    # =========================================================================
    # Local Qwen
    # =========================================================================

    def _load_qwen(self, model_key: str, precision: str, verbose: bool, hf_token: str = ""):
        if (SID_PromptSynthesis._vlm_wrapper is not None and
            SID_PromptSynthesis._current_model == model_key and
            SID_PromptSynthesis._current_precision == precision):
            if verbose:
                log("Synthesis", "Reusing loaded Qwen")
            return

        if SID_PromptSynthesis._vlm_wrapper is not None:
            self._unload_qwen()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        # Set HF token for gated model access
        if hf_token:
            from ..core.download import set_hf_token
            set_hf_token(hf_token)

        from ..core.models import ModelFactory
        start = log_start("Synthesis", f"Loading {model_key} ({precision})")
        SID_PromptSynthesis._vlm_wrapper = ModelFactory.get(model_key, precision=precision)
        SID_PromptSynthesis._vlm_wrapper.load()
        SID_PromptSynthesis._current_model = model_key
        SID_PromptSynthesis._current_precision = precision
        log_end("Synthesis", "Model loaded", start)

    def _unload_qwen(self):
        import gc
        if SID_PromptSynthesis._vlm_wrapper is not None:
            start = log_start("Synthesis", "Releasing model")
            from ..core.models import ModelFactory
            if SID_PromptSynthesis._current_model:
                ModelFactory.release(SID_PromptSynthesis._current_model)
            SID_PromptSynthesis._vlm_wrapper = None
            SID_PromptSynthesis._current_model = None
            SID_PromptSynthesis._current_precision = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            cleanup_memory(aggressive=True)
            log_end("Synthesis", "Model released", start)

    def _run_qwen(self, image: Image.Image, prompt: str, max_tokens: int, verbose: bool) -> str:
        from ..core.models.base import GenerationConfig, CaptionMode
        gen_config = GenerationConfig(max_tokens=max_tokens, temperature=0.3, do_sample=True)
        result = SID_PromptSynthesis._vlm_wrapper.generate(
            image=image, mode=CaptionMode.DETAILED, config=gen_config,
            custom_prompt=prompt, system_prompt=SYSTEM_PROMPT,
        )
        return result.strip() if result else ""

    # =========================================================================
    # Cloud APIs
    # =========================================================================

    def _run_gemini(self, image: Image.Image, prompt: str, model: str, api_key: str, max_tokens: int) -> str:
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("google-generativeai not installed. Run: pip install google-generativeai")
        if not api_key:
            raise ValueError("API key required for Gemini")
        genai.configure(api_key=api_key)
        model_instance = genai.GenerativeModel(model)
        response = model_instance.generate_content(
            [f"{SYSTEM_PROMPT}\n\n{prompt}", image],
            generation_config=genai.types.GenerationConfig(max_output_tokens=max_tokens, temperature=0.3)
        )
        return response.text.strip() if response.text else ""

    def _run_openai(self, image: Image.Image, prompt: str, model: str, api_key: str, max_tokens: int) -> str:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai not installed. Run: pip install openai")
        if not api_key:
            raise ValueError("API key required for OpenAI")
        client = OpenAI(api_key=api_key)
        image_b64 = self._pil_to_base64(image)
        response = client.chat.completions.create(
            model=model, max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                ]}
            ]
        )
        return response.choices[0].message.content.strip() if response.choices else ""

    def _run_anthropic(self, image: Image.Image, prompt: str, model: str, api_key: str, max_tokens: int) -> str:
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic not installed. Run: pip install anthropic")
        if not api_key:
            raise ValueError("API key required for Anthropic")
        client = anthropic.Anthropic(api_key=api_key)
        image_b64 = self._pil_to_base64(image)
        response = client.messages.create(
            model=model, max_tokens=max_tokens, system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_b64}},
                {"type": "text", "text": prompt}
            ]}]
        )
        return response.content[0].text.strip() if response.content else ""

    # =========================================================================
    # Main
    # =========================================================================

    def _run_vlm(self, image: Image.Image, prompt: str, max_tokens: int,
                 provider: str, model_id: str, api_key: str, verbose: bool) -> str:
        if provider == "local":
            return self._run_qwen(image, prompt, max_tokens, verbose)
        elif provider == "gemini":
            return self._run_gemini(image, prompt, model_id, api_key, max_tokens)
        elif provider == "openai":
            return self._run_openai(image, prompt, model_id, api_key, max_tokens)
        elif provider == "anthropic":
            return self._run_anthropic(image, prompt, model_id, api_key, max_tokens)
        raise ValueError(f"Unknown provider: {provider}")

    def synthesize(
        self,
        image: torch.Tensor,
        vlm_model: str = "[Local] Qwen2.5-VL-3B Abliterated",
        metadata: str = "",
        api_key: str = "",
        precision: str = "4-Bit",
        release_vram: bool = True,
        verbose: bool = False,
        hf_token: str = "",
    ) -> tuple[torch.Tensor, str, str]:
        """
        Extract detailed description from image.

        Single pass: Very detailed, high-resolution description.
        """
        total_start = log_start("Synthesis", f"Starting with {vlm_model}")
        pil_image = self._tensor_to_pil(image)

        # Parse input metadata
        parsed_metadata = {}
        if metadata and metadata.strip():
            try:
                parsed_metadata = json.loads(metadata)
            except json.JSONDecodeError as e:
                log_error("Synthesis", f"Error parsing metadata: {e}")

        provider = self._get_provider(vlm_model)
        model_id = self._get_model_id(vlm_model)

        # Progress: load (if local) + 1 pass
        total_steps = 2 if provider == "local" else 1
        pbar = comfy.utils.ProgressBar(total_steps)

        log("Synthesis", f"Image: {pil_image.width}x{pil_image.height}, Provider: {provider}")

        # Load local model if needed
        if provider == "local":
            self._load_qwen(model_id, precision, verbose, hf_token)
            pbar.update(1)

        # Single pass: Detailed description
        desc_start = log_start("Synthesis", "Extracting detailed description")
        description = self._run_vlm(pil_image, DESCRIPTION_PROMPT, 4096, provider, model_id, api_key, verbose)
        if description:
            parsed_metadata["vlm_description"] = description
            log_end("Synthesis", "Description extracted", desc_start, f"{len(description)} chars")
        pbar.update(1)

        # Release VRAM
        if provider == "local" and release_vram:
            self._unload_qwen()

        # Output is just the description
        prompt_output = description if description else "No description generated"

        # Add synthesis info
        parsed_metadata["synthesis"] = {"model": vlm_model, "provider": provider}

        output_metadata = json.dumps(parsed_metadata, indent=2, ensure_ascii=False)

        del pil_image
        cleanup_memory()

        log_end("Synthesis", "Complete", total_start)

        if verbose:
            log("Synthesis", f"Output preview:\n{prompt_output[:500]}...")

        return (image, output_metadata, prompt_output)
