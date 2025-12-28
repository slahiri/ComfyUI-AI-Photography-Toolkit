"""
SID Prompt Synthesis Node - Stage 2: Generate detailed image descriptions.

Supports multiple VLM providers:
- Local Qwen models (use hf_token for gated models)
- Google Gemini, OpenAI, Anthropic (use api_key)

Uses optimized prompts for Qwen VLM to generate comprehensive, detailed
image descriptions in natural flowing prose.

Outputs:
- metadata: JSON with tagger data + vlm_description + synthesis info
- prompt: Full detailed VLM description (verbose natural language)
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

SYSTEM_PROMPT = """You are an expert image captioner specializing in detailed, verbose descriptions for AI image generation. Write in flowing natural language with rich detail."""

DESCRIPTION_PROMPT = """Describe this image in extensive detail for AI image generation. Write a comprehensive, flowing description covering ALL of the following aspects in order:

SUBJECT: Who or what is the main focus? Describe count, approximate age, gender, ethnicity, body type, and any distinguishing features in detail.

FACE & EXPRESSION: Describe facial features precisely - face shape, eye color and shape, nose, lips, eyebrows. Note the expression, emotion conveyed, and any makeup details.

HAIR: Describe the exact hair color with nuance (e.g., "warm chestnut brown with golden highlights"), length, texture, style, and how it falls or is arranged.

CLOTHING & ACCESSORIES: Describe every visible garment in detail - type, color, material, fit, pattern, texture. Include all jewelry, accessories, shoes, bags, etc.

POSE & BODY LANGUAGE: Describe body position, hand placement, stance, gestures, gaze direction, and any movement or action.

ENVIRONMENT & SETTING: Is it indoor/outdoor? Describe the location, background elements, props, objects, atmosphere, weather if visible.

LIGHTING: Describe light direction, quality (soft/hard/dramatic), color temperature (warm/cool), shadows, highlights, and overall lighting mood.

COMPOSITION & FRAMING: What shot type (close-up, medium, full body, wide)? Camera angle, depth of field, what's in focus vs blurred.

STYLE & MOOD: Overall photographic style (editorial, candid, portrait, etc.), color palette, mood, and artistic quality.

Write as continuous, detailed prose - not bullet points or JSON. Be extremely thorough and descriptive. Describe everything you can see."""

COMPOSE_SYSTEM = """You are an expert prompt engineer for Z-Image AI image generation. You synthesize image analysis data into optimally structured prompts."""

COMPOSE_PROMPT_TEMPLATE = """Compose an optimal Z-Image prompt by synthesizing the following analysis data.

## VALIDATED TAGS (high confidence detections):
{canonical_tags}

## FLORENCE CAPTIONS (factual base):
{florence_data}

## DETAILED VLM DESCRIPTION:
{vlm_description}

## TECHNICAL DATA:
- Shot type: {shot_type}
- Lighting: {lighting}
- Style: {style}

---

## Z-IMAGE PROMPT STRUCTURE (follow this order):

1. **SHOT & SUBJECT**: Shot type, main subject (who/what), count, key attributes
2. **APPEARANCE**: Age, gender, ethnicity, body type, face, expression, hair (color with nuance, style)
3. **CLOTHING & ACCESSORIES**: Garments, colors, materials, fit, jewelry, accessories
4. **ENVIRONMENT**: Location, background elements, props, atmosphere
5. **LIGHTING**: Direction, quality (soft/hard), color temperature, shadows, highlights
6. **MOOD & STYLE**: Photographic style, color palette, overall mood
7. **TECHNICAL**: Camera angle, depth of field, focus, framing
8. **CONSTRAINTS**: Quality requirements (sharp, detailed, professional)

## RULES:
- Target length: 80-250 words (optimal for Z-Image)
- Write as flowing prose, NOT bullet points
- Be SPECIFIC with colors, materials, textures
- Prioritize VALIDATED TAGS over VLM description
- Do NOT contradict validated tags
- Do NOT include "no watermark" or negative instructions (Z-Image doesn't support negatives)
- Focus on 3-5 key visual concepts

Write the composed prompt now:"""

def analyze_tag_coverage(canonical: dict, description: str, verbose: bool = False) -> dict:
    """Analyze which canonical tags are covered in the description.

    Args:
        canonical: The canonical structure from metadata
        description: The text description to check against (VLM or Florence)
        verbose: Enable detailed logging

    Returns:
        dict with matched_tags, missing_tags, coverage statistics
    """
    if not canonical or not description:
        return {"error": "Missing canonical or description"}

    description_lower = description.lower()

    matched_tags = {}
    missing_tags = []

    # Categories to check (skip meta categories)
    check_categories = ["subject", "scene", "composition", "lighting", "style"]

    total_tags = 0
    matched_count = 0

    for category in check_categories:
        if category not in canonical:
            continue

        cat_tags = canonical[category]
        if not isinstance(cat_tags, dict):
            continue

        for tag, confidence in cat_tags.items():
            total_tags += 1

            # Normalize tag for matching
            tag_lower = tag.lower().replace("_", " ")
            tag_words = tag_lower.split()

            # Check for match - tag appears in description
            # Try exact match first, then word-by-word
            found = False
            match_context = ""

            if tag_lower in description_lower:
                found = True
                # Extract context around match
                idx = description_lower.find(tag_lower)
                start = max(0, idx - 20)
                end = min(len(description), idx + len(tag_lower) + 20)
                match_context = description[start:end].strip()
            else:
                # Try matching key words (for multi-word tags)
                if len(tag_words) >= 2:
                    # Check if most words appear
                    words_found = sum(1 for w in tag_words if len(w) > 2 and w in description_lower)
                    if words_found >= len(tag_words) * 0.7:
                        found = True
                        match_context = f"[partial: {words_found}/{len(tag_words)} words]"

            if found:
                matched_count += 1
                matched_tags[tag] = {
                    "confidence": confidence,
                    "category": category,
                    "context": match_context[:50] if match_context else ""
                }
            else:
                missing_tags.append({
                    "tag": tag,
                    "confidence": confidence,
                    "category": category
                })

    # Sort missing by confidence (highest first - most important to add)
    missing_tags.sort(key=lambda x: -x["confidence"])

    coverage = matched_count / total_tags if total_tags > 0 else 0

    result = {
        "total_tags": total_tags,
        "matched_count": matched_count,
        "missing_count": len(missing_tags),
        "coverage": round(coverage, 3),
        "matched_tags": matched_tags,
        "missing_tags": missing_tags[:20],  # Top 20 missing
    }

    if verbose:
        log("Synthesis", f"Tag coverage: {matched_count}/{total_tags} ({coverage:.1%})")
        log("Synthesis", f"Top missing: {[t['tag'] for t in missing_tags[:5]]}")

    return result


class SID_PromptSynthesis:
    """
    Stage 2: Generate detailed image descriptions using VLM.

    Uses Qwen VLM (or cloud APIs) with optimized prompts to generate
    comprehensive, verbose descriptions in natural flowing prose.

    Inputs:
    - image: Input image tensor
    - metadata: SID_METADATA from analysis node (optional, preserved in output)

    Outputs:
    - metadata: JSON with tagger data + vlm_description + synthesis info
    - prompt: Full detailed description (verbose natural language)
    """

    CATEGORY = "SID Nodes"
    RETURN_TYPES = ("SID_METADATA", "STRING")
    RETURN_NAMES = ("metadata", "prompt")
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
                "synthesis_enabled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable VLM synthesis. If disabled, passthrough metadata and use Florence description."
                }),
                "vlm_model": (VLM_MODELS, {
                    "default": "[Local] Qwen2.5-VL-3B Abliterated",
                    "tooltip": "VLM model: [Local] runs on GPU, others need api_key"
                }),
            },
            "optional": {
                "metadata": ("SID_METADATA", {"tooltip": "Metadata from SID_ImageAnalysis"}),
                "api_key": ("STRING", {"default": "", "tooltip": "API key for Gemini/OpenAI/Anthropic"}),
                "precision": (QWEN_PRECISION, {"default": "4-Bit", "tooltip": "Local model precision"}),
                "max_tokens": ("INT", {
                    "default": 1024,
                    "min": 128,
                    "max": 16384,
                    "step": 128,
                    "tooltip": "Maximum tokens for VLM output"
                }),
                "temperature": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider",
                    "tooltip": "VLM sampling temperature (lower = more deterministic)"
                }),
                "verbose": ("BOOLEAN", {"default": False, "tooltip": "Enable detailed logging with quality output"}),
                "release_vram": ("BOOLEAN", {"default": True, "tooltip": "Release local VLM after use"}),
                "hf_token": ("STRING", {"default": "", "tooltip": "HuggingFace token for gated local models"}),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffff,
                    "tooltip": "Seed for reproducibility"
                }),
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

    def _run_qwen(self, image: Image.Image, prompt: str, max_tokens: int, temperature: float, verbose: bool) -> str:
        from ..core.models.base import GenerationConfig, CaptionMode
        gen_config = GenerationConfig(max_tokens=max_tokens, temperature=temperature, do_sample=True)
        result = SID_PromptSynthesis._vlm_wrapper.generate(
            image=image, mode=CaptionMode.DETAILED, config=gen_config,
            custom_prompt=prompt, system_prompt=SYSTEM_PROMPT,
        )
        return result.strip() if result else ""

    # =========================================================================
    # Cloud APIs
    # =========================================================================

    def _run_gemini(self, image: Image.Image, prompt: str, model: str, api_key: str, max_tokens: int, temperature: float) -> str:
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
            generation_config=genai.types.GenerationConfig(max_output_tokens=max_tokens, temperature=temperature)
        )
        return response.text.strip() if response.text else ""

    def _run_openai(self, image: Image.Image, prompt: str, model: str, api_key: str, max_tokens: int, temperature: float) -> str:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai not installed. Run: pip install openai")
        if not api_key:
            raise ValueError("API key required for OpenAI")
        client = OpenAI(api_key=api_key)
        image_b64 = self._pil_to_base64(image)
        response = client.chat.completions.create(
            model=model, max_tokens=max_tokens, temperature=temperature,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                ]}
            ]
        )
        return response.choices[0].message.content.strip() if response.choices else ""

    def _run_anthropic(self, image: Image.Image, prompt: str, model: str, api_key: str, max_tokens: int, temperature: float) -> str:
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic not installed. Run: pip install anthropic")
        if not api_key:
            raise ValueError("API key required for Anthropic")
        client = anthropic.Anthropic(api_key=api_key)
        image_b64 = self._pil_to_base64(image)
        response = client.messages.create(
            model=model, max_tokens=max_tokens, temperature=temperature, system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_b64}},
                {"type": "text", "text": prompt}
            ]}]
        )
        return response.content[0].text.strip() if response.content else ""

    # =========================================================================
    # Main
    # =========================================================================

    def _run_vlm(self, image: Image.Image, prompt: str, max_tokens: int, temperature: float,
                 provider: str, model_id: str, api_key: str, verbose: bool) -> str:
        if provider == "local":
            return self._run_qwen(image, prompt, max_tokens, temperature, verbose)
        elif provider == "gemini":
            return self._run_gemini(image, prompt, model_id, api_key, max_tokens, temperature)
        elif provider == "openai":
            return self._run_openai(image, prompt, model_id, api_key, max_tokens, temperature)
        elif provider == "anthropic":
            return self._run_anthropic(image, prompt, model_id, api_key, max_tokens, temperature)
        raise ValueError(f"Unknown provider: {provider}")

    def synthesize(
        self,
        image: torch.Tensor,
        synthesis_enabled: bool = True,
        vlm_model: str = "[Local] Qwen2.5-VL-3B Abliterated",
        metadata: str = "",
        api_key: str = "",
        precision: str = "4-Bit",
        max_tokens: int = 1024,
        temperature: float = 0.3,
        verbose: bool = False,
        release_vram: bool = True,
        hf_token: str = "",
        seed: int = 0,
    ) -> tuple[str, str]:
        """
        Extract detailed description from image using VLM.

        If synthesis_enabled=False, passes through metadata and uses Florence description.

        Returns:
            tuple[str, str]: (metadata_json, prompt_text)
        """
        # Clear VRAM from previous nodes (Analysis loads many models)
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        cleanup_memory(aggressive=True)
        log("Synthesis", "VRAM cleared before synthesis")

        # Parse input metadata (from SID_ImageAnalysis)
        parsed_metadata = {}
        if metadata and metadata.strip():
            try:
                parsed_metadata = json.loads(metadata)
            except json.JSONDecodeError as e:
                log_error("Synthesis", f"Error parsing metadata: {e}")

        # === PASSTHROUGH MODE ===
        if not synthesis_enabled:
            log("Synthesis", "Synthesis disabled - passthrough mode")

            # Use Florence description as prompt
            prompt_output = parsed_metadata.get("florence_description", "")
            if not prompt_output:
                prompt_output = parsed_metadata.get("florence_caption", "No description available")

            # Run coverage analysis against Florence
            canonical = parsed_metadata.get("canonical", {})
            if canonical and prompt_output:
                coverage = analyze_tag_coverage(canonical, prompt_output, verbose)
                parsed_metadata["tag_coverage"] = coverage
                parsed_metadata["tag_coverage"]["source"] = "florence"

                # Print coverage statistics
                print("=" * 60)
                print("Tag Coverage Analysis (Florence)")
                print("=" * 60)
                print(f"{'Total Tags:':<20} {coverage['total_tags']}")
                print(f"{'Matched:':<20} {coverage['matched_count']}")
                print(f"{'Missing:':<20} {coverage['missing_count']}")
                print(f"{'Coverage:':<20} {coverage['coverage']:.1%}")
                print("-" * 60)
                if coverage.get('missing_tags'):
                    print("Top Missing Tags:")
                    for t in coverage['missing_tags'][:10]:
                        print(f"  [{t['category']}] {t['tag']} ({t['confidence']:.2f})")
                print("=" * 60)

            parsed_metadata["synthesis"] = {"enabled": False, "source": "florence"}
            output_metadata = json.dumps(parsed_metadata, indent=2, ensure_ascii=False)

            # Final cleanup (passthrough mode)
            if release_vram:
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                cleanup_memory(aggressive=True)
                log("Synthesis", "VRAM cleared post-execution (passthrough)")

            return (output_metadata, prompt_output)

        # === SYNTHESIS MODE ===
        total_start = log_start("Synthesis", f"Starting with {vlm_model}")
        pil_image = self._tensor_to_pil(image)

        # Set random seed for reproducibility (numpy requires seed < 2^32)
        import random
        seed_32bit = seed % (2**32)
        random.seed(seed_32bit)
        np.random.seed(seed_32bit)
        torch.manual_seed(seed_32bit)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_32bit)

        # Log received canonical structure info
        if verbose and "canonical" in parsed_metadata:
            canonical = parsed_metadata["canonical"]
            log("Synthesis", f"Received canonical categories: {list(canonical.keys())}")
            for cat, tags in canonical.items():
                if tags and cat != "below_threshold":
                    tag_count = len(tags) if isinstance(tags, dict) else 0
                    log("Synthesis", f"  {cat}: {tag_count} tags")

        provider = self._get_provider(vlm_model)
        model_id = self._get_model_id(vlm_model)

        # Progress: load (if local) + 1 pass + coverage
        total_steps = 3 if provider == "local" else 2
        pbar = comfy.utils.ProgressBar(total_steps)

        log("Synthesis", f"Image: {pil_image.width}x{pil_image.height}, Provider: {provider}")

        # Load local model if needed
        if provider == "local":
            self._load_qwen(model_id, precision, verbose, hf_token)
            pbar.update(1)

        # Single pass: Get detailed description from VLM
        desc_start = log_start("Synthesis", "Extracting detailed description")
        raw_description = self._run_vlm(pil_image, DESCRIPTION_PROMPT, max_tokens, temperature, provider, model_id, api_key, verbose)
        description = ""

        if raw_description:
            description = raw_description.strip()
            log_end("Synthesis", "VLM response received", desc_start, f"{len(description)} chars")
            parsed_metadata["vlm_description"] = description
        else:
            log_end("Synthesis", "No VLM response", desc_start)

        pbar.update(1)

        # Release VRAM
        if provider == "local" and release_vram:
            self._unload_qwen()

        # Run coverage analysis
        canonical = parsed_metadata.get("canonical", {})
        if canonical and description:
            coverage_start = log_start("Synthesis", "Analyzing tag coverage")
            coverage = analyze_tag_coverage(canonical, description, verbose)
            coverage["source"] = "vlm"
            parsed_metadata["tag_coverage"] = coverage
            log_end("Synthesis", "Coverage analysis", coverage_start, f"{coverage['coverage']:.1%}")

            # Print coverage statistics
            print("=" * 60)
            print("Tag Coverage Analysis (VLM)")
            print("=" * 60)
            print(f"{'Total Tags:':<20} {coverage['total_tags']}")
            print(f"{'Matched:':<20} {coverage['matched_count']}")
            print(f"{'Missing:':<20} {coverage['missing_count']}")
            print(f"{'Coverage:':<20} {coverage['coverage']:.1%}")
            print("-" * 60)
            if coverage.get('missing_tags'):
                print("Top Missing Tags:")
                for t in coverage['missing_tags'][:10]:
                    print(f"  [{t['category']}] {t['tag']} ({t['confidence']:.2f})")
            print("=" * 60)

        pbar.update(1)

        # Output is the full VLM description
        prompt_output = description if description else "No description generated"

        if verbose:
            if description:
                log("Synthesis", f"VLM description generated: {len(description)} chars")
                log("Synthesis", f"Preview: {description[:300]}...")
            else:
                log("Synthesis", "WARNING: No VLM description generated")

        # Add synthesis info to metadata
        parsed_metadata["synthesis"] = {
            "enabled": True,
            "model": vlm_model,
            "provider": provider,
            "seed": seed_32bit,
            "description_length": len(description) if description else 0,
        }

        output_metadata = json.dumps(parsed_metadata, indent=2, ensure_ascii=False)

        del pil_image

        # Final cleanup
        if release_vram:
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            cleanup_memory(aggressive=True)
            log("Synthesis", "VRAM cleared post-execution")
        else:
            cleanup_memory()

        log_end("Synthesis", "Complete", total_start)

        if verbose:
            log("Synthesis", f"Output preview:\n{prompt_output[:500]}...")

        return (output_metadata, prompt_output)
