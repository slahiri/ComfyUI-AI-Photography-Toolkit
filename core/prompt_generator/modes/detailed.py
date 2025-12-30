"""
Detailed mode - Multi-pass LLM prompt generation with full enhancements.

Executes taggers, then uses 3 LLM passes to build comprehensive
descriptions with tag injection, scene-specific prompts, and detail
enhancements for each component group.

Works with any LLM (local or API). Uses extended thinking when
available (Anthropic API) for improved quality.
"""

import base64
import io
import time
from typing import Any, Dict, Optional, List, Tuple

import numpy as np
from PIL import Image

from .base import BaseMode
from .standard import StandardMode
from ..types import GeneratorResult, TaggerResults, Decisions
from ..decisions import get_engine
from ..templates import get_loader, get_selector, get_builder
from ..templates.tag_injector import get_unmapped_tags, format_tags_for_prompt, get_debug_info, get_female_anatomy_tags
from ..templates.scene_prompts import format_scene_prompt, get_scene_group
from ..templates.detail_prompts import (
    get_animal_prompt,
    get_clothing_prompt,
    get_tag_enhancement,
    ANIMAL_PROMPTS,
    CLOTHING_PROMPTS,
)
from ..taggers import get_runner


class DetailedMode(BaseMode):
    """
    Detailed mode implementation with full enhancements.

    Uses 3 LLM passes to build comprehensive descriptions:
    - Pass 1: Subject + Appearance (with tag-driven enhancements)
    - Pass 2: Clothing + Pose + Expression (with clothing-specific prompts)
    - Pass 3: Environment + Lighting + Composition (with scene-specific prompts)

    Includes all Standard mode capabilities:
    - Tag injection in each pass
    - Scene-specific environment prompts
    - Animal/clothing/tag-driven detail enhancements
    - Inclusion analysis on final output

    Falls back to Standard mode if:
    - Any error occurs during execution
    """

    @property
    def name(self) -> str:
        return "detailed"

    @property
    def requires_llm(self) -> bool:
        return True

    @property
    def requires_taggers(self) -> bool:
        return True

    @property
    def requires_reasoning(self) -> bool:
        return False  # Reasoning is optional, not required

    def execute(
        self,
        image: Any = None,
        llm_model: Any = None,
        tagger_results: Optional[TaggerResults] = None,
        decisions: Optional[Decisions] = None,
        prompt_config: Optional[Dict[str, Any]] = None,
        tag_threshold: float = 0.5,
        max_injected_tags: int = 30,
        **kwargs
    ) -> GeneratorResult:
        """
        Execute Detailed mode with full enhancements.

        Uses 3 LLM passes to build comprehensive descriptions.
        Works with any LLM (local or API). Uses extended thinking
        when available (Anthropic API) for improved quality.

        Args:
            image: Image tensor from ComfyUI
            llm_model: LLMModelConfig with provider settings
            tagger_results: Pre-computed tagger results
            decisions: Pre-computed decisions
            prompt_config: Optional custom system/user prompts
            **kwargs: Additional parameters (tag_threshold, etc.)

        Returns:
            GeneratorResult with multi-pass generated description
        """
        if image is None:
            return GeneratorResult(
                prompt="[No image provided]",
                metadata={"mode": "detailed", "error": "No image provided"}
            )

        if llm_model is None:
            return GeneratorResult(
                prompt="[No LLM model provided]",
                metadata={"mode": "detailed", "error": "No LLM model provided"}
            )

        start_time = time.time()
        timing = {}
        pass_outputs = {}
        pass_details = {}

        try:
            # Step 1: Run taggers if not provided
            tagger_start = time.time()
            if tagger_results is None:
                runner = get_runner()
                tagger_results = runner.run_standard(image)
            timing["taggers_ms"] = int((time.time() - tagger_start) * 1000)

            # Step 2: Make decisions if not provided
            decision_start = time.time()
            if decisions is None:
                engine = get_engine()
                decisions = engine.make_decisions(tagger_results)
            timing["decisions_ms"] = int((time.time() - decision_start) * 1000)

            # Step 3: Get tag debug info for metadata
            tag_debug_info = get_debug_info(tagger_results, tag_threshold)

            # Step 4: Get scene info
            scene_type = decisions.scene_type
            scene_group = get_scene_group(scene_type) if scene_type else None
            scene_prompt = format_scene_prompt(scene_type) if scene_type else None

            # Step 5: Get detected tags for injection
            detected_tags = get_unmapped_tags(
                tagger_results,
                threshold=tag_threshold,
                max_tags=max_injected_tags,
            )

            # Inject female anatomy tags for women subjects (NudeNet often misses these)
            subject_type = decisions.subject_type.value if hasattr(decisions.subject_type, 'value') else str(decisions.subject_type)
            anatomy_tags = get_female_anatomy_tags(subject_type, detected_tags)
            if anatomy_tags:
                # Merge anatomy tags with detected tags (anatomy tags go first for emphasis)
                detected_tags = anatomy_tags + list(detected_tags)

            tags_str = format_tags_for_prompt(detected_tags, show_confidence=True) if detected_tags else ""

            # Step 6: Convert image to base64
            base64_image = self._image_to_base64(image, llm_model)

            # Step 7: Build and execute passes based on subject type
            llm_start = time.time()
            passes = self._get_passes(decisions, tagger_results, tags_str, scene_prompt, tag_threshold)

            for pass_config in passes:
                pass_name = pass_config["name"]
                pass_prompt = pass_config["prompt"]

                response = self._call_llm(
                    llm_model=llm_model,
                    base64_image=base64_image,
                    prompt=pass_prompt,
                    prompt_config=prompt_config,
                )

                cleaned = self._clean_response(response)
                pass_outputs[pass_name] = cleaned
                pass_details[pass_name] = {
                    "sections": pass_config.get("sections", []),
                    "enhancements": pass_config.get("enhancements", []),
                    "tags_injected": pass_config.get("tags_injected", False),
                }

            timing["llm_ms"] = int((time.time() - llm_start) * 1000)

            # Step 8: Assemble final prompt
            prompt = self._assemble_passes(pass_outputs)

            # Step 9: Analyze tag inclusion in output
            inclusion_start = time.time()
            from ..validation import analyze_inclusion
            injected_tags = [
                (t["tag"], t["confidence"])
                for t in tag_debug_info.get("injected_tags", [])
            ]
            inclusion_analysis = analyze_inclusion(injected_tags, prompt)
            timing["inclusion_analysis_ms"] = int((time.time() - inclusion_start) * 1000)

            # Print inclusion summary
            self._print_inclusion_summary(inclusion_analysis)

            total_time = int((time.time() - start_time) * 1000)

            return GeneratorResult(
                prompt=prompt,
                metadata={
                    "mode": "detailed",
                    "source": "llm_multi_pass_enhanced",
                    "llm_provider": llm_model.provider,
                    "llm_model": llm_model.model,
                    "processing": {
                        "taggers_executed": True,
                        "florence_executed": False,
                        "llm_executed": True,
                        "llm_passes": len(passes),
                    },
                    "decisions": decisions.to_dict(),
                    "passes": pass_details,
                    "pass_outputs": pass_outputs,
                    "scene": {
                        "detected": scene_type,
                        "group": scene_group,
                        "prompt_used": scene_prompt[:100] + "..." if scene_prompt and len(scene_prompt) > 100 else scene_prompt,
                    } if scene_type else None,
                    "tag_injection": tag_debug_info,
                    "inclusion_analysis": inclusion_analysis,
                    "taggers": tagger_results.to_dict(),
                    "timing": {
                        **timing,
                        "total_ms": total_time,
                    }
                }
            )

        except Exception as e:
            error_msg = str(e)
            print(f"[Detailed] Error: {error_msg}")
            import traceback
            traceback.print_exc()

            # Fall back to Standard mode on error
            print("[Detailed] Falling back to Standard mode")
            return StandardMode().execute(
                image, llm_model, tagger_results, decisions, prompt_config, **kwargs
            )

    def _get_passes(
        self,
        decisions: Decisions,
        tagger_results: TaggerResults,
        tags_str: str,
        scene_prompt: Optional[str],
        tag_threshold: float,
    ) -> List[Dict[str, Any]]:
        """
        Build pass configurations based on decisions and detected content.

        Returns list of pass configs with enhanced prompts.
        """
        subject_type = decisions.subject_type.value
        detected_animal = decisions.wildlife_type  # wildlife_type holds animal name
        passes = []

        # =====================================================================
        # PASS 1: Subject + Appearance
        # =====================================================================
        pass1_sections = ["subject", "appearance", "hair", "face", "eyes", "skin"]
        pass1_enhancements = []

        if subject_type in ["person", "woman", "man", "couple", "group"]:
            pass1_prompt = """Analyze the subject's identity and physical appearance in this image.

Describe in detail:
1. Subject identification (gender, apparent age, ethnicity impression)
2. Hair (style, color, length, texture, how it frames the face)
3. Face shape and distinctive features
4. Eyes (color, shape, expression, what they convey)
5. Skin tone and any notable features"""

            # Add tag-driven enhancements for hair/eyes/face
            hair_enhancements = self._get_tag_enhancements_for_section("hair", tagger_results, tag_threshold)
            eyes_enhancements = self._get_tag_enhancements_for_section("eyes", tagger_results, tag_threshold)
            face_enhancements = self._get_tag_enhancements_for_section("face", tagger_results, tag_threshold)

            if hair_enhancements:
                pass1_prompt += f"\n\nFor hair details: {hair_enhancements}"
                pass1_enhancements.append({"type": "tag", "section": "hair"})
            if eyes_enhancements:
                pass1_prompt += f"\n\nFor eye details: {eyes_enhancements}"
                pass1_enhancements.append({"type": "tag", "section": "eyes"})
            if face_enhancements:
                pass1_prompt += f"\n\nFor facial details: {face_enhancements}"
                pass1_enhancements.append({"type": "tag", "section": "face"})

        elif subject_type == "animal" and detected_animal:
            # Animal-specific prompts
            animal_prompts = get_animal_prompt(detected_animal)
            if animal_prompts:
                pass1_prompt = f"""Analyze this {detected_animal} in the image.

{animal_prompts.get('species', 'Identify the species and any breed characteristics.')}

{animal_prompts.get('appearance', 'Describe the physical appearance in detail.')}

{animal_prompts.get('expression', 'Describe the expression and demeanor.')}"""
                pass1_enhancements.append({"type": "animal", "animal": detected_animal})
            else:
                pass1_prompt = """Analyze the animal subject in this image.

Describe:
1. Species and any breed/variety identification
2. Physical characteristics (size, coloring, markings)
3. Fur/feathers/scales texture and condition
4. Expression and body language"""
        else:
            # Object/scene subject
            pass1_prompt = """Analyze the main subject in this image.

Describe:
1. What is the primary subject/focus
2. Key characteristics and details
3. Condition, materials, textures
4. Notable features or unique aspects"""

        # Add detected tags to pass 1
        if tags_str:
            pass1_prompt += f"""

Detected visual elements to incorporate: {tags_str}
Include these naturally in your description where relevant."""

        pass1_prompt += """

Write as flowing descriptive prose suitable for AI image generation.
Be specific and detailed. Only describe what is actually visible."""

        passes.append({
            "name": "subject_appearance",
            "sections": pass1_sections,
            "prompt": pass1_prompt,
            "enhancements": pass1_enhancements,
            "tags_injected": bool(tags_str),
        })

        # =====================================================================
        # PASS 2: Body + Clothing + Pose
        # =====================================================================
        pass2_sections = ["body", "clothing", "pose", "expression", "accessories"]
        pass2_enhancements = []

        if subject_type in ["person", "woman", "man", "couple", "group"]:
            pass2_prompt = """Analyze the clothing, body, and pose in this image.

Based on what's visible, describe:
1. Body type and build (if visible)
2. All clothing items in detail (style, color, material, fit)
3. Pose and body position
4. Expression and gesture
5. Accessories (jewelry, bags, glasses, etc.)"""

            # Add clothing-specific enhancements
            clothing_enhancements = self._get_clothing_enhancements(tagger_results, tag_threshold)
            if clothing_enhancements:
                pass2_prompt += f"\n\nClothing details to focus on:\n{clothing_enhancements}"
                pass2_enhancements.append({"type": "clothing"})

            # Add tag-driven enhancements for pose/expression
            pose_enhancements = self._get_tag_enhancements_for_section("pose", tagger_results, tag_threshold)
            expression_enhancements = self._get_tag_enhancements_for_section("expression", tagger_results, tag_threshold)

            if pose_enhancements:
                pass2_prompt += f"\n\nFor pose: {pose_enhancements}"
                pass2_enhancements.append({"type": "tag", "section": "pose"})
            if expression_enhancements:
                pass2_prompt += f"\n\nFor expression: {expression_enhancements}"
                pass2_enhancements.append({"type": "tag", "section": "expression"})

        elif subject_type == "animal":
            pass2_prompt = """Analyze the animal's body and pose in this image.

Describe:
1. Body structure and proportions
2. Pose and position
3. Movement or stillness
4. Any accessories (collar, harness, etc.)"""
        else:
            pass2_prompt = """Analyze secondary elements and arrangement in this image.

Describe:
1. Arrangement and composition of elements
2. Supporting objects or items
3. Spatial relationships
4. Any text or labels visible"""

        # Add detected tags to pass 2
        if tags_str:
            pass2_prompt += f"""

Detected elements: {tags_str}
Reference these where applicable."""

        pass2_prompt += """

Write as flowing descriptive prose. Only describe what is actually visible."""

        passes.append({
            "name": "body_clothing_pose",
            "sections": pass2_sections,
            "prompt": pass2_prompt,
            "enhancements": pass2_enhancements,
            "tags_injected": bool(tags_str),
        })

        # =====================================================================
        # PASS 3: Environment + Lighting + Composition
        # =====================================================================
        pass3_sections = ["environment", "lighting", "composition", "atmosphere"]
        pass3_enhancements = []

        # Use scene-specific prompt if available
        if scene_prompt:
            env_instruction = scene_prompt
            pass3_enhancements.append({"type": "scene", "scene": decisions.scene_type})
        else:
            env_instruction = """Describe the setting and background: location type,
notable features, atmosphere, and any contextual elements."""

        pass3_prompt = f"""Analyze the environment, lighting, and composition in this image.

1. Setting/Environment:
{env_instruction}

2. Lighting:
Describe the lighting quality, direction, color temperature, shadows, highlights,
and how it affects the mood.

3. Composition:
Describe the camera framing, angle, depth of field, subject placement,
and any compositional techniques used.

4. Atmosphere:
Overall mood, feeling, and visual impact of the image."""

        # Add sky/weather info if available
        if decisions.weather_condition:
            pass3_prompt += f"\n\nSky/Weather detected: {decisions.weather_condition}. Describe how this affects the scene."
            pass3_enhancements.append({"type": "weather", "weather": decisions.weather_condition})

        # Add landmark info if available
        if decisions.landmark_name:
            pass3_prompt += f"\n\nLandmark detected: {decisions.landmark_name}. Include this in your environment description."
            pass3_enhancements.append({"type": "landmark", "landmark": decisions.landmark_name})

        pass3_prompt += """

Write as flowing descriptive prose suitable for AI image generation."""

        passes.append({
            "name": "environment_lighting_composition",
            "sections": pass3_sections,
            "prompt": pass3_prompt,
            "enhancements": pass3_enhancements,
            "tags_injected": False,
        })

        return passes

    def _get_tag_enhancements_for_section(
        self,
        section: str,
        tagger_results: TaggerResults,
        threshold: float,
    ) -> str:
        """Get tag-driven enhancement text for a section."""
        # Use the get_tag_enhancement function with correct signature
        result = get_tag_enhancement(section, tagger_results, threshold)
        if result:
            trigger_tag, enhanced_prompt = result
            return enhanced_prompt
        return ""

    def _get_clothing_enhancements(
        self,
        tagger_results: TaggerResults,
        threshold: float,
    ) -> str:
        """Get clothing-specific enhancement prompts."""
        # Use the get_clothing_prompt function with correct signature
        result = get_clothing_prompt(tagger_results, threshold)
        if result:
            clothing_type, clothing_prompt = result
            return f"- {clothing_type}: {clothing_prompt}"
        return ""

    def _image_to_base64(self, image_tensor: Any, llm_model: Any) -> str:
        """Convert image tensor to base64 string."""
        if hasattr(image_tensor, 'cpu'):
            img_np = image_tensor[0].cpu().numpy()
        else:
            img_np = np.array(image_tensor[0])

        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
        else:
            img_np = img_np.astype(np.uint8)

        pil_image = Image.fromarray(img_np)

        # Resize for API efficiency (1024x1024 max for reasoning models)
        target_pixels = 1024 * 1024
        current_pixels = pil_image.width * pil_image.height
        if current_pixels > target_pixels:
            scale = (target_pixels / current_pixels) ** 0.5
            new_width = max(64, (int(pil_image.width * scale) // 8) * 8)
            new_height = max(64, (int(pil_image.height * scale) // 8) * 8)
            pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG", quality=95)
        return base64.standard_b64encode(buffer.getvalue()).decode("utf-8")

    def _call_llm(
        self,
        llm_model: Any,
        base64_image: str,
        prompt: str,
        prompt_config: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Make LLM call. Uses extended thinking when available (Anthropic API)."""
        client = self._get_client(llm_model)

        # System prompt optimized for Z-Image/Flux generation
        system_prompt = """You are an expert visual analyst creating prompts for AI image generation (Z-Image/Flux).

CRITICAL RULES:
- Use CONCRETE, SPECIFIC, VISUAL descriptions only
- Describe what you SEE, not what you interpret or feel
- NO poetic, philosophical, abstract, or emotional language
- NO metaphors, symbolism, or artistic interpretation
- NO phrases like "captures the essence", "evokes a sense of", "speaks to"
- NO commentary about meaning, mood interpretation, or artistic intent
- Focus on: subject details, colors, textures, lighting, composition, materials

Be comprehensive and precise. Describe observable physical details only."""

        if prompt_config and prompt_config.get("system_prompt"):
            system_prompt = prompt_config["system_prompt"]

        if hasattr(client, 'messages'):
            # Anthropic API - use thinking if supported
            supports_reasoning = getattr(llm_model, 'supports_reasoning', False)

            if supports_reasoning:
                # With extended thinking
                response = client.messages.create(
                    model=llm_model.model,
                    max_tokens=llm_model.max_tokens,
                    thinking={"type": "enabled", "budget_tokens": min(4000, llm_model.max_tokens // 2)},
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
                            {"type": "text", "text": prompt}
                        ]
                    }]
                )
                # Extract text from response (skip thinking blocks)
                for block in response.content:
                    if hasattr(block, 'text'):
                        return block.text
                return ""
            else:
                # Without thinking
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
                            {"type": "text", "text": prompt}
                        ]
                    }]
                )
                return response.content[0].text
        else:
            # OpenAI-compatible API (including local models)
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
                            {"type": "text", "text": prompt}
                        ]
                    }
                ]
            )

            if not response or not response.choices or not response.choices[0].message:
                raise ValueError("LLM returned empty response")

            return response.choices[0].message.content

    def _get_client(self, llm_model: Any) -> Any:
        """Get LLM client based on provider."""
        provider = llm_model.provider.lower()

        if provider == "anthropic":
            import anthropic
            import httpx
            timeout = httpx.Timeout(timeout=300.0, connect=30.0)
            return anthropic.Anthropic(api_key=llm_model.api_key, timeout=timeout)

        elif provider == "local":
            # Local VLM models (Qwen2-VL, etc.)
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
            # OpenAI-compatible API (OpenAI, Gemini, Ollama, LMStudio, etc.)
            from openai import OpenAI
            return OpenAI(
                api_key=llm_model.api_key or "not-needed",
                base_url=llm_model.api_url if llm_model.api_url else None,
            )

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

    def _assemble_passes(self, pass_outputs: Dict[str, str]) -> str:
        """Assemble final prompt from pass outputs."""
        parts = []
        order = ["subject_appearance", "body_clothing_pose", "environment_lighting_composition"]

        for pass_name in order:
            if pass_name in pass_outputs and pass_outputs[pass_name]:
                parts.append(pass_outputs[pass_name])

        return " ".join(parts)

    def _print_inclusion_summary(self, analysis: Dict[str, Any]) -> None:
        """Print a summary of tag inclusion analysis."""
        if not analysis.get("available"):
            return

        score = analysis.get("inclusion_score", 0)
        included = analysis.get("included_count", 0)
        total = analysis.get("total_count", 0)
        method = analysis.get("method", "unknown")

        # Color code based on score
        if score >= 0.8:
            status = "EXCELLENT"
        elif score >= 0.6:
            status = "GOOD"
        elif score >= 0.4:
            status = "FAIR"
        else:
            status = "LOW"

        print(f"\n[Detailed Mode - Inclusion Analysis] Score: {score:.1%} ({status}) - {included}/{total} tags reflected")
        print(f"  Method: {method}")

        summary = analysis.get("summary", {})
        if summary.get("included"):
            included_tags = ", ".join(summary["included"][:5])
            if len(summary["included"]) > 5:
                included_tags += f" (+{len(summary['included']) - 5} more)"
            print(f"  Included: {included_tags}")

        if summary.get("missing"):
            missing_tags = ", ".join(summary["missing"][:5])
            if len(summary["missing"]) > 5:
                missing_tags += f" (+{len(summary['missing']) - 5} more)"
            print(f"  Missing: {missing_tags}")
