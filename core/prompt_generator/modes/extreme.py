"""
Extreme mode - 6-pass LLM prompt generation with maximum detail.

Each pass focuses on a single aspect of the 6-part Z-Image structure:
1. Subject - physical features, age, ethnicity, build, hair, skin
2. Clothing - garments, fabrics, colors, fit, accessories
3. Pose & Expression - body position, gesture, facial expression, gaze
4. Scene - setting, background, props, atmosphere
5. Lighting - quality, direction, color temperature, shadows
6. Camera - shot type, angle, framing, depth of field, lens

Works with any LLM (local or API).
"""

import base64
import io
import time
from typing import Any, Dict, Optional, List

import numpy as np
from PIL import Image

from .base import BaseMode
from .standard import StandardMode
from ..types import GeneratorResult, TaggerResults, Decisions
from ..decisions import get_engine
from ..templates.tag_injector import get_unmapped_tags, format_tags_for_prompt, get_debug_info, get_female_anatomy_tags
from ..templates.scene_prompts import format_scene_prompt, get_scene_group
from ..templates.detail_prompts import get_animal_prompt, get_clothing_prompt, get_tag_enhancement
from ..taggers import get_runner


class ExtremeMode(BaseMode):
    """
    Extreme mode with 6 focused LLM passes.

    Each pass generates detailed content for a single section:
    1. Subject (physical features)
    2. Clothing (garments, accessories)
    3. Pose & Expression
    4. Scene (environment, background)
    5. Lighting
    6. Camera (framing, composition)

    Falls back to Standard mode on error.
    """

    @property
    def name(self) -> str:
        return "extreme"

    @property
    def requires_llm(self) -> bool:
        return True

    @property
    def requires_taggers(self) -> bool:
        return True

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
        Execute Extreme mode with 6 focused passes.

        Args:
            image: Image tensor from ComfyUI
            llm_model: LLMModelConfig with provider settings
            tagger_results: Pre-computed tagger results
            decisions: Pre-computed decisions
            prompt_config: Optional custom system/user prompts
            **kwargs: Additional parameters

        Returns:
            GeneratorResult with 6-pass generated description
        """
        if image is None:
            return GeneratorResult(
                prompt="[No image provided]",
                metadata={"mode": "extreme", "error": "No image provided"}
            )

        if llm_model is None:
            return GeneratorResult(
                prompt="[No LLM model provided]",
                metadata={"mode": "extreme", "error": "No LLM model provided"}
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

            # Step 3: Get tag debug info
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

            # Inject female anatomy tags for women subjects
            subject_type = decisions.subject_type.value if hasattr(decisions.subject_type, 'value') else str(decisions.subject_type)
            anatomy_tags = get_female_anatomy_tags(subject_type, detected_tags)
            if anatomy_tags:
                detected_tags = anatomy_tags + list(detected_tags)

            tags_str = format_tags_for_prompt(detected_tags, show_confidence=True) if detected_tags else ""

            # Step 6: Convert image to base64
            base64_image = self._image_to_base64(image, llm_model)

            # Step 7: Build and execute 6 passes
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
                    "section": pass_config.get("section", pass_name),
                    "enhancements": pass_config.get("enhancements", []),
                }

            timing["llm_ms"] = int((time.time() - llm_start) * 1000)

            # Step 8: Assemble final prompt
            prompt = self._assemble_passes(pass_outputs)

            # Step 9: Analyze tag inclusion
            inclusion_start = time.time()
            from ..validation import analyze_inclusion
            injected_tags = [
                (t["tag"], t["confidence"])
                for t in tag_debug_info.get("injected_tags", [])
            ]
            inclusion_analysis = analyze_inclusion(injected_tags, prompt)
            timing["inclusion_analysis_ms"] = int((time.time() - inclusion_start) * 1000)

            self._print_inclusion_summary(inclusion_analysis)

            total_time = int((time.time() - start_time) * 1000)

            return GeneratorResult(
                prompt=prompt,
                metadata={
                    "mode": "extreme",
                    "source": "llm_6_pass",
                    "llm_provider": llm_model.provider,
                    "llm_model": llm_model.model,
                    "processing": {
                        "taggers_executed": True,
                        "florence_executed": False,
                        "llm_executed": True,
                        "llm_passes": 6,
                    },
                    "decisions": decisions.to_dict(),
                    "passes": pass_details,
                    "pass_outputs": pass_outputs,
                    "scene": {
                        "detected": scene_type,
                        "group": scene_group,
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
            print(f"[Extreme] Error: {error_msg}")
            import traceback
            traceback.print_exc()

            # Fall back to Standard mode on error
            print("[Extreme] Falling back to Standard mode")
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
        """Build 6 focused pass configurations."""
        subject_type = decisions.subject_type.value
        passes = []

        # =====================================================================
        # PASS 1: SUBJECT - Physical features, identity
        # =====================================================================
        if subject_type in ["person", "woman", "man", "couple", "group"]:
            pass1_prompt = """Analyze ONLY the subject's physical identity in this image.

Describe with technical precision:
- Gender and apparent age range (20s, 30s, etc.)
- Ethnicity/racial features
- Face shape (oval, round, square, heart)
- Skin tone (fair, medium, olive, tan, brown, dark)
- Hair: color, length, texture (straight/wavy/curly), style
- Distinctive facial features (high cheekbones, full lips, strong jaw, etc.)

Use concrete visual terms only. No poetic language.
Output 2-3 sentences of specific physical description."""

        elif subject_type == "animal":
            animal_type = decisions.wildlife_type or "animal"
            pass1_prompt = f"""Analyze ONLY the {animal_type}'s physical identity.

Describe with technical precision:
- Species and breed if identifiable
- Size and build
- Fur/feather/scale coloring and patterns
- Distinctive physical features

Use concrete visual terms only. Output 2-3 sentences."""

        else:
            pass1_prompt = """Analyze ONLY the main subject's identity.

Describe with technical precision:
- What is the primary subject
- Key identifying characteristics
- Materials and textures
- Condition and notable features

Use concrete visual terms only. Output 2-3 sentences."""

        # Add relevant tags for subject
        if tags_str:
            pass1_prompt += f"\n\nDetected elements: {tags_str}"

        passes.append({
            "name": "subject",
            "section": "Subject",
            "prompt": pass1_prompt,
            "enhancements": [],
        })

        # =====================================================================
        # PASS 2: CLOTHING - Garments, accessories
        # =====================================================================
        if subject_type in ["person", "woman", "man", "couple", "group"]:
            pass2_prompt = """Analyze ONLY the clothing and accessories in this image.

Describe with technical precision:
- Each garment: type, color, material/fabric
- Fit: tight, loose, tailored, flowing
- Style: casual, formal, streetwear, elegant, etc.
- Accessories: jewelry, bags, glasses, hats, shoes
- Notable details: patterns, textures, brand elements

Use concrete visual terms only. No poetic language.
Output 2-3 sentences focused on what is worn."""

            # Add clothing-specific enhancements
            clothing_result = get_clothing_prompt(tagger_results, tag_threshold)
            if clothing_result:
                clothing_type, clothing_detail = clothing_result
                pass2_prompt += f"\n\nDetected clothing: {clothing_type}. {clothing_detail}"

        elif subject_type == "animal":
            pass2_prompt = """Analyze any accessories or coverings on this animal.

Describe: collar, harness, saddle, clothing, decorations if present.
If no accessories, describe the animal's natural covering (fur pattern, markings).

Use concrete visual terms only. Output 1-2 sentences."""

        else:
            pass2_prompt = """Analyze secondary elements and styling.

Describe any accessories, decorations, or supporting elements.
Use concrete visual terms only. Output 1-2 sentences."""

        passes.append({
            "name": "clothing",
            "section": "Clothing",
            "prompt": pass2_prompt,
            "enhancements": [],
        })

        # =====================================================================
        # PASS 3: POSE & EXPRESSION - Body position, emotion
        # =====================================================================
        if subject_type in ["person", "woman", "man", "couple", "group"]:
            pass3_prompt = """Analyze ONLY the pose and expression in this image.

POSE - describe with precision:
- Body orientation (facing camera, 3/4 angle, profile, back)
- Posture (standing, sitting, leaning, lying)
- Arm/hand position
- Leg position if visible
- Weight distribution

EXPRESSION - describe:
- Facial expression (smiling, serious, neutral, pensive)
- Eye direction (looking at camera, away, down)
- Mouth (open, closed, smiling, parted)
- Overall mood conveyed

Use concrete visual terms only. Output 2-3 sentences."""

        elif subject_type == "animal":
            pass3_prompt = """Analyze ONLY the animal's pose and demeanor.

Describe:
- Body position (standing, sitting, lying, running)
- Head orientation and ear position
- Eye expression and alertness
- Overall demeanor (relaxed, alert, playful, curious)

Use concrete visual terms only. Output 2-3 sentences."""

        else:
            pass3_prompt = """Analyze the arrangement and orientation.

Describe how the subject is positioned and oriented.
Use concrete visual terms only. Output 1-2 sentences."""

        passes.append({
            "name": "pose_expression",
            "section": "Pose & Expression",
            "prompt": pass3_prompt,
            "enhancements": [],
        })

        # =====================================================================
        # PASS 4: SCENE - Environment, background
        # =====================================================================
        if scene_prompt:
            env_instruction = scene_prompt
        else:
            env_instruction = """Describe the setting and background:
- Location type (studio, outdoor, indoor, urban, nature)
- Background elements and props
- Atmosphere and environment details
- Time of day if apparent"""

        pass4_prompt = f"""Analyze ONLY the scene and environment in this image.

{env_instruction}

Focus on:
- What is in the background
- Setting type and location
- Props or environmental elements
- Atmospheric conditions (clear, hazy, foggy)

Use concrete visual terms only. No poetic descriptions.
Output 2-3 sentences about the environment."""

        # Add weather/landmark info
        enhancements = []
        if decisions.weather_condition:
            pass4_prompt += f"\n\nWeather detected: {decisions.weather_condition}"
            enhancements.append({"type": "weather", "value": decisions.weather_condition})
        if decisions.landmark_name:
            pass4_prompt += f"\n\nLandmark detected: {decisions.landmark_name}"
            enhancements.append({"type": "landmark", "value": decisions.landmark_name})

        passes.append({
            "name": "scene",
            "section": "Scene",
            "prompt": pass4_prompt,
            "enhancements": enhancements,
        })

        # =====================================================================
        # PASS 5: LIGHTING - Quality, direction, color
        # =====================================================================
        pass5_prompt = """Analyze ONLY the lighting in this image.

Describe with technical precision:
- Light quality: soft/diffused vs hard/direct
- Light direction: front, side, back, overhead, below
- Color temperature: warm (golden/orange) vs cool (blue/white)
- Key light source: natural sun, window, studio, artificial
- Shadow characteristics: soft, hard, minimal, dramatic
- Highlights: where light falls brightest
- Overall lighting mood: high-key, low-key, natural, dramatic

Use camera/photography terms (e.g., "side lighting at 45 degrees",
"warm 3200K color temperature", "soft diffused window light").
Output 2-3 sentences about lighting only."""

        lighting_enhancements = []
        if decisions.is_golden_hour:
            pass5_prompt += "\n\nGolden hour lighting detected - describe the warm tones and long shadows."
            lighting_enhancements.append({"type": "golden_hour"})
        if decisions.is_dramatic_lighting:
            pass5_prompt += "\n\nDramatic lighting detected - describe the high contrast and shadows."
            lighting_enhancements.append({"type": "dramatic"})

        passes.append({
            "name": "lighting",
            "section": "Lighting",
            "prompt": pass5_prompt,
            "enhancements": lighting_enhancements,
        })

        # =====================================================================
        # PASS 6: CAMERA - Framing, angle, composition
        # =====================================================================
        pass6_prompt = """Analyze ONLY the camera and composition in this image.

Describe with technical precision:
- Shot type: ECU (extreme close-up), CU (close-up), MCU (medium close-up),
  MS (medium shot), MFS (medium full), FS (full shot), LS (long shot)
- Camera angle: eye-level, high angle, low angle, bird's eye, worm's eye
- Framing: centered, rule of thirds, symmetrical, asymmetrical
- Depth of field: shallow (blurred background, f/1.4-2.8),
  medium (some blur, f/4-5.6), deep (sharp throughout, f/8-16)
- Subject placement in frame
- Any lens effects: bokeh, distortion, compression

Use camera terminology (f-stops, focal lengths, shot types).
Output 2-3 sentences about camera/composition only."""

        camera_enhancements = []
        if decisions.camera_angle:
            pass6_prompt += f"\n\nDetected camera angle: {decisions.camera_angle}"
            camera_enhancements.append({"type": "angle", "value": decisions.camera_angle})
        if decisions.depth_of_field:
            pass6_prompt += f"\n\nDetected DOF: {decisions.depth_of_field}"
            camera_enhancements.append({"type": "dof", "value": decisions.depth_of_field})

        passes.append({
            "name": "camera",
            "section": "Camera",
            "prompt": pass6_prompt,
            "enhancements": camera_enhancements,
        })

        return passes

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

        # Resize for API efficiency
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
        """Make LLM call for a single pass."""
        client = self._get_client(llm_model)

        # System prompt for focused single-section analysis
        system_prompt = """You are an expert visual analyst creating prompts for AI image generation.

CRITICAL RULES:
- Describe ONLY what is asked in the prompt - ignore other aspects
- Use CONCRETE, SPECIFIC, VISUAL descriptions only
- NO poetic, philosophical, or emotional language
- NO metaphors or artistic interpretation
- Use technical photography/camera terms where applicable
- Be precise and literal
- Keep output focused and concise (2-3 sentences max)"""

        if prompt_config and prompt_config.get("system_prompt"):
            system_prompt = prompt_config["system_prompt"]

        # Determine max tokens: 256 for local, user-specified for API
        is_local = llm_model.provider.lower() == "local"
        max_tokens = 256 if is_local else llm_model.max_tokens

        if hasattr(client, 'messages'):
            # Anthropic API - uses user-specified tokens
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
            # OpenAI-compatible API - local uses 256 fixed, API uses user-specified
            response = client.chat.completions.create(
                model=llm_model.model,
                max_tokens=max_tokens,
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
            from ....llm_providers.sid_llm_local import LocalModelClient
            extra = llm_model.extra_params or {}
            return LocalModelClient(
                model_name=llm_model.model,
                quantization=extra.get("quantization", "4-bit"),
                device=extra.get("device", "auto"),
                attention_mode=extra.get("attention_mode", "auto"),
                keep_model_loaded=extra.get("keep_model_loaded", True),
                repetition_penalty=extra.get("repetition_penalty", 1.2),
                top_p=extra.get("top_p", 0.9),
                hf_token=extra.get("hf_token"),
            )

        else:
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
            r'^(?:Here is|Here\'s|This is|The image shows|In this image)[:\s]*',
            r'^(?:I can see|I see|Looking at|Based on)[:\s]*',
            r'^(?:The subject|The person|The woman|The man)[:\s]*',
        ]
        for pattern in preambles:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)

        # Remove markdown
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'\*([^*]+)\*', r'\1', text)

        # Clean whitespace
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    def _assemble_passes(self, pass_outputs: Dict[str, str]) -> str:
        """Assemble final prompt from 6 pass outputs."""
        parts = []
        order = ["subject", "clothing", "pose_expression", "scene", "lighting", "camera"]

        for pass_name in order:
            if pass_name in pass_outputs and pass_outputs[pass_name]:
                parts.append(pass_outputs[pass_name])

        return " ".join(parts)

    def _print_inclusion_summary(self, analysis: Dict[str, Any]) -> None:
        """Print tag inclusion summary."""
        if not analysis.get("available"):
            return

        score = analysis.get("inclusion_score", 0)
        included = analysis.get("included_count", 0)
        total = analysis.get("total_count", 0)

        if score >= 0.8:
            status = "EXCELLENT"
        elif score >= 0.6:
            status = "GOOD"
        elif score >= 0.4:
            status = "FAIR"
        else:
            status = "LOW"

        print(f"\n[Extreme Mode - Inclusion] Score: {score:.1%} ({status}) - {included}/{total} tags")
