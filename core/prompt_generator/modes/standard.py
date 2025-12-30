"""
Standard mode - Tag-driven template-based LLM prompt generation.

Executes taggers to make decisions, then uses templates to build
a focused LLM prompt for comprehensive image description.
"""

import base64
import io
import time
from typing import Any, Dict, Optional, List

import numpy as np
from PIL import Image

from .base import BaseMode
from ..types import GeneratorResult, TaggerResults, Decisions
from ..decisions import get_engine
from ..templates import get_loader, get_selector, get_builder
from ..taggers import get_runner


class StandardMode(BaseMode):
    """
    Standard mode implementation.

    Executes taggers to analyze the image, makes decisions based on
    tag results, selects appropriate template, and makes a single
    LLM call with a focused prompt.
    """

    @property
    def name(self) -> str:
        return "standard"

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
        **kwargs
    ) -> GeneratorResult:
        """
        Execute Standard mode.

        Args:
            image: Image tensor from ComfyUI
            llm_model: LLMModelConfig with provider settings
            tagger_results: Pre-computed tagger results (or None to compute)
            decisions: Pre-computed decisions (or None to compute)
            prompt_config: Optional custom system/user prompts
            **kwargs: Additional parameters

        Returns:
            GeneratorResult with LLM-generated description
        """
        if image is None:
            return GeneratorResult(
                prompt="[No image provided]",
                metadata={"mode": "standard", "error": "No image provided"}
            )

        if llm_model is None:
            return GeneratorResult(
                prompt="[No LLM model provided]",
                metadata={"mode": "standard", "error": "No LLM model provided"}
            )

        start_time = time.time()
        timing = {}

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

            # Step 3: Select template based on subject type
            selector = get_selector()
            template = selector.select_template(decisions.subject_type)

            # Step 4: Get active sections based on decisions
            active_sections = selector.get_active_sections(
                template, decisions, tagger_results.wd14
            )

            # Step 5: Build LLM prompt from template (with detected tags + enhancements)
            tag_threshold = kwargs.get("tag_threshold", 0.5)
            builder = get_builder()
            llm_prompt = builder.build_llm_prompt(
                template,
                active_sections,
                decisions,
                tagger_results=tagger_results,
                tag_threshold=tag_threshold,
            )
            prompt_enhancements = builder.get_last_enhancements()

            # Get debug info for metadata
            from ..templates.tag_injector import get_debug_info
            from ..templates.scene_prompts import get_scene_group, format_scene_prompt
            tag_debug_info = get_debug_info(tagger_results, tag_threshold)

            # Get scene info for metadata
            scene_type = decisions.scene_type
            scene_group = get_scene_group(scene_type) if scene_type else None
            scene_prompt_used = format_scene_prompt(scene_type) if scene_type else None

            # Step 6: Convert image to base64
            base64_image = self._image_to_base64(image, llm_model)

            # Step 7: Make LLM call
            llm_start = time.time()
            response = self._call_llm(
                llm_model=llm_model,
                base64_image=base64_image,
                prompt=llm_prompt,
                prompt_config=prompt_config,
            )
            timing["llm_ms"] = int((time.time() - llm_start) * 1000)

            # Step 8: Clean response
            prompt = self._clean_response(response)

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
                    "mode": "standard",
                    "source": "llm_template",
                    "llm_provider": llm_model.provider,
                    "llm_model": llm_model.model,
                    "processing": {
                        "taggers_executed": True,
                        "florence_executed": False,
                        "llm_executed": True,
                        "llm_passes": 1,
                    },
                    "decisions": decisions.to_dict(),
                    "template": {
                        "selected": decisions.subject_type.value,
                        "sections_used": active_sections,
                    },
                    "enhancements": prompt_enhancements if prompt_enhancements else None,
                    "scene": {
                        "detected": scene_type,
                        "group": scene_group,
                        "prompt_used": scene_prompt_used[:100] + "..." if scene_prompt_used and len(scene_prompt_used) > 100 else scene_prompt_used,
                    } if scene_type else None,
                    "tag_injection": tag_debug_info,
                    "inclusion_analysis": inclusion_analysis,
                    "llm_prompt": llm_prompt,
                    "taggers": tagger_results.to_dict(),
                    "timing": {
                        **timing,
                        "total_ms": total_time,
                    }
                }
            )

        except Exception as e:
            error_msg = str(e)
            print(f"[Standard] Error: {error_msg}")
            import traceback
            traceback.print_exc()

            return GeneratorResult(
                prompt=f"[Standard mode error: {error_msg}]",
                metadata={
                    "mode": "standard",
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

        # Resize for provider
        pil_image = self._resize_for_provider(pil_image, llm_model.provider)

        # Convert to base64
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG", quality=95)
        return base64.standard_b64encode(buffer.getvalue()).decode("utf-8")

    def _resize_for_provider(self, image: Image.Image, provider: str) -> Image.Image:
        """Resize image to optimal size for provider."""
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
            new_width = (new_width // 8) * 8
            new_height = (new_height // 8) * 8
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        return image

    def _call_llm(
        self,
        llm_model: Any,
        base64_image: str,
        prompt: str,
        prompt_config: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Make LLM call with image and prompt."""
        client = self._get_client(llm_model)

        # Default system prompt
        default_system = """You are an expert visual analyst. Describe the image based on the specific aspects requested.
Write as a single flowing paragraph suitable for AI image generation.
Be specific and detailed. Do not include meta-commentary about the image."""

        # Use custom prompts if provided
        if prompt_config:
            system_prompt = prompt_config.get("system_prompt", default_system)
            # If custom user_prompt is provided, use it instead of template-built prompt
            if prompt_config.get("user_prompt"):
                prompt = prompt_config["user_prompt"]
        else:
            system_prompt = default_system

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
                        {"type": "text", "text": prompt}
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
                hf_token=extra.get("hf_token"),
            )

        else:
            from openai import OpenAI
            return OpenAI(
                api_key=llm_model.api_key or "not-needed",
                base_url=llm_model.api_url if llm_model.api_url else None,
            )

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

        print(f"\n[Inclusion Analysis] Score: {score:.1%} ({status}) - {included}/{total} tags reflected")
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
