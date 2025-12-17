# -*- coding: utf-8 -*-
"""
ComfyUI-AI-Photography-Toolkit - Prompt Debug Agent Node

Evaluates prompt generation quality using the connected LLM.
Uses a single API call for fast, efficient evaluation.

Features:
- Evaluates prompt quality against Z-Image best practices
- Analyzes source/output image alignment
- Provides recommendations and improved prompt
- Stores results for review

Author: Siddhartha Lahiri
Email: siddhartha.lahiri@gmail.com
License: MIT
"""

import json
import time
import base64
import io
from typing import Any, Dict, Tuple
from PIL import Image

from comfy_api.latest import io as comfy_io

from .llm_providers.llm_model_type import LLMModelConfig


# Custom type for LLM model input
LLM_MODEL_Type = comfy_io.Custom("LLM_MODEL")


class SID_PromptDebugAgent(comfy_io.ComfyNode):
    """
    Prompt Debug Agent for evaluating and improving prompt generation.

    Uses the connected LLM to:
    - Analyze source and output images
    - Evaluate prompt against best practices
    - Score quality and provide recommendations
    - Generate improved prompt

    Inputs:
    - source_image: Original input image
    - output_image: Generated result image
    - prompt: The generated prompt to evaluate
    - llm_model: LLM configuration (uses same model that generated prompt)
    - enable_debug: Toggle to enable/disable debugging
    """

    @classmethod
    def define_schema(cls) -> comfy_io.Schema:
        return comfy_io.Schema(
            node_id="SID_PromptDebugAgent",
            display_name="SID Prompt Debug Agent",
            category="SID Photography Toolkit/Debug",
            description="Evaluate prompt quality using the connected LLM",
            is_output_node=True,
            inputs=[
                comfy_io.Image.Input(
                    "source_image",
                    tooltip="Original source image that was analyzed"
                ),
                comfy_io.Image.Input(
                    "output_image",
                    tooltip="Generated output image (result of the prompt)"
                ),
                comfy_io.String.Input(
                    "prompt",
                    tooltip="Connect from prompt generator output"
                ),
                LLM_MODEL_Type.Input(
                    "llm_model",
                    tooltip="LLM configuration (same as prompt generator)"
                ),
                comfy_io.Boolean.Input(
                    "enable_debug",
                    default=True,
                    display_name="Enable Debug",
                    tooltip="Enable/disable debug evaluation (when off, passes through without evaluation)"
                ),
            ],
            outputs=[
                comfy_io.String.Output(
                    "full_report",
                    display_name="full_report",
                    tooltip="Complete evaluation report as JSON"
                ),
                comfy_io.Float.Output(
                    "quality_score",
                    display_name="quality_score",
                    tooltip="Overall prompt quality score (0-10)"
                ),
                comfy_io.String.Output(
                    "quick_summary",
                    display_name="quick_summary",
                    tooltip="Brief human-readable evaluation summary"
                ),
                comfy_io.String.Output(
                    "suggested_prompt",
                    display_name="suggested_prompt",
                    tooltip="Improved prompt based on evaluation"
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        source_image,
        output_image,
        prompt: str,
        llm_model: LLMModelConfig,
        enable_debug: bool = True,
    ) -> Tuple[str, float, str, str]:
        """
        Execute the debug evaluation.

        Returns:
            Tuple of (full_report, quality_score, quick_summary, suggested_prompt)
        """
        # Skip if debug is disabled
        if not enable_debug:
            print("[SID-Debug] Debug disabled, passing through...")
            return (
                json.dumps({"status": "skipped", "reason": "debug_disabled"}),
                0.0,
                "Debug evaluation skipped (disabled)",
                prompt
            )

        print("[SID-Debug] Starting prompt evaluation...")
        start_time = time.time()

        # Run two-call evaluation (1: analysis, 2: improved prompt)
        try:
            evaluation = cls._evaluate_two_calls(
                source_image=source_image,
                output_image=output_image,
                prompt=prompt,
                llm_model=llm_model,
            )
        except Exception as e:
            error_msg = f"Evaluation failed: {e}"
            print(f"[SID-Debug] {error_msg}")
            import traceback
            traceback.print_exc()
            return (
                json.dumps({"error": error_msg}),
                0.0,
                f"Error: {error_msg}",
                prompt
            )

        # Add timing
        evaluation["timing"] = {
            "total_seconds": round(time.time() - start_time, 2)
        }

        # Save results
        try:
            cls._save_results(source_image, output_image, prompt, llm_model, evaluation)
        except Exception as e:
            print(f"[SID-Debug] Warning: Failed to save results: {e}")

        # Extract results
        full_report = json.dumps(evaluation, indent=2, ensure_ascii=False)
        quality_score = cls._extract_score(evaluation)
        quick_summary = cls._generate_summary(evaluation, quality_score)
        suggested_prompt = cls._extract_improved_prompt(evaluation, prompt)

        print(f"[SID-Debug] Evaluation complete in {evaluation['timing']['total_seconds']:.1f}s. Score: {quality_score:.1f}/10")

        return (full_report, quality_score, quick_summary, suggested_prompt)

    @classmethod
    def _evaluate_two_calls(
        cls,
        source_image,
        output_image,
        prompt: str,
        llm_model: LLMModelConfig,
    ) -> Dict[str, Any]:
        """Run evaluation with two API calls: 1) Analysis, 2) Improved prompt."""

        # Convert images to base64
        source_b64 = cls._image_to_base64(source_image)
        output_b64 = cls._image_to_base64(output_image)

        # Get provider info
        provider = llm_model.provider.lower()
        api_key = llm_model.api_key
        model = llm_model.model
        api_url = llm_model.api_url

        print(f"[SID-Debug] Using {provider}/{model} for evaluation...")

        # === CALL 1: Analysis ===
        print("[SID-Debug] Step 1/2: Analyzing prompt quality...")

        analysis_system = """You are an expert prompt evaluator for Z-Image/Flux image generation.

Analyze the given prompt by comparing the SOURCE image (what was analyzed) with the OUTPUT image (what was generated).

Score these aspects (0-10):
1. **Source Alignment**: How well does the prompt describe what's in the source image?
2. **Output Quality**: How well did the output match the prompt intent?
3. **Structure**: Is the prompt well-structured (subject, setting, lighting, style)?
4. **Specificity**: Does it use concrete visual terms instead of vague words?

Z-Image Best Practices:
- Start with subject and framing (e.g., "close-up portrait of...")
- Include pose, expression, lighting, and camera angle
- Use specific visual vocabulary (not abstract words like "beautiful")
- End with "no text, no watermark"

Respond ONLY with this JSON structure:
{
  "scores": {
    "source_alignment": {"score": 8, "reason": "brief reason"},
    "output_quality": {"score": 7, "reason": "brief reason"},
    "structure": {"score": 8, "reason": "brief reason"},
    "specificity": {"score": 7, "reason": "brief reason"},
    "overall": {"score": 7.5, "reason": "brief summary"}
  },
  "issues": ["issue 1", "issue 2"],
  "recommendations": ["recommendation 1", "recommendation 2"],
  "missing_elements": ["element visible in source but not in prompt"],
  "hallucinated_elements": ["element in prompt but not in source"]
}"""

        analysis_user = f"""Analyze this prompt:

PROMPT TO EVALUATE:
{prompt}

The first image is the SOURCE (original input), the second is the OUTPUT (generated result).
Identify what's good, what's missing, and what needs improvement."""

        # Make analysis call
        analysis_result = cls._call_llm(provider, api_key, api_url, model, analysis_system, analysis_user, source_b64, output_b64)
        analysis = cls._parse_json_response(analysis_result)

        # === CALL 2: Generate Improved Prompt ===
        print("[SID-Debug] Step 2/2: Generating improved prompt...")

        # Build context from analysis
        issues_text = "\n".join(f"- {issue}" for issue in analysis.get("issues", []))
        recommendations_text = "\n".join(f"- {rec}" for rec in analysis.get("recommendations", []))
        missing_text = "\n".join(f"- {elem}" for elem in analysis.get("missing_elements", []))

        improve_system = """You are an expert Z-Image prompt writer. Your task is to write an improved version of a prompt based on analysis feedback.

Z-Image Best Practices:
- ALWAYS start with framing and subject (e.g., "medium shot portrait of a...")
- ALWAYS include pose (facing camera, three-quarter view, etc.)
- ALWAYS include expression for humans
- Use specific visual vocabulary: skin tones (warm ivory, golden tan), hair (styled back, wavy auburn), lighting (golden hour, Rembrandt lighting)
- Describe clothing with length (jacket ending at waist, dress extending to knees)
- Identify accessories precisely (ornate silver pendant, leather messenger bag)
- End with "no text, no watermark"
- Do NOT use abstract words like "beautiful", "stunning", "elegant"
- Do NOT include meta tags like "8K", "masterpiece", "best quality"

Output ONLY the improved prompt text, nothing else. No explanations, no JSON, just the prompt."""

        improve_user = f"""Write an improved version of this prompt based on the analysis.

ORIGINAL PROMPT:
{prompt}

ISSUES FOUND:
{issues_text if issues_text else "None identified"}

RECOMMENDATIONS:
{recommendations_text if recommendations_text else "None"}

MISSING ELEMENTS (visible in source but not described):
{missing_text if missing_text else "None identified"}

Look at the SOURCE image and write a complete, improved prompt that addresses all issues. The prompt should be 80-200 words and follow Z-Image best practices."""

        # Make improvement call (with images for reference)
        improved_prompt = cls._call_llm(provider, api_key, api_url, model, improve_system, improve_user, source_b64, output_b64)

        # Clean up the improved prompt (remove any JSON wrapper or quotes)
        improved_prompt = improved_prompt.strip()
        if improved_prompt.startswith('"') and improved_prompt.endswith('"'):
            improved_prompt = improved_prompt[1:-1]
        if improved_prompt.startswith("```"):
            # Remove code blocks
            lines = improved_prompt.split("\n")
            improved_prompt = "\n".join(line for line in lines if not line.startswith("```"))

        # Combine results
        evaluation = analysis.copy()
        evaluation["improved_prompt"] = improved_prompt.strip()
        evaluation["evaluator"] = {"provider": provider, "model": model}
        evaluation["api_calls"] = 2

        return evaluation

    @classmethod
    def _call_llm(cls, provider: str, api_key: str, api_url: str, model: str, system: str, user: str, img1_b64: str, img2_b64: str) -> str:
        """Route to appropriate LLM provider."""
        if provider == "anthropic":
            return cls._call_anthropic(api_key, model, system, user, img1_b64, img2_b64)
        elif provider in ["openai", "groq", "together", "fireworks", "openrouter", "gemini", "mistral", "deepseek", "cerebras", "xai"]:
            return cls._call_openai_compatible(api_key, api_url, model, system, user, img1_b64, img2_b64)
        elif provider in ["ollama", "lmstudio"]:
            return cls._call_local_api(api_url, model, system, user, img1_b64, img2_b64)
        elif provider == "local":
            # Local transformers models - try to use API URL if available, otherwise error
            if api_url:
                return cls._call_local_api(api_url, model, system, user, img1_b64, img2_b64)
            else:
                raise ValueError(
                    f"Debug evaluation requires an API-based model. "
                    f"Local transformers model '{model}' doesn't support multi-image comparison. "
                    f"Please use Ollama, LM Studio, or a cloud provider (Anthropic, OpenAI, etc.) for debug evaluation."
                )
        else:
            raise ValueError(f"Unsupported provider for debug: {provider}. Supported: anthropic, openai, groq, together, fireworks, openrouter, ollama, lmstudio")

    @classmethod
    def _call_anthropic(cls, api_key: str, model: str, system: str, user: str, img1_b64: str, img2_b64: str) -> str:
        """Call Anthropic API with images."""
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)

        response = client.messages.create(
            model=model,
            max_tokens=2048,
            system=system,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": user},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img1_b64}},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img2_b64}},
                ]
            }]
        )

        return response.content[0].text

    @classmethod
    def _call_openai_compatible(cls, api_key: str, api_url: str, model: str, system: str, user: str, img1_b64: str, img2_b64: str) -> str:
        """Call OpenAI-compatible API with images."""
        import openai

        client = openai.OpenAI(api_key=api_key, base_url=api_url if api_url else None)

        response = client.chat.completions.create(
            model=model,
            max_tokens=2048,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": [
                    {"type": "text", "text": user},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img1_b64}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img2_b64}"}},
                ]}
            ]
        )

        return response.choices[0].message.content

    @classmethod
    def _call_local_api(cls, api_url: str, model: str, system: str, user: str, img1_b64: str, img2_b64: str) -> str:
        """Call local API (Ollama/LM Studio) with images."""
        import openai

        client = openai.OpenAI(api_key="not-needed", base_url=api_url)

        response = client.chat.completions.create(
            model=model,
            max_tokens=2048,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": [
                    {"type": "text", "text": user},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img1_b64}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img2_b64}"}},
                ]}
            ]
        )

        return response.choices[0].message.content

    @classmethod
    def _image_to_base64(cls, image) -> str:
        """Convert image to base64 string."""
        if isinstance(image, str):
            return image

        # Convert tensor to PIL
        if hasattr(image, 'cpu'):
            import numpy as np
            if len(image.shape) == 4:
                image = image[0]
            np_image = (image.cpu().numpy() * 255).astype(np.uint8)
            pil_image = Image.fromarray(np_image)
        elif isinstance(image, Image.Image):
            pil_image = image
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        # Resize if too large (for faster API calls)
        max_size = 1024
        if max(pil_image.size) > max_size:
            pil_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    @classmethod
    def _parse_json_response(cls, response: str) -> Dict[str, Any]:
        """Parse JSON from LLM response."""
        import re

        # Try to find JSON block
        if "```json" in response:
            match = re.search(r'```json\s*([\s\S]*?)\s*```', response)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    pass

        # Try to find raw JSON
        if "{" in response:
            match = re.search(r'\{[\s\S]*\}', response)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass

        return {"raw_response": response, "parse_error": "Could not parse JSON"}

    @classmethod
    def _save_results(cls, source_image, output_image, prompt: str, llm_model: LLMModelConfig, evaluation: Dict[str, Any]):
        """Save evaluation results to disk."""
        from pathlib import Path
        from datetime import datetime

        # Get debug_results directory
        base_dir = Path(__file__).parent / "debug_results"
        base_dir.mkdir(exist_ok=True)

        # Create session folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = base_dir / f"session_{timestamp}"
        session_dir.mkdir(exist_ok=True)

        # Save evaluation JSON
        eval_path = session_dir / "evaluation.json"
        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(evaluation, f, indent=2, ensure_ascii=False)

        # Save prompt
        prompt_path = session_dir / "prompt.txt"
        with open(prompt_path, "w", encoding="utf-8") as f:
            f.write(prompt)

        # Save model config
        config_path = session_dir / "model_config.json"
        model_config = cls._extract_model_config(llm_model)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(model_config, f, indent=2)

        print(f"[SID-Debug] Results saved to: {session_dir}")

    @classmethod
    def _extract_model_config(cls, llm_model: LLMModelConfig) -> Dict[str, Any]:
        """Extract model configuration from LLMModelConfig."""
        if isinstance(llm_model, dict):
            return llm_model

        config = {
            "provider": getattr(llm_model, "provider", "unknown"),
            "model": getattr(llm_model, "model", "unknown"),
            "temperature": getattr(llm_model, "temperature", 0.7),
        }

        # Extract extra params if available
        extra_params = {}
        for attr in ["top_p", "top_k", "max_tokens", "repetition_penalty"]:
            if hasattr(llm_model, attr):
                value = getattr(llm_model, attr)
                if value is not None:
                    extra_params[attr] = value

        if extra_params:
            config["extra_params"] = extra_params

        # Include endpoint info if local model
        if hasattr(llm_model, "endpoint") and llm_model.endpoint:
            config["endpoint"] = llm_model.endpoint

        # Include model metadata if available
        if hasattr(llm_model, "metadata") and llm_model.metadata:
            config["metadata"] = llm_model.metadata

        return config

    @classmethod
    def _extract_score(cls, evaluation: Dict[str, Any]) -> float:
        """Extract overall score from evaluation."""
        # Try different paths where score might be stored (ordered by new structure)
        score_paths = [
            ["scores", "overall", "score"],  # New structure
            ["scores", "overall"],  # If score is direct value
            ["evaluation", "scores", "overall", "score"],
            ["overall_score"],
            ["evaluation", "overall_score"],
        ]

        for path in score_paths:
            current = evaluation
            try:
                for key in path:
                    current = current[key]
                if isinstance(current, (int, float)):
                    return float(current)
                # Handle dict with "score" key
                if isinstance(current, dict) and "score" in current:
                    return float(current["score"])
            except (KeyError, TypeError):
                continue

        # Default score if not found
        return 5.0

    @classmethod
    def _generate_summary(cls, evaluation: Dict[str, Any], score: float) -> str:
        """Generate a human-readable summary of the evaluation."""
        lines = [f"Overall Score: {score:.1f}/10"]

        # Try to extract key insights
        try:
            # Get scoring breakdown
            scores = evaluation.get("evaluation", {}).get("scores", {})
            if not scores:
                scores = evaluation.get("scores", {})

            for category, data in scores.items():
                if category != "overall" and isinstance(data, dict):
                    cat_score = data.get("score", "N/A")
                    lines.append(f"  - {category.replace('_', ' ').title()}: {cat_score}")

        except Exception:
            pass

        # Try to get recommendations
        try:
            recommendations = evaluation.get("recommendations", [])
            if not recommendations:
                recommendations = evaluation.get("evaluation", {}).get("recommendations", [])

            # Handle both list and dict formats
            if isinstance(recommendations, dict):
                improvements = recommendations.get("prompt_improvements", [])
            else:
                improvements = recommendations if isinstance(recommendations, list) else []

            if improvements:
                lines.append(f"\nTop Recommendations ({len(improvements)} total):")
                for rec in improvements[:3]:
                    if isinstance(rec, str):
                        rec_text = rec[:80] + "..." if len(rec) > 80 else rec
                        lines.append(f"  - {rec_text}")
                    elif isinstance(rec, dict):
                        rec_text = rec.get('suggestion', str(rec))[:80]
                        lines.append(f"  - {rec_text}...")

        except Exception:
            pass

        # Add timing info if available
        try:
            timing = evaluation.get("timing", {})
            if timing:
                lines.append(f"\nEvaluation Time: {timing.get('total_seconds', 0):.1f}s")
        except Exception:
            pass

        return "\n".join(lines)

    @classmethod
    def _extract_improved_prompt(cls, evaluation: Dict[str, Any], original: str) -> str:
        """Extract the improved prompt suggestion from evaluation."""
        # Try different paths (ordered by likelihood based on expected JSON structure)
        paths = [
            ["improved_prompt"],  # Direct key (expected from new structure)
            ["evaluation", "improved_prompt"],
            ["recommendations", "improved_prompt"],
            ["suggested_prompt"],
            ["recommendations", "suggested_prompt"],
        ]

        for path in paths:
            current = evaluation
            try:
                for key in path:
                    current = current[key]
                if isinstance(current, str) and len(current) > 10:
                    return current
            except (KeyError, TypeError):
                continue

        # Try to find in raw_response if JSON parsing failed
        raw = evaluation.get("raw_response", "")
        if raw and "improved_prompt" in raw.lower():
            # Try to extract from raw text
            import re
            match = re.search(r'"improved_prompt"\s*:\s*"([^"]+)"', raw)
            if match:
                return match.group(1)

        # Return original if no improvement found
        return original
