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
from .llm_providers.sid_llm_api import LLM_MODEL_Type
from .sid_prompt_generator_v2 import GENERATION_METADATA_Type, SID_PROMPT_Type


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
                SID_PROMPT_Type.Input(
                    "prompt",
                    tooltip="Connect from prompt generator output"
                ),
                LLM_MODEL_Type.Input(
                    "llm_model",
                    tooltip="LLM configuration for running debug evaluation"
                ),
                GENERATION_METADATA_Type.Input(
                    "source_metadata",
                    optional=True,
                    tooltip="Connect from prompt generator metadata output. Stores all generation settings with debug results."
                ),
                comfy_io.Boolean.Input(
                    "enable_debug",
                    default=True,
                    display_name="Enable Debug",
                    tooltip="Enable/disable debug evaluation (when off, passes through without evaluation)"
                ),
                comfy_io.Boolean.Input(
                    "analyze_template",
                    default=True,
                    display_name="Template Effectiveness",
                    tooltip="Analyze template effectiveness and provide improvement suggestions. Disable for faster analysis."
                ),
                comfy_io.Boolean.Input(
                    "generate_improved",
                    default=True,
                    display_name="Improved Prompt",
                    tooltip="Generate an improved version of the prompt. Disable for faster analysis."
                ),
                comfy_io.Boolean.Input(
                    "print_to_console",
                    default=False,
                    display_name="Print to Console",
                    tooltip="Print detailed analysis results to the console/terminal."
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
                comfy_io.Float.Output(
                    "template_score",
                    display_name="template_score",
                    tooltip="Template effectiveness score (0-10)"
                ),
                comfy_io.String.Output(
                    "template_suggestions",
                    display_name="template_suggestions",
                    tooltip="Suggestions for improving the TOML template"
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
        source_metadata: Dict[str, Any] = None,
        enable_debug: bool = True,
        analyze_template: bool = True,
        generate_improved: bool = True,
        print_to_console: bool = False,
    ) -> Tuple[str, float, str, str, float, str]:
        """
        Execute the intelligent debug evaluation.

        Args:
            analyze_template: If True, analyze template effectiveness (Phase 2)
            generate_improved: If True, generate improved prompt (Phase 3)
            print_to_console: If True, print detailed analysis to console
            If both analyze_template and generate_improved are False, only deep image analysis is performed (Phase 1)

        Returns:
            Tuple of (full_report, quality_score, quick_summary, suggested_prompt, template_score, template_suggestions)
        """
        # Skip if debug is disabled
        if not enable_debug:
            print("[SID-Debug] Debug disabled, passing through...")
            return comfy_io.NodeOutput(
                json.dumps({"status": "skipped", "reason": "debug_disabled"}),
                0.0,
                "Debug evaluation skipped (disabled)",
                prompt,
                0.0,
                "Debug disabled - no template analysis performed"
            )

        # =================================================================
        # VALIDATION: Require API-based reasoning models for intelligent debug
        # =================================================================
        provider = llm_model.provider.lower()
        model_name = getattr(llm_model, 'model', 'Unknown')

        # Check 1: Must be API-based (not local transformers without API endpoint)
        if provider == "local" and not llm_model.api_url:
            raise ValueError(
                f"INTELLIGENT DEBUG requires an API-based LLM model.\n\n"
                f"Local transformers models cannot perform multi-image comparison needed for thorough debug.\n\n"
                f"Recommended API providers:\n"
                f"  - Anthropic: Claude with extended thinking\n"
                f"  - OpenAI: o1, o3 series (reasoning models)\n"
                f"  - Google: Gemini 2.0 Flash Thinking\n"
                f"  - Ollama/LM Studio: Via API endpoint\n\n"
                f"Connect an API-based LLM node to the debug agent."
            )

        # Check 2: Must support reasoning for thorough analysis
        supports_reasoning = getattr(llm_model, 'supports_reasoning', False)
        if not supports_reasoning:
            raise ValueError(
                f"INTELLIGENT DEBUG requires a reasoning-capable model.\n\n"
                f"Model '{model_name}' does not support extended thinking/reasoning.\n"
                f"Reasoning enables thorough, multi-step analysis of prompts and templates.\n\n"
                f"Recommended reasoning models:\n"
                f"  - Anthropic: Claude with extended thinking enabled\n"
                f"  - OpenAI: o1, o1-pro, o3, o3-mini\n"
                f"  - Google: Gemini 2.0 Flash Thinking\n"
                f"  - Local: Qwen3-VL-*-Thinking variants (via Ollama)\n\n"
                f"Select a reasoning-capable model in your LLM node."
            )

        # Determine analysis mode
        if not analyze_template and not generate_improved:
            mode_str = "Deep Analysis Only (Phase 1)"
            phases = 1
        elif analyze_template and not generate_improved:
            mode_str = "Analysis + Template Evaluation (Phases 1-2)"
            phases = 2
        elif not analyze_template and generate_improved:
            mode_str = "Analysis + Improved Prompt (Phases 1,3)"
            phases = 2
        else:
            mode_str = "Full Evaluation (Phases 1-3)"
            phases = 3

        print(f"[SID-Debug] Using reasoning model: {provider}/{model_name}")
        print(f"[SID-Debug] Mode: {mode_str}")
        print("[SID-Debug] Starting intelligent prompt evaluation...")
        start_time = time.time()

        # Run evaluation with specified phases
        try:
            evaluation = cls._evaluate_with_phases(
                source_image=source_image,
                output_image=output_image,
                prompt=prompt,
                llm_model=llm_model,
                source_metadata=source_metadata,
                analyze_template=analyze_template,
                generate_improved=generate_improved,
                print_to_console=print_to_console,
            )
        except Exception as e:
            error_msg = f"Evaluation failed: {e}"
            print(f"[SID-Debug] {error_msg}")
            import traceback
            traceback.print_exc()
            return comfy_io.NodeOutput(
                json.dumps({"error": error_msg}),
                0.0,
                f"Error: {error_msg}",
                prompt,
                0.0,
                f"Error during evaluation: {error_msg}"
            )

        # Add timing and mode info
        evaluation["timing"] = {
            "total_seconds": round(time.time() - start_time, 2)
        }
        evaluation["mode"] = {
            "analyze_template": analyze_template,
            "generate_improved": generate_improved,
            "phases_run": phases,
            "description": mode_str,
        }

        # Save results
        try:
            cls._save_results(source_image, output_image, prompt, llm_model, evaluation, source_metadata)
        except Exception as e:
            print(f"[SID-Debug] Warning: Failed to save results: {e}")

        # Extract results
        full_report = json.dumps(evaluation, indent=2, ensure_ascii=False)
        quality_score = cls._extract_score(evaluation)
        quick_summary = cls._generate_summary(evaluation, quality_score)
        suggested_prompt = cls._extract_improved_prompt(evaluation, prompt) if generate_improved else prompt

        # Extract template-specific results
        template_score = cls._extract_template_score(evaluation) if analyze_template else 0.0
        template_suggestions = cls._format_template_suggestions(evaluation) if analyze_template else "Template analysis disabled"

        print(f"[SID-Debug] Evaluation complete in {evaluation['timing']['total_seconds']:.1f}s.")
        if analyze_template:
            print(f"[SID-Debug] Prompt Score: {quality_score:.1f}/10, Template Score: {template_score:.1f}/10")
        else:
            print(f"[SID-Debug] Quality Score: {quality_score:.1f}/10")

        return comfy_io.NodeOutput(
            full_report,
            quality_score,
            quick_summary,
            suggested_prompt,
            template_score,
            template_suggestions
        )

    @classmethod
    def _evaluate_with_phases(
        cls,
        source_image,
        output_image,
        prompt: str,
        llm_model: LLMModelConfig,
        source_metadata: Dict[str, Any] = None,
        analyze_template: bool = True,
        generate_improved: bool = True,
        print_to_console: bool = False,
    ) -> Dict[str, Any]:
        """
        Run intelligent evaluation with configurable phases.

        Phase 1: Deep Image Analysis (always runs)
        Phase 2: Template & Prompt Evaluation (if analyze_template=True)
        Phase 3: Generate Improvements (if generate_improved=True)

        Args:
            analyze_template: Run template effectiveness analysis
            generate_improved: Generate improved prompt
            print_to_console: Print detailed results to console

        Returns comprehensive evaluation based on enabled phases.
        """
        # Convert images to base64
        source_b64 = cls._image_to_base64(source_image)
        output_b64 = cls._image_to_base64(output_image)

        # Get provider info
        provider = llm_model.provider.lower()
        api_key = llm_model.api_key
        model = llm_model.model
        api_url = llm_model.api_url

        # Extract template info from source metadata
        template_info = None
        if source_metadata:
            gen_settings = source_metadata.get("generation_settings", {})
            template_info = gen_settings.get("template_info", {})

        if template_info:
            print(f"[SID-Debug] Template: {template_info.get('name', 'Unknown')} ({template_info.get('path', 'N/A')})")

        # Determine total phases for progress display
        total_phases = 1 + (1 if analyze_template else 0) + (1 if generate_improved else 0)
        current_phase = 0

        # === PHASE 1: Deep Image Analysis (always runs) ===
        current_phase += 1
        print(f"[SID-Debug] Phase {current_phase}/{total_phases}: Deep image analysis with reasoning...")

        phase1_system = """You are an expert visual analyst with deep reasoning capability.

Use your extended thinking to thoroughly analyze both images before responding.

Your task: Analyze the SOURCE image (first) and OUTPUT image (second) in detail.

REASONING PROCESS (think through this carefully):
1. Identify EVERY visible element in the SOURCE image
2. Note the exact framing, crop, and composition
3. Catalog all visual details: subject, clothing, pose, expression, lighting, background
4. Compare element-by-element with the OUTPUT image
5. Note: preservation quality, distortions, additions, omissions

After thorough analysis, respond with this JSON:
{
  "source_analysis": {
    "framing": "describe exact frame boundaries and crop",
    "subject": "detailed subject description",
    "clothing": ["item 1 with details", "item 2"],
    "pose_expression": "body angle, gaze direction, expression",
    "lighting": "lighting type, direction, quality",
    "background": "background description",
    "notable_details": ["specific detail 1", "specific detail 2"]
  },
  "output_analysis": {
    "framing": "how output framing compares",
    "subject_preservation": "how well subject was preserved",
    "differences": ["difference 1", "difference 2"],
    "distortions": ["any anatomical or visual distortions"],
    "quality_assessment": "overall quality notes"
  },
  "comparison": {
    "preserved_elements": ["elements that transferred well"],
    "lost_elements": ["elements missing in output"],
    "added_elements": ["elements not in source but in output"],
    "transformation_quality": "0-10 score with reason"
  }
}"""

        phase1_user = """Analyze both images thoroughly.
First image: SOURCE (the original that was analyzed to create a prompt)
Second image: OUTPUT (the result generated from that prompt)

Take your time to identify every detail before responding."""

        # Make Phase 1 call with reasoning
        phase1_result = cls._call_llm_with_reasoning(
            provider, api_key, api_url, model,
            phase1_system, phase1_user,
            source_b64, output_b64,
            reasoning_budget=8000
        )
        image_analysis = cls._parse_json_response(phase1_result)

        # Console output for Phase 1
        if print_to_console:
            print("\n" + "="*80)
            print("PHASE 1: DEEP IMAGE ANALYSIS")
            print("="*80)
            source_analysis = image_analysis.get("source_analysis", {})
            print(f"\n[SOURCE IMAGE]")
            print(f"  Framing: {source_analysis.get('framing', 'N/A')}")
            print(f"  Subject: {source_analysis.get('subject', 'N/A')}")
            print(f"  Clothing: {', '.join(source_analysis.get('clothing', []))}")
            print(f"  Pose/Expression: {source_analysis.get('pose_expression', 'N/A')}")
            print(f"  Lighting: {source_analysis.get('lighting', 'N/A')}")
            print(f"  Background: {source_analysis.get('background', 'N/A')}")
            if source_analysis.get('notable_details'):
                print(f"  Notable Details: {', '.join(source_analysis.get('notable_details', []))}")

            comparison = image_analysis.get("comparison", {})
            print(f"\n[COMPARISON]")
            print(f"  Transformation Quality: {comparison.get('transformation_quality', 'N/A')}")
            if comparison.get('preserved_elements'):
                print(f"  Preserved: {', '.join(comparison.get('preserved_elements', []))}")
            if comparison.get('lost_elements'):
                print(f"  Lost: {', '.join(comparison.get('lost_elements', []))}")
            if comparison.get('added_elements'):
                print(f"  Added: {', '.join(comparison.get('added_elements', []))}")

        # Initialize evaluation results
        evaluation = {}
        improvements = {}
        api_calls = 1

        # === PHASE 2: Template & Prompt Evaluation (if enabled) ===
        if analyze_template:
            current_phase += 1
            print(f"[SID-Debug] Phase {current_phase}/{total_phases}: Evaluating prompt and template effectiveness...")

            # Build template context
            template_context = ""
            if template_info and template_info.get("system_prompt"):
                template_context = f"""
TEMPLATE USED FOR GENERATION:
- Name: {template_info.get('name', 'Unknown')}
- Path: {template_info.get('path', 'N/A')}
- Category: {template_info.get('category', 'unknown')}
- Model Size Variant: {template_info.get('model_size_used', 'large')}

TEMPLATE SYSTEM PROMPT:
{template_info.get('system_prompt', 'Not available')[:2000]}

TEMPLATE USER PROMPT:
{template_info.get('user_prompt', 'Not available')[:500]}
"""
            else:
                template_context = "\nTEMPLATE: Not available (custom override or detailed mode used)"

            phase2_system = """You are an expert prompt and template evaluator with deep reasoning capability.

Use extended thinking to thoroughly evaluate both the generated prompt AND the template that produced it.

REASONING PROCESS:
1. Read the PROMPT that was generated
2. Compare it against your IMAGE ANALYSIS findings
3. Evaluate against Z-Image best practices
4. If TEMPLATE is provided, evaluate how well the template guided the generation
5. Identify specific gaps in both prompt AND template

Z-Image Best Practices for reference:
- Start with camera angle + shot type + subject
- Include pose (body angle separate from gaze)
- Include expression for humans
- Use specific visual vocabulary (skin tones, hair descriptors, lighting terms)
- Describe clothing with specificity and length markers
- End with "no text, no watermark"
- AVOID: abstract words (beautiful, elegant), meta tags (8K, masterpiece)

Respond with this JSON:
{
  "scores": {
    "source_alignment": {"score": 0, "reason": "how well prompt describes source"},
    "output_quality": {"score": 0, "reason": "how well output matched intent"},
    "structure": {"score": 0, "reason": "prompt organization quality"},
    "specificity": {"score": 0, "reason": "concrete vs vague vocabulary"},
    "overall": {"score": 0, "reason": "overall prompt quality"},
    "template_effectiveness": {"score": 0, "reason": "how well template guided generation"}
  },
  "prompt_analysis": {
    "strengths": ["what the prompt does well"],
    "issues": ["specific issues in the prompt"],
    "missing_elements": ["visible in source, not in prompt"],
    "hallucinated_elements": ["in prompt, not in source"],
    "vocabulary_issues": ["vague or problematic word choices"]
  },
  "template_analysis": {
    "template_strengths": ["what template does well"],
    "effectiveness_issues": ["gaps in template guidance"],
    "missing_instructions": ["guidance template should include"],
    "unclear_instructions": ["confusing or vague template parts"],
    "model_size_appropriateness": "is the template well-suited for this model size?"
  }
}"""

            phase2_user = f"""Evaluate this prompt against the source image analysis and template.

GENERATED PROMPT TO EVALUATE:
{prompt}

IMAGE ANALYSIS FINDINGS:
{json.dumps(image_analysis, indent=2)}
{template_context}

Score the prompt quality AND template effectiveness (0-10 scale).
Identify specific issues in both."""

            phase2_result = cls._call_llm_with_reasoning(
                provider, api_key, api_url, model,
                phase2_system, phase2_user,
                source_b64, output_b64,
                reasoning_budget=10000
            )
            evaluation = cls._parse_json_response(phase2_result)
            api_calls += 1

            # Console output for Phase 2
            if print_to_console:
                print("\n" + "="*80)
                print("PHASE 2: PROMPT & TEMPLATE EVALUATION")
                print("="*80)
                scores = evaluation.get("scores", {})
                print(f"\n[SCORES]")
                for key, val in scores.items():
                    score = val.get("score", 0) if isinstance(val, dict) else val
                    reason = val.get("reason", "") if isinstance(val, dict) else ""
                    print(f"  {key}: {score}/10 - {reason}")

                prompt_analysis = evaluation.get("prompt_analysis", {})
                print(f"\n[PROMPT ANALYSIS]")
                if prompt_analysis.get("strengths"):
                    print(f"  Strengths: {', '.join(prompt_analysis.get('strengths', []))}")
                if prompt_analysis.get("issues"):
                    print(f"  Issues: {', '.join(prompt_analysis.get('issues', []))}")
                if prompt_analysis.get("missing_elements"):
                    print(f"  Missing: {', '.join(prompt_analysis.get('missing_elements', []))}")

                template_analysis = evaluation.get("template_analysis", {})
                print(f"\n[TEMPLATE ANALYSIS]")
                if template_analysis.get("effectiveness_issues"):
                    print(f"  Issues: {', '.join(template_analysis.get('effectiveness_issues', []))}")
                if template_analysis.get("missing_instructions"):
                    print(f"  Missing: {', '.join(template_analysis.get('missing_instructions', []))}")

        # === PHASE 3: Generate Improvements (if enabled) ===
        improved_prompt = prompt  # Default to original
        if generate_improved:
            current_phase += 1
            print(f"[SID-Debug] Phase {current_phase}/{total_phases}: Generating improved prompt...")

            # Build context from Phase 2 results (or use image analysis if Phase 2 skipped)
            if analyze_template and evaluation:
                prompt_issues = evaluation.get("prompt_analysis", {}).get("issues", [])
                template_issues = evaluation.get("template_analysis", {}).get("effectiveness_issues", [])
                missing_elements = evaluation.get("prompt_analysis", {}).get("missing_elements", [])
            else:
                # Use image analysis comparison to identify issues
                comparison = image_analysis.get("comparison", {})
                prompt_issues = []
                template_issues = []
                missing_elements = comparison.get("lost_elements", [])

            issues_text = "\n".join(f"- {i}" for i in prompt_issues) if prompt_issues else "None identified"
            template_issues_text = "\n".join(f"- {i}" for i in template_issues) if template_issues else "None identified"
            missing_text = "\n".join(f"- {m}" for m in missing_elements) if missing_elements else "None identified"

            phase3_system = """You are an expert prompt writer with reasoning capability.

Use extended thinking to craft an IMPROVED PROMPT that addresses all identified issues.

For the IMPROVED PROMPT:
- ALWAYS start with camera angle + shot type + subject
- Include pose, expression, lighting with specific vocabulary
- Address all missing elements from the source image
- Remove any hallucinated elements
- Use Z-Image best practices
- 80-200 words, end with "no text, no watermark"

Respond with this JSON:
{
  "improved_prompt": "your complete rewritten prompt here",
  "changes_made": ["list of key changes from original"],
  "summary": "brief 1-2 sentence summary of improvements"
}"""

            phase3_user = f"""Generate an improved prompt based on the analysis.

ORIGINAL PROMPT:
{prompt}

ISSUES TO FIX:
{issues_text}

MISSING FROM SOURCE:
{missing_text}

SOURCE IMAGE ANALYSIS:
{json.dumps(image_analysis.get('source_analysis', {}), indent=2)}

Look at the SOURCE image and write a complete improved prompt."""

            phase3_result = cls._call_llm_with_reasoning(
                provider, api_key, api_url, model,
                phase3_system, phase3_user,
                source_b64, output_b64,
                reasoning_budget=8000
            )
            improvements = cls._parse_json_response(phase3_result)
            api_calls += 1

            # Clean up improved prompt
            improved_prompt = improvements.get("improved_prompt", prompt)
            if isinstance(improved_prompt, str):
                improved_prompt = improved_prompt.strip()
                if improved_prompt.startswith('"') and improved_prompt.endswith('"'):
                    improved_prompt = improved_prompt[1:-1]
                if improved_prompt.startswith("```"):
                    lines = improved_prompt.split("\n")
                    improved_prompt = "\n".join(line for line in lines if not line.startswith("```"))

            # Console output for Phase 3
            if print_to_console:
                print("\n" + "="*80)
                print("PHASE 3: IMPROVED PROMPT")
                print("="*80)
                print(f"\n{improved_prompt}")
                if improvements.get("changes_made"):
                    print(f"\n[CHANGES MADE]")
                    for change in improvements.get("changes_made", []):
                        print(f"  - {change}")

        # Build final evaluation based on which phases ran
        final_evaluation = {
            "image_analysis": image_analysis,
            "original_prompt": prompt,
            "evaluator": {
                "provider": provider,
                "model": model,
                "reasoning_enabled": True,
            },
            "api_calls": api_calls,
        }

        # Add Phase 2 results if it ran
        if analyze_template and evaluation:
            final_evaluation["scores"] = evaluation.get("scores", {})
            final_evaluation["prompt_analysis"] = evaluation.get("prompt_analysis", {})
            final_evaluation["template_analysis"] = {
                "template_used": {
                    "name": template_info.get("name") if template_info else None,
                    "path": template_info.get("path") if template_info else None,
                    "category": template_info.get("category") if template_info else None,
                    "model_size": template_info.get("model_size_used") if template_info else None,
                },
                **evaluation.get("template_analysis", {}),
            }

        # Add Phase 3 results if it ran
        if generate_improved:
            final_evaluation["improved_prompt"] = improved_prompt
            final_evaluation["template_suggestions"] = improvements.get("template_suggestions", {})
            final_evaluation["improvement_summary"] = improvements.get("summary", "")
        else:
            final_evaluation["improved_prompt"] = prompt

        # Console summary
        if print_to_console:
            print("\n" + "="*80)
            print("EVALUATION COMPLETE")
            print("="*80)
            print(f"  API Calls: {api_calls}")
            if analyze_template and evaluation:
                overall = evaluation.get("scores", {}).get("overall", {})
                score = overall.get("score", 0) if isinstance(overall, dict) else overall
                print(f"  Overall Score: {score}/10")
            print("")

        return final_evaluation

    @classmethod
    def _call_llm_with_reasoning(
        cls,
        provider: str,
        api_key: str,
        api_url: str,
        model: str,
        system: str,
        user: str,
        img1_b64: str,
        img2_b64: str,
        reasoning_budget: int = 8000
    ) -> str:
        """Call LLM with reasoning/extended thinking enabled."""
        if provider == "anthropic":
            return cls._call_anthropic_with_reasoning(api_key, model, system, user, img1_b64, img2_b64, reasoning_budget)
        elif provider in ["openai", "groq", "together", "fireworks", "openrouter", "gemini", "mistral", "deepseek", "cerebras", "xai"]:
            # OpenAI o1/o3 models have built-in reasoning
            return cls._call_openai_compatible_reasoning(api_key, api_url, model, system, user, img1_b64, img2_b64)
        elif provider in ["ollama", "lmstudio"]:
            return cls._call_local_api(api_url, model, system, user, img1_b64, img2_b64)
        elif provider == "local" and api_url:
            return cls._call_local_api(api_url, model, system, user, img1_b64, img2_b64)
        else:
            raise ValueError(f"Unsupported provider for reasoning debug: {provider}")

    @classmethod
    def _call_anthropic_with_reasoning(
        cls,
        api_key: str,
        model: str,
        system: str,
        user: str,
        img1_b64: str,
        img2_b64: str,
        reasoning_budget: int = 8000
    ) -> str:
        """Call Anthropic API with extended thinking enabled."""
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)

        # Use extended thinking for thorough analysis
        response = client.messages.create(
            model=model,
            max_tokens=16000,  # Higher for reasoning + output
            thinking={
                "type": "enabled",
                "budget_tokens": reasoning_budget
            },
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": f"System instructions: {system}\n\n{user}"},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img1_b64}},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img2_b64}},
                ]
            }]
        )

        # Extract the text response (not the thinking blocks)
        for block in response.content:
            if block.type == "text":
                return block.text

        return str(response.content)

    @classmethod
    def _call_openai_compatible_reasoning(
        cls,
        api_key: str,
        api_url: str,
        model: str,
        system: str,
        user: str,
        img1_b64: str,
        img2_b64: str
    ) -> str:
        """Call OpenAI-compatible API (o1/o3 have built-in reasoning)."""
        import openai

        client = openai.OpenAI(api_key=api_key, base_url=api_url if api_url else None)

        # o1/o3 models handle reasoning internally
        response = client.chat.completions.create(
            model=model,
            max_completion_tokens=8192,  # Higher limit for reasoning models
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
    def _save_results(cls, source_image, output_image, prompt: str, llm_model: LLMModelConfig, evaluation: Dict[str, Any], source_metadata: Dict[str, Any] = None):
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

        # Save source metadata and extract generation model config + template info
        generation_model_config = None
        template_info = None
        if source_metadata and isinstance(source_metadata, dict):
            # Save full source metadata
            source_meta_path = session_dir / "source_metadata.json"
            with open(source_meta_path, "w", encoding="utf-8") as f:
                json.dump(source_metadata, f, indent=2, ensure_ascii=False)
            # Extract generation model config from source metadata
            generation_model_config = source_metadata.get("model_config")
            # Extract template info from generation_settings
            gen_settings = source_metadata.get("generation_settings", {})
            template_info = gen_settings.get("template_info")
            print(f"[SID-Debug] Source metadata saved")

        # Save generation model config (from source_metadata, not llm_model which is for evaluation)
        config_path = session_dir / "generation_model.json"
        if generation_model_config:
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(generation_model_config, f, indent=2, ensure_ascii=False)
        else:
            # Fallback: no source metadata provided, note this in the file
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump({"note": "Source metadata not connected - generation model unknown"}, f, indent=2)

        # Save template info separately for easy access
        template_path = session_dir / "template_info.json"
        if template_info:
            with open(template_path, "w", encoding="utf-8") as f:
                json.dump(template_info, f, indent=2, ensure_ascii=False)
            print(f"[SID-Debug] Template info saved: {template_info.get('name', 'Unknown')}")
        else:
            with open(template_path, "w", encoding="utf-8") as f:
                json.dump({"note": "Template info not available (custom override or metadata not connected)"}, f, indent=2)

        # Save evaluation model config (the LLM used for debug evaluation)
        eval_model_path = session_dir / "evaluation_model.json"
        eval_model_config = cls._extract_model_config(llm_model)
        with open(eval_model_path, "w", encoding="utf-8") as f:
            json.dump(eval_model_config, f, indent=2)

        # Save source image
        try:
            source_pil = cls._tensor_to_pil(source_image)
            source_path = session_dir / "source.jpg"
            source_pil.save(source_path, "JPEG", quality=90)
        except Exception as e:
            print(f"[SID-Debug] Warning: Could not save source image: {e}")

        # Save output image
        try:
            output_pil = cls._tensor_to_pil(output_image)
            output_path = session_dir / "output.jpg"
            output_pil.save(output_path, "JPEG", quality=90)
        except Exception as e:
            print(f"[SID-Debug] Warning: Could not save output image: {e}")

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
    def _tensor_to_pil(cls, tensor):
        """Convert ComfyUI tensor to PIL Image."""
        from PIL import Image
        import numpy as np

        if tensor is None:
            return None

        # Handle batched tensor (B, H, W, C)
        if len(tensor.shape) == 4:
            tensor = tensor[0]

        # Convert to numpy and scale to 0-255
        np_image = (tensor.cpu().numpy() * 255).astype(np.uint8)

        return Image.fromarray(np_image)

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

    @classmethod
    def _extract_template_score(cls, evaluation: Dict[str, Any]) -> float:
        """Extract template effectiveness score from evaluation."""
        # Try different paths where template score might be stored
        score_paths = [
            ["scores", "template_effectiveness", "score"],
            ["scores", "template_effectiveness"],
            ["template_analysis", "score"],
            ["template_score"],
        ]

        for path in score_paths:
            current = evaluation
            try:
                for key in path:
                    current = current[key]
                if isinstance(current, (int, float)):
                    return float(current)
                if isinstance(current, dict) and "score" in current:
                    return float(current["score"])
            except (KeyError, TypeError):
                continue

        # Default score if not found
        return 5.0

    @classmethod
    def _format_template_suggestions(cls, evaluation: Dict[str, Any]) -> str:
        """Format template suggestions as human-readable text."""
        lines = []

        # Get template info
        template_analysis = evaluation.get("template_analysis", {})
        template_used = template_analysis.get("template_used", {})

        if template_used and template_used.get("name"):
            lines.append(f"Template: {template_used.get('name')} ({template_used.get('path', 'N/A')})")
            lines.append(f"Category: {template_used.get('category', 'unknown')}")
            lines.append(f"Model Size: {template_used.get('model_size', 'unknown')}")
            lines.append("")

        # Get suggestions from evaluation
        suggestions = evaluation.get("template_suggestions", {})

        # Priority additions
        priority_additions = suggestions.get("priority_additions", [])
        if priority_additions:
            lines.append("PRIORITY ADDITIONS:")
            for item in priority_additions[:5]:
                if isinstance(item, dict):
                    instruction = item.get("instruction", str(item))
                    reason = item.get("reason", "")
                    lines.append(f"  + {instruction}")
                    if reason:
                        lines.append(f"    Reason: {reason}")
                elif isinstance(item, str):
                    lines.append(f"  + {item}")
            lines.append("")

        # Clarifications needed
        clarifications = suggestions.get("clarifications_needed", [])
        if clarifications:
            lines.append("CLARIFICATIONS NEEDED:")
            for item in clarifications[:3]:
                if isinstance(item, dict):
                    current = item.get("current", "")
                    suggested = item.get("suggested", "")
                    lines.append(f"  Current: {current[:100]}...")
                    lines.append(f"  Suggested: {suggested[:100]}...")
                    lines.append("")
                elif isinstance(item, str):
                    lines.append(f"  - {item}")
            lines.append("")

        # Structure improvements
        structure = suggestions.get("structure_improvements", "")
        if structure:
            lines.append("STRUCTURE IMPROVEMENTS:")
            lines.append(f"  {structure[:300]}...")
            lines.append("")

        # Model size notes
        model_notes = suggestions.get("model_size_notes", "")
        if model_notes:
            lines.append("MODEL SIZE NOTES:")
            lines.append(f"  {model_notes[:200]}...")
            lines.append("")

        # Improvement summary
        summary = evaluation.get("improvement_summary", "")
        if summary:
            lines.append("SUMMARY:")
            lines.append(f"  {summary}")

        # Effectiveness issues from template_analysis
        effectiveness_issues = template_analysis.get("effectiveness_issues", [])
        if effectiveness_issues and not lines:
            lines.append("TEMPLATE EFFECTIVENESS ISSUES:")
            for issue in effectiveness_issues[:5]:
                lines.append(f"  - {issue}")

        if not lines:
            return "No template suggestions available (template info not provided or no issues found)"

        return "\n".join(lines)
