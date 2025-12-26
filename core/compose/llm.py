"""LLM-based prompt composition using various providers."""

import json
from typing import Dict, List, Optional, Any

from .base import (
    BaseComposeGenerator, ComposeConfig, ComposeResult, ComposeMode,
    ComposeAnalytics
)
from ..log import log, log_start, log_end, log_error


class LLMGenerator(BaseComposeGenerator):
    """
    LLM-based prompt generator using various providers.

    Supports local models (Qwen) and cloud APIs (Anthropic, OpenAI, Gemini).
    Best quality output but slower (500ms - 3s).
    Non-deterministic (temperature-based variation).
    """

    def __init__(self, style: str = "z-image", config: Optional[ComposeConfig] = None,
                 caption_source: str = "Auto", provider: str = "local", model: str = None,
                 api_key: str = None, hf_token: str = None):
        super().__init__(style, config, caption_source)
        self.provider = provider
        self.model = model  # model_id passed directly from node
        self.api_key = api_key
        self.hf_token = hf_token

        # Local model instance (lazy loaded)
        self._local_model = None

    def generate(self, metadata: Dict) -> ComposeResult:
        """Generate prompt using LLM."""
        analytics = ComposeAnalytics()

        # Extract tags for context
        tags, tag_sources = self.extract_tags_with_tracking(metadata, analytics)

        # Mark all tags as included (LLM sees everything)
        for t in analytics.all_tags:
            if t.status == "pending":
                t.status = "included"
                t.category = "llm_context"

        analytics.florence_used = bool(self.get_florence_caption(metadata))
        if analytics.florence_used:
            analytics.florence_sections_used = ["full_caption"]

        # Build LLM prompt
        llm_prompt = self._build_llm_prompt(metadata, tags)

        # Generate based on provider
        start = log_start("Compose", f"LLM generation ({self.provider}/{self.model})")

        try:
            if self.provider == "local":
                prompt = self._generate_local(llm_prompt, metadata)
            elif self.provider == "anthropic":
                prompt = self._generate_anthropic(llm_prompt)
            elif self.provider == "openai":
                prompt = self._generate_openai(llm_prompt)
            elif self.provider == "gemini":
                prompt = self._generate_gemini(llm_prompt)
            else:
                raise ValueError(f"Unknown provider: {self.provider}")

            log_end("Compose", "LLM generation complete", start)

        except Exception as e:
            log_error("Compose", f"LLM generation failed: {e}")
            # Fallback to Standard mode
            from .standard import StandardGenerator
            fallback = StandardGenerator(self.style_config.style_name, self.config)
            result = fallback.generate(metadata)
            result.metadata["llm_error"] = str(e)
            result.metadata["fallback"] = True
            return result

        # Clean up prompt
        prompt = self._clean_llm_output(prompt)

        # Finalize analytics
        analytics.finalize()

        return ComposeResult(
            prompt=prompt,
            mode=ComposeMode.ENHANCE_AI,
            style=self.style_config.style_name,
            categories_used=["llm_generated"],
            analytics=analytics,
            metadata={"provider": self.provider, "model": self.model},
        )

    def _build_llm_prompt(self, metadata: Dict, tags: Dict[str, float]) -> str:
        """Build the prompt to send to LLM."""
        # Get top tags with confidence
        sorted_tags = sorted(tags.items(), key=lambda x: x[1], reverse=True)[:30]
        tags_str = ', '.join([f"{t[0]} ({t[1]:.2f})" for t in sorted_tags])

        # Get caption based on caption_source setting
        caption = self._get_caption_for_llm(metadata)

        # Get photography info
        photo = self.get_photography_info(metadata)
        photo_str = ', '.join([f"{k}: {v:.0%}" for k, v in photo.items() if isinstance(v, (int, float)) and v > 0.5][:8])

        # Get composition info
        comp = self.get_composition_info(metadata)
        comp_str = ', '.join([f"{k}: {v:.0%}" for k, v in comp.items() if isinstance(v, (int, float)) and v > 0.5][:5])

        # Get image dimensions
        info = self.get_image_info(metadata)
        width = info.get('width', 0)
        height = info.get('height', 0)

        # Fill in template
        user_prompt = self.style_config.llm_user_prompt.format(
            width=width,
            height=height,
            tags=tags_str or "No tags available",
            florence=caption or "Not available",
            photography=photo_str or "Not available",
            composition=comp_str or "Not available",
        )

        return user_prompt

    def _get_caption_for_llm(self, metadata: Dict) -> str:
        """Get caption based on caption_source setting, with philosophical filtering."""
        source = self.caption_source

        # Tags Only - no caption
        if source == "Tags Only":
            return ''

        # Synthesis (VLM)
        if source == "Synthesis (VLM)":
            vlm_desc = metadata.get('vlm_description', '')
            if vlm_desc:
                return self.filter_philosophical(vlm_desc)
            return ''

        # Analysis (Florence)
        if source == "Analysis (Florence)":
            return self.filter_philosophical(self._get_florence_caption(metadata))

        # Auto - include both if available
        parts = []
        vlm_desc = metadata.get('vlm_description', '')
        florence = self._get_florence_caption(metadata)

        if vlm_desc:
            parts.append(f"VLM Description:\n{self.filter_philosophical(vlm_desc)}")
        if florence:
            parts.append(f"Florence Caption:\n{self.filter_philosophical(florence)}")

        return '\n\n'.join(parts) if parts else ''

    def _get_florence_caption(self, metadata: Dict) -> str:
        """Get best Florence caption."""
        florence_desc = metadata.get('florence_description', '')
        florence_caption = metadata.get('florence_caption', '')

        if florence_desc and len(florence_desc) > 50:
            return florence_desc
        if florence_caption:
            return florence_caption
        return ''

    def _generate_local(self, prompt: str, metadata: Dict) -> str:
        """Generate using local text-only LLM (not VLM)."""
        from ..models.text_llm import TextLLMModel

        # model is already the config_name (e.g., "qwen25_text_3b")
        model_id = self.model

        # Lazy load model
        if self._local_model is None:
            # Set HF token if provided (before loading)
            if self.hf_token:
                from ..download import set_hf_token
                set_hf_token(self.hf_token)

            self._local_model = TextLLMModel(
                config_name=model_id,
                precision="4bit",
            )

        # Generate with system prompt (no artificial output limit)
        result = self._local_model.generate(
            prompt=prompt,
            system_prompt=self.style_config.llm_system_prompt,
            max_tokens=4096,
            temperature=self.config.temperature,
        )

        return result

    def _generate_anthropic(self, prompt: str) -> str:
        """Generate using Anthropic Claude."""
        if not self.api_key:
            raise ValueError("Anthropic API key required")

        import anthropic

        client = anthropic.Anthropic(api_key=self.api_key)

        message = client.messages.create(
            model=self.model,
            max_tokens=4096,
            temperature=self.config.temperature,
            system=self.style_config.llm_system_prompt,
            messages=[{"role": "user", "content": prompt}],
        )

        return message.content[0].text

    def _generate_openai(self, prompt: str) -> str:
        """Generate using OpenAI GPT."""
        if not self.api_key:
            raise ValueError("OpenAI API key required")

        import openai

        client = openai.OpenAI(api_key=self.api_key)

        response = client.chat.completions.create(
            model=self.model,
            max_tokens=4096,
            temperature=self.config.temperature,
            messages=[
                {"role": "system", "content": self.style_config.llm_system_prompt},
                {"role": "user", "content": prompt},
            ],
        )

        return response.choices[0].message.content

    def _generate_gemini(self, prompt: str) -> str:
        """Generate using Google Gemini."""
        if not self.api_key:
            raise ValueError("Gemini API key required")

        import google.generativeai as genai

        genai.configure(api_key=self.api_key)

        model = genai.GenerativeModel(
            model_name=self.model,
            system_instruction=self.style_config.llm_system_prompt,
        )

        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=4096,
                temperature=self.config.temperature,
            ),
        )

        return response.text

    def _clean_llm_output(self, text: str) -> str:
        """Clean up LLM output."""
        # Remove common artifacts
        text = text.strip()

        # Remove markdown code blocks if present
        if text.startswith("```"):
            lines = text.split('\n')
            if len(lines) > 2:
                text = '\n'.join(lines[1:-1])

        # Remove quotes if the entire text is quoted
        if (text.startswith('"') and text.endswith('"')) or \
           (text.startswith("'") and text.endswith("'")):
            text = text[1:-1]

        # Remove "Here is..." preambles
        prefixes = [
            "Here is the prompt:",
            "Here's the prompt:",
            "Here is a prompt:",
            "Generated prompt:",
            "Z-Image prompt:",
        ]
        for prefix in prefixes:
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip()

        return text

    def unload(self):
        """Unload local model if loaded."""
        if self._local_model is not None:
            self._local_model.unload()
            self._local_model = None
