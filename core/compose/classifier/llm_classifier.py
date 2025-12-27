"""LLM-based classification (Layer 4.5).

Layer 4.5: Uses a small LLM to classify tokens that couldn't be
classified by dictionary or embedding methods. Batches multiple
tokens for efficiency.
"""

import json
from typing import Optional, Dict, List, Tuple

from .categories import CanonicalCategory, SubjectDetailType
from .base import TokenClassification
from ..tokenizer.base import ImageToken


# =============================================================================
# Classification Prompt Templates
# =============================================================================

SYSTEM_PROMPT = """You are an image token classifier. Your task is to classify image description tokens into predefined categories.

Categories:
1. QUALITY_BOOSTERS - Image quality terms (high quality, detailed, sharp, 8k, masterpiece)
2. SUBJECT - Main subject (person, animal, vehicle, object, "no humans")
3. SUBJECT_DETAILS - Subject attributes (hair, eyes, face, body, clothing, skin, accessories)
4. ACTION_POSE - Actions and poses (standing, sitting, looking at viewer, holding)
5. ENVIRONMENT - Setting and background (indoor, outdoor, beach, studio, building)
6. LIGHTING - Light and color (soft light, shadows, warm tones, high contrast)
7. STYLE_MEDIUM - Art style (photograph, digital art, anime, realistic)
8. COMPOSITION - Camera and framing (close up, portrait, low angle, centered)
9. TECHNICAL - Technical aspects (blur, noise, watermark, resolution, nsfw)
10. META - Should be filtered (meta-commentary, parsing artifacts, nonsense)

For SUBJECT_DETAILS, also specify subcategory: hair, eyes, face, body, clothing, skin, accessories

Respond with JSON only. No explanations."""

USER_PROMPT_TEMPLATE = """Classify these image tokens into categories:

{tokens}

Respond with a JSON object where keys are the token texts and values are objects with "category" and optional "subcategory":

Example response:
{{
  "brown hair": {{"category": "SUBJECT_DETAILS", "subcategory": "hair"}},
  "standing": {{"category": "ACTION_POSE"}},
  "watermark": {{"category": "TECHNICAL"}},
  "the image shows": {{"category": "META"}}
}}"""


class LLMClassifier:
    """Layer 4.5: LLM-based token classifier.

    Uses a small LLM to classify tokens that couldn't be classified
    by faster methods. Batches tokens for efficiency.
    """

    def __init__(
        self,
        provider: str = "local",
        model: str = None,
        api_key: str = None,
        max_batch_size: int = 20,
    ):
        """Initialize the LLM classifier.

        Args:
            provider: "local", "anthropic", "openai", or "gemini"
            model: Model name/ID (e.g., "qwen25_text_0.5b", "gpt-4o-mini")
            api_key: API key for cloud providers
            max_batch_size: Maximum tokens to classify in one LLM call
        """
        self.provider = provider
        self.model = model or self._get_default_model(provider)
        self.api_key = api_key
        self.max_batch_size = max_batch_size
        self._local_model = None

    def _get_default_model(self, provider: str) -> str:
        """Get default model for provider."""
        defaults = {
            "local": "qwen25_text_0.5b",  # Small, fast local model
            "anthropic": "claude-3-haiku-20240307",
            "openai": "gpt-4o-mini",
            "gemini": "gemini-1.5-flash",
        }
        return defaults.get(provider, "qwen25_text_0.5b")

    def classify_batch(
        self, tokens: List[ImageToken]
    ) -> List[Optional[TokenClassification]]:
        """Classify a batch of tokens using LLM.

        Args:
            tokens: List of tokens to classify

        Returns:
            List of classifications (None for tokens that couldn't be classified)
        """
        if not tokens:
            return []

        # Split into batches if needed
        results = []
        for i in range(0, len(tokens), self.max_batch_size):
            batch = tokens[i : i + self.max_batch_size]
            batch_results = self._classify_single_batch(batch)
            results.extend(batch_results)

        return results

    def _classify_single_batch(
        self, tokens: List[ImageToken]
    ) -> List[Optional[TokenClassification]]:
        """Classify a single batch of tokens."""
        # Build token list for prompt
        token_texts = [t.text for t in tokens]
        tokens_str = "\n".join(f"- {t}" for t in token_texts)

        user_prompt = USER_PROMPT_TEMPLATE.format(tokens=tokens_str)

        try:
            # Generate classification
            response = self._generate(user_prompt)

            # Parse response
            classifications = self._parse_response(response, tokens)
            return classifications

        except Exception as e:
            print(f"Warning: LLM classification failed: {e}")
            return [None] * len(tokens)

    def _generate(self, prompt: str) -> str:
        """Generate response from LLM."""
        if self.provider == "local":
            return self._generate_local(prompt)
        elif self.provider == "anthropic":
            return self._generate_anthropic(prompt)
        elif self.provider == "openai":
            return self._generate_openai(prompt)
        elif self.provider == "gemini":
            return self._generate_gemini(prompt)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _generate_local(self, prompt: str) -> str:
        """Generate using local LLM."""
        from ...models.text_llm import TextLLMModel

        if self._local_model is None:
            self._local_model = TextLLMModel(
                config_name=self.model,
                precision="4bit",
            )

        return self._local_model.generate(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT,
            max_tokens=1024,
            temperature=0.1,  # Low temperature for consistent classification
        )

    def _generate_anthropic(self, prompt: str) -> str:
        """Generate using Anthropic Claude."""
        if not self.api_key:
            raise ValueError("Anthropic API key required")

        import anthropic

        client = anthropic.Anthropic(api_key=self.api_key)
        message = client.messages.create(
            model=self.model,
            max_tokens=1024,
            temperature=0.1,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text

    def _generate_openai(self, prompt: str) -> str:
        """Generate using OpenAI."""
        if not self.api_key:
            raise ValueError("OpenAI API key required")

        import openai

        client = openai.OpenAI(api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model,
            max_tokens=1024,
            temperature=0.1,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
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
            system_instruction=SYSTEM_PROMPT,
        )
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=1024,
                temperature=0.1,
            ),
        )
        return response.text

    def _parse_response(
        self, response: str, tokens: List[ImageToken]
    ) -> List[Optional[TokenClassification]]:
        """Parse LLM response into classifications."""
        # Try to extract JSON from response
        try:
            # Clean up response - remove markdown code blocks
            response = response.strip()
            if response.startswith("```"):
                lines = response.split("\n")
                response = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
            if response.startswith("json"):
                response = response[4:].strip()

            data = json.loads(response)
        except json.JSONDecodeError:
            # Try to find JSON object in response
            import re
            match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except:
                    return [None] * len(tokens)
            else:
                return [None] * len(tokens)

        # Map responses back to tokens
        results = []
        for token in tokens:
            token_text = token.text
            classification_data = data.get(token_text)

            if not classification_data:
                # Try case-insensitive match
                for key, value in data.items():
                    if key.lower() == token_text.lower():
                        classification_data = value
                        break

            if not classification_data:
                results.append(None)
                continue

            # Parse category
            category_str = classification_data.get("category", "").upper()
            category = self._parse_category(category_str)

            if category is None or category == CanonicalCategory.FILTERED_META:
                # META means filter it
                results.append(
                    TokenClassification(
                        token=token,
                        primary_category=CanonicalCategory.FILTERED_META,
                        confidence=0.8,
                        classifier_layer=4.5,
                        metadata={"classification_method": "llm", "llm_category": "META"},
                    )
                )
                continue

            # Parse subcategory if present
            subcategory = None
            if category == CanonicalCategory.SUBJECT_DETAILS:
                subcat_str = classification_data.get("subcategory", "")
                subcategory = self._parse_subcategory(subcat_str)

            results.append(
                TokenClassification(
                    token=token,
                    primary_category=category,
                    subcategory=subcategory,
                    confidence=0.85,  # LLM classifications have good confidence
                    classifier_layer=4.5,
                    metadata={"classification_method": "llm"},
                )
            )

        return results

    def _parse_category(self, category_str: str) -> Optional[CanonicalCategory]:
        """Parse category string to enum."""
        category_map = {
            "QUALITY_BOOSTERS": CanonicalCategory.QUALITY_BOOSTERS,
            "QUALITY": CanonicalCategory.QUALITY_BOOSTERS,
            "SUBJECT": CanonicalCategory.SUBJECT,
            "SUBJECT_DETAILS": CanonicalCategory.SUBJECT_DETAILS,
            "ACTION_POSE": CanonicalCategory.ACTION_POSE,
            "ACTION": CanonicalCategory.ACTION_POSE,
            "POSE": CanonicalCategory.ACTION_POSE,
            "ENVIRONMENT": CanonicalCategory.ENVIRONMENT,
            "LIGHTING": CanonicalCategory.LIGHTING,
            "STYLE_MEDIUM": CanonicalCategory.STYLE_MEDIUM,
            "STYLE": CanonicalCategory.STYLE_MEDIUM,
            "COMPOSITION": CanonicalCategory.COMPOSITION,
            "TECHNICAL": CanonicalCategory.TECHNICAL,
            "META": CanonicalCategory.FILTERED_META,
            "FILTERED": CanonicalCategory.FILTERED_META,
        }
        return category_map.get(category_str.upper().replace(" ", "_"))

    def _parse_subcategory(self, subcat_str: str) -> Optional[SubjectDetailType]:
        """Parse subcategory string to enum."""
        subcat_map = {
            "HAIR": SubjectDetailType.HAIR,
            "EYES": SubjectDetailType.EYES,
            "FACE": SubjectDetailType.FACE,
            "BODY": SubjectDetailType.BODY,
            "CLOTHING": SubjectDetailType.CLOTHING,
            "CLOTHES": SubjectDetailType.CLOTHING,
            "SKIN": SubjectDetailType.SKIN,
            "ACCESSORIES": SubjectDetailType.ACCESSORIES,
            "ACCESSORY": SubjectDetailType.ACCESSORIES,
        }
        return subcat_map.get(subcat_str.upper())

    def unload(self):
        """Unload local model if loaded."""
        if self._local_model is not None:
            self._local_model.unload()
            self._local_model = None


# Global instance (lazy initialized)
_llm_classifier: Optional[LLMClassifier] = None


def get_llm_classifier(
    provider: str = "local",
    model: str = None,
    api_key: str = None,
) -> LLMClassifier:
    """Get or create the global LLM classifier instance."""
    global _llm_classifier
    if _llm_classifier is None:
        _llm_classifier = LLMClassifier(provider=provider, model=model, api_key=api_key)
    return _llm_classifier


def classify_batch_by_llm(
    tokens: List[ImageToken],
    provider: str = "local",
    model: str = None,
    api_key: str = None,
) -> List[Optional[TokenClassification]]:
    """Layer 4.5: Classify tokens using LLM.

    Args:
        tokens: List of tokens to classify
        provider: LLM provider ("local", "anthropic", "openai", "gemini")
        model: Model name/ID
        api_key: API key for cloud providers

    Returns:
        List of classifications
    """
    classifier = LLMClassifier(provider=provider, model=model, api_key=api_key)
    return classifier.classify_batch(tokens)
