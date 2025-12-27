"""Unified composition pipeline using the new tokenizer/classifier/assembler/validator modules.

This pipeline provides two modes:
1. NLP Mode: Rule-based processing (Phases 2-5)
   - Tokenization → Classification → Assembly → Validation
   - Fast, deterministic, no external dependencies

2. LLM Mode: AI-enhanced processing
   - Uses NLP pipeline as base
   - Enhances output with LLM for natural language quality
   - Supports local (Qwen) and cloud (Anthropic, OpenAI, Gemini) providers
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any

from PIL import Image

from .tokenizer import (
    TokenBatch,
    extract_all_tagger_tokens,
    extract_all_analyzer_tokens,
    extract_all_caption_tokens,
    extract_from_rag,
    normalize_batch,
)
from .classifier import (
    ClassifierConfig,
    ClassifiedImage,
    classify_batch,
    classify_batch_hybrid,
    resolve_conflicts,
    get_classification_stats,
)
from .assembler import (
    AssemblerConfig,
    AssembledPrompt,
    PromptStyle,
    assemble_prompt,
)
from .validator import (
    ValidatorConfig,
    ValidationResult,
    validate_prompt,
)


class ComposeMode(Enum):
    """Prompt composition mode."""
    NLP = "nlp"  # Rule-based (fast, deterministic)
    LLM = "llm"  # AI-enhanced (slower, better quality)


@dataclass
class PipelineConfig:
    """Configuration for the composition pipeline."""
    # Mode
    mode: ComposeMode = ComposeMode.NLP

    # Tokenizer settings
    min_confidence: float = 0.3
    split_captions: bool = False

    # Classifier settings
    use_embeddings: bool = False  # Layer 4 (requires sentence-transformers)
    use_llm_classifier: bool = False  # Layer 4.5 for classification

    # RAG settings
    use_rag: bool = False  # Visual RAG for vocabulary enhancement
    rag_threshold: float = 0.7  # Minimum similarity for RAG retrieval
    rag_top_k: int = 3  # Results per category

    # Assembler settings
    prompt_style: PromptStyle = PromptStyle.NATURAL
    max_tokens_per_category: int = 50  # High default to preserve information

    # Validator settings
    max_clip_tokens: int = 225
    auto_correct: bool = True

    # LLM settings (for LLM mode)
    llm_provider: str = "local"  # local, anthropic, openai, gemini
    llm_model: Optional[str] = None
    llm_api_key: Optional[str] = None
    llm_temperature: float = 0.7


@dataclass
class PipelineResult:
    """Result from the composition pipeline."""
    prompt: str
    mode: ComposeMode
    word_count: int = 0
    token_count: int = 0
    quality_score: float = 1.0
    categories_used: List[str] = field(default_factory=list)

    # Intermediate results
    tokens_extracted: int = 0
    tokens_classified: int = 0
    classification_stats: Dict[str, Any] = field(default_factory=dict)
    validation_issues: int = 0

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.word_count = len(self.prompt.split())


class ComposePipeline:
    """Unified composition pipeline with NLP and LLM modes.

    Usage:
        pipeline = ComposePipeline(config)
        result = pipeline.compose(metadata)
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        """Initialize the pipeline.

        Args:
            config: Pipeline configuration
        """
        self.config = config or PipelineConfig()
        self._llm_model = None

    def compose(
        self,
        metadata: Dict,
        image: Optional[Image.Image] = None,
    ) -> PipelineResult:
        """Compose a prompt from image metadata.

        Args:
            metadata: Image analysis metadata dict
            image: Optional PIL Image for RAG retrieval

        Returns:
            PipelineResult with the generated prompt
        """
        # Parse metadata if string
        if isinstance(metadata, str):
            import json
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                metadata = {}

        if self.config.mode == ComposeMode.NLP:
            return self._compose_nlp(metadata, image)
        else:
            return self._compose_llm(metadata, image)

    def _compose_nlp(
        self,
        metadata: Dict,
        image: Optional[Image.Image] = None,
    ) -> PipelineResult:
        """NLP mode: Rule-based composition pipeline."""

        # Phase 2: Tokenization
        token_batch = self._tokenize(metadata, image)

        # Phase 3: Classification
        classified = self._classify(token_batch)
        stats = get_classification_stats(classified)

        # Phase 4: Assembly
        assembled = self._assemble(classified)

        # Phase 5: Validation
        validated = self._validate(assembled)

        return PipelineResult(
            prompt=validated.prompt,
            mode=ComposeMode.NLP,
            quality_score=validated.quality_score,
            categories_used=assembled.categories_used,
            tokens_extracted=len(token_batch.tokens),
            tokens_classified=stats["total_tokens"],
            classification_stats=stats,
            validation_issues=len(validated.issues),
            metadata={
                "processed_percent": stats.get("processed_percent", 0),
                "corrections_made": validated.corrections_made,
            }
        )

    def _compose_llm(
        self,
        metadata: Dict,
        image: Optional[Image.Image] = None,
    ) -> PipelineResult:
        """LLM mode: AI-enhanced composition."""

        # First run NLP pipeline to get structured data
        token_batch = self._tokenize(metadata, image)
        classified = self._classify(token_batch)
        stats = get_classification_stats(classified)
        assembled = self._assemble(classified)

        # Build context for LLM from classified tokens
        context = self._build_llm_context(classified, metadata)

        # Generate enhanced prompt with LLM
        enhanced_prompt = self._generate_with_llm(context, assembled.prompt)

        # Validate the enhanced prompt
        from .assembler import AssembledPrompt
        enhanced_assembled = AssembledPrompt(
            prompt=enhanced_prompt,
            style=self.config.prompt_style,
            sections=assembled.sections,
        )
        validated = self._validate(enhanced_assembled)

        return PipelineResult(
            prompt=validated.prompt,
            mode=ComposeMode.LLM,
            quality_score=validated.quality_score,
            categories_used=assembled.categories_used,
            tokens_extracted=len(token_batch.tokens),
            tokens_classified=stats["total_tokens"],
            classification_stats=stats,
            validation_issues=len(validated.issues),
            metadata={
                "processed_percent": stats.get("processed_percent", 0),
                "llm_provider": self.config.llm_provider,
                "llm_model": self.config.llm_model,
                "base_prompt_words": assembled.word_count,
            }
        )

    def _tokenize(
        self,
        metadata: Dict,
        image: Optional[Image.Image] = None,
    ) -> TokenBatch:
        """Phase 2: Extract and normalize tokens."""
        # Extract from all sources
        tagger_batch = extract_all_tagger_tokens(metadata, min_confidence=0.0)
        analyzer_batch = extract_all_analyzer_tokens(metadata, min_confidence=0.0)
        caption_batch = extract_all_caption_tokens(
            metadata, split_into_clauses=self.config.split_captions
        )

        # Combine all tokens
        combined = TokenBatch()
        combined.image_info = metadata.get("image_info", {})
        combined.add_all(tagger_batch.tokens)
        combined.add_all(analyzer_batch.tokens)
        combined.add_all(caption_batch.tokens)

        # Extract from RAG if enabled and image provided
        if self.config.use_rag and image is not None:
            rag_tokens = extract_from_rag(
                image=image,
                top_k=self.config.rag_top_k,
                threshold=self.config.rag_threshold,
            )
            combined.add_all(rag_tokens)

        # Normalize
        normalized = normalize_batch(
            combined,
            min_confidence=self.config.min_confidence,
            deduplicate=True
        )

        return normalized

    def _classify(self, batch: TokenBatch) -> ClassifiedImage:
        """Phase 3: Classify tokens into categories."""
        if self.config.use_embeddings or self.config.use_llm_classifier:
            # Use hybrid classifier
            classifier_config = ClassifierConfig(
                use_embeddings=self.config.use_embeddings,
                use_llm=self.config.use_llm_classifier,
            )
            classified = classify_batch_hybrid(batch, classifier_config)
        else:
            # Use fast dictionary-only classifier
            classified = classify_batch(batch)

        # Resolve conflicts
        classified = resolve_conflicts(classified)

        return classified

    def _assemble(self, classified: ClassifiedImage) -> AssembledPrompt:
        """Phase 4: Assemble tokens into prompt."""
        config = AssemblerConfig(
            style=self.config.prompt_style,
            max_tokens_per_category=self.config.max_tokens_per_category,
        )
        return assemble_prompt(classified, config)

    def _validate(self, assembled: AssembledPrompt) -> ValidationResult:
        """Phase 5: Validate and correct prompt."""
        config = ValidatorConfig(
            max_tokens=self.config.max_clip_tokens,
            auto_correct=self.config.auto_correct,
        )
        return validate_prompt(assembled, config)

    def _build_llm_context(self, classified: ClassifiedImage, metadata: Dict) -> str:
        """Build context string for LLM from classified tokens."""
        from .classifier.categories import CanonicalCategory, CATEGORY_ORDER

        parts = []

        # Add tokens by category
        for category in CATEGORY_ORDER:
            tokens = classified.get_category(category)
            if tokens:
                cat_name = category.value.replace("_", " ").title()
                token_texts = [t.token.text for t in tokens[:8]]
                parts.append(f"{cat_name}: {', '.join(token_texts)}")

        # Add image info
        info = metadata.get("image_info", {})
        if info:
            parts.append(f"Image: {info.get('width', 0)}x{info.get('height', 0)}")

        return "\n".join(parts)

    def _generate_with_llm(self, context: str, base_prompt: str) -> str:
        """Generate enhanced prompt using LLM."""
        system_prompt = """You are an expert prompt engineer for image generation.
Your task is to enhance the given prompt to be more natural and descriptive.

Guidelines:
- Keep all the important details from the original
- Make the language flow naturally
- Use vivid, specific descriptions
- Maintain the same overall structure and meaning
- Output ONLY the enhanced prompt, no explanations"""

        user_prompt = f"""Enhance this image generation prompt:

Context (classified tokens):
{context}

Base prompt:
{base_prompt}

Enhanced prompt:"""

        provider = self.config.llm_provider

        if provider == "local":
            return self._generate_local(system_prompt, user_prompt)
        elif provider == "anthropic":
            return self._generate_anthropic(system_prompt, user_prompt)
        elif provider == "openai":
            return self._generate_openai(system_prompt, user_prompt)
        elif provider == "gemini":
            return self._generate_gemini(system_prompt, user_prompt)
        else:
            # Fallback to base prompt
            return base_prompt

    def _generate_local(self, system_prompt: str, user_prompt: str) -> str:
        """Generate using local LLM."""
        from ..models.text_llm import TextLLMModel

        model_name = self.config.llm_model or "qwen25_text_1.5b"

        if self._llm_model is None:
            self._llm_model = TextLLMModel(
                config_name=model_name,
                precision="4bit",
            )

        return self._llm_model.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            max_tokens=512,
            temperature=self.config.llm_temperature,
        )

    def _generate_anthropic(self, system_prompt: str, user_prompt: str) -> str:
        """Generate using Anthropic Claude."""
        if not self.config.llm_api_key:
            raise ValueError("Anthropic API key required")

        import anthropic

        client = anthropic.Anthropic(api_key=self.config.llm_api_key)
        model = self.config.llm_model or "claude-3-haiku-20240307"

        message = client.messages.create(
            model=model,
            max_tokens=512,
            temperature=self.config.llm_temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )

        return message.content[0].text.strip()

    def _generate_openai(self, system_prompt: str, user_prompt: str) -> str:
        """Generate using OpenAI."""
        if not self.config.llm_api_key:
            raise ValueError("OpenAI API key required")

        import openai

        client = openai.OpenAI(api_key=self.config.llm_api_key)
        model = self.config.llm_model or "gpt-4o-mini"

        response = client.chat.completions.create(
            model=model,
            max_tokens=512,
            temperature=self.config.llm_temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        return response.choices[0].message.content.strip()

    def _generate_gemini(self, system_prompt: str, user_prompt: str) -> str:
        """Generate using Google Gemini."""
        if not self.config.llm_api_key:
            raise ValueError("Gemini API key required")

        import google.generativeai as genai

        genai.configure(api_key=self.config.llm_api_key)
        model_name = self.config.llm_model or "gemini-1.5-flash"

        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_prompt,
        )

        response = model.generate_content(
            user_prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=512,
                temperature=self.config.llm_temperature,
            ),
        )

        return response.text.strip()

    def unload(self):
        """Unload any loaded models."""
        if self._llm_model is not None:
            self._llm_model.unload()
            self._llm_model = None

        # Unload RAG retriever if used
        if self.config.use_rag:
            try:
                from .rag import unload_rag_retriever
                unload_rag_retriever()
            except ImportError:
                pass


# Convenience functions
def compose_nlp(metadata: Dict, **kwargs) -> PipelineResult:
    """Compose prompt using NLP mode.

    Args:
        metadata: Image analysis metadata
        **kwargs: Additional PipelineConfig options

    Returns:
        PipelineResult
    """
    config = PipelineConfig(mode=ComposeMode.NLP, **kwargs)
    pipeline = ComposePipeline(config)
    return pipeline.compose(metadata)


def compose_llm(
    metadata: Dict,
    provider: str = "local",
    model: str = None,
    api_key: str = None,
    **kwargs
) -> PipelineResult:
    """Compose prompt using LLM mode.

    Args:
        metadata: Image analysis metadata
        provider: LLM provider (local, anthropic, openai, gemini)
        model: Model name
        api_key: API key for cloud providers
        **kwargs: Additional PipelineConfig options

    Returns:
        PipelineResult
    """
    config = PipelineConfig(
        mode=ComposeMode.LLM,
        llm_provider=provider,
        llm_model=model,
        llm_api_key=api_key,
        **kwargs
    )
    pipeline = ComposePipeline(config)
    return pipeline.compose(metadata)
