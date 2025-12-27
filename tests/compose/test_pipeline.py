"""Test the unified composition pipeline with NLP and LLM modes."""

import json
import sys
from pathlib import Path
from types import ModuleType
import importlib.util


def setup_modules(compose_path: Path):
    """Setup module hierarchy for imports."""
    # Add project root to path
    project_root = compose_path.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Create package structure
    packages = [
        ("core", compose_path.parent),
        ("core.compose", compose_path),
        ("core.compose.tokenizer", compose_path / "tokenizer"),
        ("core.compose.classifier", compose_path / "classifier"),
        ("core.compose.assembler", compose_path / "assembler"),
        ("core.compose.validator", compose_path / "validator"),
    ]

    for name, path in packages:
        if name not in sys.modules:
            pkg = ModuleType(name)
            pkg.__path__ = [str(path)]
            sys.modules[name] = pkg


def load_module(name: str, path: Path):
    """Load a module from path."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    module.__package__ = ".".join(name.split(".")[:-1])
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    print("=" * 70)
    print("Unified Pipeline Test: NLP and LLM Modes")
    print("=" * 70)

    compose_path = Path(__file__).parent.parent.parent / "core" / "compose"
    setup_modules(compose_path)

    # Load all required modules in order and link to parent packages
    # Tokenizer
    base_mod = load_module("core.compose.tokenizer.base", compose_path / "tokenizer" / "base.py")
    load_module("core.compose.tokenizer.normalizer", compose_path / "tokenizer" / "normalizer.py")
    load_module("core.compose.tokenizer.tagger_extractor", compose_path / "tokenizer" / "tagger_extractor.py")
    load_module("core.compose.tokenizer.analyzer_extractor", compose_path / "tokenizer" / "analyzer_extractor.py")
    load_module("core.compose.tokenizer.caption_extractor", compose_path / "tokenizer" / "caption_extractor.py")
    tokenizer_init = load_module("core.compose.tokenizer.__init__", compose_path / "tokenizer" / "__init__.py")

    # Link tokenizer exports to package
    tokenizer_pkg = sys.modules["core.compose.tokenizer"]
    for attr in ["TokenBatch", "TokenType", "TokenSource", "ImageToken",
                 "extract_all_tagger_tokens", "extract_all_analyzer_tokens",
                 "extract_all_caption_tokens", "normalize_batch"]:
        if hasattr(tokenizer_init, attr):
            setattr(tokenizer_pkg, attr, getattr(tokenizer_init, attr))

    # Classifier
    load_module("core.compose.classifier.categories", compose_path / "classifier" / "categories.py")
    load_module("core.compose.classifier.base", compose_path / "classifier" / "base.py")
    load_module("core.compose.classifier.deterministic", compose_path / "classifier" / "deterministic.py")
    load_module("core.compose.classifier.dictionary", compose_path / "classifier" / "dictionary.py")
    classifier_init = load_module("core.compose.classifier.__init__", compose_path / "classifier" / "__init__.py")

    # Link classifier exports
    classifier_pkg = sys.modules["core.compose.classifier"]
    for attr in ["ClassifierConfig", "ClassifiedImage", "classify_batch",
                 "classify_batch_hybrid", "resolve_conflicts", "get_classification_stats"]:
        if hasattr(classifier_init, attr):
            setattr(classifier_pkg, attr, getattr(classifier_init, attr))

    # Assembler
    load_module("core.compose.assembler.base", compose_path / "assembler" / "base.py")
    load_module("core.compose.assembler.rules", compose_path / "assembler" / "rules.py")
    load_module("core.compose.assembler.standard", compose_path / "assembler" / "standard.py")
    assembler_init = load_module("core.compose.assembler.__init__", compose_path / "assembler" / "__init__.py")

    # Link assembler exports
    assembler_pkg = sys.modules["core.compose.assembler"]
    for attr in ["AssemblerConfig", "AssembledPrompt", "PromptStyle", "assemble_prompt"]:
        if hasattr(assembler_init, attr):
            setattr(assembler_pkg, attr, getattr(assembler_init, attr))

    # Validator
    load_module("core.compose.validator.base", compose_path / "validator" / "base.py")
    load_module("core.compose.validator.rules", compose_path / "validator" / "rules.py")
    load_module("core.compose.validator.standard", compose_path / "validator" / "standard.py")
    validator_init = load_module("core.compose.validator.__init__", compose_path / "validator" / "__init__.py")

    # Link validator exports
    validator_pkg = sys.modules["core.compose.validator"]
    for attr in ["ValidatorConfig", "ValidationResult", "validate_prompt"]:
        if hasattr(validator_init, attr):
            setattr(validator_pkg, attr, getattr(validator_init, attr))

    # Pipeline
    pipeline_module = load_module("core.compose.compose_pipeline", compose_path / "compose_pipeline.py")

    ComposePipeline = pipeline_module.ComposePipeline
    PipelineConfig = pipeline_module.PipelineConfig
    ComposeMode = pipeline_module.ComposeMode
    PromptStyle = pipeline_module.PromptStyle
    compose_nlp = pipeline_module.compose_nlp

    # Load test metadata
    sample_path = Path(__file__).parent / "sample_metadata.json"
    if not sample_path.exists():
        print(f"\n[SKIP] sample_metadata.json not found")
        return

    with open(sample_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    # Test NLP Mode
    print("\n" + "=" * 70)
    print("Test 1: NLP Mode (Rule-based)")
    print("=" * 70)

    config_nlp = PipelineConfig(
        mode=ComposeMode.NLP,
        prompt_style=PromptStyle.NATURAL,
        min_confidence=0.3,
    )
    pipeline_nlp = ComposePipeline(config_nlp)
    result_nlp = pipeline_nlp.compose(metadata)

    print(f"\nMode: {result_nlp.mode.value}")
    print(f"Words: {result_nlp.word_count}")
    print(f"Quality Score: {result_nlp.quality_score:.2f}")
    print(f"Tokens Extracted: {result_nlp.tokens_extracted}")
    print(f"Classification: {result_nlp.classification_stats.get('processed_percent', 0)}% processed")
    print(f"Validation Issues: {result_nlp.validation_issues}")
    print(f"Categories: {', '.join(result_nlp.categories_used)}")

    print(f"\nPrompt Preview:")
    prompt = result_nlp.prompt
    if len(prompt) > 300:
        print(f"  {prompt[:300]}...")
    else:
        print(f"  {prompt}")

    # Test NLP Mode with different styles
    print("\n" + "=" * 70)
    print("Test 2: NLP Mode with TAGS Style")
    print("=" * 70)

    config_tags = PipelineConfig(
        mode=ComposeMode.NLP,
        prompt_style=PromptStyle.TAGS,
    )
    pipeline_tags = ComposePipeline(config_tags)
    result_tags = pipeline_tags.compose(metadata)

    print(f"\nTAGS Style ({result_tags.word_count} words):")
    print(f"  {result_tags.prompt[:200]}..." if len(result_tags.prompt) > 200 else f"  {result_tags.prompt}")

    # Test convenience function
    print("\n" + "=" * 70)
    print("Test 3: Using compose_nlp() Convenience Function")
    print("=" * 70)

    result_conv = compose_nlp(metadata, prompt_style=PromptStyle.HYBRID)
    print(f"\nHYBRID Style ({result_conv.word_count} words):")
    print(f"  {result_conv.prompt[:200]}..." if len(result_conv.prompt) > 200 else f"  {result_conv.prompt}")

    # Note about LLM mode
    print("\n" + "=" * 70)
    print("Note: LLM Mode")
    print("=" * 70)
    print("""
LLM mode requires additional setup:
- Local: Qwen model will auto-download on first use
- Anthropic: Set llm_api_key in config
- OpenAI: Set llm_api_key in config
- Gemini: Set llm_api_key in config

Example:
    config = PipelineConfig(
        mode=ComposeMode.LLM,
        llm_provider="anthropic",
        llm_model="claude-3-haiku-20240307",
        llm_api_key="your-key",
    )
    result = ComposePipeline(config).compose(metadata)
""")

    print("\n" + "=" * 70)
    print("Pipeline Test Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
