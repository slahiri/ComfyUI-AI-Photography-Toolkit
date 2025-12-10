# Changelog

All notable changes to ComfyUI-AI-Photography-Toolkit will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [4.1.0] - 2024-12-10

### Added
- **High-resolution GGUF models** - all local models now support 1024x1024+ images natively:
  - Qwen3-VL 2B Q4 (4GB VRAM) - fast, dynamic resolution
  - Qwen2.5-VL 7B Q4/Q8 (6-10GB VRAM) - excellent quality, dynamic resolution
  - Llama 3.2 Vision 11B Q4 (8GB VRAM) - Meta's latest, 1120x1120 native
  - Pixtral 12B Q4 (10GB VRAM) - Mistral's vision model, 1024x1024 native
  - MiniCPM-V 2.6 (10GB VRAM) - multilingual, dynamic resolution
- **Max Image Size option** for GGUF provider - resize images before encoding for faster processing:
  - 512 (default): ~4x faster encoding
  - 768: ~2x faster encoding
  - 1024: ~1.5x faster encoding
  - Original: no resize (full quality, slowest)
- **`max_image_resolution` field** in GGUF model metadata for documentation
- **Image caching infrastructure** - tracks image hashes for future optimization

### Changed
- **Removed max_tokens slider** from all LLM provider nodes (Anthropic, OpenAI, Grok, GGUF)
  - Now uses model's max_output_tokens from metadata directly
  - Eliminates the hardcoded 8192 cap that was limiting some models
- **Updated GGUF model list** - removed LLaVA 1.5/1.6 models (limited to 336x336/672x672 per tile)
- **Default GGUF model** changed from Moondream to Qwen2.5-VL 7B Q4
- **Simplified GGUF node** - removed quality_mode preset selector

### Removed
- LLaVA 1.5 7B, 13B models (can't handle 1024x1024 images)
- LLaVA 1.6 Mistral 7B, 34B models (can't handle 1024x1024 images)
- Moondream2 model (replaced with higher resolution alternatives)

## [4.0.1] - 2024-12-10

### Added
- **Read-only prompt display** on V2 generator node - shows generated prompt after execution
- **Max output tokens metadata** for all LLM providers with model-specific limits:
  - Anthropic: Claude Sonnet/Opus 4.x (64K), Claude 3.5 (8K)
  - OpenAI: GPT-4o (16K), o1 series (32K-100K)
  - Grok: grok-2-vision (32K), grok-vision-beta (8K)
- **Max tokens guardrail** - automatically caps user's max_tokens to model's limit
- **User prompt integration** into analysis pipeline (both agentic and iterative modes):
  - Component analysis now receives user requirements as priority instructions
  - Agentic mode includes "CRITICAL: USER'S SPECIAL REQUEST" section
  - Final prompt shows user focus with `[USER FOCUS: ...]` prefix
- **Logging for user requests** in both iterative and agentic pipelines

### Changed
- `_analyze_component()` now accepts optional `user_prompt` parameter
- `_build_agentic_prompt()` includes user requirements as high-priority analysis instructions
- `_assemble_prompt()` structures user requirements prominently at prompt start
- LLM provider nodes now store `model_max_output_tokens` in `extra_params`

### Technical
- Added `get_max_output_tokens()` class method to all LLM providers
- V2 generator validates max_tokens against model limits at execution time
- `is_output_node=True` added to V2 schema for UI text display

## [4.3.0] - 2024-12-09

### Added
- **High-resolution prompt generation** for near-accurate image reproduction from source
- **Detailed subject analysis** with ethnicity detection and heritage identification
- **Precise skin tone extraction** including undertones (warm/cool/neutral) and surface quality
- **Comprehensive facial feature analysis**:
  - Eyes: color, shape, gaze direction, expression, eyelid position
  - Eyebrows: shape, thickness, grooming, arch, color
  - Nose: shape, bridge, tip, nostril visibility, proportions
  - Lips: shape, upper/lower lip ratio, natural color, position
- **Detailed pose analysis** for all body parts:
  - Head and neck positioning with tilt/rotation
  - Shoulder alignment and arm positions
  - Hand positions with palm orientation and finger details
  - Torso, hip, and leg positioning
  - Feet position and weight distribution
- **Clothing coverage analysis**:
  - Neckline type and depth
  - Cleavage and sideboob visibility levels
  - Back and midriff exposure
  - Leg and thigh visibility
  - Gaps, slits, and fabric behavior
- **Cosmetics/makeup extraction**:
  - Foundation and skin coverage
  - Eye makeup (shadow, liner, mascara)
  - Lip color and finish
  - Blush and contouring
- **Structured JSON output** with categorized data sections:
  - metadata (generator, version, timestamp, processing time)
  - model (provider, model name, template, settings)
  - settings (detail level, content detail, focus options)
  - image (dimensions, aspect ratio, color info)
  - classification, subject, cosmetics, pose, clothing
  - environment, lighting, colors, style
  - generated_prompt with word/token counts
  - zimage_recommendations
- **LLM interaction tracking** - all raw queries and responses included in output for debugging
- **Token-optimized prompts** that respect max token limits efficiently

### Changed
- Enhanced Claude template with forensic-level image analysis prompts
- Enhanced Local GGUF template with multi-iteration Deep mode (6 focused analysis stages)
- Improved compound color descriptors for accuracy (e.g., "warm honey-golden blonde")

## [4.2.0] - 2024-11-XX

### Added
- High-quality GGUF models: LLaVA 1.6 34B, Qwen2.5-VL 7B, LLaVA 1.6 Mistral 7B
- Support for Qwen2-VL chat format

### Fixed
- Moondream2 download URLs (correct HuggingFace repo)
- GGUF provider API key validation error

## [4.1.0] - 2024-11-XX

### Added
- `SID_OpenAI_Compatible_LLM` node - supports OpenAI, Together AI, LM Studio, Ollama
- `SID_Grok_LLM` node - xAI Grok vision models
- `SID_GGUF_LLM` node - local GGUF models with auto-download
- Auto-install `llama-cpp-python` with GPU detection (NVIDIA CUDA, Apple Metal, AMD ROCm, CPU fallback)
- Split into Basic (`SID_ZImagePromptGenerator`) and Advanced (`SID_ZImagePromptGenerator_Advanced`) nodes
- Welcome message shows dependency installation status on load

## [4.0.0] - 2024-11-XX

### Added
- Initial Z-Image optimized prompt generator
- 6-stage agentic pipeline for comprehensive image analysis
- Anthropic Claude API support
- Persistent disk caching for repeated analysis
- Multiple detail levels: Quick (1 LLM call), Standard (2 calls), Deep (3 calls)
- Focus options: subject, environment, lighting, colors, mood
- Content detail levels: minimal, standard, detailed, explicit
- Prompt modes: Image Only, Prompt Guides, Prompt First, Prompt Dominates

### Technical
- SOLID principles architecture with template pattern
- BasePromptTemplate abstraction for provider-specific prompts
- Factory pattern for template selection
- ComfyUI V3 API compatibility
