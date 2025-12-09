# Changelog

All notable changes to ComfyUI-AI-Photography-Toolkit will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
