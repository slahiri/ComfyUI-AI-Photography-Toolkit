# ComfyUI-AI-Photography-Toolkit

AI-powered Z-Image prompt generator for ComfyUI. Analyzes images and generates flowing narrative prompts optimized for Z-Image models.

**Version:** 4.1.0
**Author:** Siddhartha Lahiri

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/slahiri/ComfyUI-AI-Photography-Toolkit.git
```

Restart ComfyUI. All dependencies install automatically on first load.

## Nodes

### Prompt Generators

| Node | Description |
|------|-------------|
| **SID_ZImagePromptGenerator** | Built-in Anthropic Claude |
| **SID_ZImagePromptGenerator_Advanced** | Connect any LLM provider |

### LLM Providers (for Advanced node)

| Node | Provider | API Key |
|------|----------|---------|
| **SID_Anthropic_LLM** | Anthropic Claude | [console.anthropic.com](https://console.anthropic.com) |
| **SID_OpenAI_Compatible_LLM** | OpenAI / GPT / Together AI / LM Studio | [platform.openai.com](https://platform.openai.com) or local |
| **SID_Grok_LLM** | xAI Grok | [console.x.ai](https://console.x.ai) |
| **SID_GGUF_LLM** | Local GGUF models | Not required |

## Quick Start

### Option 1: Basic Node (Anthropic)

```
Image → SID_ZImagePromptGenerator → zimage_prompt
```

Enter your Anthropic API key in the node.

### Option 2: Advanced Node (Any Provider)

```
SID_Anthropic_LLM ─┐
                   ├→ SID_ZImagePromptGenerator_Advanced → zimage_prompt
        Image ─────┘
```

## Setup by Provider

| Provider | Node | API Key | Models / Notes |
|----------|------|---------|----------------|
| **Anthropic** | SID_ZImagePromptGenerator or SID_Anthropic_LLM | [console.anthropic.com](https://console.anthropic.com) | claude-sonnet-4-5, claude-haiku-4-5 |
| **OpenAI** | SID_OpenAI_Compatible_LLM | [platform.openai.com](https://platform.openai.com) | gpt-4o, gpt-4o-mini, gpt-4-turbo |
| **xAI Grok** | SID_Grok_LLM | [console.x.ai](https://console.x.ai) | grok-2-vision-1212, grok-vision-beta |
| **LM Studio** | SID_OpenAI_Compatible_LLM | Not required | Set `api_url`: `http://localhost:1234/v1` |
| **Local GGUF** | SID_GGUF_LLM | Not required | Auto-download with GPU detection |

### Local GGUF Models

`llama-cpp-python` is **auto-installed** with GPU detection:
- **NVIDIA** - Installs CUDA 12.1 or 12.2+ wheel automatically
- **Apple Silicon** - Installs with Metal support (M1/M2/M3)
- **AMD** - Installs with ROCm support (Linux)
- **CPU** - Falls back to CPU-only build

Models **auto-download** when `auto_download` is enabled (default: ON).

**Model Location:** `ComfyUI/models/LLM/GGUF/`

| Model | VRAM | Manual Download |
|-------|------|-----------------|
| moondream2-q4_k_m | ~4GB | [HuggingFace](https://huggingface.co/vikhyatk/moondream2) |
| llava-v1.5-7b-q4_k_m | ~8GB | [HuggingFace](https://huggingface.co/mys/ggml_llava-v1.5-7b) |
| minicpm-v-2_6-q4_k_m | ~10GB | [HuggingFace](https://huggingface.co/openbmb/MiniCPM-V-2_6-gguf) |
| llava-v1.5-13b-q4_k_m | ~16GB | [HuggingFace](https://huggingface.co/mys/ggml_llava-v1.5-13b) |

**Manual Installation:** Download both the model `.gguf` file AND the `mmproj` (vision encoder) file to `ComfyUI/models/LLM/GGUF/`

## Node Settings

| Setting | Options |
|---------|---------|
| detail_level | Quick (1 call), Standard (2), Deep (3) |
| focus_override | Auto-detect, Portrait, Landscape, etc. |
| content_detail | minimal, standard, detailed, explicit |
| prompt_mode | Image Only, Prompt Guides, Prompt First, Prompt Dominates |

## Outputs

| Output | Description |
|--------|-------------|
| zimage_prompt | Generated narrative prompt |
| structured_data | JSON classification |
| debug_log | Processing details |

## Dependencies

All dependencies are **auto-installed** on first load:
- `anthropic` - Claude API
- `openai` - OpenAI/Grok API
- `pyyaml` - Config
- `requests` - HTTP
- `llama-cpp-python` - Local GGUF inference (with GPU auto-detection)

## License

MIT License
