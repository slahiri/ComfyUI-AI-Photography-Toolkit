# ComfyUI-AI-Photography-Toolkit

AI-powered Z-Image prompt generator for ComfyUI. Analyzes images and generates flowing narrative prompts optimized for Z-Image models.

**Version:** 4.1.0
**Author:** Siddhartha Lahiri

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/slahiri/ComfyUI-AI-Photography-Toolkit.git
```

Restart ComfyUI. Dependencies install automatically on first load.

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
| **Local GGUF** | SID_GGUF_LLM | Not required | Models auto-download (see below) |

### Local GGUF Models

Models download automatically to `ComfyUI/models/LLM/GGUF/`

| Model | VRAM |
|-------|------|
| moondream2-q4_k_m | ~4GB |
| llava-v1.5-7b-q4_k_m | ~8GB |
| minicpm-v-2_6-q4_k_m | ~10GB |
| llava-v1.5-13b-q4_k_m | ~16GB |

**Install llama-cpp-python:**

```bash
# NVIDIA CUDA 12.1
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121

# NVIDIA CUDA 12.2+
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu122

# CPU only
pip install llama-cpp-python

# AMD ROCm
CMAKE_ARGS="-DGGML_HIPBLAS=on" pip install llama-cpp-python
```

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

**Auto-installed:**
- `anthropic` - Claude API
- `openai` - OpenAI/Grok API
- `pyyaml` - Config
- `requests` - HTTP

**Manual (for GGUF):**
- `llama-cpp-python`

## License

MIT License
