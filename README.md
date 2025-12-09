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

### Anthropic Claude

1. Get API key: [console.anthropic.com](https://console.anthropic.com)
2. Use **SID_ZImagePromptGenerator** or **SID_Anthropic_LLM**

### OpenAI / GPT-4

1. Get API key: [platform.openai.com](https://platform.openai.com)
2. Use **SID_OpenAI_Compatible_LLM**
3. Models: `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`

### xAI Grok

1. Get API key: [console.x.ai](https://console.x.ai)
2. Use **SID_Grok_LLM**

### LM Studio / Local Server

1. Start LM Studio server
2. Use **SID_OpenAI_Compatible_LLM**
3. Set `api_url`: `http://localhost:1234/v1`
4. Leave `api_key` empty

### Local GGUF Models

1. Install llama-cpp-python (see below)
2. Use **SID_GGUF_LLM**
3. Models auto-download to `ComfyUI/models/LLM/GGUF/`

**Available Models:**

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
