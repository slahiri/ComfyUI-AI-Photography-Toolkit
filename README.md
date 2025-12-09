# ComfyUI-AI-Photography-Toolkit

AI-powered Z-Image prompt generator for ComfyUI. Analyzes images and generates flowing narrative prompts optimized for Z-Image models.

![Workflow Screenshot](docs/images/workflow-screenshot.png)

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
| **SID_ZImagePromptGenerator** | Built-in Anthropic Claude (simple setup) |
| **SID_ZImagePromptGenerator_Advanced** | Connect any LLM provider node |

### LLM Provider Nodes (for Advanced)

| Node | Provider | API Key Required |
|------|----------|------------------|
| **SID_Anthropic_LLM** | Anthropic Claude | Yes |
| **SID_OpenAI_Compatible_LLM** | OpenAI, Together AI, LM Studio, Ollama | Depends |
| **SID_Grok_LLM** | xAI Grok | Yes |
| **SID_GGUF_LLM** | Local GGUF models | No |

## Quick Start

### Option 1: Basic Node (Anthropic only)

```
Image → SID_ZImagePromptGenerator → zimage_prompt
```

### Option 2: Advanced Node (Any Provider)

```
LLM Provider Node ─┐
                   ├→ SID_ZImagePromptGenerator_Advanced → zimage_prompt
        Image ─────┘
```

---

## Setup by Provider

### Anthropic Claude (Cloud)

**Node:** `SID_ZImagePromptGenerator` (Basic) or `SID_Anthropic_LLM` (Advanced)

| Setting | Value |
|---------|-------|
| API Key | Get from [console.anthropic.com](https://console.anthropic.com) |
| Models | `claude-sonnet-4-5-20241022`, `claude-haiku-4-5-20241022` |

---

### OpenAI / GPT-4 (Cloud)

**Node:** `SID_OpenAI_Compatible_LLM`

| Setting | Value |
|---------|-------|
| api_key | Get from [platform.openai.com](https://platform.openai.com) |
| api_url | `https://api.openai.com/v1` (default) |
| model | `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo` |

---

### xAI Grok (Cloud)

**Node:** `SID_Grok_LLM`

| Setting | Value |
|---------|-------|
| api_key | Get from [console.x.ai](https://console.x.ai) |
| model | `grok-2-vision-1212`, `grok-vision-beta` |

---

### Together AI (Cloud)

**Node:** `SID_OpenAI_Compatible_LLM`

| Setting | Value |
|---------|-------|
| api_key | Get from [api.together.xyz](https://api.together.xyz) |
| api_url | `https://api.together.xyz/v1` |
| model | Select from dropdown or use `custom_model` |
| custom_model | `meta-llama/Llama-Vision-Free`, `meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo` |

---

### LM Studio (Local)

**Node:** `SID_OpenAI_Compatible_LLM`

| Setting | Value |
|---------|-------|
| api_key | Leave empty |
| api_url | `http://localhost:1234/v1` |
| custom_model | Your loaded model name |

**Setup:** Start LM Studio → Load a vision model → Start local server

---

### Ollama (Local)

**Node:** `SID_OpenAI_Compatible_LLM` (OpenAI-compatible mode)

| Setting | Value |
|---------|-------|
| api_key | Leave empty |
| api_url | `http://localhost:11434/v1` |
| custom_model | `llava`, `llava:13b`, `bakllava`, `llama3.2-vision` |

**Setup:** 
```bash
ollama pull llava
ollama serve
```

---

### Local GGUF Models

**Node:** `SID_GGUF_LLM`

| Setting | Value |
|---------|-------|
| model | Select from dropdown |
| auto_download | `True` (default) - downloads model automatically |
| n_gpu_layers | `-1` (all on GPU), `0` (CPU only) |

**GPU Auto-Detection:** `llama-cpp-python` installs automatically with correct GPU support:
- **NVIDIA** → CUDA 12.1/12.2+ wheel
- **Apple Silicon** → Metal support (M1/M2/M3)
- **AMD** → ROCm support (Linux)
- **CPU** → Fallback

**Available Models:**

| Model | VRAM | Manual Download |
|-------|------|-----------------|
| moondream2-q4_k_m | ~4GB | [HuggingFace](https://huggingface.co/vikhyatk/moondream2) |
| llava-v1.5-7b-q4_k_m | ~8GB | [HuggingFace](https://huggingface.co/mys/ggml_llava-v1.5-7b) |
| minicpm-v-2_6-q4_k_m | ~10GB | [HuggingFace](https://huggingface.co/openbmb/MiniCPM-V-2_6-gguf) |
| llava-v1.5-13b-q4_k_m | ~16GB | [HuggingFace](https://huggingface.co/mys/ggml_llava-v1.5-13b) |

**Model Location:** `ComfyUI/models/LLM/GGUF/`

**Manual Install:** Download both `.gguf` model AND `mmproj` (vision encoder) files.

---

## Node Settings

| Setting | Options | Description |
|---------|---------|-------------|
| detail_level | Quick, Standard, Deep | LLM calls: 1, 2, or 3 |
| focus_override | Auto-detect, Portrait, Landscape, etc. | Force specific genre |
| content_detail | minimal, standard, detailed, explicit | Body/clothing detail (NSFW) |
| prompt_mode | Image Only, Prompt Guides, Prompt First, Prompt Dominates | How user prompt interacts |
| max_tokens | 50-500 | Response length |
| temperature | 0.0-2.0 | Creativity (0=focused, 2=creative) |

## Outputs

| Output | Description |
|--------|-------------|
| image | Pass-through input image |
| zimage_prompt | Generated narrative prompt |
| width / height | Image dimensions |
| structured_data | JSON classification and attributes |
| debug_log | Processing details |

## Dependencies

All dependencies **auto-install** on first load:
- `anthropic` - Claude API
- `openai` - OpenAI/Grok/Together AI API
- `pyyaml` - Config
- `requests` - HTTP
- `llama-cpp-python` - Local GGUF (with GPU auto-detection)

## Changelog

### Version 4.1.0
- Add `SID_OpenAI_Compatible_LLM` - supports OpenAI, Together AI, LM Studio, Ollama
- Add `SID_Grok_LLM` - xAI Grok vision models
- Add `SID_GGUF_LLM` - local GGUF models with auto-download
- Auto-install `llama-cpp-python` with GPU detection (NVIDIA/Apple/AMD/CPU)
- Split into Basic and Advanced prompt generator nodes
- Welcome message shows dependency status on load

### Version 4.0.0
- Initial Z-Image optimized prompt generator
- 6-stage agentic pipeline
- Anthropic Claude support
- Persistent disk caching

## License

MIT License
