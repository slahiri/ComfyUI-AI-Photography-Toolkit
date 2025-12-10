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

## Sample Workflows

Ready-to-use workflow examples are included in the `sample_workflows/` folder:

| Workflow | Description |
|----------|-------------|
| [sample_workflow.json](sample_workflows/sample_workflow.json) | Anthropic Claude with Z-Image Turbo rendering |
| [sample_workflow_gguf.json](sample_workflows/sample_workflow_gguf.json) | Local GGUF model (no API key required) |

**To use:** Load in ComfyUI via `Load` button or drag & drop. Replace `YOUR_ANTHROPIC_API_KEY` with your actual key.

---

## Setup by Provider

### Anthropic Claude (Cloud)

**Node:** `SID_ZImagePromptGenerator` (Basic) or `SID_Anthropic_LLM` (Advanced)

| Setting | Value |
|---------|-------|
| API Key | Get from [console.anthropic.com](https://console.anthropic.com) |
| Models | `claude-sonnet-4-5-20250929`, `claude-haiku-4-5-20251001`, `claude-opus-4-1-20250805` |

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

**Available Models (High Resolution):**

All models support 1024x1024+ images natively for detailed analysis.

| Model | VRAM | Max Resolution | Quality | Download |
|-------|------|----------------|---------|----------|
| **qwen3-vl-2b-q4** | ~4GB | Dynamic | Good | [HuggingFace](https://huggingface.co/bartowski/Qwen3-VL-2B-Instruct-GGUF) |
| **qwen2.5-vl-7b-q4** | ~6GB | Dynamic | Better | [HuggingFace](https://huggingface.co/Mungert/Qwen2.5-VL-7B-Instruct-GGUF) |
| **llama-3.2-vision-11b-q4** | ~8GB | 1120x1120 | Excellent | [HuggingFace](https://huggingface.co/leafspark/Llama-3.2-11B-Vision-Instruct-GGUF) |
| **qwen2.5-vl-7b-q8** | ~10GB | Dynamic | Excellent | [HuggingFace](https://huggingface.co/Mungert/Qwen2.5-VL-7B-Instruct-GGUF) |
| **minicpm-v-2_6-q4** | ~10GB | Dynamic | Excellent | [HuggingFace](https://huggingface.co/openbmb/MiniCPM-V-2_6-gguf) |
| **pixtral-12b-q4** | ~10GB | 1024x1024 | Excellent | [HuggingFace](https://huggingface.co/second-state/Pixtral-12B-2409-GGUF) |

**Recommendations by VRAM:**
- **4-6GB VRAM:** qwen3-vl-2b-q4 (fast, good quality)
- **6-8GB VRAM:** qwen2.5-vl-7b-q4, llama-3.2-vision-11b-q4
- **10GB+ VRAM:** qwen2.5-vl-7b-q8, minicpm-v-2_6, pixtral-12b (best quality)

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
| temperature | 0.0-2.0 | Creativity (0=focused, 2=creative) |
| max_image_size | 512, 768, 1024, Original | GGUF only: resize before encoding (~4x faster at 512) |

## Outputs

| Output | Description |
|--------|-------------|
| image | Pass-through input image |
| zimage_prompt | Generated narrative prompt |
| width / height | Image dimensions |
| structured_data | Comprehensive JSON with all analysis data (see below) |
| debug_log | Processing details |

### Structured Data Output

The `structured_data` output contains categorized JSON with:

| Section | Contents |
|---------|----------|
| `metadata` | Generator version, timestamp, processing time |
| `model` | Provider, model name, template used, settings |
| `settings` | Detail level, content detail, focus options |
| `image` | Dimensions, aspect ratio, color information |
| `classification` | Image type, genre, subject count |
| `subject` | Ethnicity, skin tone, facial features (eyes, nose, lips, eyebrows with gaze direction) |
| `cosmetics` | Makeup details (foundation, eyes, lips, blush) |
| `pose` | Body positioning (head, arms, hands, legs, feet with palm orientation) |
| `clothing` | Garments, materials, coverage analysis (neckline, exposure levels, gaps) |
| `environment` | Setting, background, props |
| `lighting` | Type, direction, quality, color temperature |
| `colors` | Dominant colors, palette, mood |
| `style` | Photography style, mood, atmosphere |
| `generated_prompt` | Final prompt with word/token counts |
| `llm_interactions` | All raw LLM queries and responses for debugging |
| `zimage_recommendations` | Suggested Z-Image parameters |

## Dependencies

All dependencies **auto-install** on first load:
- `anthropic` - Claude API
- `openai` - OpenAI/Grok/Together AI API
- `pyyaml` - Config
- `requests` - HTTP
- `llama-cpp-python` - Local GGUF (with GPU auto-detection)

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for full version history.

### Version 4.1.0 (Latest)
- **High-resolution GGUF models** - all local models support 1024x1024+ images natively
- **Max Image Size option** for GGUF - resize images before encoding (~4x faster at 512)
- **New models**: Qwen3-VL 2B, Llama 3.2 Vision 11B, Pixtral 12B
- **Removed max_tokens slider** - uses model's optimal limit automatically
- **Read-only prompt display** on V2 generator node
- **User prompt integration** into agentic analysis pipeline

## License

MIT License
