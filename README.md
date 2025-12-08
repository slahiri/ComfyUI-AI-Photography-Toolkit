# ComfyUI-AI-Photography-Toolkit

A collection of AI-powered photography and image generation tools for ComfyUI. All nodes are prefixed with `SID_` for easy identification.

## Features

### Current Nodes

#### SID_ZImagePromptGenerator

Agentic multi-stage image analyzer that generates Z-Image compatible narrative prompts. Supports multiple AI providers including Anthropic Claude, Ollama (local), and Grok (xAI).

**Key Features:**
- **6-Stage Agentic Pipeline**: Classification → Metadata → Attribute Mapping → Detailed Analysis → Prompt Composition → Z-Image Optimization
- **Multi-Provider Support**: Anthropic (Claude), Ollama (local models), Grok (xAI)
- **Z-Image Optimized**: Generates flowing narrative prompts (no keyword lists or meta-tags)
- **NSFW Support**: Content detail levels from minimal to explicit
- **Smart Caching**: Persistent disk cache saves API calls
- **56+ Photography Genres**: Across 6 categories (People, Events, Nature, Commercial, Artistic, Lifestyle)
- **11 Shot Framings**: From Extreme Close-Up (ECU) to Very Long Shot (VLS)

### Future Nodes (Planned)
- SID_ZImagePromptEnhancer - Text-to-text prompt enhancement for Z-Image

## Installation

### Method 1: Manual Installation (Recommended)

1. Navigate to your ComfyUI custom_nodes directory:
   ```bash
   cd ComfyUI/custom_nodes
   ```

2. Clone this repository:
   ```bash
   git clone https://github.com/slahiri/ComfyUI-AI-Photography-Toolkit.git
   ```

3. Install dependencies:
   ```bash
   cd ComfyUI-AI-Photography-Toolkit
   pip install -r requirements.txt
   ```

4. Restart ComfyUI

### Method 2: Via ComfyUI Manager

1. Open ComfyUI Manager
2. Search for "ComfyUI-AI-Photography-Toolkit"
3. Click Install
4. Restart ComfyUI

## Usage

### SID_ZImagePromptGenerator

#### 1. Choose Your AI Provider

| Provider | Models | API Key | URL |
|----------|--------|---------|-----|
| **Anthropic** | claude-sonnet-4-5, claude-haiku-4-5, claude-opus-4-1 | Required ([console.anthropic.com](https://console.anthropic.com)) | Auto |
| **Ollama** | llava, moondream, bakllava (local) | Not required | `http://localhost:11434` |
| **Grok** | grok-2-vision, grok-vision-beta | Required ([console.x.ai](https://console.x.ai)) | Auto |

#### 2. Ollama Models by VRAM

| VRAM Tier | Models | Memory Required |
|-----------|--------|-----------------|
| **Low** | `ollama/moondream`, `ollama/llava:7b`, `ollama/bakllava` | ~4-8GB |
| **Mid** | `ollama/llava:13b`, `ollama/llava-llama3` | ~12-16GB |
| **High** | `ollama/llava:34b`, `ollama/llama3.2-vision` | ~24GB+ |

#### 3. Node Configuration

**API Settings:**
- `ai_provider`: Select provider (auto-detected from model)
- `api_key`: Your API key (leave empty for Ollama)
- `model`: Select model (provider auto-detected from prefix)
- `api_url`: Override URL (optional, defaults provided)

**Analysis Options:**
- `detail_level`: Quick (1 LLM call), Standard (2 calls), Deep (3 calls)
- `focus_override`: Force specific genre (Portrait, Landscape, Product, etc.)
- `content_detail`: minimal, standard, detailed, explicit (NSFW)

**Prompt Direction:**
- `user_prompt`: Optional guidance text
- `prompt_mode`:
  - "Image Only" - Analyze image, ignore prompt
  - "Prompt Guides Analysis" - Image primary, prompt guides emphasis
  - "Prompt First, Image Fills Gaps" - Prompt primary, image supplements
  - "Prompt Dominates" - Prompt foundation, minimal image details

**Focus Toggles:**
- `focus_subject`: Include subject description
- `focus_environment`: Include background/environment
- `focus_lighting`: Include lighting analysis
- `focus_colors`: Include colors and materials
- `focus_mood`: Include mood/atmosphere
- `include_text_quotes`: Quote visible text with "quotes"

**Generation Settings:**
- `max_tokens`: Target prompt length (50-500)
- `temperature`: Creativity (0.0=focused, 1.0=creative)
- `seed`: Reproducibility seed
- `seed_mode`: fixed, randomize, increment, decrement
- `cache_prompt`: Enable persistent disk caching

#### 4. Outputs

| Output | Description |
|--------|-------------|
| `image` | Pass-through of input image |
| `zimage_prompt` | Z-Image compatible narrative prompt |
| `width` | Image width in pixels |
| `height` | Image height in pixels |
| `structured_data` | JSON with classification and attributes |
| `image_metadata` | JSON with Z-Image recommendations |
| `debug_log` | Stage-by-stage processing details |

### Example Workflow

```
Image Input → SID_ZImagePromptGenerator → zimage_prompt → Z-Image Model
                      ↓
              structured_data → (optional) Further processing
```

### Z-Image Best Practices

Z-Image-Turbo (6B parameter model) works best with:
- **Narrative prompts**: Flowing descriptions, not keyword lists
- **No meta-tags**: Avoid "8K, masterpiece, best quality"
- **No negative prompts**: Z-Image doesn't use them (guidance_scale=0.0)
- **Visible elements only**: Describe what's actually in the image

## Configuration

### Ollama Setup (Local Models)

1. Install Ollama: https://ollama.ai
2. Pull a vision model:
   ```bash
   ollama pull llava:7b      # Low VRAM
   ollama pull llava:13b     # Mid VRAM
   ollama pull llava:34b     # High VRAM
   ```
3. Ollama runs on `http://localhost:11434` by default

### API Key Security

**Important:** Never commit your API key to version control!

Options for storing your API key:
1. Enter it directly in the node (least secure)
2. Use environment variables (recommended)
3. Use a secrets management system (production)

## Troubleshooting

### "ERROR: anthropic library not installed"
```bash
pip install anthropic>=0.39.0
```

### "ERROR: openai library not installed" (for Grok)
```bash
pip install openai
```

### "Ollama connection refused"
- Ensure Ollama is running: `ollama serve`
- Check URL is correct: `http://localhost:11434`
- Verify model is pulled: `ollama list`

### "API Error (AuthenticationError)"
- Verify API key is correct
- Check API key has not expired
- Ensure account has credits

### Empty or Invalid Prompts
- Try increasing `max_tokens`
- Check the debug_log output
- Ensure image is valid

## Requirements

- ComfyUI (latest version recommended)
- Python 3.10+
- API key for chosen provider (except Ollama)

## Dependencies

- `anthropic>=0.39.0` - Anthropic Claude API
- `openai` - Grok API (OpenAI-compatible)
- `requests` - Ollama API
- `pyyaml>=6.0` - Configuration loading
- `pillow>=10.0.0` - Image processing
- `numpy>=1.24.0` - Array operations

## Changelog

### Version 4.0.0 (2025-01-XX) - Z-IMAGE & MULTI-PROVIDER

**New Node: SID_ZImagePromptGenerator**
- Complete rewrite for Z-Image compatibility
- 6-stage agentic pipeline for intelligent image analysis
- Generates flowing narrative prompts (not keyword lists)
- No meta-tags, no negative prompts (Z-Image optimized)

**Multi-Provider Support**
- **Anthropic**: Claude Sonnet 4.5, Haiku 4.5, Opus 4.1
- **Ollama**: Local vision models with VRAM tiers
  - Low (~4-8GB): moondream, llava:7b, bakllava
  - Mid (~12-16GB): llava:13b, llava-llama3
  - High (~24GB+): llava:34b, llama3.2-vision
- **Grok**: xAI vision models (grok-2-vision, grok-vision-beta)

**Smart Features**
- Persistent disk caching (saves API calls)
- 56+ photography genres across 6 categories
- 11 shot framing types (ECU to VLS)
- Content detail levels (minimal to explicit/NSFW)
- Seed modes: fixed, randomize, increment, decrement
- 4 prompt modes for user prompt integration

**Outputs**
- Image pass-through
- Z-Image prompt
- Width/height
- Structured data (JSON)
- Image metadata with Z-Image recommendations
- Debug log

**Breaking Changes**
- Removed old SID_AIPromptGenerator node
- New node uses different output structure

### Previous Versions

See git history for changelog of versions 1.0.0 - 3.0.1.

## Credits

Created by Siddhartha Lahiri

Special thanks to:
- ComfyUI team for the amazing framework
- Anthropic, Ollama, and xAI for AI APIs
- The ComfyUI community

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Roadmap

- [x] Z-Image optimized prompt generation
- [x] Multi-provider support (Anthropic, Ollama, Grok)
- [x] Persistent caching
- [x] NSFW content support
- [ ] SID_ZImagePromptEnhancer (text-to-text)
- [ ] Batch processing
- [ ] More AI providers (Gemini, etc.)

## License

MIT License - See LICENSE file for details

---

**Generate better Z-Image prompts with AI!**
