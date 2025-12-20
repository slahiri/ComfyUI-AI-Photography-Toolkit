# Research: Available Captioning Nodes in ComfyUI

Scan of `D:\ComfyUI\custom_nodes` - December 2024

---

## Florence-2 Based Nodes

### comfyui-florence2
**Location:** `D:\ComfyUI\custom_nodes\comfyui-florence2`
**Files:** `nodes.py` (31KB), `modeling_florence2.py` (128KB)

**Nodes:**
| Node | Inputs | Outputs | Description |
|------|--------|---------|-------------|
| `DownloadAndLoadFlorence2Model` | model selection | FL2MODEL | Load Florence-2 model |
| `DownloadAndLoadFlorence2Lora` | LoRA selection | PEFTLORA | Load LoRA adapter |
| `Florence2ModelLoader` | path | FL2MODEL | Load from local path |
| `Florence2Run` | model, image, task | IMAGE, MASK, STRING, JSON | Run Florence-2 tasks |

**Supported Tasks:** Caption, detailed caption, more detailed caption, object detection, segmentation, OCR

---

### comfyui_tagger
**Location:** `D:\ComfyUI\custom_nodes\comfyui_tagger`
**Files:** `nodes.py` (26KB)

**Nodes:**
| Node | Category | Description |
|------|----------|-------------|
| `DownloadAndLoadFlorence2Model` | Florence2 | Model loader |
| `DownloadAndLoadFlorence2Lora` | Florence2 | LoRA loader |
| `Florence2ModelLoader` | Florence2 | Local model loader |
| `Florence2Run` | Florence2 | Run captioning |
| `Batch_text_save` | file | Save captions to files |

---

### comfyui_caption_this
**Location:** `D:\ComfyUI\custom_nodes\comfyui_caption_this`
**Files:** `florence2_caption.py` (11KB), `janus_pro_caption.py` (8KB)

**Nodes:**
| Node | Category | Description |
|------|----------|-------------|
| `Florence2ModelLoader` | Florence2Caption | Load model |
| `Florence2DescribeImage` | Florence2Caption | Single image caption |
| `Florence2CaptionImageUnderDirectory` | Florence2Caption | Batch directory processing |

**Also includes:** Janus Pro support, SigLIP encoder files

---

## JoyCaption

### ComfyUI-JoyCaption
**Location:** `D:\ComfyUI\custom_nodes\ComfyUI-JoyCaption`
**Files:** `JC.py` (20KB), `JC_GGUF.py` (18KB), `CaptionTools.py` (7KB)

**Nodes:**
| Node | Category | Outputs | Description |
|------|----------|---------|-------------|
| `JoyCaptionExtraOptions` | JoyCaption | JOYCAPTION_EXTRA_OPTIONS | Configure extra options |
| `JoyCaption` | JoyCaption | STRING | Main captioning node |
| `JoyCaptionAlpha` | JoyCaption | STRING, STRING | Alpha version with dual output |

**Features:**
- 10 caption modes (descriptive, straightforward, SD prompt, MidJourney, booru, etc.)
- GGUF quantized model support (Q4_K_M)
- Extra options for customization
- llama.cpp backend support

---

## QwenVL Nodes

### ComfyUI-QwenVL
**Location:** `D:\ComfyUI\custom_nodes\ComfyUI-QwenVL`
**Files:** `AILab_QwenVL.py` (21KB)

**Nodes:**
| Node | Category | Description |
|------|----------|-------------|
| `AILab_QwenVL` | AILab/QwenVL | Basic QwenVL captioning |
| `AILab_QwenVL_Advanced` | AILab/QwenVL | Advanced with more options |

**Features:**
- Quantization options (FP16, 8-bit, 4-bit)
- Custom prompt support
- Temperature control

---

### ComfyUI_QwenVL_PromptCaption
**Location:** `D:\ComfyUI\custom_nodes\ComfyUI_QwenVL_PromptCaption`
**Files:** `qwen_25.py` (20KB), `qwen_3.py` (20KB), `ovis_25.py` (12KB)

**Nodes:**
| Node | Category | Description |
|------|----------|-------------|
| `Qwen25Caption` | image/caption | Qwen 2.5 VL captioning |
| `Qwen25CaptionBatch` | image/caption | Batch processing |
| (Qwen3 variants) | image/caption | Qwen 3 support |

**Features:**
- Qwen 2.5 and Qwen 3 support
- Batch captioning
- BBox/vision processing utilities

---

### aistudynow-qwenvl
**Location:** `D:\ComfyUI\custom_nodes\aistudynow-qwenvl`

Alternative QwenVL implementation.

---

## BLIP Nodes

### ComfyUI-Blip
**Location:** `D:\ComfyUI\custom_nodes\ComfyUI-Blip`
**Files:** `BlipCaption.py` (10KB)

**Nodes:**
| Node | Category | Description |
|------|----------|-------------|
| `BlipCaption` | AILab/BlipCaption | Simple BLIP captioning |
| `BlipCaptionAdvanced` | AILab/BlipCaption | Advanced with more options |

---

### blip-comfyui
**Location:** `D:\ComfyUI\custom_nodes\blip-comfyui`

Alternative BLIP implementation.

---

## Tagger Nodes

### ComfyUI-Miaoshouai-Tagger
**Location:** `D:\ComfyUI\custom_nodes\ComfyUI-Miaoshouai-Tagger`
**Files:** `nodes.py` (16KB)

**Nodes:**
| Node | Category | Outputs | Description |
|------|----------|---------|-------------|
| `Tagger` | MiaoshouAI Tagger | IMAGE, STRING×3, INT | Tag extraction |
| `SaveTags` | MiaoshouAI Tagger | STRING | Save tags to file |
| `FluxCLIPTextEncode` | MiaoshouAI Tagger | CONDITIONING×2, STRING×3 | Encode for Flux |
| `CaptionAnalyzer` | MiaoshouAI Tagger | STRING | Analyze captions |

**Features:**
- WD14 tagger integration
- Florence-2 tagger
- Flux-specific encoding
- Caption analysis

---

## Multi-Provider LLM Nodes

### comfyui-if_llm
**Location:** `D:\ComfyUI\custom_nodes\comfyui-if_llm`
**Files:** `IFLLMNode.py` (77KB), 15+ API files

**API Providers:**
| Provider | File | Description |
|----------|------|-------------|
| Anthropic | `anthropic_api.py` | Claude models |
| OpenAI | `openai_api.py` | GPT-4, GPT-4o |
| Gemini | `gemini_api.py` | Google Gemini |
| Groq | `groq_api.py` | Groq inference |
| HuggingFace | `HF_api.py`, `huggingface_api.py` | HF models |
| Ollama | `ollama_api.py` | Local Ollama |
| LM Studio | `lms_api.py` | Local LM Studio |
| Kobold | `kobold_api.py` | KoboldAI |
| TextGen | `textgen_api.py` | Text Generation WebUI |
| Transformers | `transformers_api.py` (69KB) | Local transformers |
| Mistral | `mistral_api.py` | Mistral AI |
| DeepSeek | `deepseek_api.py` | DeepSeek |
| xAI | `xai_api.py` | Grok |
| vLLM | `vllm_api.py` | vLLM server |
| llama.cpp | `llamacpp_api.py` | Local llama.cpp |

**Main Nodes:**
- `IFLLMNode` - Main LLM interface
- `IFLLMDisplayTextNode` - Display results
- `IFLLMLoadImagesNodeS` - Load images for vision
- `IFLLMSaveTextNode` - Save outputs

---

### llm-toolkit
**Location:** `D:\ComfyUI\custom_nodes\llm-toolkit`
**Files:** 80+ files, `llmtoolkit_utils.py` (96KB)

**API Providers:**
| Provider | File |
|----------|------|
| Anthropic | `anthropic_api.py` |
| OpenAI | `openai_api.py` (44KB) |
| Gemini | `gemini_api.py` (13KB) |
| Groq | `groq_api.py` |
| Ollama | `ollama_api.py` |
| OpenRouter | `openrouter_api.py` |
| DeepSeek | `deepseek_api.py` |
| Transformers | `transformers_api.py` (27KB) |
| BFL | `bfl_api.py` |

**Key Nodes:**
- `generate_text` (59KB) - Main text generation
- `generate_image` - Image generation
- `prompt_manager` (28KB) - Prompt management
- `api_provider_selector` (33KB) - Provider selection
- `model_list_fetcher` - Dynamic model lists

---

## Summary Table

| Package | Model/Type | Nodes | Batch | API | Local | Special Features |
|---------|------------|-------|-------|-----|-------|------------------|
| comfyui-florence2 | Florence-2 | 4 | ❌ | ❌ | ✅ | LoRA, multi-task |
| comfyui_tagger | Florence-2 | 5 | ✅ | ❌ | ✅ | File saving |
| comfyui_caption_this | Florence-2, Janus | 3 | ✅ | ❌ | ✅ | Directory batch |
| ComfyUI-JoyCaption | JoyCaption | 3 | ❌ | ❌ | ✅ | 10 modes, GGUF |
| ComfyUI-QwenVL | Qwen-VL | 2 | ❌ | ❌ | ✅ | Quantization |
| ComfyUI_QwenVL_PromptCaption | Qwen 2.5/3 | 2+ | ✅ | ❌ | ✅ | Multi-version |
| ComfyUI-Blip | BLIP | 2 | ❌ | ❌ | ✅ | Simple/Advanced |
| ComfyUI-Miaoshouai-Tagger | WD14/Florence | 4 | ❌ | ❌ | ✅ | Flux encode |
| comfyui-if_llm | Multi-provider | 10+ | ❌ | ✅ | ✅ | 15+ providers |
| llm-toolkit | Multi-provider | 50+ | ❌ | ✅ | ✅ | Full LLM toolkit |

---

## Observations

### Strengths of Existing Nodes
1. **Good model coverage**: Florence-2, JoyCaption, QwenVL, BLIP all available
2. **Multiple implementations**: Choice of packages for same models
3. **API support**: if_llm and llm-toolkit cover most cloud providers
4. **Quantization**: GGUF and 4-bit support in several packages

### Gaps Identified
1. **No unified pipeline**: Each node is standalone, no multi-stage
2. **Limited output control**: Most output raw text, no format options
3. **No quality tiers**: User must manually select models
4. **No hybrid output**: Tags + description must be manually combined
5. **No validation**: No caption quality checking
6. **No model-specific formatting**: No presets for SD/SDXL/Flux/Z-Image
7. **Limited batch intelligence**: Basic loops, no skip/resume/filter

### Most Feature-Rich
1. **comfyui-if_llm** - Most API providers, extensive codebase
2. **llm-toolkit** - Comprehensive LLM toolkit with many utilities
3. **ComfyUI-JoyCaption** - Best diffusion-specific captioning
4. **ComfyUI-Miaoshouai-Tagger** - Good tagger with Flux support

---

## File Sizes (Complexity Indicator)

| Package | Largest File | Size |
|---------|--------------|------|
| comfyui-florence2 | modeling_florence2.py | 128 KB |
| comfyui-if_llm | utils.py | 79 KB |
| llm-toolkit | llmtoolkit_utils.py | 96 KB |
| ComfyUI-JoyCaption | JC.py | 20 KB |
| ComfyUI_QwenVL_PromptCaption | vision_process.py | 21 KB |
