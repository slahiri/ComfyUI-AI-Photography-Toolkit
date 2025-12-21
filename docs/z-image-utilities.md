# Z-Image Utilities Node Reference

A prompt enhancement toolkit optimized for **Z-Image** (Alibaba's Tongyi-MAI text-to-image model). Uses LLMs to transform simple prompts into detailed visual descriptions.

## Nodes

| Node | Purpose |
|------|---------|
| **Z-Image API Config** | Configure LLM backend (OpenRouter/Local/Direct HuggingFace) |
| **Z-Image Options** | Set inference params (temperature, top_p, seed, etc.) |
| **Z-Image Prompt Enhancer** | Core node - transforms prompts using LLM |
| **Z-Image Prompt Enhancer + CLIP** | Same as above but outputs CLIP conditioning directly |
| **Z-Image Unload Models** | Free VRAM by clearing cached models |
| **Z-Image Clear Sessions** | Clear conversation history |

---

## Node Details

### Z-Image API Config

Configures the LLM connection for prompt enhancement.

**Inputs:**
- `provider`: "openrouter", "local", or "direct"
- `model`: Model identifier or HuggingFace repo ID
- `api_key`: OpenRouter API key (if using OpenRouter)
- `local_endpoint`: LLM server URL (if using local)
- `quantization`: 4bit, 8bit, or none (for direct loading)
- `device`: auto, cuda, cpu, or mps (for direct loading)

**Output:** Config object passed to enhancer nodes

---

### Z-Image Options

Sets advanced inference parameters with enable flags.

**Inputs:**
- `temperature`: 0.0-2.0 (default: 0.7)
- `top_p`: 0.0-1.0 (default: 0.9)
- `top_k`: 0-100 (default: 40)
- `seed`: Random seed for reproducibility
- `repeat_penalty`: 0.5-2.0 (default: 1.1)
- `max_tokens`: 256-8192 (default: 2048)
- `debug_mode`: Enable detailed logging

**Output:** Options dictionary (only enabled options included)

---

### Z-Image Prompt Enhancer

Core prompt enhancement node.

**Inputs:**
- `config`: Configuration from API Config node (required)
- `prompt`: Text to enhance (required)
- `prompt_template`: "auto", "chinese", or "english"
- `options`: From Options node (optional)
- `image`: ComfyUI image tensor for vision models (optional)
- `retry_count`: 0-10 retries on failure
- `max_output_length`: 0-10000 characters (0=unlimited, default: 6000)
- `session_id`: For multi-turn conversations
- `reset_session`: Clear conversation history
- `keep_model_loaded`: Cache model in memory
- `utf8_sanitize`: Convert to ASCII-safe characters

**Outputs:** Enhanced prompt (string) + debug log (string)

---

### Z-Image Prompt Enhancer + CLIP

Combines prompt enhancement with CLIP encoding.

**Inputs:** Same as Prompt Enhancer + `clip` (CLIP model from checkpoint loader)

**Outputs:** CLIP conditioning + enhanced prompt + debug log

---

### Z-Image Unload Models

Frees GPU memory by unloading cached models.

**Inputs:** `unload_all` (boolean, default: True)

**Output:** Status message

---

### Z-Image Clear Sessions

Clears conversation history.

**Inputs:**
- `clear_all`: Clear all sessions (default: True)
- `session_id`: Specific session to clear

**Output:** Status message

---

## Backend Options

### 1. OpenRouter (Cloud)

Easiest setup, no local infrastructure needed.

- **Free tier**: `qwen/qwen3-235b-a22b:free`
- Rate limiting handled automatically
- Pay-per-token model

### 2. Local API Servers

Works with Ollama, LM Studio, vLLM, text-generation-webui.

- No API costs, full privacy
- Requires running separate server
- Common ports: 11434 (Ollama), 1234 (LM Studio), 8000 (vLLM)

### 3. Direct HuggingFace Loading

Loads models directly into ComfyUI process.

- No separate server needed
- Full control over quantization
- VRAM requirements (approximate):
  - 8B model @ 4-bit: ~6GB
  - 8B model @ 8-bit: ~10GB
  - 8B model @ FP16: ~16GB

---

## Z-Image Specialization

Z-Image is a text-to-image model from Tongyi-MAI (Alibaba) optimized for high-quality visual generation.

**Token Limits:**
- Default: 512 tokens
- Maximum: 1024 tokens
- Node warns when prompts exceed limits

**Prompt Style:**
- Requires detailed, concrete visual descriptions
- No metaphors or emotional language
- No meta-tags (8K, masterpiece, etc.)
- Precise layout and text placement instructions

**Output Cleaning:**
- Removes `<think>...</think>` blocks from reasoning models
- Extracts final visual descriptions
- Handles Chinese/English reasoning indicators
- Removes markdown artifacts

---

## Recommended Models

| Model | Provider | Notes |
|-------|----------|-------|
| `Goekdeniz-Guelmez/Josiefied-Qwen3-8B-abliterated-v1` | Local/Direct | Best overall, NSFW-capable |
| `Qwen/Qwen3-8B` | Local/Direct | Official Qwen, SFW only |
| `qwen/qwen3-235b-a22b:free` | OpenRouter | High quality, no local VRAM |

---

## Workflow Examples

### Standard Workflow
```
[Checkpoint Loader] → [Z-Image API Config] → [Z-Image Prompt Enhancer] → [CLIP Text Encode] → [KSampler]
                                      ↑
                       [Z-Image Options] (optional)
```

### Streamlined with CLIP Integration
```
[Checkpoint Loader] → [Z-Image Prompt Enhancer + CLIP] → [KSampler]
                                   ↑
                       [Z-Image API Config]
```

---

## Key Features

- **Vision model support** - Process images with VLMs
- **Session management** - Multi-turn conversations with history
- **Auto-quantization** - Downgrades to 4-bit if VRAM insufficient
- **Smart retry logic** - Exponential backoff for API failures
- **Language detection** - Auto-selects Chinese or English template
- **Model caching** - Persists across executions for faster runs
- **Comprehensive logging** - Debug output saved to `z_image_debug.log`

---

## Technical Details

### LLM Client Architecture

Three client implementations:
- **OpenRouterClient** - HTTP calls to OpenRouter API
- **LocalLLMClient** - OpenAI-compatible API servers
- **DirectLocalModelClient** - HuggingFace transformers loading

### Memory Management

- `get_device_info()` - CUDA/MPS/CPU detection with VRAM reporting
- `enforce_quantization()` - Auto-downgrade if insufficient memory
- `clear_gpu_memory()` - Garbage collection and cache clearing

### Session Management

- `ChatSession` class with conversation history
- Auto-cleanup of sessions older than 24 hours
- Thread-safe access

---

## Dependencies

**Required for direct model loading:**
- `transformers >= 4.30.0`
- `accelerate >= 0.20.0`
- `bitsandbytes >= 0.41.0`
- `huggingface-hub >= 0.16.0`

**Optional:**
- `PIL/Pillow` - Image processing
- `numpy` - Image tensor operations
- `psutil` - System memory info

OpenRouter and Local providers work without these dependencies.

---

## File Locations

- **Models**: `ComfyUI/models/LLM/Z-Image/`
- **Logs**: `z_image_debug.log`
- **Workflows**: `custom_nodes/Comfyui-Z-Image-Utilities/workflows/`

---

## Source

- GitHub: https://github.com/Koko-boya/Comfyui-Z-Image-Utilities
- System prompt derived from Z-Image-Turbo HuggingFace Space (Tongyi-MAI)
