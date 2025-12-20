# Supported Models (MVP)

## Quick
Fast captioning, lower quality. Good for batch processing.

| Model | Type | Notes |
|-------|------|-------|
| `MiaoshouAI/Florence-2-large-PromptGen-v2.0` | Local | Fast, lightweight |
| `Salesforce/blip-image-captioning-large` | Local | Classic captioner |

## Standard
Balanced quality and speed. Good for most use cases.

| Model | Type | Notes |
|-------|------|-------|
| `mradermacher/llama-joycaption-beta-one-hf-llava-GGUF` | Local | Q4_K_M quantized |
| `Salesforce/blip2-opt-2.7b` | Local | Good quality |
| `Qwen/Qwen2.5-VL-3B-Instruct` | Local | Qwen vision-language |
| `Qwen/Qwen2.5-VL-7B-Instruct` | Local | 4-bit quantized |

## Detailed
High quality, slower. Best for single image analysis.

| Model | Type | Notes |
|-------|------|-------|
| `fancyfeast/llama-joycaption-beta-one-hf-llava` | Local | Full precision |
| `Ertugrul/Qwen2.5-VL-7B-Captioner-Relaxed` | Local | Fine-tuned for captions |
| `Salesforce/blip2-opt-6.7b` | Local | Larger BLIP2 |

## Extreme (API)
Best quality, requires API keys. Cloud-based.

| Provider | URL | Notes |
|----------|-----|-------|
| Gemini | https://ai.google.dev/ | Google AI |
| OpenAI | https://platform.openai.com/ | GPT-4o Vision |
| Claude | https://console.anthropic.com/ | Anthropic |
| Grok | https://console.x.ai/ | xAI |

---

# Final Captioning Tiers (Detailed Reference)

| Tier | Model | VRAM / Cost | Speed | Best For |
|------|-------|-------------|-------|----------|
| **Quick** | Florence-2-large-PromptGen v2.0 | ~1.5 GB | ⚡⚡⚡ | Bulk tagging, initial passes |
| | BLIP-image-captioning-large | ~1 GB | ⚡⚡⚡ | Simple baseline |
| **Standard** | JoyCaption Beta One Q4_K_M | ~5-7 GB | ⚡⚡ | Balanced quality/speed |
| | BLIP-2-opt-2.7b | ~7 GB | ⚡⚡ | Conditional captioning |
| | Qwen2.5-VL-3B-Instruct | ~6-8 GB | ⚡⚡ | Multilingual |
| | Qwen2.5-VL-7B (4-bit) | ~6-8 GB | ⚡ | Better Qwen, quantized |
| **Detailed** | JoyCaption Beta One FP16 | ~16 GB | 🐢 | Best for diffusion training |
| | Qwen2.5-VL-7B-Captioner-Relaxed | ~16 GB | 🐢 | T2I optimized |
| | Qwen2.5-VL-7B (8-bit) | ~10 GB | 🐢 | Comfortable fit |
| | BLIP-2-opt-6.7b | ~14 GB | 🐢 | Larger BLIP-2 |
| **Extreme (API)** | Gemini 2.5 Pro | ~$1.25/M in | 🐌 | Best vision overall |
| | GPT-4o | ~$5/M in | 🐌 | Creative, detailed |
| | Claude Sonnet 4 | ~$3/M in | 🐌 | Follows format precisely |
| | Grok 4 | ~$0.20/M in | 🐌 | Cheapest API option |
