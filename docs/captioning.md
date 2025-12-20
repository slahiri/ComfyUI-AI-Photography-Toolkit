# Image Captioning with Vision Language Models (VLMs)

A comprehensive guide to VLMs for image captioning, with model-specific details and best practices.

## Table of Contents
- [What are VLMs?](#what-are-vlms)
- [Why VLMs for Captioning?](#why-vlms-for-captioning)
- [Captioning Approaches](#captioning-approaches)
  - [Booru Tags (Danbooru/Gelbooru)](#booru-tags-danboorugelbooru)
  - [e621 Tags](#e621-tags)
  - [Natural Language Captions](#natural-language-captions)
  - [Hybrid Approach](#hybrid-approach)
  - [Model-Specific Recommendations](#model-specific-recommendations)
- [Best Practices](#best-practices)
- [Model-Specific Details](#model-specific-details)
  - [Quick Tier](#quick-tier)
  - [Standard Tier](#standard-tier)
  - [Detailed Tier](#detailed-tier)
  - [Extreme (API) Tier](#extreme-api-tier)
- [Prompting Techniques](#prompting-techniques)
- [Vision Model Landscape](#vision-model-landscape)
  - [Vision Foundation Models](#vision-foundation-models)
  - [Contrastive Vision-Language Models](#contrastive-vision-language-models)
  - [Diffusion-Optimized Captioners](#diffusion-optimized-captioners)
  - [Tag-Based Models](#tag-based-models)
  - [Workflow Patterns](#workflow-patterns)
- [Common Pitfalls](#common-pitfalls)

---

## What are VLMs?

Vision Language Models (VLMs) are multimodal AI models that can process both images and text simultaneously. Unlike traditional image classifiers that output fixed labels, VLMs can:

- Generate natural language descriptions of images
- Answer questions about visual content
- Follow specific formatting instructions
- Understand context and nuance

**Key architectures include:**
- **Encoder-Decoder**: Image encoder (ViT) + Text decoder (LLM)
- **Cross-Modal Fusion**: Deep integration of visual and text features
- **Mixture of Experts (MoE)**: Specialized sub-networks for different tasks

## Why VLMs for Captioning?

### For Diffusion Model Training

Automated descriptive captions enable training diffusion models on wider ranges of images without manual labeling. High-quality captions improve generation quality (ref: DALL-E 3 paper).

### Caption Quality Matters

| Caption Quality | Training Result |
|-----------------|-----------------|
| Generic tags | Model learns surface patterns |
| Detailed descriptions | Model learns relationships, composition, style |
| Structured format | Consistent, predictable outputs |

---

## Captioning Approaches

Different diffusion models expect different caption formats. Choosing the right approach is critical for training quality.

### Booru Tags (Danbooru/Gelbooru)

Booru tags are structured metadata keywords from anime imageboard sites. They use a comma-separated format with underscores for multi-word concepts.

**Format:**
```
1girl, long_hair, purple_hair, blue_eyes, school_uniform, sitting, classroom, looking_at_viewer
```

**Tag Categories:**
| Category | Examples |
|----------|----------|
| Subject | `1girl`, `1boy`, `multiple_girls`, `solo` |
| Hair | `long_hair`, `blonde_hair`, `twintails`, `ponytail` |
| Eyes | `blue_eyes`, `red_eyes`, `heterochromia` |
| Clothing | `school_uniform`, `dress`, `armor`, `nude` |
| Expression | `smile`, `blush`, `crying`, `angry` |
| Pose | `sitting`, `standing`, `lying`, `from_behind` |
| Setting | `outdoors`, `classroom`, `beach`, `night` |
| Meta | `highres`, `absurdres`, `masterpiece`, `best_quality` |

**Strengths:**
- Precise control over specific attributes
- Well-understood by anime models (SD1.5, Pony, NAI)
- Extensive tag vocabularies available
- Fast to generate with taggers (WD14, DeepDanbooru)

**Limitations:**
- Ambiguity: "1girl, 1boy, kimono" - who wears the kimono?
- No relationship/composition information
- Limited for realistic/photographic content
- Newer models (Flux, SD3) prefer natural language

**Tools:**
- [WD14 Tagger](https://huggingface.co/SmilingWolf/wd-v1-4-vit-tagger-v2)
- [DeepDanbooru](https://github.com/KichangKim/DeepDanbooru)
- [a1111-sd-webui-tagcomplete](https://github.com/DominikDoom/a1111-sd-webui-tagcomplete)

---

### e621 Tags

e621 uses a similar booru-style system but with vocabulary specific to furry/anthropomorphic content.

**Format:**
```
anthro, female, canine, wolf, blue_fur, yellow_eyes, sitting, forest, solo
```

**Key Differences from Danbooru:**
| Aspect | Danbooru | e621 |
|--------|----------|------|
| Focus | Anime/manga | Furry/anthro |
| Species tags | Limited | Extensive (`canine`, `feline`, `dragon`) |
| Body tags | Human-focused | Anthro-focused (`fur`, `scales`, `feathers`) |
| Rating system | `safe`, `questionable`, `explicit` | `safe`, `questionable`, `explicit` |

**Common Tag Categories:**
| Category | Examples |
|----------|----------|
| Species | `canine`, `feline`, `dragon`, `avian`, `equine` |
| Body | `fur`, `scales`, `feathers`, `anthro`, `feral` |
| Colors | `blue_fur`, `white_scales`, `orange_feathers` |
| Features | `wings`, `tail`, `horns`, `claws` |

**Conversion:**
Tools exist to convert between Danbooru and e621 tag formats:
- [danbooru-e621-tag-list-processor](https://github.com/DraconicDragon/danbooru-e621-tag-list-processor)

---

### Natural Language Captions

Full sentence descriptions that read like natural text. Preferred for newer models.

**Format:**
```
A young woman with long purple hair and blue eyes sits at a wooden desk in a sunlit classroom. She wears a navy school uniform with a red ribbon and looks directly at the viewer with a gentle smile. Soft afternoon light streams through the windows behind her.
```

**Strengths:**
- Clear relationships and composition ("woman wearing kimono stands next to man in suit")
- Better generalization to unseen concepts
- Required for Flux, SD3, PixArt Sigma
- Handles complex scenes naturally
- Better spatial/compositional understanding

**Limitations:**
- Slower to generate (requires VLM)
- More tokens = higher training cost
- May be too verbose for some use cases
- Less precise for specific attributes

**Best Practices:**
- Describe what IS in the image, not what isn't
- Use clear, complete phrases
- Include: subject, setting, lighting, mood, colors
- Be visually grounded (describe what you see)

---

### Hybrid Approach

Combines tags for structure with natural language for context.

**Format:**
```
1girl, purple_hair, blue_eyes, school_uniform. A young woman sits at her desk in a sunlit classroom, looking at the viewer with a gentle smile. Soft afternoon lighting.
```

**When to Use:**
- SDXL (handles both reasonably well)
- When you need precise attributes + context
- Dual-encoder models (CLIP + T5)

**Strategy:**
- Tags for structural elements: `1girl`, `sitting`, `indoors`
- Natural language for complex descriptions
- Keep tags first, then sentences

---

### Model-Specific Recommendations

| Model | Recommended Format | Notes |
|-------|-------------------|-------|
| **SD 1.5** | Booru tags | Trained primarily on tagged data |
| **NAI** | Booru tags | Optimized for Danbooru tags |
| **Pony** | e621/Booru tags | Trained on e621 + Danbooru |
| **SDXL** | Hybrid or Tags | Handles both, tags more precise |
| **Flux** | Natural language | T5 encoder expects sentences |
| **SD3** | Natural language | Trained on natural captions |
| **PixArt** | Natural language | Sentence-based training |
| **Z-Image** | Structured natural language | Long detailed prompts, NO negative prompts |

**Flux-Specific Tips:**
- Use dual captioning: WD14 tags + natural language
- Captions should match your prompting style
- Longer, descriptive sentences work best
- "Clear, complete phrases that are visually grounded"

**Z-Image-Specific Tips:**
- Uses S3-DiT (Scalable Single-Stream Diffusion Transformer) backbone
- **No negative prompts** - guidance scale is 0.0, negatives are ignored
- Long, detailed prompts work best (80-250 words, up to 1024 tokens)
- Structure prompts like camera directions: shot type, subject, clothing, environment, lighting, mood
- Embed constraints positively: "fully clothed, no watermark, correct anatomy"
- Excellent for photorealistic generation and bilingual text rendering (English/Chinese)
- Be very specific - the model is "unopinionated" and needs explicit direction

**Z-Image Prompt Structure:**
```
[Shot type], [Subject description], [Age/appearance], [Clothing details],
[Environment/background], [Lighting], [Mood/atmosphere], [Style/medium],
[Technical notes], [Safety constraints: no watermark, no text, correct anatomy]
```

---

## Best Practices

### Image Preparation
1. **Resolution**: Most models work best with 384px-1024px images
2. **Format**: JPEG, PNG, WebP supported by most models
3. **Quality**: Clear, well-lit images produce better captions
4. **Aspect Ratio**: Some models handle varied ratios better than others

### Prompting Guidelines
1. **Be specific** about what you want described
2. **Provide structure** if you need formatted output
3. **Use examples** (few-shot) for consistent style
4. **Place images before text** in the prompt (when possible)

### Output Considerations
1. **Token limits** vary by model (256-4096 typical)
2. **Temperature 0** for consistent, deterministic output
3. **Post-processing** may be needed for formatting

---

## Model-Specific Details

---

### Quick Tier

#### Florence-2-large-PromptGen v2.0
`MiaoshouAI/Florence-2-large-PromptGen-v2.0`

**Architecture**: Sequence-to-sequence with DaViT encoder
**Parameters**: 0.77B
**VRAM**: ~1.5 GB

**Strengths:**
- Extremely fast inference (~30ms)
- State-of-the-art zero-shot captioning (CIDEr 133.0 on COCO)
- Multiple task modes via prompts
- Excellent for bulk processing

**Limitations:**
- Limited context understanding for complex scenes
- May struggle with multiple objects and relationships
- Less detailed than larger models
- Watermark recognition not always reliable

**Best Prompts:**
```
<CAPTION>           # Brief caption
<DETAILED_CAPTION>  # More detailed
<MORE_DETAILED_CAPTION>  # Most detailed
```

**Best For:** Initial tagging passes, bulk processing, quick previews

---

#### BLIP-image-captioning-large
`Salesforce/blip-image-captioning-large`

**Architecture**: ViT-L/16 encoder + text decoder
**Parameters**: ~470M
**VRAM**: ~1 GB

**Strengths:**
- Very lightweight and fast
- Good baseline quality
- Supports conditional captioning (with text prompts)
- Real-time capable (milliseconds on GPU)

**Limitations:**
- Less detailed than newer models
- Basic vocabulary compared to LLM-based models
- No instruction following

**Best Prompts:**
```python
# Unconditional
processor(image, return_tensors="pt")

# Conditional (guided)
processor(image, "a photograph of", return_tensors="pt")
```

**Best For:** Simple baseline, real-time applications, low-resource environments

---

### Standard Tier

#### JoyCaption Beta One (GGUF Q4_K_M)
`mradermacher/llama-joycaption-beta-one-hf-llava-GGUF`

**Architecture**: LLaVA-based (Llama + CLIP)
**Parameters**: ~7B (quantized to ~4B effective)
**VRAM**: ~5-7 GB

**Strengths:**
- Built specifically for diffusion training
- Free, open, and uncensored
- 10 different caption modes
- Equal coverage of SFW and NSFW content
- Broad diversity in training data

**Limitations:**
- Quantized version has some quality loss
- Slower than Florence-2
- May be verbose in some modes

**Caption Modes:**
| Mode | Description |
|------|-------------|
| `descriptive` | Detailed, verbose descriptions |
| `straightforward` | Balanced detail (recommended) |
| `stable_diffusion_prompt` | SD-style tags |
| `midjourney` | MJ-style prompts |
| `booru` | Booru tag format |

**Best For:** Balanced quality/speed, diffusion training datasets

---

#### BLIP-2-opt-2.7b
`Salesforce/blip2-opt-2.7b`

**Architecture**: Q-Former bridge + OPT-2.7B LLM
**Parameters**: ~2.7B
**VRAM**: ~7 GB

**Strengths:**
- Bridges vision encoder to LLM
- Conditional captioning with prompts
- Good instruction following
- Scalable architecture

**Limitations:**
- Larger than BLIP-1
- Can be slow on CPU

**Best Prompts:**
```
"Describe this image in detail."
"What is shown in this photograph?"
"Generate a caption for this image:"
```

**Best For:** Conditional captioning, prompt-guided descriptions

---

#### Qwen2.5-VL-3B-Instruct
`Qwen/Qwen2.5-VL-3B-Instruct`

**Architecture**: Qwen2.5 LLM + Vision encoder
**Parameters**: 3B
**VRAM**: ~6-8 GB

**Strengths:**
- Excellent multilingual support
- Good instruction following
- Handles varied aspect ratios well
- Natural language grounding

**Limitations:**
- Smaller than 7B variant
- Less detailed for complex scenes

**Prompt Format:**
```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": "Describe this image in detail."}
        ]
    }
]
```

**Best For:** Multilingual applications, varied aspect ratios

---

#### Qwen2.5-VL-7B-Instruct (4-bit)
`Qwen/Qwen2.5-VL-7B-Instruct`

**Architecture**: Qwen2.5 LLM + Vision encoder
**Parameters**: 7B (quantized to ~4B effective)
**VRAM**: ~6-8 GB with 4-bit quantization

**Strengths:**
- Better quality than 3B
- Fits in 8GB VRAM with quantization
- Robust 480px-2560px resolution handling
- Object detection + grounding support

**Limitations:**
- Slower than smaller variants
- 4-bit quantization affects quality slightly
- Object detection accuracy decreases outside optimal resolution

**Best For:** Quality-focused with limited VRAM

---

### Detailed Tier

#### JoyCaption Beta One (FP16)
`fancyfeast/llama-joycaption-beta-one-hf-llava`

**Architecture**: LLaVA-based (Llama + CLIP)
**Parameters**: ~7B (full precision)
**VRAM**: ~16 GB

**Strengths:**
- Highest quality JoyCaption
- No quantization artifacts
- Trained on 2.4M samples
- DPO-optimized (score improved 5.14 → 7.03)

**Limitations:**
- High VRAM requirement
- Slower inference
- Requires good GPU

**Best For:** Final dataset preparation, highest quality captions for training

---

#### Qwen2.5-VL-7B-Captioner-Relaxed
`Ertugrul/Qwen2.5-VL-7B-Captioner-Relaxed`

**Architecture**: Fine-tuned Qwen2.5-VL-7B
**Parameters**: 7B
**VRAM**: ~16 GB

**Strengths:**
- Fine-tuned specifically for T2I captioning
- Relaxed constraints (less restrictive)
- Enhanced detail generation
- Natural language output optimized for image generation

**Limitations:**
- High VRAM
- May be too verbose for some uses

**Best For:** Text-to-image training datasets, detailed descriptions

---

#### BLIP-2-opt-6.7b
`Salesforce/blip2-opt-6.7b`

**Architecture**: Q-Former + OPT-6.7B LLM
**Parameters**: ~6.7B
**VRAM**: ~14 GB

**Strengths:**
- Larger LLM backbone
- Better language generation
- More nuanced descriptions

**Limitations:**
- High VRAM
- Slower than smaller BLIP-2

**Best For:** High-quality conditional captioning

---

### Extreme (API) Tier

#### Gemini 2.5 Pro
**Provider**: Google AI
**URL**: https://ai.google.dev/
**Cost**: ~$1.25/M input tokens

**Strengths:**
- Best overall vision understanding
- Up to 1M input tokens (3600 images/prompt)
- Enhanced object detection with bounding boxes
- Multimodal output (text + images)
- Spatial understanding improvements

**Limitations:**
- API latency
- Cost at scale
- Rate limits

**Token Usage:**
- 258 tokens if both dimensions ≤ 384px
- Larger images tiled into 768x768 (258 tokens each)

**Best For:** Complex scene understanding, batch processing via API

---

#### GPT-4o
**Provider**: OpenAI
**URL**: https://platform.openai.com/
**Cost**: ~$5/M input tokens

**Strengths:**
- Excellent creative descriptions
- Strong instruction following
- Good structured output (JSON, etc.)
- Few-shot learning works well

**Limitations:**
- Higher cost
- Cannot identify specific individuals
- Spatial reasoning limitations

**Best Practices:**
- Use system prompts for consistent style
- Provide few-shot examples for formatting
- Use ontology/schema for structured output
- Include image before text in messages

**Best For:** Creative, detailed captions; structured output

---

#### Claude Sonnet 4
**Provider**: Anthropic
**URL**: https://console.anthropic.com/
**Cost**: ~$3/M input tokens

**Strengths:**
- Strongest vision model from Anthropic
- Excellent visual reasoning
- Accurate text transcription from images
- Follows format instructions precisely
- Charts/graphs interpretation

**Limitations:**
- Cannot identify individuals
- Precise spatial measurements difficult
- Requires human oversight for critical uses

**Best Practices:**
- Place images before text
- Use clear, structured queries
- Supported formats: JPEG, PNG, GIF, WebP

**Best For:** Format-precise outputs, visual reasoning, chart interpretation

---

#### Grok 4
**Provider**: xAI
**URL**: https://console.x.ai/
**Cost**: ~$0.20/M input tokens

**Strengths:**
- Cheapest API option
- Good general captioning
- Fast response times

**Limitations:**
- Less established than competitors
- Fewer advanced features

**Best For:** Cost-sensitive API usage, high-volume processing

---

## Prompting Techniques

### Zero-Shot
Simple direct instruction:
```
Describe this image in detail.
```

### Few-Shot
Provide examples for consistent style:
```
Example 1: [image] → "A woman with long brown hair wearing a red dress..."
Example 2: [image] → "A man in his 30s with short black hair..."
Now describe: [target image]
```

### Structured Output
Request specific format:
```
Describe this image using the following structure:
- Subject:
- Setting:
- Lighting:
- Mood:
- Colors:
```

### Chain-of-Thought
For complex analysis:
```
First, identify the main subject.
Then, describe the background.
Finally, note the lighting and mood.
```

---

## Vision Model Landscape

Beyond the captioning models listed above, here's a broader view of vision models and their roles in image understanding workflows.

### Vision Foundation Models

General-purpose vision backbones that produce features for downstream tasks.

#### DINOv2 (Meta)
Self-supervised Vision Transformer producing excellent general-purpose embeddings.

| Aspect | Details |
|--------|---------|
| **Type** | Self-supervised ViT |
| **Parameters** | ViT-S/B/L/G variants |
| **Training** | 142M images, no labels |
| **Strengths** | Classification, retrieval, segmentation, depth estimation |

**Key Features:**
- Works across domains without fine-tuning
- Student-teacher training mechanism
- Both image-level and patch-level learning
- Zero-shot learning capabilities

**Best For:** Feature extraction, transfer learning, when you need robust visual embeddings

**Resources:** [DINOv2 GitHub](https://github.com/facebookresearch/dinov2) | [Meta AI Blog](https://ai.meta.com/blog/dino-v2-computer-vision-self-supervised-learning/)

---

#### InternVL 2.5 (Shanghai AI Lab)
State-of-the-art open-source multimodal model rivaling GPT-4o.

| Aspect | Details |
|--------|---------|
| **Type** | ViT-MLP-LLM architecture |
| **Parameters** | 1B to 78B variants |
| **Resolution** | Up to 448×448 tiles (dynamic) |
| **Strengths** | Multi-image, video, OCR, reasoning |

**Key Features:**
- First open-source to achieve >70% on MMMU benchmark
- Progressive scaling training strategy
- Supports English and Chinese
- MPO (Mixed Preference Optimization) variants available

**Best For:** Complex scene understanding, document analysis, video comprehension

**Resources:** [InternVL GitHub](https://github.com/OpenGVLab/InternVL) | [HuggingFace](https://huggingface.co/OpenGVLab/InternVL2_5-78B)

---

#### EVA-02 (BAAI)
Scaled-up vision transformer with excellent transfer learning.

| Aspect | Details |
|--------|---------|
| **Type** | Vision Transformer |
| **Strengths** | Downstream classification, minimal fine-tuning needed |

**Best For:** When you need strong classification with efficient fine-tuning

---

### Contrastive Vision-Language Models

Models that align image and text embeddings in a shared space.

#### SigLIP / SigLIP 2 (Google)
Improved CLIP using sigmoid loss instead of softmax.

| Aspect | Details |
|--------|---------|
| **Type** | Contrastive VLM |
| **Improvement** | Better calibration, more efficient training |
| **Batch Size** | Works well with smaller batches |

**Key Differences from CLIP:**
- Sigmoid loss vs softmax contrastive loss
- Better zero-shot classification on ImageNet
- More memory efficient (2x batch size possible)
- Multilingual support (SigLIP 2)

**SigLIP 2 Improvements:**
- Better zero-shot classification
- Improved transfer performance for VLMs
- Better localization and dense prediction

**Best For:** Zero-shot classification, efficient training with limited compute

**Resources:** [SigLIP 2 Blog](https://huggingface.co/blog/siglip2) | [HuggingFace](https://huggingface.co/google/siglip-so400m-patch14-384)

---

#### CLIP (OpenAI)
The classic contrastive vision-language model.

| Aspect | Details |
|--------|---------|
| **Type** | Contrastive VLM |
| **Strengths** | Zero-shot classification, embedding similarity |
| **Ecosystem** | Extensive community tooling |

**Best For:** When you need broad compatibility and community support

---

#### CoCa (Google)
Combines contrastive and captioning objectives.

**Best For:** Tasks requiring both classification and generation

---

### Diffusion-Optimized Captioners

Models specifically designed or fine-tuned for generating prompts for diffusion models.

#### JoyCaption
Purpose-built for diffusion model training captions.

| Aspect | Details |
|--------|---------|
| **Architecture** | LLaVA-based (Llama + CLIP) |
| **Modes** | 10 different caption styles |
| **Training** | 2.4M samples, DPO-optimized |

**Caption Modes:**
| Mode | Output Style |
|------|--------------|
| `descriptive` | Detailed, verbose natural language |
| `straightforward` | Balanced detail (recommended) |
| `stable_diffusion_prompt` | SD-style prompts |
| `midjourney` | MJ-style prompts |
| `booru` | Danbooru tag format |

**Best For:** Training data for Flux, SDXL, SD models

**Resources:** [JoyCaption GitHub](https://github.com/fpgaminer/joycaption) | [Civitai Article](https://civitai.com/articles/14672/joycaption-beta-one-release)

---

#### CogVLM2 / GLM-4V (Tsinghua/Zhipu)
Strong visual understanding with artistic style recognition.

| Aspect | Details |
|--------|---------|
| **Architecture** | ViT + Adapter + LLM |
| **CogVLM2 Base** | Llama3-8B backbone |
| **GLM-4V** | GLM-9B backbone, 13B total |
| **Resolution** | Up to 1344×1344 (CogVLM2), 1120×1120 (GLM-4V) |

**Key Features:**
- GPT-4V equivalent performance
- Excellent OCR capabilities
- Video understanding (CogVLM2-Video)
- Bilingual (English/Chinese)

**Best For:** Detailed descriptions, style identification, OCR-heavy images

**Resources:** [CogVLM2 GitHub](https://github.com/THUDM/CogVLM2) | [GLM-4 GitHub](https://github.com/THUDM/GLM-4)

---

#### LLaVA-NeXT / LLaVA 1.6
Solid VLM with good ComfyUI integration.

| Aspect | Details |
|--------|---------|
| **Architecture** | CLIP + Vicuna/Llama |
| **Strengths** | Good balance of quality and speed |
| **Integration** | Strong ComfyUI node support |

**Best For:** General-purpose captioning with easy integration

---

### Tag-Based Models

Fast taggers for booru-style output.

#### WD14 Tagger / WD Tagger (SmilingWolf)
Danbooru-style tag extraction optimized for anime/illustration.

| Aspect | Details |
|--------|---------|
| **Type** | Image classifier |
| **Output** | Danbooru tags + ratings |
| **Speed** | Very fast |
| **Accuracy** | ~50% better than DeepDanbooru |

**Available Models:**
| Model | Notes |
|-------|-------|
| `wd-v1-4-moat-tagger-v2` | Newest, MOAT architecture |
| `wd-v1-4-convnext-tagger-v2` | Most popular |
| `wd-v1-4-vit-tagger-v2` | ViT-based |

**Best For:** Anime/illustration tagging, fast batch processing, SDXL/Pony prompts

**Resources:** [HuggingFace Space](https://huggingface.co/spaces/SmilingWolf/wd-tagger) | [ComfyUI Node](https://github.com/pythongosssss/ComfyUI-WD14-Tagger)

---

#### DeepDanbooru
Original Danbooru tagger, still widely used.

| Aspect | Details |
|--------|---------|
| **Type** | CNN classifier |
| **Output** | Danbooru tags |
| **Speed** | Fast |

**Best For:** Legacy workflows, when WD14 unavailable

---

### Model Comparison Table

| Model | Type | Best For | Speed | VRAM |
|-------|------|----------|-------|------|
| **DINOv2** | Foundation | Feature extraction | Fast | Low |
| **InternVL 2.5** | VLM | Complex scenes, OCR | Medium | High |
| **SigLIP** | Contrastive | Zero-shot classification | Fast | Low |
| **JoyCaption** | Captioner | Diffusion training | Medium | Medium |
| **CogVLM2** | VLM | Detailed descriptions | Slow | High |
| **WD14 Tagger** | Tagger | Anime tags | Very Fast | Low |
| **Florence-2** | Multi-task | General captioning | Fast | Low |
| **Qwen2.5-VL** | VLM | Multilingual, instruction | Medium | Medium |

---

### Workflow Patterns

#### Two-Stage Captioning (Recommended for SD/SDXL)
```
Image → WD Tagger (tags) + VLM (description) → LLM combines → Final Prompt
```

#### Direct Captioning (Recommended for Flux)
```
Image → Qwen2-VL / JoyCaption → Optional cleanup → Flux
```

#### Hybrid for SDXL
```
Image → WD14 (structural tags) + Florence-2 (description) → Merge
```

#### Production Pipeline
```
Batch Images → Florence-2 (quick pass) → Filter → JoyCaption (detailed) → Training
```

### ComfyUI Nodes

| Node | Model | Purpose |
|------|-------|---------|
| `ComfyUI-JoyCaption` | JoyCaption | Diffusion-optimized captions |
| `ComfyUI-Florence2` | Florence-2 | Multi-task captioning |
| `ComfyUI-WD14-Tagger` | WD14 | Booru tag extraction |
| `ComfyUI-LLaVA-Captioner` | LLaVA | General VLM captioning |
| `ComfyUI-InternVL` | InternVL | Advanced VLM tasks |

---

## Common Pitfalls

### 1. Hallucinations
VLMs may invent details not present in images. Always verify critical outputs.

### 2. Copy-from-Examples
Small models may copy few-shot examples instead of analyzing the image. Use diverse examples or zero-shot for small models.

### 3. Resolution Issues
- Too small: Missing details
- Too large: Slow processing, may be resized anyway

### 4. Verbosity
Some models are overly verbose. Use temperature=0 and explicit length instructions.

### 5. Format Breaking
Models may not consistently follow format instructions. Consider post-processing.

---

## References

### General VLM Resources
- [Vision Language Models Explained - HuggingFace](https://huggingface.co/blog/vlms)
- [VLMs 2025 - HuggingFace](https://huggingface.co/blog/vlms-2025)
- [Vision Language Models Guide - Encord](https://encord.com/blog/vision-language-models-guide/)
- [Awesome VLM Architectures - GitHub](https://github.com/gokayfem/awesome-vlm-architectures)

### Model-Specific
- [Florence-2 - AssemblyAI](https://www.assemblyai.com/blog/florence-2-how-it-works-how-to-use)
- [JoyCaption - GitHub](https://github.com/fpgaminer/joycaption)
- [JoyCaption Beta One - Civitai](https://civitai.com/articles/14672/joycaption-beta-one-release)
- [BLIP - Salesforce](https://www.salesforce.com/blog/blip-bootstrapping-language-image-pretraining/)
- [Qwen2.5-VL Captioner - HuggingFace](https://huggingface.co/Ertugrul/Qwen2.5-VL-7B-Captioner-Relaxed)

### API Providers
- [Claude Vision Docs](https://docs.claude.com/en/docs/build-with-claude/vision)
- [Gemini Image Understanding](https://ai.google.dev/gemini-api/docs/image-understanding)
- [GPT-4 Image Captioning Cookbook](https://cookbook.openai.com/examples/tag_caption_images_with_gpt4v)
