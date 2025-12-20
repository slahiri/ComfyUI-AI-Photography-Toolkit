# Qwen VL Prompting: Comprehensive Analysis

## Executive Summary

Qwen VL (Vision-Language) is Alibaba's family of multimodal large language models designed to process both visual and textual information. The family has evolved through several generations: **Qwen-VL → Qwen2-VL → Qwen2.5-VL → Qwen3-VL**, with each iteration bringing significant improvements in capabilities, prompting patterns, and output formats.

This analysis covers prompting best practices, model-specific techniques, and practical implementation patterns across the Qwen VL family.

---

## 1. Model Family Overview

### Current Generation (As of Late 2025)

| Model | Sizes | Key Features |
|-------|-------|--------------|
| **Qwen3-VL** | 2B, 4B, 8B, 32B, 30B-A3B (MoE), 235B-A22B (MoE) | Thinking mode, 256K context, 32-language OCR, 3D grounding |
| **Qwen2.5-VL** | 3B, 7B, 32B, 72B | Absolute coordinates, QwenVL HTML format, visual agent capabilities |
| **Qwen2-VL** | 2B, 7B, 72B | Dynamic resolution, M-RoPE, 20+ min video understanding |

### Model Variants

Each generation offers two key variants:
- **Instruct**: Optimized for speed and direct responses
- **Thinking**: Enhanced reasoning with explicit chain-of-thought (15-25% improvement on complex tasks)

---

## 2. Core Prompting Architecture

### Message Format Structure

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},  # Optional
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "path_or_url"},
            {"type": "text", "text": "Your prompt here"}
        ]
    }
]
```

### Multi-Modal Input Types

```python
# Image from URL
{"type": "image", "image": "https://example.com/image.jpg"}

# Image from local path
{"type": "image", "image": "file:///path/to/image.jpg"}

# Video input
{"type": "video", "video": "path_or_url", "fps": 1}

# With resolution control
{
    "type": "image",
    "image": "path_or_url",
    "resized_height": 280,
    "resized_width": 420,
    "min_pixels": 256*28*28,
    "max_pixels": 1280*28*28
}
```

---

## 3. Task-Specific Prompting Patterns

### 3.1 Object Detection & Visual Grounding

**Pattern: Request structured JSON output with explicit format specification**

```
Detect all [OBJECT_TYPE] in the image and return their locations in the form of coordinates.
The format of output should be like {"bbox_2d": [x1, y1, x2, y2], "label": "object_name", "sub_label": "attribute"}.
```

**Example Output:**
```json
[
    {"bbox_2d": [341, 258, 397, 360], "label": "motorcyclist", "sub_label": "not wearing helmet"},
    {"bbox_2d": [212, 332, 274, 448], "label": "motorcyclist", "sub_label": "wearing helmet"}
]
```

**Key Insight:** Qwen2.5-VL and Qwen3-VL use **absolute pixel coordinates** (not normalized 0-1), making direct annotation possible.

### 3.2 Keypoint Detection

**Pattern: Request point-based localization**

```
Identify [OBJECTS] and detect their key points such as [FEATURES], returning them in the form of points.
The primary label is [MAIN_IDENTIFIER], and the secondary labels include [FEATURES_LIST].
```

**Example Output:**
```json
[
    {"point_2d": ["394", "105"], "label": "LeBron James", "label2": "head"},
    {"point_2d": ["876", "131"], "label": "Stephen Curry", "label2": "head"}
]
```

### 3.3 Counting with Verification

**Pattern: Detection-first counting**

```
Count the number of [OBJECTS] in the figure. To ensure accuracy, first detect their key points, then give the total number.
```

This approach forces the model to enumerate before counting, dramatically improving accuracy.

### 3.4 OCR & Text Extraction

**Basic OCR:**
```
Read all texts in the image, output in lines.
```

**Text Spotting with Localization:**
```
Spotting all the text in the image with line-level, and output in JSON format.
```

**Example Output:**
```json
[
    {"bbox_2d": [108, 175, 496, 230], "text_content": "AuntieAnne's"},
    {"bbox_2d": [49, 429, 252, 450], "text_content": "CINNAMON SUGAR"}
]
```

**Multilingual OCR:** Qwen3-VL supports 32 languages including Arabic, Chinese, Japanese, Korean, and most European languages.

### 3.5 Key Information Extraction (KIE)

**Pattern: Specify exact fields to extract**

```
Extract following information from the receipt: ['invoice_number', 'date', 'total_amount', 'vendor_name'], output in JSON.
```

**Structured Extraction Prompt:**
```
Extract the invoice number, train number, departure station, arrival station, departure date and time,
seat number, seat class, ticket price, ID card number, and passenger name from the train ticket image.
Accurately extract the key information without omissions or fabrications.
Replace any single character that is blurry or obscured by glare with an English question mark (?).
```

### 3.6 Document Parsing (QwenVL HTML Format)

**Pattern: Request specialized HTML output**

```
QwenVL HTML
```

Or for enhanced parsing:
```
QwenVL HTML with image caption
```

**Output Features:**
- `data-bbox` attributes for spatial positioning
- Semantic HTML tags (`<h1>`, `<p>`, `<div class="formula">`)
- Embedded LaTeX for mathematical content
- Image captions with natural language descriptions

### 3.7 Video Understanding

**Event Localization:**
```
Give the query: '[EVENT_DESCRIPTION]', when does the described content occur in the video?
Use seconds for time format.
```

**Structured Video Captioning:**
```
Localize a series of activity events in the video, output the start and end timestamp for each event,
and describe each event with sentences. Provide the result in json format with 'mm:ss.ff' format for time depiction.
```

**Long Video Comprehension:**
```
Could you provide a comprehensive overview of the competition's progress?
```

Qwen3-VL supports up to 256K tokens (expandable to 1M), enabling 1+ hour video understanding.

---

## 4. Advanced Prompting Techniques

### 4.1 Thinking Mode Control (Qwen3-VL)

**Enable explicit reasoning:**
```
/think
[Your complex reasoning question]
```

**Disable for speed:**
```
/no_think
[Your simple question]
```

**Via API:**
```python
# Thinking mode
enable_thinking=True
Temperature=0.6, TopP=0.95, TopK=20, MinP=0

# Non-thinking mode
enable_thinking=False
Temperature=0.7, TopP=0.8, TopK=20, MinP=0
```

**Use Cases:**
- **Thinking**: Mathematical reasoning, complex document analysis, multi-step inference
- **Non-Thinking**: Simple VQA, basic OCR, object identification

### 4.2 System Message Patterns

**For Chart Analysis:**
```python
system_message = """You are a Vision Language Model specialized in interpreting visual data from chart images.
Your task is to analyze the provided chart image and respond to queries with concise answers,
usually a single word, number, or short phrase. The charts include a variety of types
(e.g., line charts, bar charts) and contain colors, labels, and text.
Focus on delivering accurate, succinct answers based on the visual information."""
```

**For Document Processing:**
```python
system_message = """You are an expert document analyzer. Extract information accurately
and provide structured output in JSON format. If information is unclear or missing,
indicate uncertainty rather than guessing."""
```

### 4.3 Grounding Trigger Phrases

Research indicates specific phrases trigger different grounding behaviors:

| Trigger | Effect |
|---------|--------|
| `"Find [X]"` | Object detection mode |
| `"with grounding"` | Enables bounding box output |
| `"Describe image in details with grounding"` | Full image captioning with spatial annotations |
| `"Detect..."` | Object detection with localization |
| `"Locate..."` | Point or bounding box output |

### 4.4 Resolution Optimization

```python
# For detailed analysis
min_pixels = 512 * 28 * 28
max_pixels = 2048 * 28 * 28

# For speed/efficiency
min_pixels = 256 * 28 * 28
max_pixels = 1280 * 28 * 28

processor = AutoProcessor.from_pretrained(
    model_name,
    min_pixels=min_pixels,
    max_pixels=max_pixels
)
```

---

## 5. Output Format Control

### 5.1 JSON Output Patterns

**Explicit Format Specification:**
```
Output the results in JSON format with the following structure:
{
    "field1": "description",
    "field2": ["list", "items"],
    "confidence": 0.0-1.0
}
```

**Array Output:**
```
Return all detected objects as a JSON array.
```

### 5.2 Multiple Choice Standardization

```
Please show your choice in the answer field with only the choice letter, e.g., "answer": "C".
```

### 5.3 Table Extraction

```
Extract the table from this image and output it as JSON.
```

---

## 6. Model-Specific Considerations

### Qwen3-VL Specifics

- **Context**: 256K tokens native, expandable to 1M
- **OCR**: 32 languages with low-light and rotation robustness
- **3D Grounding**: Supports spatial reasoning for embodied AI
- **Visual Coding**: Can generate HTML/CSS/JS from mockups
- **Agent Capabilities**: Computer/mobile use, GUI interaction

### Qwen2.5-VL Specifics

- **Coordinates**: Uses actual pixel dimensions (not normalized)
- **Special Format**: QwenVL HTML for document parsing
- **Timestamps**: Absolute time encoding for video
- **Visual Agent**: Native support without task-specific finetuning

### Qwen2-VL Specifics

- **Special Tokens**: `<|object_ref_start|>`, `<|object_ref_end|>`, `<|box_start|>`, `<|box_end|>`
- **Coordinate Format**: Normalized to [0, 1000] range
- **Legacy Prompts**: `"Generate the caption in English with grounding:"`

---

## 7. Common Anti-Patterns to Avoid

### ❌ Vague Prompts
```
# Bad
What's in this image?

# Good
Identify all vehicles in this image and return their locations as bounding boxes in JSON format.
```

### ❌ Missing Format Specification
```
# Bad
Extract the text.

# Good
Extract all text from this document and output in JSON format with bounding boxes.
```

### ❌ Greedy Decoding with Thinking Mode
```python
# Bad - causes repetition and degradation
temperature=0.0, enable_thinking=True

# Good
temperature=0.6, top_p=0.95, enable_thinking=True
```

### ❌ Ignoring Resolution Constraints
```python
# Bad - may exceed memory or lose detail
# No min/max_pixels specified

# Good
min_pixels=256*28*28, max_pixels=1280*28*28
```

---

## 8. Performance Optimization

### Sampling Parameters

| Mode | Temperature | TopP | TopK | MinP |
|------|-------------|------|------|------|
| Thinking | 0.6 | 0.95 | 20 | 0 |
| Non-Thinking | 0.7 | 0.8 | 20 | 0 |
| Precise OCR | 0.0-0.3 | 0.8 | 20 | 0 |

### Preventing Infinite Loops
If encountering repetition, set `presence_penalty=1.5`

### Memory Management
- Use quantized models (AWQ, GPTQ, FP8) for deployment
- Enable flash_attention_2 for multi-image/video scenarios
- Consider tensor parallelism for 72B+ models

---

## 9. Deployment Recommendations

### Framework Options

| Framework | Use Case |
|-----------|----------|
| **vLLM** | Production serving with high throughput |
| **SGLang** | Research and complex pipelines |
| **Ollama** | Local development and testing |
| **LMStudio** | Desktop experimentation |

### vLLM Example
```bash
vllm serve Qwen/Qwen3-VL-8B-Instruct \
    --enable-auto-tool-choice \
    --tool-call-parser hermes
```

---

## 10. Practical Prompt Templates

### Template: Document Information Extraction
```
You are analyzing a [DOCUMENT_TYPE]. Extract the following fields:
- [FIELD_1]: [description]
- [FIELD_2]: [description]
- [FIELD_3]: [description]

Rules:
1. If a field is unclear, use "?" for uncertain characters
2. Output in JSON format
3. Include confidence scores where possible
```

### Template: Visual Grounding with Context
```
Task: [TASK_TYPE]
Object of Interest: [OBJECT_DESCRIPTION]
Output Format: {"bbox_2d": [x1, y1, x2, y2], "label": "name", "attributes": {...}}
Additional Context: [CONTEXT]
```

### Template: Video Event Extraction
```
Analyze this video and:
1. Identify all distinct events/activities
2. For each event, provide:
   - Start timestamp (mm:ss.ff)
   - End timestamp (mm:ss.ff)
   - Description
3. Output as JSON array
```

---

## 11. Comparison: When to Use Which Variant

| Task | Recommended Model | Reasoning |
|------|-------------------|-----------|
| Simple VQA | Qwen3-VL-Instruct (4B/8B) | Fast, efficient |
| Complex Math | Qwen3-VL-Thinking (8B+) | 15-25% accuracy improvement |
| Document OCR | Qwen2.5-VL (7B+) | Proven QwenVL HTML format |
| Long Video | Qwen3-VL (any) | 256K+ context |
| Edge/Mobile | Qwen3-VL-2B | Optimized for edge AI |
| Production | Qwen2.5-VL-72B or Qwen3-VL-32B | Best accuracy/speed balance |

---

## 12. Known Limitations

1. **Audio**: No audio processing in videos
2. **Counting Accuracy**: Complex scenes may have counting errors
3. **Individual/IP Recognition**: Limited coverage of specific celebrities/brands
4. **Complex Instructions**: Multi-step instructions may require decomposition
5. **Data Timeliness**: Training data cutoff affects recognition of recent entities

---

## 13. Detailed Image Description & Captioning

This section covers best practices for generating detailed, high-quality image descriptions using Qwen VL models—particularly useful for creating training datasets for Stable Diffusion, Flux, and other text-to-image models.

### 13.1 Basic Caption Prompts (Simple to Complex)

| Prompt | Detail Level | Use Case |
|--------|--------------|----------|
| `"Describe this image."` | Medium | General captioning |
| `"Describe this image in detail."` | High | Detailed analysis |
| `"What is in this image? Include details."` | Medium-High | Balanced captioning |
| `"Provide a detailed caption for the image."` | High | Training data generation |
| `"Generate the caption in English with grounding:"` | High + Spatial | Caption with bounding boxes |

### 13.2 Structured Prompt Formula for Rich Descriptions

**Formula Pattern:**
```
[Main subject] + [action/pose] + [location/setting] + [lighting] + [style] + [mood/atmosphere] + [additional details]
```

**Example Prompt:**
```
Describe this image in detail, covering:
1. Main subject (appearance, clothing, pose)
2. Environment and background
3. Lighting conditions
4. Color palette
5. Artistic style or photographic technique
6. Mood and atmosphere
7. Any text or signage visible
```

### 13.3 Professional Captioning System Prompts

**For Text-to-Image Training Data:**
```python
system_message = """You are an expert image captioner for AI training datasets.
Generate detailed, natural language descriptions that capture:
- Subject details (appearance, clothing, expression, pose)
- Spatial relationships and positions using natural language
- Environment, setting, and background elements
- Lighting quality and direction
- Color palette and visual style
- Mood, atmosphere, and emotional tone
- Camera angle and composition

Output format: Flowing prose suitable for text-to-image model training.
Avoid: Abstract concepts, marketing language, technical jargon."""
```

**For Tag Generation (Booru-style):**
```
Your task is to generate a clean list of comma-separated tags for a text-to-image AI,
based *only* on the visual information in the image.
Limit output to maximum 50 unique tags.
Focus on: subject, pose, clothing, environment, colors, lighting, composition.
Do not include abstract concepts, interpretations, or technical jargon.
Avoid repeating tags.
```

### 13.4 Prompt Templates by Use Case

#### General Detailed Description
```
Describe this image in comprehensive detail, including the main subject,
their appearance and actions, the setting and environment, lighting conditions,
color palette, and overall mood. Use natural, flowing language.
```

#### For Stable Diffusion/Flux Training
```
Create a detailed caption for this image suitable for AI image generation training.
Include: subject description with specific details, pose and expression,
clothing and accessories, environment and background, lighting type and direction,
art style or photographic technique, color scheme, and atmosphere.
Format as a single flowing paragraph.
```

#### For Video Frame Description
```
Summarize the key events in this video sequence. For each major event, describe:
- What happens
- Who is involved
- The visual details of the scene
- The approximate timing within the sequence
```

#### Cinematic/Video Generation Prompts
```json
{
  "shot": {
    "composition": "string - lens, framing, depth of field",
    "camera_motion": "string - movement description"
  },
  "subject": {
    "description": "string - detailed appearance",
    "wardrobe": "string or null"
  },
  "scene": {
    "location": "string",
    "time_of_day": "string",
    "environment": "string"
  },
  "visual_details": {
    "action": "string",
    "action_sequence": ["0-1s: ...", "1-2s: ..."],
    "props": "string or null"
  },
  "cinematography": {
    "lighting": "string",
    "color_grading": "string"
  }
}
```

### 13.5 Key Elements to Request in Descriptions

| Category | Elements to Include |
|----------|---------------------|
| **Subject** | Gender, age, ethnicity, body type, hair, facial features, expression, pose |
| **Clothing** | Style, color, material, accessories, jewelry |
| **Environment** | Location, indoor/outdoor, architecture, nature elements |
| **Lighting** | Natural/artificial, direction, quality (soft/hard), time of day |
| **Composition** | Camera angle, framing, depth of field, focus point |
| **Style** | Photorealistic, illustration, anime, painting style, artistic medium |
| **Mood** | Atmosphere, emotional tone, energy level |
| **Colors** | Dominant palette, accents, saturation, contrast |
| **Technical** | Resolution quality, grain, bokeh, motion blur |

### 13.6 Specialized Captioning Models

Several fine-tuned Qwen VL models are optimized specifically for captioning:

| Model | Base | Specialization |
|-------|------|----------------|
| `Ertugrul/Qwen2.5-VL-7B-Captioner-Relaxed` | Qwen2.5-VL-7B | Text-to-image dataset creation, relaxed constraints |
| `prithivMLmods/Qwen3-VL-8B-Abliterated-Caption-it` | Qwen3-VL-8B | Uncensored detailed captioning |
| `impactframes/Qwen2-VL-7B-Captioner` | Qwen2-VL-7B | Enhanced detail with natural language positions |

**Using Captioner-Relaxed:**
```python
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Describe this image."}
        ]
    }
]
```

### 13.7 Caption with Spatial Grounding

**Prompt for Caption + Bounding Boxes:**
```
Generate the caption in English with grounding:
```

**Output Format:**
```
<ref>Woman</ref><box>(451,379),(731,806)</box> and <ref>her dog</ref><box>(219,424),(576,896)</box> playing on the beach
```

**For Natural Language Positions (No Coordinates):**
```
Describe this image in detail, specifying the location of each subject
using natural language (e.g., "in the foreground", "to the left", "in the background").
```

### 13.8 Controlling Output Length and Detail

| Parameter | Effect |
|-----------|--------|
| `max_new_tokens=128` | Brief caption (1-2 sentences) |
| `max_new_tokens=256` | Standard detail (paragraph) |
| `max_new_tokens=512` | Comprehensive description |
| `max_new_tokens=1024+` | Exhaustive analysis |

**Prompt Modifiers for Length:**
- "briefly describe" → shorter output
- "describe in detail" → longer output
- "provide a comprehensive analysis" → exhaustive output
- "in one sentence" → minimal output

### 13.9 Multi-Image Comparison Captioning

```
Compare these images and describe:
1. Common elements across all images
2. Key differences between them
3. The relationship or narrative connecting them
```

### 13.10 Best Practices Summary

1. **Be Explicit About Format**: Specify whether you want prose, tags, JSON, or structured output
2. **Request Specific Categories**: List the exact elements you want described (subject, lighting, mood, etc.)
3. **Use System Messages**: For consistent output across many images, use a detailed system prompt
4. **Control Resolution**: Higher image resolution → more detail captured → better descriptions
5. **Match Model to Task**: Use Captioner-Relaxed variants for training data, base models for general VQA
6. **Iterate on Prompt Phrasing**: Caption quality varies with prompt wording—test variations
7. **Consider Temperature**: Lower temp (0.3-0.5) for consistent, factual descriptions; higher (0.7+) for creative captions

### 13.11 ComfyUI Integration Notes

For your ComfyUI workflows, the **ComfyUI-QwenVL** extension provides preset prompts:

| Preset | Purpose |
|--------|---------|
| `🏷️ Tag List` | Comma-separated tags for T2I |
| `📝 Detailed Caption` | Flowing prose description |
| `🎬 Video Prompt` | Cinematic JSON structure |
| `🧩 Prompt Refine & Expand` | Enhance existing prompts |

**Custom Prompt in ComfyUI:**
```
Based on the [observed element] and [setting], create a [style] text-to-image prompt
emphasizing [specific qualities] and [lighting/mood preferences].
```

---

## References

1. Qwen2-VL Technical Report (arXiv:2409.12191)
2. Qwen2.5-VL Technical Report (arXiv:2502.13923)
3. Qwen3 Technical Report (arXiv:2505.09388)
4. GitHub: https://github.com/QwenLM/Qwen3-VL
5. Official Blog: https://qwenlm.github.io/blog/qwen2.5-vl/
6. Ertugrul/Qwen2.5-VL-7B-Captioner-Relaxed: https://huggingface.co/Ertugrul/Qwen2.5-VL-7B-Captioner-Relaxed
7. ComfyUI-QwenVL: https://github.com/1038lab/ComfyUI-QwenVL
8. prithivMLmods/Qwen3-VL-8B-Abliterated-Caption-it: https://huggingface.co/prithivMLmods/Qwen3-VL-8B-Abliterated-Caption-it
