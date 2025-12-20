# SID Photography Toolkit v5.0 - Architecture

## Overview

Clean, modular captioning toolkit with Qwen-focused model lineup.

**API**: ComfyUI V1 (Legacy) - Maximum compatibility with all ComfyUI versions.

---

## Nodes (3 Total)

| Node | Purpose | Required |
|------|---------|----------|
| **SID_Caption** | Core captioning | Yes |
| **SID_CaptionAdvanced** | Model selection within tier | No |
| **SID_CaptionOptions** | Customization options | No |

**Note:** WD14 tagging is embedded in the pipeline (Balanced/Detailed tiers), not a separate node.

---

## Node Specifications

### SID_Caption (Simple)

```
┌─────────────────────────────────────────┐
│  SID_Caption                            │
├─────────────────────────────────────────┤
│ image ●─────────────────────────────    │
│ tags ●ₒₚₜ ──────────────────────────    │
│ options ●ₒₚₜ ───────────────────────    │
│                                         │
│ Quality    [▼ Balanced            ]     │
│            ├─ Fast (single pass)        │
│            ├─ Balanced (double pass)    │
│            └─ Detailed (double pass+)   │
│                                         │
│ Style      [▼ Natural             ]     │
│            ├─ Natural                   │
│            ├─ Tags                      │
│            └─ Hybrid                    │
│                                         │
│ Length     [▼ Medium              ]     │
│            ├─ Short (~50 words)         │
│            ├─ Medium (~120 words)       │
│            └─ Long (~250 words)         │
│                                         │
├─────────────────────────────────────────┤
│ caption ●───────────────────────────    │
└─────────────────────────────────────────┘
```

**Inputs:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| image | IMAGE | Yes | Input image |
| tags | STRING | No | External tags (auto-generated in Balanced/Detailed) |
| options | SID_OPTIONS | No | From SID_CaptionOptions |
| quality | dropdown | Yes | Fast/Balanced/Detailed |
| style | dropdown | Yes | Natural/Tags/Hybrid |
| length | dropdown | Yes | Short/Medium/Long |

**Outputs:**
| Name | Type | Description |
|------|------|-------------|
| caption | STRING | Generated caption |

---

### SID_CaptionAdvanced (Power Users)

```
┌─────────────────────────────────────────┐
│  SID_CaptionAdvanced                    │
├─────────────────────────────────────────┤
│ image ●─────────────────────────────    │
│ tags ●ₒₚₜ ──────────────────────────    │
│ options ●ₒₚₜ ───────────────────────    │
│                                         │
│ Quality    [▼ Balanced            ]     │
│                                         │
│ Model      [▼ Qwen-7B-Captioner   ]     │
│            ├─ ── Fast ──                │
│            ├─ Qwen-3B                   │
│            ├─ Florence-2-PromptGen      │
│            ├─ ── Balanced ──            │
│            ├─ Qwen-7B-Captioner ⭐       │
│            ├─ Qwen-7B                   │
│            ├─ JoyCaption                │
│            ├─ ── Detailed ──            │
│            ├─ Qwen-7B-Captioner-F16 ⭐   │
│            ├─ Qwen-7B-F16               │
│            └─ JoyCaption-F16            │
│                                         │
│ Quantization [▼ Q4               ]      │
│              ├─ Q4 (4-bit, ~25% VRAM)   │
│              ├─ Q8 (8-bit, ~50% VRAM)   │
│              └─ F16 (full precision)    │
│                                         │
│ Style      [▼ Natural             ]     │
│ Length     [▼ Medium              ]     │
│                                         │
├─────────────────────────────────────────┤
│ caption ●───────────────────────────    │
│ tags_used ●─────────────────────────    │
└─────────────────────────────────────────┘
```

**Additional Inputs:**
| Name | Type | Description |
|------|------|-------------|
| model | dropdown | Specific model within tier |
| quantization | dropdown | Q4/Q8/F16 |

**Additional Outputs:**
| Name | Type | Description |
|------|------|-------------|
| tags_used | STRING | Tags used in context |

---

### SID_CaptionOptions

```
┌─────────────────────────────────────────┐
│  SID_CaptionOptions                     │
├─────────────────────────────────────────┤
│ ☐ include_lighting                      │
│ ☐ include_pose                          │
│ ☐ include_colors                        │
│ ☐ include_composition                   │
│ ☐ exclude_background                    │
│                                         │
│ ── Target Model ──                      │
│ ☐ for_flux                              │
│ ☐ for_sdxl                              │
│ ☐ for_pony                              │
│                                         │
│ custom [                          ]     │
│                                         │
├─────────────────────────────────────────┤
│ options ●───────────────────────────    │
└─────────────────────────────────────────┘
```

---

## Pass Strategy (Embedded Tagging)

| Quality | Strategy | Pass 1 | Pass 2 |
|---------|----------|--------|--------|
| **Fast** | Single | VLM only | - |
| **Balanced** | Double | WD14 tags | VLM + context |
| **Detailed** | Double+ | WD14 + Florence | VLM + context |

### Flow Diagram

```
FAST (Single Pass):
  Image ──► Qwen-3B ──► Caption

BALANCED (Double Pass):
  Image ──► WD14 ──► tags
    │                  │
    └──► Qwen-7B ◄─────┘ (with tag context)
              │
              ▼
           Caption

DETAILED (Double Pass+):
  Image ──► WD14 ──────────► tags
    │                          │
    ├──► Florence-2 ──► quick  │
    │                  caption │
    │                     │    │
    └──► Qwen-7B-F16 ◄────┴────┘ (with full context)
              │
              ▼
           Caption
```

---

## Model Registry

### Fast Tier (~2-4GB VRAM)

| Model | Repo | Default |
|-------|------|---------|
| **Qwen-3B** | `Qwen/Qwen2.5-VL-3B-Instruct` | ⭐ Yes |
| Florence-2-PromptGen | `MiaoshouAI/Florence-2-large-PromptGen-v2.0` | No |

### Balanced Tier (~6-10GB VRAM)

| Model | Repo | Default |
|-------|------|---------|
| **Qwen-7B-Captioner** | `Ertugrul/Qwen2.5-VL-7B-Captioner-Relaxed` | ⭐ Yes |
| Qwen-7B | `Qwen/Qwen2.5-VL-7B-Instruct` | No |
| JoyCaption | `fancyfeast/llama-joycaption-beta-one-hf-llava` | No |

### Detailed Tier (~12-20GB VRAM)

| Model | Repo | Default |
|-------|------|---------|
| **Qwen-7B-Captioner-F16** | `Ertugrul/Qwen2.5-VL-7B-Captioner-Relaxed` | ⭐ Yes |
| Qwen-7B-F16 | `Qwen/Qwen2.5-VL-7B-Instruct` | No |
| JoyCaption-F16 | `fancyfeast/llama-joycaption-beta-one-hf-llava` | No |

### Tagger

| Model | Repo |
|-------|------|
| WD-SwinV2 | `SmilingWolf/wd-swinv2-tagger-v3` |

---

## Quantization

| Level | Method | VRAM Savings | Platforms |
|-------|--------|--------------|-----------|
| **Q4** | BitsAndBytes 4-bit NF4 | ~75% | CUDA only |
| **Q8** | BitsAndBytes 8-bit | ~50% | CUDA only |
| **F16** | bfloat16 | None | All (CUDA, MPS, CPU) |

---

## File Structure

```
ComfyUI-AI-Photography-Toolkit/
├── __init__.py                 # Extension entry point
├── requirements.txt
├── pyproject.toml
│
├── nodes/
│   ├── __init__.py
│   ├── caption.py              # SID_Caption + SID_CaptionAdvanced
│   └── options.py              # SID_CaptionOptions
│
├── core/
│   ├── __init__.py
│   ├── pipeline.py             # CaptionPipeline (orchestration)
│   ├── platform.py             # GPU/platform detection
│   ├── prompts.py              # Prompt building
│   ├── output.py               # Output cleaning
│   │
│   └── models/
│       ├── __init__.py
│       ├── factory.py          # ModelFactory
│       ├── base.py             # BaseCaptionModel
│       ├── qwen.py             # QwenVLModel
│       ├── joycaption.py       # JoyCaptionModel
│       ├── florence.py         # FlorenceModel
│       └── tagger.py           # WD14Tagger (embedded, used by pipeline)
│
├── config/
│   ├── models.json             # Model definitions
│   ├── styles.json             # Style templates
│   └── options.json            # Option definitions
│
└── docs/
    ├── architecture.md         # This file
    ├── models.md               # Model reference
    ├── research.md             # Existing nodes research
    └── history.md              # Version history analysis
```

---

## Dependencies

```txt
# requirements.txt

transformers>=4.45.0
accelerate>=0.25.0
bitsandbytes>=0.41.0
torch>=2.0.0
Pillow>=9.0.0
```

**No llama.cpp. No GGUF. Transformers ecosystem only.**

---

## Flow-Based Pipeline Architecture

### Concept

Each pipeline is a sequence of independent, swappable steps.

```
┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐
│  Step   │──►│  Step   │──►│  Step   │──►│  Step   │
└─────────┘   └─────────┘   └─────────┘   └─────────┘

PipelineContext flows through all steps
```

### Core Components

| Component | Purpose |
|-----------|---------|
| `PipelineContext` | Data object flowing through steps |
| `PipelineStep` | Base class for steps |
| `Pipeline` | Runner that executes steps |

### Predefined Flows

```
Fast (single pass):
  build_prompt → caption_vlm → clean_output

Balanced (double pass):
  extract_tags → build_prompt → caption_vlm → clean_output

Detailed (double pass+):
  extract_tags → quick_caption → build_prompt → caption_vlm → clean_output

Future RAG (Release 2.0):
  extract_tags → search_examples → build_prompt → caption_vlm → clean_output
```

### Step Implementations

| Step | Description |
|------|-------------|
| `ExtractTagsStep` | WD14 tagger |
| `QuickCaptionStep` | Florence-2 quick caption |
| `SearchExamplesStep` | RAG search (future) |
| `BuildPromptStep` | Build VLM prompt with context |
| `CaptionVLMStep` | Main VLM inference |
| `CleanOutputStep` | Clean and format caption |

### Benefits

| Benefit | How |
|---------|-----|
| Easy to modify | Add/remove steps |
| Easy to test | Test each step independently |
| Easy to extend | Just add new Step class |
| RAG-ready | Insert `SearchExamplesStep` |
| API-ready | Insert `APIProviderStep` |

### File Structure

```
core/
  pipeline/
    __init__.py         # Exports get_pipeline()
    base.py             # PipelineContext, PipelineStep
    runner.py           # Pipeline class
    steps.py            # All step implementations
    flows.py            # Predefined pipelines
```

---

## Usage Patterns

### Simple
```
Image ──► SID_Caption ──► Caption
          (Quality: Balanced)
```

### With Options
```
SID_CaptionOptions ──┐
                     │
Image ──► SID_Caption ──► Caption
```

### Balanced/Detailed (Embedded Tags)
```
Image ──► SID_Caption
          (Quality: Balanced)
               │
               ▼ (pipeline internally runs WD14)
            Caption (with tag context)
```

### Power User
```
SID_CaptionOptions ─────────────┐
                                │
Image ──► SID_CaptionAdvanced ◄─┘
           (Quality: Detailed)
           (Model: Qwen-7B-Captioner-F16)
           (Quant: F16)
                │
                ▼ (pipeline: WD14 + Florence + VLM)
             Caption
```

---

## VRAM Requirements

| Quality | Pass 1 | Pass 2 | Peak VRAM |
|---------|--------|--------|-----------|
| Fast | - | Qwen-3B Q4 | ~4GB |
| Balanced | WD14 (~0.5GB) | Qwen-7B Q4 (~6GB) | ~6GB |
| Detailed | WD14 + Florence (~2.5GB) | Qwen-7B F16 (~16GB) | ~16GB |

**Pass 1 unloads before Pass 2** - sequential, not parallel.

---

## Platform Support

| Platform | GPU | Quantization | Status |
|----------|-----|--------------|--------|
| Windows | NVIDIA CUDA | Q4, Q8, F16 | Full support |
| Linux | NVIDIA CUDA | Q4, Q8, F16 | Full support |
| macOS | Apple Silicon | F16 only | Limited (no BitsAndBytes) |
| Any | CPU | F16 only | Slow, not recommended |

---

## Line Count Estimate

| Component | Lines |
|-----------|-------|
| nodes/ | ~300 |
| core/models/ | ~400 |
| core/ (other) | ~200 |
| config/ (JSON) | ~100 |
| **Total** | **~1000** |

---

## ComfyUI API

**Using V1 (Legacy) API** for maximum compatibility.

| Aspect | V1 Approach |
|--------|-------------|
| Registration | `NODE_CLASS_MAPPINGS` dict |
| Inputs | `INPUT_TYPES()` classmethod |
| Outputs | `RETURN_TYPES` tuple |
| Execution | `FUNCTION` points to method |
| Category | `CATEGORY` string |

### Node Registration Pattern

```
__init__.py
├── imports node classes from nodes/
├── NODE_CLASS_MAPPINGS = {"SID_Caption": SID_Caption, ...}
└── NODE_DISPLAY_NAME_MAPPINGS = {"SID_Caption": "SID Caption", ...}
```

### Why V1

| Reason | Explanation |
|--------|-------------|
| Compatibility | Works on all ComfyUI versions |
| Stability | API won't change |
| Adoption | Most custom nodes use V1 |
| V3 evolving | V3 API still maturing |

---

## Roadmap

| Release | Features |
|---------|----------|
| **1.0** | Core captioning, Qwen models, WD14, Options, Flow-based pipeline |
| **2.0** | RAG Basic (tag-based retrieval, curated examples) |
| **3.0** | RAG Enhanced (CV features + User learning) |
| **4.0** | API providers (Claude, GPT-4o, Gemini) |

### Release Progression

```
1.0 Foundation        2.0 RAG Basic         3.0 RAG Enhanced       4.0 API
─────────────────────────────────────────────────────────────────────────────
VLM Captioning    →   + Tag retrieval   →   + CV features      →   + Cloud
WD14 Tagger           + Curated examples    + User ratings         + Claude
Flow Pipeline         + SQLite + FTS5       + Learning             + GPT-4o
                                            + Pose/Framing         + Gemini
```

---

## Future: RAG Integration (Release 2.0 + 3.0)

### Overview

Progressive RAG enhancement using SQLite + FTS5 (built-in, zero dependencies).

```
Release 2.0 (Basic):
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│    Image    │────►│  WD14 Tags  │────►│  RAG Search │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                               │
                                        Tag-matched examples
                                               │
                                               ▼
                                    ┌─────────────────┐
                                    │  VLM + Examples │
                                    └─────────────────┘

Release 3.0 (Enhanced):
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│    Image    │────►│  WD14 Tags  │────►│             │
└──────┬──────┘     └─────────────┘     │             │
       │                                │  Enhanced   │
       │            ┌─────────────┐     │  RAG Search │
       └───────────►│ CV Analysis │────►│             │
                    └─────────────┘     └──────┬──────┘
                                               │
                                        Tag + CV matched examples
                                        + User-rated boost
                                               │
                                               ▼
                                    ┌─────────────────┐
                                    │  VLM + Examples │
                                    └─────────────────┘
```

### Why SQLite + FTS5

| Factor | Benefit |
|--------|---------|
| Zero dependencies | Built into Python |
| Fast full-text search | FTS5 ~2ms for 10K records |
| Portable | Single .db file ships with node |
| Scalable | Handles 100K+ records |
| Extensible | Add columns for CV/ratings later |

### Database Schema Evolution

```sql
-- Release 2.0: Basic RAG
CREATE TABLE examples (
    id INTEGER PRIMARY KEY,
    category TEXT,
    tags TEXT,              -- JSON array
    caption TEXT,
    source TEXT             -- "curated"
);

CREATE VIRTUAL TABLE examples_fts USING fts5(caption, tags);

-- Release 3.0: Add CV + User Learning
ALTER TABLE examples ADD COLUMN framing TEXT;      -- "close-up", "full-body"
ALTER TABLE examples ADD COLUMN pose TEXT;         -- "standing", "sitting"
ALTER TABLE examples ADD COLUMN composition TEXT;  -- "centered", "rule-of-thirds"
ALTER TABLE examples ADD COLUMN source TEXT;       -- "curated" | "user"

CREATE TABLE user_ratings (
    id INTEGER PRIMARY KEY,
    example_id INTEGER,
    rating INTEGER,         -- 1-5 stars
    user_edit TEXT,         -- User's corrected caption
    timestamp DATETIME,
    FOREIGN KEY (example_id) REFERENCES examples(id)
);

CREATE TABLE user_preferences (
    key TEXT PRIMARY KEY,
    value TEXT              -- JSON
);
```

### Planned Structure

```
data/
  examples.db              # Ships with node (~100KB, 500 examples)

~/.sid_toolkit/
  user_data.db             # User ratings + learned examples (Release 3.0)
```

### Release 2.0: Basic RAG

**Tag-based retrieval from curated examples.**

| Component | Description |
|-----------|-------------|
| 500 curated examples | Ship with node |
| Tag matching | WD14 tags → FTS5 search |
| Few-shot prompting | Inject 3 examples into VLM prompt |

Pipeline step:
```
extract_tags → search_examples → build_prompt → caption_vlm → clean_output
```

### Release 3.0: Enhanced RAG

**CV features + User learning on top of Basic RAG.**

| Component | Description |
|-----------|-------------|
| CV Analysis | Framing, pose, composition detection |
| CV-enhanced search | Match by structure, not just tags |
| User ratings | Rate captions good/bad |
| Learning | Good captions → add to examples |
| Preference memory | Remember user style preferences |

Pipeline step:
```
extract_tags → analyze_cv → search_examples_enhanced → build_prompt → caption_vlm → clean_output
```

### Example Categories

| Category | Examples | Matching |
|----------|----------|----------|
| portrait_closeup | 50 | tags + framing:close-up |
| portrait_medium | 50 | tags + framing:medium |
| portrait_fullbody | 50 | tags + framing:full-body |
| landscape | 50 | tags + composition |
| fashion | 50 | tags + pose:standing |
| product | 50 | tags + composition:centered |
| artistic | 50 | tags + style |
| general | 50 | fallback |

**Total: ~400-500 curated examples**

### Integration Points (Release 1.0 Ready)

```python
# core/prompts.py - examples parameter ready
def build_prompt(style, length, tags, options, examples=None, cv_context=None):
    base = STYLE_TEMPLATES[style][length]

    if examples:
        base += format_examples(examples)

    if cv_context:
        base += format_cv_context(cv_context)

    return base

# core/pipeline/steps.py - step ready to insert
class SearchExamplesStep(PipelineStep):
    name = "search_examples"

    def should_run(self, ctx):
        return ctx.options.get("use_rag", False)  # Disabled by default until 2.0
```

### CV Analysis (Release 3.0)

Lightweight detection using:

| Option | Model | Size | Detects |
|--------|-------|------|---------|
| MediaPipe | Face/Pose | ~10MB | Face landmarks, body pose |
| YOLO-World | YOLO | ~50MB | Objects, people |
| OpenCV | Haar/DNN | ~5MB | Basic face detection |

**Recommendation:** MediaPipe (lightweight, cross-platform)
