# ComfyUI-AI-Photography-Toolkit

## Project Overview
A ComfyUI custom node toolkit for AI-powered image analysis and prompt generation. Provides multi-model tagging, VLM captioning, and prompt enhancement.

## Current Version
- Branch: `dev/4.3.0`
- Tag: `v4.3.0`

## Architecture

### Core Components

```
core/
├── models/           # VLM caption models
│   ├── florence.py   # Florence-2 PromptGen (uses kijai's bundled impl)
│   ├── qwen.py       # Qwen2-VL
│   └── base.py       # BaseCaptionModel
├── taggers/          # Image analysis taggers
│   ├── wd14.py       # WD14 tagger (ONNX)
│   ├── joytag.py     # JoyTag (ONNX, 448x448)
│   ├── nudenet.py    # NudeNet v2
│   ├── iqa.py        # NIMA/MUSIQ/BRISQUE quality
│   ├── saliency.py   # DINOv2 + CV fallback
│   ├── composition.py # CADB composition
│   ├── fashion/      # Fashion composite tagger
│   │   ├── fashion_clip.py
│   │   ├── yolov8_clothing.py
│   │   ├── yolos_fashionpedia.py
│   │   ├── segformer_clothes.py
│   │   └── wargon_classifier.py
│   └── pose/         # Pose composite tagger
│       └── mediapipe_pose.py
├── tagging_pipeline.py  # Orchestrates all taggers
└── config/           # Model configs (YAML)
```

### Nodes
- `SID_TaggerConfig` - Configure taggers and thresholds
- `SID_ZImagePromptGenerator` - Main prompt generation node

## Key Technical Decisions

### Florence-2 Loading (transformers 4.50+)
- Uses kijai's `comfyui-florence2` bundled implementation
- Must register both model AND processor classes in sys.modules
- See `_load_kijai_florence_class()` and `_register_processor_from_model()`

### Dependency Management
- Standard `requirements.txt` approach (ComfyUI-Manager compatible)
- No auto-install code - removed all subprocess/pip machinery
- torch/torchvision NOT in requirements (user installs with correct CUDA)

### JSON Output Format
```json
{
  "module_name": {
    "tags": ["tag1", "tag2", "tag3"],
    "attributes": { "count": 3, "model": "...", ... }
  }
}
```

## Recent Fixes (Dec 2024)

1. **Florence-2 processor loading** - Register `Florence2Processor` from model's local `processing_florence2.py`

2. **Saliency without opencv-contrib** - Uses edge + color contrast fallback, DINOv2 uses feature norms not attention

3. **JoyTag ONNX** - Fixed to use correct 448x448 input size

4. **Removed RAM++** - Incompatible with transformers 4.50+ (`apply_chunking_to_forward` removed)

5. **WD14 model list** - Fixed invalid `wd-vit-large-tagger-v3`, added `wd-v1-4-vit-tagger-v2`

## Available WD14 Models
- wd-swinv2-tagger-v3 (default, best)
- wd-vit-tagger-v3
- wd-convnext-tagger-v3
- wd-eva02-large-tagger-v3
- wd-v1-4-moat-tagger-v2
- wd-v1-4-swinv2-tagger-v2
- wd-v1-4-convnext-tagger-v2
- wd-v1-4-vit-tagger-v2

## Fashion Vocabulary
Expanded to include 50+ ethnic/traditional wear terms:
- Indian: sari, lehenga, salwar kameez, kurta, sherwani, dupatta
- Middle Eastern: abaya, hijab, thobe, kaftan
- Asian: kimono, hanbok, cheongsam, ao dai

See `core/taggers/vocabularies.json` for full list.

## Known Limitations
- FashionCLIP trained on Western fashion - ethnic wear detection may be limited
- MediaPipe pose may fail on some systems (reinstall mediapipe)
- CADB composition model unavailable (uses fallback CV analysis)

## Development Commands
```bash
# Clone on new machine
git clone git@github.com:slahiri/ComfyUI-AI-Photography-Toolkit.git
cd ComfyUI-AI-Photography-Toolkit
git checkout dev/4.3.0

# Install dependencies (in ComfyUI venv)
pip install -r requirements.txt
```

## File Locations
- Models: `D:\ComfyUI\models\LLM\` (Florence, Qwen)
- Taggers: `D:\ComfyUI\models\taggers\` (WD14, JoyTag)
- Config: `core/config/*.yaml`
- Vocabularies: `core/taggers/vocabularies.json`

## Active Development: Prompt Composition Rewrite

**Status**: In Progress (Dec 2024)

The `SID_PromptCompose` node is being rewritten with a new tokenization → classification → assembly pipeline.

### Documentation

| Document | Purpose |
|----------|---------|
| [docs/image_tokenization_spec.md](docs/image_tokenization_spec.md) | Full technical specification (Phases 1-6) |
| [docs/implementation_plan.md](docs/implementation_plan.md) | Implementation roadmap with task checkboxes |

### Quick Summary

**Problem**: Current `StandardGenerator` produces broken output:
- Grammar errors ("her eyes are brown eyes")
- Meta-commentary not filtered ("in the middle of the image")
- Missing content, redundant phrases

**Solution**: New pipeline with:
1. **Tokenization** - Extract from all metadata sources into `ImageToken` format
2. **Classification** - 5-layer cascade assigns tokens to 9 canonical categories
3. **Assembly (Rule-Based)** - Templates + phrase mappings + grammar rules
4. **Assembly (LLM)** - Optional enhancement using LLM providers

### Modes
- `Standard` → Rule-based assembly (fast, free, deterministic)
- `Enhance with AI` → LLM-based assembly (higher quality, costs money)

### Key Files (New Structure)
```
core/compose/
├── tokenizer/      # Phase 1-2: Extract and normalize tokens
├── classifier/     # Phase 3: Assign to canonical categories
├── assembler/      # Phase 5-6: Rule-based and LLM assembly
├── validator/      # Quality validation (optional)
└── pipeline.py     # Main orchestrator
```

**Resume Work**: Start with [implementation_plan.md](docs/implementation_plan.md)
