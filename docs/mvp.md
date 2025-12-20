# Release 1.0 MVP - Implementation Plan

## Goals

Build a clean, functional captioning toolkit with:
- 3 ComfyUI nodes (Caption, CaptionAdvanced, CaptionOptions)
- Qwen-focused model lineup
- Integrated WD14 tagging (in pipeline, not separate node)
- Flow-based pipeline
- V1 API compatibility

---

## Scope

### In Scope (Release 1.0)

| Feature | Description |
|---------|-------------|
| SID_Caption | Simple captioning node |
| SID_CaptionAdvanced | Model/quantization selection |
| SID_CaptionOptions | Customization options |
| Embedded tagging | WD14/Florence tags in pipeline (Balanced/Detailed tiers) |
| Flow pipeline | Step-based execution |
| Model factory | Qwen, JoyCaption, Florence |
| Platform detection | CUDA/MPS/CPU, quantization support |
| Output cleaning | Remove artifacts, format caption |

### Out of Scope (Future Releases)

| Feature | Release |
|---------|---------|
| RAG examples | 2.0 |
| CV analysis | 3.0 |
| User learning | 3.0 |
| API providers | 4.0 |

---

## File Structure

```
ComfyUI-AI-Photography-Toolkit/
├── __init__.py                     # Entry point, NODE_CLASS_MAPPINGS
├── requirements.txt                # Dependencies
├── pyproject.toml                  # Package config
│
├── nodes/
│   ├── __init__.py                 # Export all nodes
│   ├── caption.py                  # SID_Caption, SID_CaptionAdvanced
│   └── options.py                  # SID_CaptionOptions
│
├── core/
│   ├── __init__.py
│   ├── platform.py                 # GPU/platform detection
│   ├── prompts.py                  # Prompt building
│   ├── output.py                   # Output cleaning
│   │
│   ├── pipeline/
│   │   ├── __init__.py             # Export get_pipeline()
│   │   ├── base.py                 # PipelineContext, PipelineStep
│   │   ├── runner.py               # Pipeline class
│   │   ├── steps.py                # Step implementations
│   │   └── flows.py                # Predefined pipelines
│   │
│   └── models/
│       ├── __init__.py             # Export ModelFactory
│       ├── factory.py              # ModelFactory class
│       ├── base.py                 # BaseCaptionModel
│       ├── qwen.py                 # QwenVLModel
│       ├── joycaption.py           # JoyCaptionModel
│       ├── florence.py             # FlorenceModel
│       └── tagger.py               # WD14Tagger (embedded, used by pipeline)
│
├── config/
│   ├── models.json                 # Model definitions
│   ├── styles.json                 # Style templates
│   └── options.json                # Option definitions
│
└── docs/
    ├── architecture.md
    ├── models.md
    ├── research.md
    ├── history.md
    └── mvp.md                      # This file
```

---

## Implementation Phases

### Phase 1: Foundation (~200 lines)

**Goal:** Basic structure, platform detection, entry point.

| File | Description | Lines |
|------|-------------|-------|
| `__init__.py` | Entry point, version, node registration | ~30 |
| `core/platform.py` | GPU detection, quantization support | ~80 |
| `core/__init__.py` | Core exports | ~10 |
| `requirements.txt` | Dependencies | ~10 |

**Deliverable:** Node loads in ComfyUI, detects GPU.

---

### Phase 2: Model Layer (~300 lines)

**Goal:** Model factory, base class, model implementations.

| File | Description | Lines |
|------|-------------|-------|
| `core/models/base.py` | BaseCaptionModel, Quantization enum | ~80 |
| `core/models/factory.py` | ModelFactory, registry, cache | ~80 |
| `core/models/qwen.py` | QwenVLModel | ~60 |
| `core/models/florence.py` | FlorenceModel | ~50 |
| `core/models/joycaption.py` | JoyCaptionModel | ~60 |
| `core/models/tagger.py` | WD14Tagger | ~80 |
| `core/models/__init__.py` | Exports | ~10 |

**Deliverable:** Can load models and run inference.

---

### Phase 3: Pipeline (~200 lines)

**Goal:** Flow-based pipeline with steps.

| File | Description | Lines |
|------|-------------|-------|
| `core/pipeline/base.py` | PipelineContext, PipelineStep | ~50 |
| `core/pipeline/runner.py` | Pipeline class | ~40 |
| `core/pipeline/steps.py` | All step implementations | ~80 |
| `core/pipeline/flows.py` | Predefined pipelines | ~30 |
| `core/pipeline/__init__.py` | Exports | ~10 |

**Deliverable:** Can run caption pipeline programmatically.

---

### Phase 4: Prompts & Output (~100 lines)

**Goal:** Prompt building, output cleaning.

| File | Description | Lines |
|------|-------------|-------|
| `core/prompts.py` | build_prompt(), style templates | ~60 |
| `core/output.py` | clean_caption(), formatting | ~40 |

**Deliverable:** Clean, formatted captions.

---

### Phase 5: Nodes (~150 lines)

**Goal:** ComfyUI node implementations.

| File | Description | Lines |
|------|-------------|-------|
| `nodes/caption.py` | SID_Caption, SID_CaptionAdvanced | ~100 |
| `nodes/options.py` | SID_CaptionOptions | ~40 |
| `nodes/__init__.py` | Exports | ~10 |

**Deliverable:** All 3 nodes working in ComfyUI.

---

### Phase 6: Config (~100 lines JSON)

**Goal:** Externalized configuration.

| File | Description |
|------|-------------|
| `config/models.json` | Model repos, tiers, defaults |
| `config/styles.json` | Style templates by length |
| `config/options.json` | Option definitions |

**Deliverable:** Easy to modify without code changes.

---

## Detailed Implementation

### Phase 1: Foundation

#### `__init__.py`

```python
"""
SID Photography Toolkit v5.0
AI-powered captioning for ComfyUI
"""

__version__ = "5.0.0"

# Import nodes
from .nodes import (
    SID_Caption,
    SID_CaptionAdvanced,
    SID_CaptionOptions,
)

# V1 API registration
NODE_CLASS_MAPPINGS = {
    "SID_Caption": SID_Caption,
    "SID_CaptionAdvanced": SID_CaptionAdvanced,
    "SID_CaptionOptions": SID_CaptionOptions,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SID_Caption": "SID Caption",
    "SID_CaptionAdvanced": "SID Caption (Advanced)",
    "SID_CaptionOptions": "SID Caption Options",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

# Startup message
from .core.platform import get_system_info
info = get_system_info()
print(f"[SID-Toolkit] v{__version__} loaded")
print(f"[SID-Toolkit] GPU: {info.gpu_name} | Quantization: {', '.join(info.available_quants)}")
```

#### `core/platform.py`

```python
"""Platform detection for GPU and quantization support."""

import platform
import torch
from dataclasses import dataclass
from enum import Enum

class GPUType(Enum):
    CUDA = "cuda"
    MPS = "mps"
    CPU = "cpu"

@dataclass
class SystemInfo:
    platform: str
    gpu_type: GPUType
    gpu_name: str
    vram_gb: float
    supports_bnb: bool
    available_quants: list[str]

_SYSTEM_INFO = None

def detect_system() -> SystemInfo:
    """Detect system capabilities."""
    plat = platform.system().lower()

    # GPU detection
    if torch.cuda.is_available():
        gpu_type = GPUType.CUDA
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        supports_bnb = True
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        gpu_type = GPUType.MPS
        gpu_name = "Apple Silicon"
        vram_gb = 0
        supports_bnb = False
    else:
        gpu_type = GPUType.CPU
        gpu_name = "CPU"
        vram_gb = 0
        supports_bnb = False

    # Available quantizations
    quants = ["F16"]
    if supports_bnb:
        quants.extend(["Q8", "Q4"])

    return SystemInfo(
        platform=plat,
        gpu_type=gpu_type,
        gpu_name=gpu_name,
        vram_gb=round(vram_gb, 1),
        supports_bnb=supports_bnb,
        available_quants=quants
    )

def get_system_info() -> SystemInfo:
    """Get cached system info."""
    global _SYSTEM_INFO
    if _SYSTEM_INFO is None:
        _SYSTEM_INFO = detect_system()
    return _SYSTEM_INFO

def get_device() -> str:
    """Get torch device string."""
    info = get_system_info()
    return info.gpu_type.value
```

#### `requirements.txt`

```txt
transformers>=4.45.0
accelerate>=0.25.0
bitsandbytes>=0.41.0
torch>=2.0.0
Pillow>=9.0.0
```

---

### Phase 2: Model Layer

#### `core/models/base.py`

```python
"""Base model class and types."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from PIL import Image
import torch

class Quantization(Enum):
    F16 = "F16"
    Q8 = "Q8"
    Q4 = "Q4"

@dataclass
class ModelConfig:
    name: str
    repo: str
    quantization: Quantization

@dataclass
class GenerationConfig:
    max_tokens: int = 300
    temperature: float = 0.6
    top_p: float = 0.9

class BaseCaptionModel(ABC):
    """Base class for all caption models."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.processor = None
        self.loaded = False

    @abstractmethod
    def load(self) -> None:
        pass

    @abstractmethod
    def generate(self, image: Image.Image, prompt: str,
                 config: GenerationConfig = None) -> str:
        pass

    def unload(self) -> None:
        if self.model:
            del self.model
        if self.processor:
            del self.processor
        self.model = None
        self.processor = None
        self.loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _get_quant_config(self):
        """Get BitsAndBytes config."""
        from ..platform import get_system_info
        info = get_system_info()
        quant = self.config.quantization

        if quant == Quantization.F16 or not info.supports_bnb:
            return torch.bfloat16, None
        elif quant == Quantization.Q8:
            from transformers import BitsAndBytesConfig
            return torch.float16, BitsAndBytesConfig(load_in_8bit=True)
        elif quant == Quantization.Q4:
            from transformers import BitsAndBytesConfig
            return torch.float16, BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
```

#### `core/models/factory.py`

```python
"""Model factory for creating caption models."""

from .base import BaseCaptionModel, ModelConfig, Quantization
from .qwen import QwenVLModel
from .florence import FlorenceModel
from .joycaption import JoyCaptionModel

MODEL_REGISTRY = {
    # Fast tier
    "Qwen-3B": {
        "class": QwenVLModel,
        "repo": "Qwen/Qwen2.5-VL-3B-Instruct",
    },
    "Florence-2-PromptGen": {
        "class": FlorenceModel,
        "repo": "MiaoshouAI/Florence-2-large-PromptGen-v2.0",
    },
    # Balanced tier
    "Qwen-7B-Captioner": {
        "class": QwenVLModel,
        "repo": "Ertugrul/Qwen2.5-VL-7B-Captioner-Relaxed",
    },
    "Qwen-7B": {
        "class": QwenVLModel,
        "repo": "Qwen/Qwen2.5-VL-7B-Instruct",
    },
    "JoyCaption": {
        "class": JoyCaptionModel,
        "repo": "fancyfeast/llama-joycaption-beta-one-hf-llava",
    },
    # Detailed tier (same models, F16 default)
    "Qwen-7B-Captioner-F16": {
        "class": QwenVLModel,
        "repo": "Ertugrul/Qwen2.5-VL-7B-Captioner-Relaxed",
    },
    "Qwen-7B-F16": {
        "class": QwenVLModel,
        "repo": "Qwen/Qwen2.5-VL-7B-Instruct",
    },
    "JoyCaption-F16": {
        "class": JoyCaptionModel,
        "repo": "fancyfeast/llama-joycaption-beta-one-hf-llava",
    },
}

TIER_DEFAULTS = {
    "Fast": "Qwen-3B",
    "Balanced": "Qwen-7B-Captioner",
    "Detailed": "Qwen-7B-Captioner-F16",
}

_MODEL_CACHE = {}

class ModelFactory:
    """Factory for creating and caching caption models."""

    @staticmethod
    def get(name: str, quantization: str = "Q4") -> BaseCaptionModel:
        if name not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model: {name}")

        quant = Quantization[quantization]
        cache_key = f"{name}_{quantization}"

        if cache_key in _MODEL_CACHE:
            return _MODEL_CACHE[cache_key]

        entry = MODEL_REGISTRY[name]
        config = ModelConfig(name=name, repo=entry["repo"], quantization=quant)
        model = entry["class"](config)

        _MODEL_CACHE[cache_key] = model
        return model

    @staticmethod
    def get_for_tier(tier: str, quantization: str = "Q4") -> BaseCaptionModel:
        name = TIER_DEFAULTS[tier]
        return ModelFactory.get(name, quantization)

    @staticmethod
    def clear_cache():
        for model in _MODEL_CACHE.values():
            model.unload()
        _MODEL_CACHE.clear()
```

---

### Phase 3: Pipeline

#### `core/pipeline/base.py`

```python
"""Pipeline base classes."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
from PIL import Image

@dataclass
class PipelineContext:
    """Data flowing through pipeline."""
    image: Any                              # PIL Image or tensor
    tags: str = ""
    quick_caption: str = ""
    examples: list[str] = field(default_factory=list)
    prompt: str = ""
    caption: str = ""
    options: dict = field(default_factory=dict)

    # Settings
    quality: str = "Balanced"
    style: str = "Natural"
    length: str = "Medium"
    model_name: str = ""
    quantization: str = "Q4"

class PipelineStep(ABC):
    """Base class for pipeline steps."""

    name: str = "base_step"

    @abstractmethod
    def execute(self, ctx: PipelineContext) -> PipelineContext:
        pass

    def should_run(self, ctx: PipelineContext) -> bool:
        return True
```

#### `core/pipeline/runner.py`

```python
"""Pipeline runner."""

from .base import PipelineContext, PipelineStep

class Pipeline:
    """Executes a sequence of steps."""

    def __init__(self, steps: list[PipelineStep]):
        self.steps = steps

    def run(self, ctx: PipelineContext) -> PipelineContext:
        for step in self.steps:
            if step.should_run(ctx):
                ctx = step.execute(ctx)
        return ctx

    def add_step(self, step: PipelineStep, after: str = None):
        if after:
            idx = next(i for i, s in enumerate(self.steps) if s.name == after)
            self.steps.insert(idx + 1, step)
        else:
            self.steps.append(step)

    def remove_step(self, name: str):
        self.steps = [s for s in self.steps if s.name != name]
```

#### `core/pipeline/steps.py`

```python
"""Pipeline step implementations."""

from .base import PipelineStep, PipelineContext

class ExtractTagsStep(PipelineStep):
    name = "extract_tags"

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        from ..models.tagger import TaggerFactory

        tagger = TaggerFactory.get_default()
        result = tagger.predict(ctx.image)
        ctx.tags = result["all_tags"]
        tagger.unload()
        return ctx

    def should_run(self, ctx: PipelineContext) -> bool:
        # Skip if tags already provided externally
        return not ctx.tags

class QuickCaptionStep(PipelineStep):
    name = "quick_caption"

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        from ..models import ModelFactory

        model = ModelFactory.get("Florence-2-PromptGen", "Q4")
        ctx.quick_caption = model.generate(ctx.image, "")
        model.unload()
        return ctx

    def should_run(self, ctx: PipelineContext) -> bool:
        return ctx.quality == "Detailed"

class BuildPromptStep(PipelineStep):
    name = "build_prompt"

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        from ..prompts import build_prompt

        ctx.prompt = build_prompt(
            style=ctx.style,
            length=ctx.length,
            tags=ctx.tags,
            quick_caption=ctx.quick_caption,
            options=ctx.options
        )
        return ctx

class CaptionVLMStep(PipelineStep):
    name = "caption_vlm"

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        from ..models import ModelFactory

        if ctx.model_name:
            model = ModelFactory.get(ctx.model_name, ctx.quantization)
        else:
            model = ModelFactory.get_for_tier(ctx.quality, ctx.quantization)

        ctx.caption = model.generate(ctx.image, ctx.prompt)
        return ctx

class CleanOutputStep(PipelineStep):
    name = "clean_output"

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        from ..output import clean_caption

        ctx.caption = clean_caption(ctx.caption, ctx.length)
        return ctx
```

#### `core/pipeline/flows.py`

```python
"""Predefined pipeline flows."""

from .runner import Pipeline
from .steps import (
    ExtractTagsStep,
    QuickCaptionStep,
    BuildPromptStep,
    CaptionVLMStep,
    CleanOutputStep,
)

def get_fast_pipeline() -> Pipeline:
    return Pipeline([
        BuildPromptStep(),
        CaptionVLMStep(),
        CleanOutputStep(),
    ])

def get_balanced_pipeline() -> Pipeline:
    return Pipeline([
        ExtractTagsStep(),
        BuildPromptStep(),
        CaptionVLMStep(),
        CleanOutputStep(),
    ])

def get_detailed_pipeline() -> Pipeline:
    return Pipeline([
        ExtractTagsStep(),
        QuickCaptionStep(),
        BuildPromptStep(),
        CaptionVLMStep(),
        CleanOutputStep(),
    ])

def get_pipeline(quality: str) -> Pipeline:
    pipelines = {
        "Fast": get_fast_pipeline,
        "Balanced": get_balanced_pipeline,
        "Detailed": get_detailed_pipeline,
    }
    return pipelines[quality]()
```

---

### Phase 4: Prompts & Output

#### `core/prompts.py`

```python
"""Prompt building utilities."""

STYLE_TEMPLATES = {
    "Natural": {
        "Short": "Write a brief, natural description of this image in about 50 words.",
        "Medium": "Write a detailed, natural description of this image in about 120 words.",
        "Long": "Write a comprehensive, natural description of this image in about 250 words.",
    },
    "Tags": {
        "Short": "Generate comma-separated tags for this image. About 10 tags.",
        "Medium": "Generate comma-separated tags for this image. About 20 tags.",
        "Long": "Generate comma-separated tags for this image. About 30 tags.",
    },
    "Hybrid": {
        "Short": "First list 5 key tags, then write a 30-word description.",
        "Medium": "First list 10 key tags, then write a 80-word description.",
        "Long": "First list 15 key tags, then write a 150-word description.",
    },
}

def build_prompt(
    style: str,
    length: str,
    tags: str = "",
    quick_caption: str = "",
    options: dict = None,
    examples: list[str] = None  # Future: RAG
) -> str:
    """Build prompt for VLM."""
    base = STYLE_TEMPLATES[style][length]
    parts = [base]

    # Add tag context
    if tags:
        parts.append(f"\nDetected elements: {tags}")

    # Add quick caption context
    if quick_caption:
        parts.append(f"\nQuick analysis: {quick_caption}")

    # Add options
    if options:
        option_instructions = _build_option_instructions(options)
        if option_instructions:
            parts.append(option_instructions)

    # Future: Add RAG examples
    if examples:
        example_text = "\n".join([f"- {ex}" for ex in examples])
        parts.append(f"\nReference examples:\n{example_text}")

    return "\n".join(parts)

def _build_option_instructions(options: dict) -> str:
    instructions = []

    if options.get("include_lighting"):
        instructions.append("Include lighting direction and quality.")
    if options.get("include_pose"):
        instructions.append("Describe pose and body position in detail.")
    if options.get("include_colors"):
        instructions.append("Mention specific colors and color relationships.")
    if options.get("include_composition"):
        instructions.append("Describe the composition and framing.")
    if options.get("exclude_background"):
        instructions.append("Focus on the subject, minimal background description.")
    if options.get("for_flux"):
        instructions.append("Optimize for Flux model. Natural flowing language.")
    if options.get("for_sdxl"):
        instructions.append("Optimize for SDXL. Include style keywords.")
    if options.get("for_pony"):
        instructions.append("Optimize for Pony. Use booru-style tags.")
    if options.get("custom"):
        instructions.append(options["custom"])

    return "\n" + "\n".join(instructions) if instructions else ""
```

#### `core/output.py`

```python
"""Output cleaning utilities."""

import re

LENGTH_LIMITS = {
    "Short": 80,
    "Medium": 180,
    "Long": 350,
}

def clean_caption(text: str, length: str = "Medium") -> str:
    """Clean and format caption output."""
    if not text:
        return ""

    # Remove thinking tags
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL)

    # Remove markdown
    text = re.sub(r"```[\s\S]*?```", "", text)
    text = re.sub(r"`[^`]+`", "", text)

    # Remove common prefixes
    prefixes = [
        r"^Here['']?s?\s*(?:is\s*)?(?:the\s*)?(?:a\s*)?(?:caption|description)[:\s]*",
        r"^(?:The\s*)?(?:image\s*)?(?:shows|depicts|features)[:\s]*",
    ]
    for prefix in prefixes:
        text = re.sub(prefix, "", text, flags=re.IGNORECASE)

    # Clean whitespace
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    # Truncate if needed
    max_words = LENGTH_LIMITS.get(length, 180)
    text = _truncate_words(text, max_words)

    return text

def _truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text

    truncated = " ".join(words[:max_words])

    # End at sentence boundary if possible
    last_period = truncated.rfind(". ")
    if last_period > len(truncated) * 0.7:
        return truncated[:last_period + 1]

    return truncated + "..."
```

---

### Phase 5: Nodes

#### `nodes/caption.py`

```python
"""Caption nodes for ComfyUI."""

from PIL import Image
import torch

def tensor_to_pil(tensor) -> Image.Image:
    """Convert ComfyUI tensor to PIL Image."""
    if len(tensor.shape) == 4:
        tensor = tensor[0]
    np_image = (tensor.cpu().numpy() * 255).astype("uint8")
    return Image.fromarray(np_image)

class SID_Caption:
    """Simple captioning node."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "quality": (["Fast", "Balanced", "Detailed"],),
                "style": (["Natural", "Tags", "Hybrid"],),
                "length": (["Short", "Medium", "Long"],),
            },
            "optional": {
                "tags": ("STRING", {"forceInput": True}),
                "options": ("SID_OPTIONS",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("caption",)
    FUNCTION = "execute"
    CATEGORY = "SID Photography Toolkit"

    def execute(self, image, quality, style, length, tags=None, options=None):
        from ..core.pipeline import get_pipeline, PipelineContext

        pil_image = tensor_to_pil(image)

        ctx = PipelineContext(
            image=pil_image,
            quality=quality,
            style=style,
            length=length,
            tags=tags or "",
            options=options or {}
        )

        pipeline = get_pipeline(quality)
        ctx = pipeline.run(ctx)

        return (ctx.caption,)


class SID_CaptionAdvanced:
    """Advanced captioning with model selection."""

    MODELS = [
        "── Fast ──",
        "Qwen-3B",
        "Florence-2-PromptGen",
        "── Balanced ──",
        "Qwen-7B-Captioner",
        "Qwen-7B",
        "JoyCaption",
        "── Detailed ──",
        "Qwen-7B-Captioner-F16",
        "Qwen-7B-F16",
        "JoyCaption-F16",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        from ..core.platform import get_system_info
        info = get_system_info()

        return {
            "required": {
                "image": ("IMAGE",),
                "quality": (["Fast", "Balanced", "Detailed"],),
                "model": (cls.MODELS,),
                "quantization": (info.available_quants,),
                "style": (["Natural", "Tags", "Hybrid"],),
                "length": (["Short", "Medium", "Long"],),
            },
            "optional": {
                "tags": ("STRING", {"forceInput": True}),
                "options": ("SID_OPTIONS",),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("caption", "tags_used")
    FUNCTION = "execute"
    CATEGORY = "SID Photography Toolkit"

    def execute(self, image, quality, model, quantization, style, length,
                tags=None, options=None):
        from ..core.pipeline import get_pipeline, PipelineContext

        pil_image = tensor_to_pil(image)

        # Handle separator lines in dropdown
        if model.startswith("──"):
            model = ""

        ctx = PipelineContext(
            image=pil_image,
            quality=quality,
            style=style,
            length=length,
            tags=tags or "",
            options=options or {},
            model_name=model,
            quantization=quantization
        )

        pipeline = get_pipeline(quality)
        ctx = pipeline.run(ctx)

        return (ctx.caption, ctx.tags)
```

#### `nodes/options.py`

```python
"""Caption options node."""

class SID_CaptionOptions:
    """Customization options for captioning."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "include_lighting": ("BOOLEAN", {"default": False}),
                "include_pose": ("BOOLEAN", {"default": False}),
                "include_colors": ("BOOLEAN", {"default": False}),
                "include_composition": ("BOOLEAN", {"default": False}),
                "exclude_background": ("BOOLEAN", {"default": False}),
                "for_flux": ("BOOLEAN", {"default": False}),
                "for_sdxl": ("BOOLEAN", {"default": False}),
                "for_pony": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "custom": ("STRING", {"default": "", "multiline": True}),
            }
        }

    RETURN_TYPES = ("SID_OPTIONS",)
    RETURN_NAMES = ("options",)
    FUNCTION = "execute"
    CATEGORY = "SID Photography Toolkit"

    def execute(self, include_lighting, include_pose, include_colors,
                include_composition, exclude_background,
                for_flux, for_sdxl, for_pony, custom=""):
        return ({
            "include_lighting": include_lighting,
            "include_pose": include_pose,
            "include_colors": include_colors,
            "include_composition": include_composition,
            "exclude_background": exclude_background,
            "for_flux": for_flux,
            "for_sdxl": for_sdxl,
            "for_pony": for_pony,
            "custom": custom,
        },)
```

---

## Testing Plan

### Unit Tests

| Component | Test |
|-----------|------|
| `platform.py` | Detect GPU correctly |
| `factory.py` | Load models, cache works |
| `pipeline/` | Steps execute in order |
| `prompts.py` | Prompt building correct |
| `output.py` | Cleaning works |

### Integration Tests

| Test | Description |
|------|-------------|
| Load in ComfyUI | Nodes appear in menu |
| Fast caption | Single pass works |
| Balanced caption | Double pass with embedded tags |
| Detailed caption | Triple pass with embedded tags + quick caption |
| Options | Options affect output |
| Embedded tagger | WD14 tags integrated correctly in pipeline |

### Manual Tests

1. Load node in ComfyUI
2. Connect image
3. Run each quality tier
4. Try each style
5. Connect SID_CaptionOptions
6. Verify captions are reasonable
7. Check Balanced/Detailed include tag context

---

## Dependencies

```txt
# requirements.txt

# Core
transformers>=4.45.0
accelerate>=0.25.0
torch>=2.0.0
Pillow>=9.0.0

# Quantization (CUDA only)
bitsandbytes>=0.41.0

# Tagger
timm>=0.9.0
```

---

## Estimated Lines

| Component | Lines |
|-----------|-------|
| `__init__.py` | ~25 |
| `core/platform.py` | ~80 |
| `core/models/` | ~350 |
| `core/pipeline/` | ~200 |
| `core/prompts.py` | ~80 |
| `core/output.py` | ~50 |
| `nodes/` | ~150 |
| **Total** | **~935** |

---

## Implementation Order

1. **Phase 1:** Foundation (platform, __init__)
2. **Phase 2:** Model layer (factory, base, implementations)
3. **Phase 3:** Pipeline (context, steps, flows)
4. **Phase 4:** Prompts & output
5. **Phase 5:** Nodes
6. **Phase 6:** Config files
7. **Testing & refinement**
