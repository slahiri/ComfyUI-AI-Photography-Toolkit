# SID Prompt Debug Agent

## Overview

The SID Prompt Debug Agent is an intelligent evaluation system that analyzes prompt generation quality by comparing source images, generated outputs, and the prompts used. It uses Claude Opus 4.5 for evaluation and optionally Tavily for web search to stay updated with latest prompting techniques.

**Key Features:**
- Expert in Z-Image prompting best practices
- Expert in general vision model prompting
- Agentic approach with tool use (LLM + Search)
- Local knowledge base that improves over time
- Model-specific recommendations
- Comprehensive scoring and evaluation

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    SID_PromptDebugAgent                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   Anthropic  │    │    Tavily    │    │  Knowledge   │      │
│  │  Claude Opus │    │    Search    │    │    Base      │      │
│  │     4.5      │    │  (Optional)  │    │   (Local)    │      │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘      │
│         │                   │                   │               │
│         └───────────────────┼───────────────────┘               │
│                             │                                   │
│                    ┌────────▼────────┐                          │
│                    │  Agent Router   │                          │
│                    └────────┬────────┘                          │
│                             │                                   │
│         ┌───────────────────┼───────────────────┐               │
│         ▼                   ▼                   ▼               │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │  Evaluate   │    │   Search    │    │   Build     │         │
│  │   Prompt    │    │   Best      │    │  Knowledge  │         │
│  │  Quality    │    │  Practices  │    │    Base     │         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Debug Mode Configuration

The debug node is only available when `debug_mode = true` in the configuration. This allows hiding the node in production releases.

**config/settings.toml:**
```toml
[development]
debug_mode = true  # Set to false for production/publish

[debug_agent]
default_save_results = true
default_update_knowledge = false
knowledge_base_path = "knowledge_base"
results_path = "debug_results"
```

**Console Output:**
- Development: `[SID Toolkit] Debug mode ENABLED - Debug nodes available`
- Production: `[SID Toolkit] Debug mode disabled - Production mode`

---

## Node Specification

### Inputs

| Input | Type | Description |
|-------|------|-------------|
| `source_image` | IMAGE | Original input image |
| `output_image` | IMAGE | AI-generated result |
| `prompt` | STRING | Generated prompt used |
| `llm_model` | LLM_MODEL | Model config that generated the prompt |
| `anthropic_api_key` | STRING | For Claude Opus 4.5 evaluation |
| `tavily_api_key` | STRING | For web search (optional) |
| `save_results` | BOOLEAN | Save to local storage |
| `update_knowledge` | BOOLEAN | Update local knowledge base |

### Outputs

| Output | Type | Description |
|--------|------|-------------|
| `evaluation` | STRING | Full evaluation text |
| `score` | FLOAT | Overall quality score (0-10) |
| `summary` | STRING | Brief summary |

### Usage Flow

```
[Source Image] ──────┐
                     │
[Output Image] ──────┤
                     │
[Prompt] ────────────┼──► [SID_PromptDebugAgent] ──► evaluation
                     │                           ──► score
[LLM_MODEL config] ──┤                           ──► summary
                     │
[API Keys] ──────────┘
```

---

## Knowledge Base

### Directory Structure

```
ComfyUI-AI-Photography-Toolkit/
└── knowledge_base/
    ├── zimage_vocabulary.json       # Z-Image specific terms
    ├── zimage_best_practices.json   # Z-Image prompting rules
    ├── vision_model_prompting.json  # General vision LLM prompting
    ├── model_specific_tips.json     # Per-model recommendations
    └── learned_insights.json        # Accumulated from debug sessions
```

### Knowledge Sources

| Source | Description | When Updated |
|--------|-------------|--------------|
| **Pre-populated** | Ships with package - curated best practices | Package release |
| **Runtime Search** | Tavily search for latest techniques | When Tavily key provided |
| **Learned** | Accumulated from debug sessions | Each evaluation (if enabled) |

---

## Knowledge Base Content

### zimage_vocabulary.json

```json
{
  "version": "1.0.0",
  "last_updated": "2025-12-14T10:30:00Z",

  "categories": {
    "lighting": {
      "terms": [
        {"term": "soft natural lighting", "effectiveness": 9.2, "use_case": "outdoor portraits"},
        {"term": "golden hour light", "effectiveness": 8.8, "use_case": "warm outdoor scenes"},
        {"term": "studio lighting", "effectiveness": 8.5, "use_case": "controlled indoor"},
        {"term": "rim lighting", "effectiveness": 8.0, "use_case": "dramatic edge definition"},
        {"term": "diffused overcast", "effectiveness": 7.8, "use_case": "even, shadowless"}
      ],
      "avoid": ["HDR lighting", "neon glow", "lens flare"]
    },

    "skin_texture": {
      "terms": [
        {"term": "realistic skin texture", "effectiveness": 9.5, "use_case": "always for humans"},
        {"term": "natural skin pores", "effectiveness": 9.3, "use_case": "close-up portraits"},
        {"term": "photorealistic skin", "effectiveness": 9.0, "use_case": "general human subjects"},
        {"term": "subtle skin imperfections", "effectiveness": 8.2, "use_case": "authentic look"}
      ],
      "avoid": ["airbrushed", "plastic skin", "porcelain"]
    },

    "fabric_materials": {
      "terms": [
        {"term": "silk", "visual_cues": "smooth, reflective, flowing drape"},
        {"term": "cotton", "visual_cues": "matte, soft texture, natural wrinkles"},
        {"term": "denim", "visual_cues": "visible weave, indigo blue, stiff structure"},
        {"term": "leather", "visual_cues": "glossy or matte, grain texture, structured"},
        {"term": "wool", "visual_cues": "fuzzy texture, matte, chunky knit patterns"},
        {"term": "linen", "visual_cues": "visible weave, natural wrinkles, breathable look"},
        {"term": "velvet", "visual_cues": "rich depth, light-absorbing, plush texture"}
      ]
    },

    "colors": {
      "descriptors": [
        {"term": "deep burgundy", "hex_range": "#800020-#722F37"},
        {"term": "navy blue", "hex_range": "#000080-#001F3F"},
        {"term": "forest green", "hex_range": "#228B22-#014421"},
        {"term": "cream white", "hex_range": "#FFFDD0-#F5F5DC"},
        {"term": "charcoal gray", "hex_range": "#36454F-#2C3E50"}
      ],
      "modifiers": ["muted", "vibrant", "pastel", "saturated", "desaturated"]
    },

    "composition": {
      "terms": [
        {"term": "centered composition", "use_case": "formal portraits"},
        {"term": "rule of thirds", "use_case": "dynamic positioning"},
        {"term": "negative space", "use_case": "minimalist, focus on subject"},
        {"term": "shallow depth of field", "use_case": "subject isolation"},
        {"term": "bokeh background", "use_case": "soft background blur"}
      ]
    },

    "technical_quality": {
      "enhancers": [
        "ultra detailed",
        "8K resolution",
        "sharp focus",
        "high dynamic range",
        "professional photography"
      ],
      "use_sparingly": ["hyper-realistic", "masterpiece", "best quality"]
    }
  }
}
```

### zimage_best_practices.json

```json
{
  "version": "1.0.0",
  "last_updated": "2025-12-14T10:30:00Z",

  "structure_rules": {
    "sentence_count": {
      "min": 3,
      "max": 6,
      "rationale": "Each sentence should describe ONE aspect"
    },
    "sentence_order": [
      "Subject description (who/what)",
      "Clothing/appearance details",
      "Pose/action",
      "Environment/background",
      "Lighting conditions",
      "Technical quality tags"
    ],
    "word_count": {
      "quick_mode": {"min": 50, "max": 150},
      "standard_mode": {"min": 150, "max": 400},
      "detailed_mode": {"min": 300, "max": 600}
    }
  },

  "must_include": {
    "human_subjects": [
      "Age/gender indicator",
      "Hair color and style",
      "Skin texture terms (for realism)",
      "Clothing with color and material",
      "Pose or action description"
    ],
    "all_images": [
      "Lighting description",
      "Background/environment",
      "At least one technical quality term"
    ]
  },

  "must_avoid": {
    "content": [
      "Philosophical commentary",
      "Narrative storytelling",
      "Emotional interpretations",
      "Metaphors and similes",
      "Business/corporate jargon"
    ],
    "structure": [
      "Comma-separated keyword lists",
      "One long run-on sentence",
      "Repetitive information",
      "Vague descriptors (beautiful, nice, good)"
    ]
  },

  "effectiveness_patterns": {
    "high_effectiveness": [
      {
        "pattern": "[Subject] with [hair description]. [Clothing with color, material, style]. [Pose/action]. [Environment]. [Lighting]. [Technical tags].",
        "score": 9.2,
        "example": "A young woman with long auburn hair in loose waves. She wears a navy blue silk blouse tucked into high-waisted cream linen trousers. Standing with arms relaxed at her sides, facing camera. Outdoor garden setting with blurred green foliage background. Soft natural daylight from the left creating gentle shadows. Realistic skin texture, natural skin pores, photorealistic, ultra detailed."
      }
    ],
    "low_effectiveness": [
      {
        "pattern": "Keyword, keyword, keyword, keyword...",
        "score": 3.5,
        "issue": "No context, relationships unclear"
      },
      {
        "pattern": "One extremely long sentence with many clauses and descriptions all connected...",
        "score": 4.2,
        "issue": "Hard to parse, priority unclear"
      }
    ]
  }
}
```

### vision_model_prompting.json

```json
{
  "version": "1.0.0",
  "last_updated": "2025-12-14T10:30:00Z",

  "general_principles": {
    "clarity": {
      "description": "Be specific and unambiguous",
      "examples": {
        "good": "red floral dress with small white roses",
        "bad": "nice dress with pattern"
      }
    },
    "hierarchy": {
      "description": "Most important details first",
      "order": ["subject", "key attributes", "environment", "style", "technical"]
    },
    "concreteness": {
      "description": "Use concrete visual terms, not abstract concepts",
      "examples": {
        "good": "warm golden sunlight from the right",
        "bad": "beautiful lighting"
      }
    }
  },

  "model_family_tips": {
    "flux": {
      "strengths": ["photorealism", "faces", "text rendering"],
      "prompt_style": "Descriptive sentences work well",
      "special_tokens": [],
      "avoid": ["excessive negative prompts"]
    },
    "sdxl": {
      "strengths": ["artistic styles", "composition"],
      "prompt_style": "Can handle keyword style",
      "special_tokens": [],
      "tips": ["Quality tags help: masterpiece, best quality"]
    },
    "sd3": {
      "strengths": ["text", "composition", "prompt following"],
      "prompt_style": "Natural language preferred",
      "tips": ["Very literal prompt interpretation"]
    }
  },

  "common_issues": {
    "hands": {
      "problem": "Malformed hands common",
      "mitigation": "Specify hand position clearly or hide hands"
    },
    "text": {
      "problem": "Text often garbled",
      "mitigation": "Use models with good text (Flux, SD3)"
    },
    "multiple_subjects": {
      "problem": "Feature mixing between subjects",
      "mitigation": "Clear separation, use position descriptors"
    }
  }
}
```

### model_specific_tips.json

```json
{
  "version": "1.0.0",
  "last_updated": "2025-12-14T10:30:00Z",

  "vision_models": {
    "Qwen2.5-VL-7B-Instruct": {
      "strengths": ["human detection", "pose analysis", "color accuracy"],
      "weaknesses": ["small accessories", "fine patterns", "fabric textures"],
      "optimal_settings": {
        "temperature": 0.2,
        "quantization": "8-bit",
        "repetition_penalty": 1.2
      },
      "prompt_tips": [
        "Explicitly ask for accessory details",
        "Request fabric material description",
        "Ask for pattern details separately"
      ]
    },
    "LLaVA-1.6-34B": {
      "strengths": ["detailed descriptions", "scene understanding"],
      "weaknesses": ["can be verbose", "occasional hallucination"],
      "optimal_settings": {
        "temperature": 0.3,
        "repetition_penalty": 1.3
      },
      "prompt_tips": [
        "Use structured output requests",
        "Set clear length limits"
      ]
    },
    "claude-sonnet-4-5-20250929": {
      "strengths": ["accuracy", "follows instructions", "structured output"],
      "weaknesses": ["conservative descriptions"],
      "optimal_settings": {
        "temperature": 0.3
      },
      "prompt_tips": [
        "Can handle complex multi-part prompts",
        "Good with JSON output"
      ]
    }
  }
}
```

---

## Agent Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                     Agent Execution Flow                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. LOAD KNOWLEDGE                                              │
│     ├─► Load local knowledge base                               │
│     └─► Check if update needed (>7 days old)                    │
│                                                                 │
│  2. SEARCH (if update_knowledge=True or outdated)               │
│     ├─► Tavily: "Z-Image prompting best practices 2025"         │
│     ├─► Tavily: "Flux SDXL prompt engineering techniques"       │
│     ├─► Tavily: "{model_name} vision model prompting tips"      │
│     └─► Update knowledge base with new findings                 │
│                                                                 │
│  3. ANALYZE SOURCE IMAGE                                        │
│     ├─► Claude: Detailed analysis of source image               │
│     ├─► Extract: subjects, clothing, colors, environment        │
│     └─► Compare against knowledge base vocabulary               │
│                                                                 │
│  4. EVALUATE PROMPT                                             │
│     ├─► Check against zimage_best_practices                     │
│     ├─► Check vocabulary usage                                  │
│     ├─► Identify missing elements                               │
│     └─► Identify hallucinated elements                          │
│                                                                 │
│  5. ANALYZE OUTPUT IMAGE                                        │
│     ├─► Claude: Compare output to source                        │
│     ├─► Track preserved vs lost elements                        │
│     └─► Assess prompt effectiveness                             │
│                                                                 │
│  6. MODEL-SPECIFIC ANALYSIS                                     │
│     ├─► Load tips for {model_name}                              │
│     ├─► Compare settings vs optimal                             │
│     └─► Generate model-specific recommendations                 │
│                                                                 │
│  7. GENERATE RECOMMENDATIONS                                    │
│     ├─► Prompt improvements (using vocabulary)                  │
│     ├─► Parameter adjustments                                   │
│     └─► Generate improved prompt example                        │
│                                                                 │
│  8. UPDATE LEARNED INSIGHTS                                     │
│     ├─► Log this evaluation to learned_insights.json            │
│     ├─► Update effectiveness scores for terms used              │
│     └─► Track model-specific patterns                           │
│                                                                 │
│  9. SAVE RESULTS                                                │
│     └─► Save complete evaluation to debug_results/              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Agent Tools

The agent uses Claude's native tool use capability with the following tools:

### Tool Definitions

| Tool | Description | Parameters |
|------|-------------|------------|
| `search_web` | Search web for latest techniques | query, category |
| `read_knowledge` | Read from local knowledge base | file, section |
| `update_knowledge` | Update local knowledge base | file, operation, data |
| `analyze_image` | Analyze source or output image | image_type, focus_areas |
| `compare_images` | Compare source and output | aspects |
| `evaluate_prompt` | Evaluate against best practices | check_vocabulary, check_structure, check_completeness |
| `get_model_tips` | Get model-specific tips | include_optimal_settings |
| `save_results` | Save evaluation to disk | include_images, include_knowledge_updates |

### Tool Schema

```python
TOOLS = [
    {
        "name": "search_web",
        "description": "Search the web for latest prompting techniques and best practices",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "category": {"type": "string", "enum": ["zimage", "vision_model", "model_specific"]}
            },
            "required": ["query"]
        }
    },
    {
        "name": "read_knowledge",
        "description": "Read from local knowledge base",
        "input_schema": {
            "type": "object",
            "properties": {
                "file": {"type": "string", "enum": [
                    "zimage_vocabulary",
                    "zimage_best_practices",
                    "vision_model_prompting",
                    "model_specific_tips",
                    "learned_insights"
                ]},
                "section": {"type": "string", "description": "Optional specific section"}
            },
            "required": ["file"]
        }
    },
    {
        "name": "update_knowledge",
        "description": "Update local knowledge base with new insights",
        "input_schema": {
            "type": "object",
            "properties": {
                "file": {"type": "string"},
                "operation": {"type": "string", "enum": ["add", "update", "increment_score"]},
                "data": {"type": "object"}
            },
            "required": ["file", "operation", "data"]
        }
    },
    {
        "name": "analyze_image",
        "description": "Analyze an image in detail",
        "input_schema": {
            "type": "object",
            "properties": {
                "image_type": {"type": "string", "enum": ["source", "output"]},
                "focus_areas": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["image_type"]
        }
    },
    {
        "name": "compare_images",
        "description": "Compare source and output images",
        "input_schema": {
            "type": "object",
            "properties": {
                "aspects": {"type": "array", "items": {"type": "string"}}
            }
        }
    },
    {
        "name": "evaluate_prompt",
        "description": "Evaluate prompt against best practices",
        "input_schema": {
            "type": "object",
            "properties": {
                "check_vocabulary": {"type": "boolean"},
                "check_structure": {"type": "boolean"},
                "check_completeness": {"type": "boolean"}
            }
        }
    },
    {
        "name": "get_model_tips",
        "description": "Get tips specific to the model that generated the prompt",
        "input_schema": {
            "type": "object",
            "properties": {
                "include_optimal_settings": {"type": "boolean"}
            }
        }
    },
    {
        "name": "save_results",
        "description": "Save evaluation results to disk",
        "input_schema": {
            "type": "object",
            "properties": {
                "include_images": {"type": "boolean"},
                "include_knowledge_updates": {"type": "boolean"}
            }
        }
    }
]
```

---

## Evaluation Criteria

### Scoring Categories

| Category | Weight | Description |
|----------|--------|-------------|
| Prompt Accuracy | 25% | Does prompt describe source image correctly? |
| Output Fidelity | 25% | Does output match source image? |
| Prompt Effectiveness | 25% | Did prompt guide output successfully? |
| Best Practices Compliance | 25% | Follows Z-Image rules? |

### Detailed Breakdown

```
1. PROMPT ACCURACY (0-10)
   - Subject description accuracy
   - Clothing description accuracy
   - Environment description accuracy
   - Technical details accuracy

2. OUTPUT FIDELITY (0-10)
   - Subject preservation
   - Clothing preservation
   - Pose preservation
   - Background preservation

3. PROMPT EFFECTIVENESS (0-10)
   - Detail translation
   - Style guidance
   - Technical guidance

4. BEST PRACTICES COMPLIANCE (0-10)
   - Structure compliance
   - Vocabulary usage
   - Must-include coverage
   - Must-avoid compliance

5. SPECIFIC CHECKS
   - Subject: face, pose, expression preserved?
   - Clothing: color, style, layers, fabric correct?
   - Environment: background, lighting matched?
   - Technical: quality, resolution, artifacts?

6. MODEL-SPECIFIC ANALYSIS
   - Known model tendencies
   - Parameter effectiveness
   - Settings vs optimal comparison
```

---

## Results Storage

### Directory Structure

```
ComfyUI-AI-Photography-Toolkit/
└── debug_results/
    └── YYYY-MM-DD_HH-MM-SS_<session_id>/
        ├── source.jpg
        ├── output.jpg
        ├── prompt.txt
        ├── evaluation.json
        └── metadata.json
```

### metadata.json

```json
{
  "timestamp": "2025-12-14T10:30:00Z",
  "model_config": {
    "provider": "local",
    "model": "Qwen2.5-VL-7B-Instruct",
    "text_model": "Qwen3-0.6B-Instruct",
    "temperature": 0.3,
    "max_tokens": 2048,
    "quantization": "4-bit",
    "repetition_penalty": 1.2
  },
  "generator_settings": {
    "analysis_mode": "standard",
    "prompt_length": 400
  },
  "system_info": {
    "platform": "win32",
    "gpu": "NVIDIA GeForce RTX 4090",
    "vram_total_gb": 24.0,
    "vram_available_gb": 18.5
  }
}
```

---

## Complete Response JSON Structure

```json
{
  "debug_session": {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "timestamp": "2025-12-14T10:30:00.000Z",
    "version": "1.0.0",
    "agent_version": "1.0.0"
  },

  "system_info": {
    "platform": "win32",
    "os_version": "Windows 11 Pro 23H2",
    "python_version": "3.11.9",
    "torch_version": "2.4.0+cu121",
    "cuda_version": "12.1",
    "gpu": {
      "name": "NVIDIA GeForce RTX 4090",
      "vram_total_gb": 24.0,
      "vram_available_gb": 18.5,
      "driver_version": "551.23"
    },
    "ram_total_gb": 64.0,
    "ram_available_gb": 48.2,
    "comfyui_version": "0.3.10"
  },

  "model_config": {
    "provider": "local",
    "model": "Qwen2.5-VL-7B-Instruct",
    "text_model": "Qwen3-0.6B-Instruct",
    "api_url": "",
    "temperature": 0.3,
    "max_tokens": 2048,
    "supports_vision": true,
    "supports_reasoning": false,
    "extra_params": {
      "quantization": "4-bit",
      "device": "cuda",
      "attention_mode": "flash_attention_2",
      "keep_model_loaded": true,
      "repetition_penalty": 1.2,
      "top_p": 0.9
    }
  },

  "generator_settings": {
    "analysis_mode": "standard",
    "prompt_length": 400,
    "generate_negative": false,
    "generate_caption": false,
    "user_guidance": ""
  },

  "inputs": {
    "source_image": {
      "path": "debug_results/2025-12-14_10-30-00_abc123/source.jpg",
      "dimensions": [1024, 768],
      "format": "JPEG",
      "file_size_kb": 245
    },
    "output_image": {
      "path": "debug_results/2025-12-14_10-30-00_abc123/output.jpg",
      "dimensions": [1024, 768],
      "format": "JPEG",
      "file_size_kb": 312
    },
    "prompt": {
      "path": "debug_results/2025-12-14_10-30-00_abc123/prompt.txt",
      "word_count": 387,
      "character_count": 2156,
      "content": "A young woman with long dark hair..."
    }
  },

  "knowledge_base_status": {
    "zimage_vocabulary": {
      "version": "1.0.0",
      "last_updated": "2025-12-14T08:00:00Z",
      "total_terms": 156
    },
    "zimage_best_practices": {
      "version": "1.0.0",
      "last_updated": "2025-12-14T08:00:00Z",
      "total_rules": 42
    },
    "vision_model_prompting": {
      "version": "1.0.0",
      "last_updated": "2025-12-14T08:00:00Z"
    },
    "model_specific_tips": {
      "version": "1.0.0",
      "models_covered": 12,
      "has_current_model": true
    },
    "search_performed": true,
    "new_insights_added": 3
  },

  "agent_steps": {
    "steps_completed": [
      {
        "step": 1,
        "name": "load_knowledge",
        "status": "success",
        "duration_ms": 45
      },
      {
        "step": 2,
        "name": "search_best_practices",
        "status": "success",
        "duration_ms": 2340,
        "searches": [
          {"query": "Z-Image prompting best practices 2025", "results": 5},
          {"query": "Flux prompt engineering 2025", "results": 4}
        ]
      },
      {
        "step": 3,
        "name": "analyze_source",
        "status": "success",
        "duration_ms": 3200,
        "elements_detected": 12
      },
      {
        "step": 4,
        "name": "evaluate_prompt",
        "status": "success",
        "duration_ms": 1500
      },
      {
        "step": 5,
        "name": "analyze_output",
        "status": "success",
        "duration_ms": 2800
      },
      {
        "step": 6,
        "name": "model_analysis",
        "status": "success",
        "duration_ms": 800
      },
      {
        "step": 7,
        "name": "generate_recommendations",
        "status": "success",
        "duration_ms": 2100
      }
    ],
    "total_duration_ms": 12785
  },

  "evaluation": {
    "evaluator": {
      "provider": "anthropic",
      "model": "claude-opus-4-5-20251101",
      "total_tokens_used": 4521
    },

    "scores": {
      "prompt_accuracy": {
        "score": 7.5,
        "max": 10,
        "breakdown": {
          "subject_description": 8.0,
          "clothing_description": 7.0,
          "environment_description": 7.5,
          "technical_details": 7.5
        }
      },
      "output_fidelity": {
        "score": 8.0,
        "max": 10,
        "breakdown": {
          "subject_preservation": 8.5,
          "clothing_preservation": 7.5,
          "pose_preservation": 8.5,
          "background_preservation": 7.5
        }
      },
      "prompt_effectiveness": {
        "score": 7.0,
        "max": 10,
        "breakdown": {
          "detail_translation": 7.0,
          "style_guidance": 7.5,
          "technical_guidance": 6.5
        }
      },
      "best_practices_compliance": {
        "score": 7.2,
        "max": 10,
        "breakdown": {
          "structure": 8.0,
          "vocabulary_usage": 6.5,
          "must_include_coverage": 7.0,
          "must_avoid_compliance": 8.5
        }
      },
      "overall": {
        "score": 7.4,
        "max": 10,
        "grade": "B"
      }
    },

    "element_analysis": {
      "detected_in_source": [
        {"element": "young woman", "category": "subject", "confidence": 0.98},
        {"element": "long dark hair", "category": "hair", "confidence": 0.95},
        {"element": "red floral dress", "category": "clothing", "confidence": 0.92},
        {"element": "silk material", "category": "fabric", "confidence": 0.75},
        {"element": "gold hoop earrings", "category": "accessory", "confidence": 0.88},
        {"element": "outdoor garden", "category": "environment", "confidence": 0.90},
        {"element": "natural daylight", "category": "lighting", "confidence": 0.93}
      ],
      "mentioned_in_prompt": [
        {"element": "young woman", "matched": true},
        {"element": "long dark hair", "matched": true},
        {"element": "red dress with floral pattern", "matched": true},
        {"element": "outdoor setting", "matched": true},
        {"element": "soft natural lighting", "matched": true}
      ],
      "missing_from_prompt": [
        {
          "element": "silk material",
          "category": "fabric",
          "suggested_term": "flowing silk fabric",
          "from_vocabulary": true
        },
        {
          "element": "gold hoop earrings",
          "category": "accessory",
          "suggested_term": "delicate gold hoop earrings",
          "from_vocabulary": false
        }
      ],
      "hallucinated_in_prompt": [],
      "vocabulary_usage": {
        "terms_from_knowledge_base": 4,
        "terms_missing_opportunity": 3,
        "effectiveness_score": 6.8
      }
    },

    "best_practices_analysis": {
      "structure": {
        "sentence_count": 5,
        "expected_range": "3-6",
        "compliant": true,
        "notes": "Good sentence structure"
      },
      "sentence_order": {
        "followed": ["subject", "clothing", "environment", "lighting"],
        "missing": ["pose/action", "technical quality tags"],
        "compliance": 0.67
      },
      "must_include": {
        "human_subjects": {
          "age_gender": true,
          "hair_description": true,
          "skin_texture": false,
          "clothing_details": true,
          "pose_action": false
        },
        "all_images": {
          "lighting": true,
          "background": true,
          "technical_quality": false
        }
      },
      "must_avoid": {
        "violations": [],
        "compliant": true
      }
    },

    "model_specific_analysis": {
      "model_info": {
        "provider": "local",
        "model_name": "Qwen2.5-VL-7B-Instruct",
        "model_type": "vision",
        "quantization": "4-bit",
        "in_knowledge_base": true
      },
      "known_model_traits": {
        "strengths": ["human detection", "pose analysis", "color accuracy"],
        "weaknesses": ["small accessories", "fine patterns", "fabric textures"]
      },
      "settings_analysis": {
        "temperature": {
          "current": 0.3,
          "optimal": 0.2,
          "assessment": "Slightly high for precision tasks",
          "impact": "May contribute to occasional imprecision"
        },
        "quantization": {
          "current": "4-bit",
          "optimal": "8-bit",
          "assessment": "4-bit may reduce detail detection",
          "impact": "Likely contributing to missed accessories"
        },
        "repetition_penalty": {
          "current": 1.2,
          "optimal": 1.2,
          "assessment": "Optimal",
          "impact": "No repetition observed"
        }
      },
      "observed_vs_expected": {
        "missed_accessories": {
          "expected": true,
          "observed": true,
          "notes": "Consistent with known weakness"
        },
        "fabric_detection": {
          "expected": "weak",
          "observed": "weak",
          "notes": "Silk material not detected, as predicted"
        }
      }
    }
  },

  "recommendations": {
    "priority": "high",

    "prompt_improvements": [
      {
        "category": "accessories",
        "issue": "Earrings not mentioned",
        "current": "(not mentioned)",
        "suggested": "delicate gold hoop earrings",
        "source": "detected in image",
        "priority": "high"
      },
      {
        "category": "fabric",
        "issue": "Material not specified",
        "current": "red dress",
        "suggested": "red silk dress with flowing drape",
        "source": "zimage_vocabulary.json",
        "priority": "high"
      },
      {
        "category": "skin_texture",
        "issue": "Missing realism tags",
        "current": "(not mentioned)",
        "suggested": "realistic skin texture, natural skin pores",
        "source": "zimage_best_practices.json",
        "priority": "medium"
      },
      {
        "category": "technical",
        "issue": "No quality enhancers",
        "current": "(not mentioned)",
        "suggested": "photorealistic, ultra detailed",
        "source": "zimage_vocabulary.json",
        "priority": "medium"
      }
    ],

    "parameter_adjustments": [
      {
        "parameter": "temperature",
        "current": 0.3,
        "suggested": 0.2,
        "reason": "Improve precision for detail detection",
        "expected_impact": "Better accessory and pattern detection"
      },
      {
        "parameter": "quantization",
        "current": "4-bit",
        "suggested": "8-bit",
        "reason": "Model's detail detection improves with higher precision",
        "expected_impact": "Better fabric and fine detail recognition",
        "vram_impact": "+2-3GB"
      }
    ],

    "system_prompt_suggestions": [
      {
        "area": "accessories",
        "suggestion": "Add to detection prompts: 'Always identify and describe jewelry, watches, and accessories'",
        "rationale": "Model consistently misses small accessories"
      },
      {
        "area": "fabric",
        "suggestion": "Add to clothing section: 'Identify fabric material (silk, cotton, wool, etc.) and texture'",
        "rationale": "Fabric detection is a known weakness"
      }
    ],

    "suggested_improved_prompt": "A young woman in her mid-twenties with long, dark flowing hair cascading past her shoulders. She wears a red silk dress with delicate floral patterns in white and pink, the fabric draping softly with natural folds. Delicate gold hoop earrings catch the light. Standing relaxed in an outdoor garden setting with lush green foliage creating a soft bokeh background. Warm natural daylight illuminates her from the front-left, creating gentle shadows that define her features. Realistic skin texture, natural skin pores, photorealistic, ultra detailed."
  },

  "search_insights": {
    "searches_performed": [
      {
        "query": "Z-Image prompting best practices 2025",
        "source": "tavily",
        "results_used": 3,
        "new_insights": [
          "Flux models respond well to 'cinematic' lighting descriptors",
          "Sentence-based prompts outperform keyword lists by 23%"
        ]
      },
      {
        "query": "Qwen2.5-VL prompting tips",
        "source": "tavily",
        "results_used": 2,
        "new_insights": [
          "Explicit structure requests improve output quality"
        ]
      }
    ],
    "knowledge_base_updates": [
      {
        "file": "zimage_vocabulary.json",
        "additions": ["cinematic lighting"],
        "updates": []
      },
      {
        "file": "model_specific_tips.json",
        "additions": [],
        "updates": ["Qwen2.5-VL-7B-Instruct tips expanded"]
      }
    ]
  },

  "learned_insights_update": {
    "session_learnings": [
      {
        "insight": "Qwen2.5-VL at 4-bit misses gold jewelry",
        "confidence": 0.85,
        "occurrences": 1
      },
      {
        "insight": "Silk fabric detection requires explicit prompting",
        "confidence": 0.75,
        "occurrences": 1
      }
    ],
    "vocabulary_effectiveness_updates": [
      {
        "term": "soft natural lighting",
        "previous_score": 9.2,
        "new_score": 9.2,
        "sample_size": 45
      }
    ]
  },

  "comparison_metrics": {
    "prompt_coverage": {
      "elements_in_source": 7,
      "elements_in_prompt": 5,
      "coverage_percentage": 71.4
    },
    "vocabulary_utilization": {
      "available_relevant_terms": 12,
      "terms_used": 4,
      "utilization_percentage": 33.3
    },
    "best_practices_compliance": {
      "total_rules": 15,
      "rules_followed": 11,
      "compliance_percentage": 73.3
    },
    "output_accuracy": {
      "elements_in_prompt": 5,
      "elements_in_output": 4,
      "accuracy_percentage": 80.0
    },
    "end_to_end": {
      "elements_in_source": 7,
      "elements_in_output": 4,
      "preservation_percentage": 57.1
    }
  },

  "full_evaluation_text": "## Prompt Evaluation Report\n\n### Overall Assessment\nThe generated prompt achieved a score of **7.4/10** (Grade: B)...\n\n### Strengths\n- Good sentence structure following Z-Image best practices\n- Accurate subject and clothing color description\n- Appropriate lighting description\n\n### Areas for Improvement\n1. **Missing Accessories**: Gold hoop earrings were not detected or mentioned\n2. **Fabric Material**: Silk material not identified - model weakness at 4-bit quantization\n3. **Technical Tags**: Missing realism enhancers (skin texture, photorealistic)\n\n### Model-Specific Notes\nQwen2.5-VL-7B at 4-bit quantization has known limitations with:\n- Small accessory detection\n- Fabric material identification\n\nConsider upgrading to 8-bit quantization if VRAM allows.\n\n### Recommended Prompt\n[See suggested_improved_prompt above]\n\n### Knowledge Base Insights\nThis session added 3 new insights to the knowledge base..."
}
```

---

## Implementation Plan

### Phase 1: Configuration & Structure
| Step | File | Description |
|------|------|-------------|
| 1.1 | `config/settings.toml` | Add `debug_mode = true` setting |
| 1.2 | `__init__.py` | Add conditional loading based on debug_mode |
| 1.3 | `debug_agent/__init__.py` | Create module structure |

### Phase 2: Knowledge Base (Pre-populated)
| Step | File | Description |
|------|------|-------------|
| 2.1 | `knowledge_base/zimage_vocabulary.json` | Curated Z-Image terms & effectiveness scores |
| 2.2 | `knowledge_base/zimage_best_practices.json` | Structure rules, must-include, must-avoid |
| 2.3 | `knowledge_base/vision_model_prompting.json` | General vision LLM prompting guidelines |
| 2.4 | `knowledge_base/model_specific_tips.json` | Per-model strengths, weaknesses, optimal settings |
| 2.5 | `knowledge_base/learned_insights.json` | Empty template for accumulated learnings |

### Phase 3: Agent Core
| Step | File | Description |
|------|------|-------------|
| 3.1 | `debug_agent/knowledge.py` | KnowledgeBase class - read/write/update JSON files |
| 3.2 | `debug_agent/tools.py` | Tool implementations (search, analyze, evaluate) |
| 3.3 | `debug_agent/agent.py` | PromptDebugAgent class - orchestration & tool loop |
| 3.4 | `debug_agent/system_info.py` | Gather system info (GPU, VRAM, OS, versions) |

### Phase 4: ComfyUI Node
| Step | File | Description |
|------|------|-------------|
| 4.1 | `sid_prompt_debug.py` | SID_PromptDebugAgent node definition |
| 4.2 | `sid_prompt_debug.py` | Input/output schema |
| 4.3 | `sid_prompt_debug.py` | Execute method - call agent |

### Phase 5: Storage & Results
| Step | File | Description |
|------|------|-------------|
| 5.1 | `debug_agent/storage.py` | Save results to debug_results/ folder |
| 5.2 | `debug_agent/storage.py` | Save images, prompt, evaluation JSON |

### File Dependency Order

```
1. config/settings.toml           (no dependencies)
2. knowledge_base/*.json          (no dependencies)
3. debug_agent/system_info.py     (no dependencies)
4. debug_agent/knowledge.py       (no dependencies)
5. debug_agent/tools.py           (depends on: knowledge.py)
6. debug_agent/storage.py         (no dependencies)
7. debug_agent/agent.py           (depends on: tools.py, knowledge.py, storage.py)
8. debug_agent/__init__.py        (exports agent)
9. sid_prompt_debug.py            (depends on: debug_agent)
10. __init__.py                   (updated - conditional import)
```

---

## Future Enhancements

- **Batch Evaluation**: Evaluate multiple prompts at once
- **A/B Testing**: Compare different prompt strategies
- **Dashboard**: Web UI for viewing aggregated results
- **Export**: Generate training data for prompt improvement
- **Auto-tune**: Automatically adjust model parameters based on learnings
