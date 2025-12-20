# Version History Analysis

Analysis of branches 4.0.0 through 4.3.0 to identify what went wrong and what could be salvaged.
Focus: **Simplicity**

---

## Complexity Growth

| Version | Python Files | Key Changes |
|---------|-------------|-------------|
| 4.0.0 | 4 | Single node, single purpose |
| 4.1.0 | 19 | +LLM providers, +prompt templates, +advanced node |
| 4.2.0 | 16 | Consolidated slightly, +local models |
| 4.3.0 | 32 | +debug agent, +web UI, +learning system |

**Pattern**: 4x → 4x → 2x file growth per version

---

## 4.0.0 - The Simple Beginning

**Files:** `__init__.py`, `sid_zimage_prompt_generator.py`, `utils/zimage_utils.py`

**What Worked:**
- Single node (`SID_ZImagePromptGenerator`) - one purpose
- All logic in one file (~1100 lines)
- Clean utility separation (caching, config, output cleaning)
- YAML config for prompts/attributes

**What Was Good:**
- `clean_zimage_output()` - removes meta-tags, thinking blocks, markdown
- `hash_image_tensor()` - fast image hashing for caching
- `get_image_metadata()` - aspect ratio, orientation detection
- Multi-provider support (Anthropic, Ollama, Grok) in single node

**Problems:**
- All 3 providers implemented inline (messy)
- Large YAML config (~700 lines) for attributes
- "Agentic pipeline" was just 2 LLM calls with hardcoded stages

---

## 4.1.0 - Abstraction Explosion

**Added:**
- `llm_providers/` - Base class + 4 providers (Anthropic, Grok, OpenAI, GGUF)
- `prompt_templates/` - Base class + 2 templates (Claude, GGUF)
- Split into Basic + Advanced nodes
- Auto-install dependencies in `__init__.py`

**What Went Wrong:**
1. **Over-abstraction**: `BaseLLMProvider` class with 10+ methods for 4 implementations
2. **Two nodes**: Basic vs Advanced confused users
3. **GGUF complexity**: GPU detection, llama-cpp-python auto-install
4. **Template inheritance**: `BasePromptTemplate` → `ClaudeTemplate` → overkill for 2 templates

**What Could Be Salvaged:**
- Provider pattern (but simpler - just functions, not classes)
- GPU detection logic (useful for local models)

---

## 4.2.0 - Bug Fix Hell

**53 commits** between 4.1.0 and 4.2.0 - mostly fixes:
- DynamicCache compatibility (5 commits)
- Flash attention detection (2 commits)
- Local model errors (Florence-2, Moondream2, QwenVL)
- VAE decode hang, KSampler slowdown
- Empty response crashes

**Added:**
- `config_loader.py` - TOML config for tier-specific prompts
- Separate `sid_llm_api.py` and `sid_llm_local.py`

**What Went Wrong:**
1. **Local model complexity**: Each model (Florence, Moondream, Qwen) needed special handling
2. **Transformers version chasing**: DynamicCache API kept changing
3. **Too many model options**: 15+ models, each with quirks
4. **Mixed concerns**: Image analysis + prompt generation + model management

**What Could Be Salvaged:**
- Anthropic API implementation (clean, stable)
- Output cleaning utilities
- TOML config pattern

---

## 4.3.0 - Feature Creep Death

**Added:**
- `debug_agent/` - 5 files for debugging/learning system
- `prompt_editor/` - Web UI for editing prompts
- `cv_analyzer.py` - OpenCV-based pre-analysis
- `negative_prompt_builder.py` - Negative prompt generation
- `zimage_vocabulary.py` - Vocabulary management
- `template_registry_loader.py` - Dynamic template loading

**Commits show the problem:**
```
improve: templates from non-excluded debug session (score 7.75)
improve: prompt quality based on evaluator learnings (score > 5 analysis)
fix: critical framing accuracy - stop hallucinating body parts beyond frame
fix: add GAZE DIRECTION guidance for accurate pose description
```

**What Went Wrong:**
1. **Prompt engineering via code**: Endless tweaks to fix LLM output
2. **Debug system bloat**: Built a learning system instead of fixing prompts
3. **CV pre-analysis**: Added complexity without improving results
4. **Model kept hallucinating**: Small models (Qwen3-VL-2B) couldn't follow instructions

**Root Cause:**
- Trying to make small local models do complex multi-stage analysis
- Building infrastructure to "learn" instead of using better models
- Feature creep to avoid admitting the core approach wasn't working

---

## What To Salvage

### Keep (Simple, Working)

| Component | Source | Lines | Notes |
|-----------|--------|-------|-------|
| `clean_zimage_output()` | 4.0.0/utils | ~60 | Removes meta-tags, thinking blocks |
| `hash_image_tensor()` | 4.0.0/utils | ~20 | Fast image hashing |
| `get_image_metadata()` | 4.0.0/utils | ~30 | Aspect ratio, orientation |
| Anthropic API call | 4.2.0/api | ~100 | Clean, stable implementation |
| Output truncation | 4.0.0/utils | ~20 | Sentence-boundary truncation |

### Discard

| Component | Reason |
|-----------|--------|
| LLM provider classes | Over-abstracted, 10+ methods for simple API calls |
| Prompt template classes | Overkill - just use string templates |
| Debug agent | Built to fix symptoms, not causes |
| CV analyzer | Added complexity, no improvement |
| Local model handlers | Too many edge cases per model |
| GGUF support | Installation complexity, limited benefit |

### Rethink

| Component | Problem | Solution |
|-----------|---------|----------|
| Multi-stage pipeline | Hardcoded stages | Single-stage with good prompts |
| Model selection | UI dropdown explosion | Tier selection (Quick/Standard/Detailed) |
| Caching | Complex key generation | Simple hash + params |
| Config | 700-line YAML | Minimal TOML |

---

## Key Lessons

1. **One node, one purpose**: Don't split into Basic/Advanced
2. **Functions over classes**: LLM calls don't need inheritance
3. **Fewer models, better prompts**: One good model > 15 quirky ones
4. **Fix prompts, not code**: Endless code changes can't fix bad prompts
5. **Local models have limits**: Small VLMs can't do complex reasoning
6. **KISS**: Simple caching, simple config, simple output

---

## Recommended Architecture (v5.0)

```
__init__.py              # Extension entry point
nodes/
  caption.py             # Single captioning node
  utilities.py           # Helper nodes (if needed)
core/
  llm.py                 # Simple LLM API calls (functions, not classes)
  output.py              # Output cleaning utilities
  cache.py               # Simple caching
config/
  prompts.toml           # Prompt templates
  models.toml            # Model definitions
```

**Principles:**
- Max 6-8 Python files
- No class hierarchies for LLM providers
- Tier-based model selection (not dropdown of 20 models)
- Single captioning node with quality tier input
- Delegate complex analysis to capable API models
