"""Module selection logic for Refine mode."""

import json
from pathlib import Path
from typing import Optional


def _load_refine_config() -> dict:
    """Load refine modules configuration."""
    config_path = Path(__file__).parent.parent.parent / "config" / "refine_modules.json"
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[RefineEngine] Warning: Could not load refine config: {e}")
        return {}


# Load config at module level
REFINE_CONFIG = _load_refine_config()


class ModuleSelector:
    """
    Selects which question modules to run based on detected tags.

    Analyzes metadata from Stage 1 (ImageAnalysis) and determines
    which modules are relevant for the image content.
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or REFINE_CONFIG
        self.modules_config = self.config.get("modules", {})

    def select(self, metadata: dict, mode: str = "adaptive") -> list[str]:
        """
        Select modules to run based on metadata and mode.

        Args:
            metadata: Metadata dict from ImageAnalysis
            mode: "sequential", "parallel", or "adaptive"

        Returns:
            List of module names to run, in priority order
        """
        if mode == "sequential" or mode == "parallel":
            # Run all modules
            return self._get_all_modules()
        elif mode == "adaptive":
            return self._select_adaptive(metadata)
        else:
            # Default to adaptive
            return self._select_adaptive(metadata)

    def _get_all_modules(self) -> list[str]:
        """Get all modules sorted by priority."""
        modules = []
        for name, config in self.modules_config.items():
            priority = config.get("priority", 99)
            modules.append((priority, name))

        modules.sort(key=lambda x: x[0])
        return [name for _, name in modules]

    def _select_adaptive(self, metadata: dict) -> list[str]:
        """
        Select modules adaptively based on detected tags.

        Always runs: subject, lighting, framing
        Conditionally runs: clothing, pose, body_detail, environment, style
        """
        selected = []

        for name, config in self.modules_config.items():
            priority = config.get("priority", 99)

            # Always run core modules
            if config.get("always_run", False):
                selected.append((priority, name))
                continue

            # Check trigger conditions
            trigger = config.get("trigger", {})
            if self._check_trigger(metadata, trigger):
                selected.append((priority, name))

        # Sort by priority
        selected.sort(key=lambda x: x[0])
        return [name for _, name in selected]

    def _check_trigger(self, metadata: dict, trigger: dict) -> bool:
        """
        Check if trigger conditions are met.

        Args:
            metadata: Full metadata dict
            trigger: Trigger config with taggers, keywords, min_confidence

        Returns:
            True if module should be activated
        """
        if not trigger:
            return False

        taggers = trigger.get("taggers", [])
        keywords = trigger.get("keywords", [])
        min_confidence = trigger.get("min_confidence", 0.4)

        # Check each specified tagger
        for tagger_name in taggers:
            if tagger_name not in metadata:
                continue

            tagger_data = metadata[tagger_name]
            if not isinstance(tagger_data, dict):
                continue

            # Check each tag in this tagger
            for tag, confidence in tagger_data.items():
                if confidence < min_confidence:
                    continue

                tag_lower = tag.lower().replace("_", " ").replace("-", " ")

                # Check if any keyword matches
                for keyword in keywords:
                    keyword_lower = keyword.lower().replace("_", " ")
                    if keyword_lower in tag_lower or tag_lower in keyword_lower:
                        return True

        return False

    def get_module_config(self, module_name: str) -> dict:
        """Get configuration for a specific module."""
        return self.modules_config.get(module_name, {})

    def build_prompt(self, module_name: str, metadata: dict, tag_threshold: float = 0.35) -> str:
        """
        Build the prompt for a module, always injecting tag context.

        Args:
            module_name: Name of the module
            metadata: Full metadata dict
            tag_threshold: Minimum confidence to include tags (0.1-0.9)

        Returns:
            Prompt string with tag context included
        """
        config = self.get_module_config(module_name)

        # Get base prompt
        prompt = config.get("prompt", "Describe this aspect of the image.")

        # Get trigger info for specific tag collection
        trigger = config.get("trigger", {})
        detected_tags = self._collect_triggered_tags(metadata, trigger, tag_threshold)

        # If module has a prompt_template with detected tags, use it
        if detected_tags and "prompt_template" in config:
            prompt = config["prompt_template"]
            tag_context = self._build_tag_context(metadata, config.get("context_tags", []), tag_threshold)
            prompt = prompt.replace("{detected_tags}", detected_tags)
            prompt = prompt.replace("{tag_context}", tag_context)

        # Build comprehensive tag summary and put it BEFORE the instructions
        tag_summary = self._build_comprehensive_tag_summary(metadata, tag_threshold)
        if tag_summary:
            # Put tags at the START as context, with VERIFICATION instruction
            tag_context_block = f"""DETECTED ITEMS TO VERIFY (check each one against the image):
{tag_summary}

INSTRUCTIONS:
1. Look at the image and CHECK each detected item above
2. If you see it, describe it accurately (correct colors, positions)
3. If a tag is wrong, ignore it
4. DO NOT copy tag labels - write natural descriptions
5. Pay attention to: exposed skin areas, layered clothing, headwear layers, hand positions
---

"""
            prompt = tag_context_block + prompt

        return prompt

    def _build_comprehensive_tag_summary(self, metadata: dict, tag_threshold: float = 0.35) -> str:
        """
        Build a comprehensive summary of ALL detected tags from metadata.
        This gives the VLM context about what was detected in the analysis stage.

        Args:
            metadata: Full metadata dict from ImageAnalysis
            tag_threshold: Minimum confidence to include tags (0.1-0.9)
        """
        if not metadata:
            return ""

        sections = []

        # First, add Florence outputs (most valuable context for prompt generation)
        # PromptGen-specific outputs first (best for prompts)
        if "florence_mixed_caption_plus" in metadata and metadata["florence_mixed_caption_plus"]:
            sections.append(f"Florence Mixed Caption+: {metadata['florence_mixed_caption_plus'].strip()[:600]}")
        elif "florence_mixed_caption" in metadata and metadata["florence_mixed_caption"]:
            sections.append(f"Florence Mixed Caption: {metadata['florence_mixed_caption'].strip()[:600]}")

        if "florence_generate_tags" in metadata and metadata["florence_generate_tags"]:
            sections.append(f"Florence Tags: {metadata['florence_generate_tags'].strip()[:400]}")

        if "florence_analyze" in metadata and metadata["florence_analyze"]:
            sections.append(f"Florence Analysis: {metadata['florence_analyze'].strip()[:400]}")

        # Standard Florence outputs as fallback/supplement
        if "florence_description" in metadata and metadata["florence_description"]:
            sections.append(f"Florence Description: {metadata['florence_description'].strip()[:400]}")
        elif "florence_caption" in metadata and metadata["florence_caption"]:
            sections.append(f"Florence Caption: {metadata['florence_caption'].strip()[:300]}")

        # Also check for other caption sources
        other_caption_keys = ["caption", "description", "vlm_caption"]
        for cap_key in other_caption_keys:
            if cap_key in metadata and metadata[cap_key]:
                cap_data = metadata[cap_key]
                if isinstance(cap_data, str) and cap_data.strip():
                    # Don't duplicate if already have florence
                    if not any("Florence" in s for s in sections):
                        sections.append(f"Image Caption: {cap_data.strip()[:400]}")
                    break

        # Priority taggers to include
        tagger_display_names = {
            "wd14": "General Tags",
            "joytag": "Content Tags",
            "fashion_yolov8": "Clothing Detection",
            "fashion_yolos": "Fashion Items",
            "fashion_segformer": "Garment Segments",
            "fashion_clip": "Fashion Style",
            "nudenet": "Body Areas",
            "pose": "Pose Detection",
            "composition": "Composition",
            "saliency": "Focus Areas",
        }

        for tagger_key, display_name in tagger_display_names.items():
            if tagger_key not in metadata:
                continue

            tagger_data = metadata[tagger_key]

            if isinstance(tagger_data, dict):
                # Filter by threshold and sort by confidence
                high_conf_tags = [
                    (tag, conf) for tag, conf in tagger_data.items()
                    if isinstance(conf, (int, float)) and conf >= tag_threshold
                ]
                if not high_conf_tags:
                    continue

                sorted_tags = sorted(high_conf_tags, key=lambda x: x[1], reverse=True)[:10]
                tags_str = ", ".join(f"{tag}" for tag, conf in sorted_tags)
                sections.append(f"{display_name}: {tags_str}")

            elif isinstance(tagger_data, str) and tagger_data.strip():
                sections.append(f"{display_name}: {tagger_data[:150]}")

            elif isinstance(tagger_data, list) and tagger_data:
                # Handle list format (some taggers return lists)
                items = [str(item) for item in tagger_data[:10]]
                sections.append(f"{display_name}: {', '.join(items)}")

        return "\n".join(sections) if sections else ""

    def _collect_triggered_tags(self, metadata: dict, trigger: dict, tag_threshold: float = 0.35) -> str:
        """Collect tags that triggered this module."""
        if not trigger:
            return ""

        taggers = trigger.get("taggers", [])
        keywords = trigger.get("keywords", [])
        # Use the higher of config min_confidence or user threshold
        min_confidence = max(trigger.get("min_confidence", 0.4), tag_threshold)

        found_tags = []

        for tagger_name in taggers:
            if tagger_name not in metadata:
                continue

            tagger_data = metadata[tagger_name]
            if not isinstance(tagger_data, dict):
                continue

            for tag, confidence in tagger_data.items():
                if confidence < min_confidence:
                    continue

                tag_lower = tag.lower().replace("_", " ").replace("-", " ")

                for keyword in keywords:
                    keyword_lower = keyword.lower().replace("_", " ")
                    if keyword_lower in tag_lower or tag_lower in keyword_lower:
                        # Format: tag (confidence%)
                        found_tags.append(f"{tag} ({confidence:.0%})")
                        break

        return ", ".join(found_tags[:15])  # Limit to 15 tags

    def _build_tag_context(self, metadata: dict, context_taggers: list[str], tag_threshold: float = 0.35) -> str:
        """Build context from specified taggers."""
        if not context_taggers:
            return ""

        context_parts = []

        for tagger_name in context_taggers:
            if tagger_name not in metadata:
                continue

            tagger_data = metadata[tagger_name]
            if isinstance(tagger_data, dict):
                # Get top tags above threshold
                filtered_tags = [(t, c) for t, c in tagger_data.items() if c >= tag_threshold]
                sorted_tags = sorted(filtered_tags, key=lambda x: x[1], reverse=True)[:5]
                if sorted_tags:
                    tags_str = ", ".join(f"{t} ({c:.0%})" for t, c in sorted_tags)
                    context_parts.append(f"{tagger_name}: {tags_str}")
            elif isinstance(tagger_data, str):
                context_parts.append(f"{tagger_name}: {tagger_data[:200]}")

        return "\n".join(context_parts)
