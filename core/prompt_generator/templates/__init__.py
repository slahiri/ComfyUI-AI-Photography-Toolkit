"""
Template system for prompt generation.

Templates define the structure of prompts based on subject type
and active sections determined by tag decisions.
"""

from .loader import ConfigLoader, get_loader
from .selector import TemplateSelector, get_selector
from .builder import PromptBuilder, get_builder
from .tag_injector import get_unmapped_tags, format_tags_for_prompt, categorize_tags, get_debug_info, get_female_anatomy_tags, FEMALE_ANATOMY_TAGS
from .scene_prompts import get_scene_prompt, get_scene_group, format_scene_prompt, SCENE_GROUPS
from .detail_prompts import (
    enhance_section_prompt,
    get_all_enhancements,
    get_animal_prompt,
    get_clothing_prompt,
    get_tag_enhancement,
    ANIMAL_PROMPTS,
    CLOTHING_PROMPTS,
    TAG_ENHANCEMENTS,
)

__all__ = [
    "ConfigLoader",
    "get_loader",
    "TemplateSelector",
    "get_selector",
    "PromptBuilder",
    "get_builder",
    "get_unmapped_tags",
    "format_tags_for_prompt",
    "categorize_tags",
    "get_debug_info",
    "get_scene_prompt",
    "get_scene_group",
    "format_scene_prompt",
    "SCENE_GROUPS",
    "enhance_section_prompt",
    "get_all_enhancements",
    "get_animal_prompt",
    "get_clothing_prompt",
    "get_tag_enhancement",
    "ANIMAL_PROMPTS",
    "CLOTHING_PROMPTS",
    "TAG_ENHANCEMENTS",
]
