"""
Template selector based on subject type and decisions.

Selects the appropriate template and active sections based on
detected subject type and tag-based decisions.
"""

from typing import Dict, List, Any, Optional
from ..types import SubjectType, Decisions, ShotType, NSFWLevel
from .loader import get_loader


class TemplateSelector:
    """
    Selects templates and active sections based on decisions.

    Uses subject type to select the base template, then evaluates
    section trigger conditions to determine which sections to include.
    """

    def __init__(self):
        """Initialize the template selector."""
        self.loader = get_loader()

    def select_template(self, subject_type: SubjectType) -> Dict[str, Any]:
        """
        Select template based on subject type.

        Args:
            subject_type: Detected subject type

        Returns:
            Template configuration dict
        """
        return self.loader.get_template(subject_type.value)

    def get_active_sections(
        self,
        template: Dict[str, Any],
        decisions: Decisions,
        tags: Dict[str, float],
    ) -> List[str]:
        """
        Determine which template sections should be active.

        Args:
            template: Template configuration
            decisions: Tag-based decisions
            tags: Raw tag dict from taggers

        Returns:
            List of active section names in order
        """
        sections = template.get("sections", {})
        assembly_order = template.get("assembly_order", list(sections.keys()))

        active_sections = []

        for section_name in assembly_order:
            if section_name not in sections:
                continue

            section_config = sections[section_name]

            if self._should_activate_section(section_config, decisions, tags):
                active_sections.append(section_name)

        return active_sections

    def _should_activate_section(
        self,
        section_config: Dict[str, Any],
        decisions: Decisions,
        tags: Dict[str, float],
    ) -> bool:
        """
        Determine if a section should be activated.

        Args:
            section_config: Section configuration from template
            decisions: Tag-based decisions
            tags: Raw tag dict

        Returns:
            True if section should be active
        """
        # Check "trigger: always"
        if section_config.get("trigger") == "always":
            return True

        # Check trigger_tags
        trigger_tags = section_config.get("trigger_tags", [])
        if trigger_tags:
            tag_names = list(tags.keys())
            if any(t in tag_names for t in trigger_tags):
                return True

        # Check trigger_conditions
        trigger_conditions = section_config.get("trigger_conditions", {})
        if trigger_conditions:
            return self._evaluate_conditions(trigger_conditions, decisions, tags)

        # Default: don't activate if no conditions specified
        return False

    def _evaluate_conditions(
        self,
        conditions: Dict[str, Any],
        decisions: Decisions,
        tags: Dict[str, float],
    ) -> bool:
        """
        Evaluate trigger conditions.

        Args:
            conditions: Condition dict from template
            decisions: Tag-based decisions
            tags: Raw tag dict

        Returns:
            True if conditions are met
        """
        results = []

        # Check shot_type condition
        if "shot_type" in conditions:
            allowed_shots = conditions["shot_type"]
            shot_match = decisions.shot_type.value in [s.lower() for s in allowed_shots] or \
                        decisions.shot_type.name in allowed_shots
            results.append(shot_match)

        # Check nsfw_level condition
        if "nsfw_level" in conditions:
            allowed_levels = conditions["nsfw_level"]
            nsfw_match = decisions.nsfw_level.value in [l.lower() for l in allowed_levels] or \
                        decisions.nsfw_level.name in allowed_levels
            results.append(nsfw_match)

        # Check scene_confidence condition
        if "scene_confidence" in conditions:
            threshold_str = conditions["scene_confidence"]
            # Parse "> 0.5" format
            if threshold_str.startswith(">"):
                threshold = float(threshold_str[1:].strip())
                # Scene confidence comes from places365 analyzer
                # For now, assume it's in decisions
                scene_confidence = float(decisions.scene_type is not None)
                results.append(scene_confidence > threshold)

        # Check not_tags condition (negative check)
        if "not_tags" in conditions:
            excluded_tags = conditions["not_tags"]
            tag_names = list(tags.keys())
            no_excluded = not any(t in tag_names for t in excluded_tags)
            results.append(no_excluded)

        # Default operator is AND
        operator = conditions.get("operator", "AND")

        if not results:
            return False

        if operator == "OR":
            return any(results)
        else:  # AND
            return all(results)


# Global singleton
_selector: Optional[TemplateSelector] = None


def get_selector() -> TemplateSelector:
    """Get the global TemplateSelector instance."""
    global _selector
    if _selector is None:
        _selector = TemplateSelector()
    return _selector
