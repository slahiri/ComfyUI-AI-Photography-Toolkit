"""
Template and decision configuration loader.

Loads YAML configuration files for templates and tag decisions.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import yaml


class ConfigLoader:
    """
    Loads and caches configuration files for the prompt generator.

    Provides access to:
    - prompt_templates.yaml: Templates for different subject types
    - tag_decisions.yaml: Rules for making decisions from tags
    """

    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize the config loader.

        Args:
            config_dir: Path to config directory. Defaults to ../config/
        """
        if config_dir is None:
            config_dir = Path(__file__).parent.parent / "config"
        self.config_dir = Path(config_dir)

        # Cached configs
        self._templates: Optional[Dict[str, Any]] = None
        self._decisions: Optional[Dict[str, Any]] = None

    def load_templates(self) -> Dict[str, Any]:
        """
        Load prompt templates configuration.

        Returns:
            Dict containing all template definitions
        """
        if self._templates is None:
            template_path = self.config_dir / "prompt_templates.yaml"
            if not template_path.exists():
                print(f"[ConfigLoader] Warning: Template file not found: {template_path}")
                self._templates = {"templates": {}}
            else:
                with open(template_path, 'r', encoding='utf-8') as f:
                    self._templates = yaml.safe_load(f) or {"templates": {}}

        return self._templates

    def load_decisions(self) -> Dict[str, Any]:
        """
        Load tag decisions configuration.

        Returns:
            Dict containing all decision rules
        """
        if self._decisions is None:
            decisions_path = self.config_dir / "tag_decisions.yaml"
            if not decisions_path.exists():
                print(f"[ConfigLoader] Warning: Decisions file not found: {decisions_path}")
                self._decisions = {}
            else:
                with open(decisions_path, 'r', encoding='utf-8') as f:
                    self._decisions = yaml.safe_load(f) or {}

        return self._decisions

    def get_template(self, subject_type: str) -> Dict[str, Any]:
        """
        Get template for a specific subject type.

        Args:
            subject_type: Subject type (woman, man, landscape, etc.)

        Returns:
            Template dict, or generic template if not found
        """
        templates = self.load_templates()
        all_templates = templates.get("templates", {})

        # Try exact match
        if subject_type.lower() in all_templates:
            return all_templates[subject_type.lower()]

        # Fallback to generic
        return all_templates.get("generic", {
            "base_prompt": "Describe this image in detail.",
            "sections": {},
            "assembly_order": [],
        })

    def get_subject_detection_rules(self) -> Dict[str, Any]:
        """Get subject detection rules."""
        decisions = self.load_decisions()
        return decisions.get("subject_detection", {})

    def get_nsfw_detection_rules(self) -> Dict[str, Any]:
        """Get NSFW detection rules."""
        decisions = self.load_decisions()
        return decisions.get("nsfw_detection", {})

    def get_shot_type_mapping(self) -> Dict[str, Any]:
        """Get shot type mapping rules."""
        decisions = self.load_decisions()
        return decisions.get("shot_type_mapping", {})

    def get_section_trigger_rules(self) -> Dict[str, Any]:
        """Get section trigger rules."""
        decisions = self.load_decisions()
        return decisions.get("section_triggers", {})

    def get_confidence_thresholds(self) -> Dict[str, float]:
        """Get confidence thresholds for taggers."""
        decisions = self.load_decisions()
        return decisions.get("confidence_thresholds", {
            "wd14": 0.35,
            "pixai": 0.4,
            "joytag": 0.5,
            "nudenet": 0.3,
        })

    def reload(self):
        """Force reload of all configurations."""
        self._templates = None
        self._decisions = None


# Global singleton instance
_loader: Optional[ConfigLoader] = None


def get_loader() -> ConfigLoader:
    """Get the global ConfigLoader instance."""
    global _loader
    if _loader is None:
        _loader = ConfigLoader()
    return _loader
