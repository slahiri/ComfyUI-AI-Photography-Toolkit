"""
Section trigger logic for template activation.

Determines which template sections should be active based on decisions.
"""

from typing import Dict, Any, List, Set
from ..types import Decisions, ShotType, NSFWLevel, TaggerResults
from ..templates.loader import get_loader


class SectionTriggerEngine:
    """
    Determines which template sections to activate.

    Evaluates section trigger rules against decisions and tags
    to determine the active sections for prompt generation.
    """

    # Sections that are always active regardless of conditions
    ALWAYS_ACTIVE = {"appearance", "expression", "lighting", "composition"}

    # Shot types where body is visible enough for body section
    BODY_VISIBLE_SHOTS = {ShotType.MS, ShotType.MFS, ShotType.FS, ShotType.MLS, ShotType.LS}

    # Shot types where pose is relevant
    POSE_RELEVANT_SHOTS = {ShotType.MCU, ShotType.MS, ShotType.MFS, ShotType.FS, ShotType.MLS, ShotType.LS}

    # Shot types where environment is prominent
    ENVIRONMENT_SHOTS = {ShotType.MLS, ShotType.LS, ShotType.ELS}

    def __init__(self):
        """Initialize the section trigger engine."""
        self.loader = get_loader()

    def get_active_sections(
        self,
        decisions: Decisions,
        tagger_results: TaggerResults,
        threshold: float = 0.35,
    ) -> List[str]:
        """
        Determine active sections based on decisions.

        Args:
            decisions: Tag-based decisions
            tagger_results: Raw tagger outputs
            threshold: Confidence threshold

        Returns:
            List of active section names
        """
        active: Set[str] = set(self.ALWAYS_ACTIVE)

        # Always include hair and face for human subjects
        if decisions.subject_type.value in ["woman", "man", "couple", "group"]:
            active.add("hair")
            active.add("face")
            active.add("eyes")

        # Body details: enabled for MS+ shots or NSFW content
        if self._should_include_body(decisions):
            active.add("body")

        # Pose details: enabled for MCU+ shots
        if self._should_include_pose(decisions):
            active.add("pose")

        # Environment: enabled unless simple background
        if self._should_include_environment(decisions, tagger_results, threshold):
            active.add("environment")

        # Clothing: enabled if clothing tags present or fashion detected
        if self._should_include_clothing(decisions, tagger_results, threshold):
            active.add("clothing")

        # Sky section: enabled if sky is visible
        if self._should_include_sky(decisions):
            active.add("sky")

        # Weather section: enabled if weather conditions detected
        if self._should_include_weather(decisions):
            active.add("weather")

        # Landmark section: enabled if landmark detected
        if self._should_include_landmark(decisions):
            active.add("landmark")

        # Golden hour section: enabled for golden/blue hour lighting
        if self._should_include_golden_hour(decisions):
            active.add("golden_hour")

        # Dramatic lighting section: enabled for dramatic lighting
        if self._should_include_dramatic_lighting(decisions):
            active.add("dramatic_lighting")

        # Camera details section: enabled when camera info available
        if self._should_include_camera_details(decisions):
            active.add("camera_details")

        # Store in decisions for reference
        decisions.template_sections = list(active)

        return decisions.template_sections

    def _should_include_body(self, decisions: Decisions) -> bool:
        """Check if body section should be included."""
        # Include if shot type shows body
        if decisions.shot_type in self.BODY_VISIBLE_SHOTS:
            return True

        # Include for suggestive/explicit content
        if decisions.nsfw_level in [NSFWLevel.SUGGESTIVE, NSFWLevel.EXPLICIT]:
            return True

        return False

    def _should_include_pose(self, decisions: Decisions) -> bool:
        """Check if pose section should be included."""
        return decisions.shot_type in self.POSE_RELEVANT_SHOTS

    def _should_include_environment(
        self,
        decisions: Decisions,
        tagger_results: TaggerResults,
        threshold: float,
    ) -> bool:
        """Check if environment section should be included."""
        # Check for simple background tags
        simple_bg_tags = ["simple_background", "white_background", "plain_background", "solid_background"]

        all_tags = {}
        for tag, conf in tagger_results.wd14.items():
            if conf >= threshold:
                all_tags[tag.lower().replace(" ", "_")] = conf

        # Exclude if simple background detected
        for bg_tag in simple_bg_tags:
            if bg_tag in all_tags:
                return False

        # Include if scene type detected
        if decisions.scene_type:
            return True

        # Include if places365 has high confidence
        if tagger_results.places365:
            max_conf = max(tagger_results.places365.values()) if tagger_results.places365 else 0
            if max_conf > 0.5:
                return True

        # Default: include environment
        return True

    def _should_include_clothing(
        self,
        decisions: Decisions,
        tagger_results: TaggerResults,
        threshold: float,
    ) -> bool:
        """Check if clothing section should be included."""
        # Check decision flag first (from deepfashion analyzer)
        if decisions.fashion_detected:
            return True

        clothing_tags = [
            "dress", "shirt", "outfit", "clothed", "wearing", "clothes",
            "suit", "jacket", "pants", "skirt", "blouse", "top", "jeans",
            "uniform", "costume", "bikini", "swimsuit", "lingerie"
        ]

        all_tags = {}
        for tag, conf in tagger_results.wd14.items():
            if conf >= threshold:
                all_tags[tag.lower().replace(" ", "_")] = conf

        for tag, conf in tagger_results.pixai.items():
            if conf >= threshold:
                key = tag.lower().replace(" ", "_")
                if key not in all_tags:
                    all_tags[key] = conf

        # Check for clothing tags
        for clothing_tag in clothing_tags:
            if clothing_tag in all_tags:
                return True

        return False

    def _should_include_sky(self, decisions: Decisions) -> bool:
        """Check if sky section should be included."""
        return decisions.sky_visible

    def _should_include_weather(self, decisions: Decisions) -> bool:
        """Check if weather section should be included."""
        return decisions.weather_condition is not None

    def _should_include_landmark(self, decisions: Decisions) -> bool:
        """Check if landmark section should be included."""
        return decisions.landmark_detected

    def _should_include_golden_hour(self, decisions: Decisions) -> bool:
        """Check if golden hour lighting section should be included."""
        return decisions.is_golden_hour

    def _should_include_dramatic_lighting(self, decisions: Decisions) -> bool:
        """Check if dramatic lighting section should be included."""
        return decisions.is_dramatic_lighting or decisions.has_strong_shadows

    def _should_include_camera_details(self, decisions: Decisions) -> bool:
        """Check if camera details section should be included."""
        return (
            decisions.camera_angle is not None or
            decisions.depth_of_field is not None or
            decisions.composition_style is not None
        )

    def evaluate_custom_rules(
        self,
        section_name: str,
        decisions: Decisions,
        tagger_results: TaggerResults,
    ) -> bool:
        """
        Evaluate custom rules for a specific section.

        Args:
            section_name: Section to evaluate
            decisions: Tag-based decisions
            tagger_results: Raw tagger outputs

        Returns:
            True if section should be active
        """
        rules = self.loader.get_section_trigger_rules()
        section_rule = rules.get(section_name, {})

        if not section_rule:
            return False

        enabled_conditions = section_rule.get("enabled_when", [])
        disabled_conditions = section_rule.get("disabled_when", [])
        operator = section_rule.get("operator", "AND")

        # Evaluate enabled conditions
        enabled_results = []
        for condition in enabled_conditions:
            result = self._evaluate_condition(condition, decisions, tagger_results)
            enabled_results.append(result)

        # Evaluate disabled conditions (if any match, disable)
        for condition in disabled_conditions:
            if self._evaluate_condition(condition, decisions, tagger_results):
                return False

        # Apply operator
        if not enabled_results:
            return False

        if operator == "OR":
            return any(enabled_results)
        else:
            return all(enabled_results)

    def _evaluate_condition(
        self,
        condition: Dict[str, Any],
        decisions: Decisions,
        tagger_results: TaggerResults,
    ) -> bool:
        """Evaluate a single condition."""
        cond_type = condition.get("condition")
        values = condition.get("values", [])
        threshold = condition.get("threshold")
        value = condition.get("value")

        if cond_type == "shot_type":
            return decisions.shot_type.name in values

        elif cond_type == "nsfw_level":
            return decisions.nsfw_level.name in values

        elif cond_type == "scene_confidence":
            max_conf = max(tagger_results.places365.values()) if tagger_results.places365 else 0
            return max_conf > (threshold or 0.5)

        elif cond_type == "tags":
            all_tags = set(tagger_results.wd14.keys())
            return bool(all_tags.intersection(set(values)))

        elif cond_type == "fashion_detected":
            return decisions.fashion_detected

        elif cond_type == "wildlife_detected":
            return decisions.wildlife_detected

        elif cond_type == "landmark_detected":
            return decisions.landmark_detected

        elif cond_type == "landmark_famous":
            return decisions.landmark_famous

        elif cond_type == "sky_visible":
            return decisions.sky_visible

        elif cond_type == "is_golden_hour":
            return decisions.is_golden_hour

        elif cond_type == "is_dramatic_lighting":
            return decisions.is_dramatic_lighting

        elif cond_type == "has_strong_shadows":
            return decisions.has_strong_shadows

        elif cond_type == "camera_angle":
            if value:
                return decisions.camera_angle == value
            return decisions.camera_angle is not None

        elif cond_type == "depth_of_field":
            if value:
                return decisions.depth_of_field == value
            return decisions.depth_of_field is not None

        elif cond_type == "composition_style":
            if value:
                return decisions.composition_style == value
            return decisions.composition_style is not None

        elif cond_type == "lighting_type":
            if value:
                return decisions.lighting_type == value
            return decisions.lighting_type is not None

        elif cond_type == "time_of_day":
            if value:
                return decisions.time_of_day == value
            return decisions.time_of_day is not None

        elif cond_type == "weather_condition":
            if value:
                return decisions.weather_condition == value
            return decisions.weather_condition is not None

        return False
