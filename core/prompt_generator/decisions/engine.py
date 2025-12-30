"""
Decision engine orchestrator.

Coordinates all decision makers to produce a complete Decisions object
from tagger outputs.
"""

from typing import Dict, Any, Optional
from ..types import Decisions, TaggerResults, SubjectType, NSFWLevel, ShotType
from .subject import SubjectDetector
from .gender import GenderDetector
from .nsfw import NSFWDetector
from .shot_type import ShotTypeDetector
from .sections import SectionTriggerEngine


class DecisionEngine:
    """
    Main decision engine that orchestrates all decision makers.

    Takes tagger results and produces a complete Decisions object
    that can be used for template selection and prompt generation.
    """

    def __init__(self):
        """Initialize the decision engine with all detectors."""
        self.subject_detector = SubjectDetector()
        self.gender_detector = GenderDetector()
        self.nsfw_detector = NSFWDetector()
        self.shot_type_detector = ShotTypeDetector()
        self.section_engine = SectionTriggerEngine()

    def make_decisions(
        self,
        tagger_results: TaggerResults,
        threshold: float = 0.35,
    ) -> Decisions:
        """
        Make all decisions from tagger results.

        Args:
            tagger_results: Combined outputs from all taggers
            threshold: Minimum confidence for tag matching

        Returns:
            Complete Decisions object
        """
        # Detect subject type
        subject_type = self.subject_detector.detect(tagger_results, threshold)

        # Detect gender (for human subjects)
        gender = None
        if subject_type in [SubjectType.WOMAN, SubjectType.MAN, SubjectType.COUPLE, SubjectType.GROUP]:
            gender = self.gender_detector.detect(tagger_results, threshold)

        # Detect NSFW level
        nsfw_level = self.nsfw_detector.detect(tagger_results, threshold)

        # Detect shot type
        shot_type = self.shot_type_detector.detect(tagger_results, threshold)

        # Get scene type from places365
        scene_type = self._get_scene_type(tagger_results)

        # Get quality level from IQA
        quality_level = self._get_quality_level(tagger_results)

        # Get lighting mood from photography analyzer
        lighting_mood = self._get_lighting_mood(tagger_results)

        # Get extended decisions from analyzers
        wildlife_info = self._get_wildlife_info(tagger_results)
        landmark_info = self._get_landmark_info(tagger_results)
        sky_weather_info = self._get_sky_weather_info(tagger_results)
        camera_info = self._get_camera_info(tagger_results)
        lighting_info = self._get_lighting_info(tagger_results)
        composition_info = self._get_composition_info(tagger_results)
        fashion_detected = self._get_fashion_detected(tagger_results)

        # Create decisions object
        decisions = Decisions(
            subject_type=subject_type,
            gender=gender,
            nsfw_level=nsfw_level,
            shot_type=shot_type,
            scene_type=scene_type,
            quality_level=quality_level,
            lighting_mood=lighting_mood,
            template_sections=[],  # Will be filled by section engine
            # Extended decisions
            wildlife_detected=wildlife_info.get("detected", False),
            wildlife_type=wildlife_info.get("type"),
            landmark_detected=landmark_info.get("detected", False),
            landmark_name=landmark_info.get("name"),
            landmark_famous=landmark_info.get("famous", False),
            sky_visible=sky_weather_info.get("sky_visible", False),
            weather_condition=sky_weather_info.get("weather"),
            time_of_day=sky_weather_info.get("time_of_day"),
            is_golden_hour=sky_weather_info.get("is_golden_hour", False),
            is_dramatic_lighting=sky_weather_info.get("is_dramatic", False) or lighting_info.get("is_dramatic", False),
            camera_angle=camera_info.get("angle"),
            depth_of_field=camera_info.get("dof"),
            composition_style=composition_info.get("style"),
            fashion_detected=fashion_detected,
            has_strong_shadows=lighting_info.get("has_strong_shadows", False),
            lighting_direction=lighting_info.get("direction"),
            lighting_type=lighting_info.get("type"),
        )

        # Determine active sections
        self.section_engine.get_active_sections(decisions, tagger_results, threshold)

        return decisions

    def _get_scene_type(self, tagger_results: TaggerResults) -> Optional[str]:
        """Extract scene type from places365 results."""
        if not tagger_results.places365:
            return None

        # Get highest confidence category
        if tagger_results.places365:
            sorted_places = sorted(
                tagger_results.places365.items(),
                key=lambda x: x[1],
                reverse=True
            )
            if sorted_places and sorted_places[0][1] > 0.3:
                return sorted_places[0][0]

        return None

    def _get_quality_level(self, tagger_results: TaggerResults) -> Optional[str]:
        """Extract quality level from IQA results."""
        if not tagger_results.iqa:
            return None

        aesthetic_score = tagger_results.iqa.get("aesthetic_score", 0)
        technical_score = tagger_results.iqa.get("technical_score", 0)

        avg_score = (aesthetic_score + technical_score) / 2

        if avg_score >= 8:
            return "PROFESSIONAL"
        elif avg_score >= 6:
            return "HIGH"
        elif avg_score >= 4:
            return "MEDIUM"
        else:
            return "LOW"

    def _get_lighting_mood(self, tagger_results: TaggerResults) -> Optional[str]:
        """Extract lighting mood from photography analyzer."""
        if not tagger_results.photography:
            return None

        # Check for specific lighting indicators
        low_key = tagger_results.photography.get("low_key", 0)
        high_key = tagger_results.photography.get("high_key", 0)
        warm = tagger_results.photography.get("warm_tones", 0)
        cool = tagger_results.photography.get("cool_tones", 0)
        dramatic = tagger_results.photography.get("dramatic", 0)
        soft = tagger_results.photography.get("soft", 0)

        # Determine mood based on highest indicator
        moods = {
            "DRAMATIC": dramatic + low_key,
            "SOFT": soft + high_key,
            "WARM": warm,
            "COOL": cool,
        }

        best_mood = max(moods, key=moods.get)
        if moods[best_mood] > 0.5:
            return best_mood

        return None

    def _get_wildlife_info(self, tagger_results: TaggerResults) -> Dict[str, Any]:
        """Extract wildlife info from wildlife analyzer."""
        if not tagger_results.wildlife:
            return {"detected": False}

        animals = tagger_results.wildlife.get("animals", {})
        if not animals:
            return {"detected": False}

        # Get top animal
        top_animal = max(animals, key=animals.get)
        top_score = animals[top_animal]

        return {
            "detected": top_score > 0.3,
            "type": top_animal if top_score > 0.3 else None,
        }

    def _get_landmark_info(self, tagger_results: TaggerResults) -> Dict[str, Any]:
        """Extract landmark info from landmarks analyzer."""
        if not tagger_results.landmarks:
            return {"detected": False}

        return {
            "detected": tagger_results.landmarks.get("detected", False),
            "name": tagger_results.landmarks.get("best_landmark"),
            "famous": tagger_results.landmarks.get("is_famous", False),
        }

    def _get_sky_weather_info(self, tagger_results: TaggerResults) -> Dict[str, Any]:
        """Extract sky/weather info from sky_weather analyzer."""
        if not tagger_results.sky_weather:
            return {"sky_visible": False}

        return {
            "sky_visible": tagger_results.sky_weather.get("sky_visible", False),
            "weather": tagger_results.sky_weather.get("weather"),
            "time_of_day": tagger_results.sky_weather.get("time_of_day"),
            "is_golden_hour": tagger_results.sky_weather.get("is_golden_hour", False),
            "is_dramatic": tagger_results.sky_weather.get("is_dramatic", False),
        }

    def _get_camera_info(self, tagger_results: TaggerResults) -> Dict[str, Any]:
        """Extract camera info from clip_camera analyzer."""
        if not tagger_results.clip_camera:
            return {}

        return {
            "angle": tagger_results.clip_camera.get("best_angle"),
            "dof": tagger_results.clip_camera.get("best_dof"),
            "perspective": tagger_results.clip_camera.get("best_perspective"),
        }

    def _get_lighting_info(self, tagger_results: TaggerResults) -> Dict[str, Any]:
        """Extract lighting info from lighting analyzer."""
        if not tagger_results.lighting:
            return {}

        # Get lighting type
        lighting_types = tagger_results.lighting.get("lighting_types", {})
        best_type = None
        if lighting_types:
            best_type = max(lighting_types, key=lighting_types.get)

        return {
            "has_strong_shadows": tagger_results.lighting.get("has_strong_shadows", False),
            "direction": tagger_results.lighting.get("dominant_direction"),
            "type": best_type,
            "is_dramatic": tagger_results.lighting.get("shadow_intensity", 0) > 0.3,
        }

    def _get_composition_info(self, tagger_results: TaggerResults) -> Dict[str, Any]:
        """Extract composition info from composition analyzer."""
        if not tagger_results.composition:
            return {}

        # Determine dominant composition style
        rule_of_thirds = tagger_results.composition.get("rule_of_thirds_score", 0)
        symmetry = tagger_results.composition.get("symmetry_score", 0)
        centered = tagger_results.composition.get("centered", False)

        style = None
        if symmetry > 0.7:
            style = "symmetrical"
        elif rule_of_thirds > 0.7:
            style = "rule_of_thirds"
        elif centered:
            style = "centered"

        return {"style": style}

    def _get_fashion_detected(self, tagger_results: TaggerResults) -> bool:
        """Check if fashion/clothing is detected."""
        # Check deepfashion results
        if tagger_results.deepfashion:
            clothing = tagger_results.deepfashion.get("clothing_type", {})
            if clothing:
                return True

        # Check fashion_color results
        if tagger_results.fashion_color:
            return len(tagger_results.fashion_color) > 0

        return False

    def get_decision_summary(self, decisions: Decisions) -> Dict[str, Any]:
        """
        Get a summary of decisions for debugging/metadata.

        Args:
            decisions: Decisions object

        Returns:
            Dict with decision summary
        """
        return {
            "subject_type": decisions.subject_type.value,
            "gender": decisions.gender,
            "nsfw_level": decisions.nsfw_level.value,
            "shot_type": decisions.shot_type.value,
            "scene_type": decisions.scene_type,
            "quality_level": decisions.quality_level,
            "lighting_mood": decisions.lighting_mood,
            "active_sections": decisions.template_sections,
            "section_count": len(decisions.template_sections),
        }


# Global singleton
_engine: Optional[DecisionEngine] = None


def get_engine() -> DecisionEngine:
    """Get the global DecisionEngine instance."""
    global _engine
    if _engine is None:
        _engine = DecisionEngine()
    return _engine
