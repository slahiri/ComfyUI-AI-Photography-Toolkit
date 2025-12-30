"""
Subject type detection from tagger outputs.

Analyzes tags to determine the primary subject of the image.
"""

from typing import Dict, Any, List, Tuple
from ..types import SubjectType, TaggerResults
from ..templates.loader import get_loader


class SubjectDetector:
    """
    Detects primary subject type from tagger outputs.

    Uses tag matching and analyzer signals to determine
    whether the image contains a woman, man, landscape, etc.
    """

    def __init__(self):
        """Initialize the subject detector."""
        self.loader = get_loader()

    def detect(
        self,
        tagger_results: TaggerResults,
        threshold: float = 0.35,
    ) -> SubjectType:
        """
        Detect subject type from tagger results.

        Args:
            tagger_results: Combined tagger outputs
            threshold: Minimum confidence for tag matching

        Returns:
            Detected SubjectType
        """
        rules = self.loader.get_subject_detection_rules()

        # Collect all tags above threshold
        all_tags = self._collect_tags(tagger_results, threshold)

        # Score each subject type
        scores: Dict[SubjectType, float] = {}

        for subject_name, rule in rules.items():
            try:
                subject_type = SubjectType(subject_name.lower())
            except ValueError:
                continue

            score = self._score_subject(subject_type, rule, all_tags, tagger_results)
            if score > 0:
                scores[subject_type] = score

        # Return highest scoring subject, or UNKNOWN
        if not scores:
            return SubjectType.UNKNOWN

        return max(scores, key=scores.get)

    def _collect_tags(
        self,
        tagger_results: TaggerResults,
        threshold: float,
    ) -> Dict[str, float]:
        """Collect all tags from all taggers above threshold."""
        all_tags = {}

        # WD14 tags
        for tag, conf in tagger_results.wd14.items():
            if conf >= threshold:
                all_tags[tag.lower().replace(" ", "_")] = conf

        # PixAI tags
        for tag, conf in tagger_results.pixai.items():
            if conf >= threshold:
                key = tag.lower().replace(" ", "_")
                if key not in all_tags or conf > all_tags[key]:
                    all_tags[key] = conf

        # JoyTag tags
        for tag, conf in tagger_results.joytag.items():
            if conf >= threshold:
                key = tag.lower().replace(" ", "_")
                if key not in all_tags or conf > all_tags[key]:
                    all_tags[key] = conf

        return all_tags

    def _score_subject(
        self,
        subject_type: SubjectType,
        rule: Dict[str, Any],
        all_tags: Dict[str, float],
        tagger_results: TaggerResults,
    ) -> float:
        """Score a subject type based on matching rules."""
        score = 0.0
        priority = rule.get("priority", 3)

        # Check trigger tags
        trigger_tags = rule.get("tags", [])
        for tag in trigger_tags:
            tag_key = tag.lower().replace(" ", "_")
            if tag_key in all_tags:
                score += all_tags[tag_key] * (4 - priority)  # Higher priority = higher multiplier

        # Check nudenet gender for WOMAN/MAN
        if subject_type == SubjectType.WOMAN:
            nudenet_gender = rule.get("nudenet_gender")
            if nudenet_gender and tagger_results.nudenet:
                detected_gender = tagger_results.nudenet.get("gender")
                if detected_gender == nudenet_gender:
                    score += 1.0

        elif subject_type == SubjectType.MAN:
            nudenet_gender = rule.get("nudenet_gender")
            if nudenet_gender and tagger_results.nudenet:
                detected_gender = tagger_results.nudenet.get("gender")
                if detected_gender == nudenet_gender:
                    score += 1.0

        # Check places365 categories for LANDSCAPE
        elif subject_type == SubjectType.LANDSCAPE:
            places_categories = rule.get("places365_categories", [])
            if places_categories and tagger_results.places365:
                for cat in places_categories:
                    cat_key = cat.lower()
                    if cat_key in tagger_results.places365:
                        score += tagger_results.places365[cat_key]

        # Check wildlife flag for ANIMAL
        elif subject_type == SubjectType.ANIMAL:
            if rule.get("wildlife_detected") and tagger_results.wildlife:
                if tagger_results.wildlife.get("detected"):
                    score += 2.0

        return score

    def get_confidence(
        self,
        tagger_results: TaggerResults,
        subject_type: SubjectType,
        threshold: float = 0.35,
    ) -> float:
        """
        Get confidence score for a specific subject type.

        Args:
            tagger_results: Combined tagger outputs
            subject_type: Subject type to score
            threshold: Minimum confidence for tag matching

        Returns:
            Confidence score (0-1)
        """
        rules = self.loader.get_subject_detection_rules()
        rule = rules.get(subject_type.name, {})

        all_tags = self._collect_tags(tagger_results, threshold)
        score = self._score_subject(subject_type, rule, all_tags, tagger_results)

        # Normalize score to 0-1 range
        max_possible = 5.0  # Rough estimate
        return min(1.0, score / max_possible)
