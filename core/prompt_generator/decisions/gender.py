"""
Gender detection from tagger outputs.

Analyzes tags and NudeNet output to determine subject gender.
"""

from typing import Dict, Any, Optional
from ..types import TaggerResults
from ..templates.loader import get_loader


class GenderDetector:
    """
    Detects subject gender from tagger outputs.

    Uses tag matching and NudeNet signals.
    """

    # Tag patterns for gender detection
    FEMALE_TAGS = ["1girl", "woman", "female", "lady", "she", "girl", "female_focus"]
    MALE_TAGS = ["1boy", "man", "male", "gentleman", "he", "boy", "male_focus"]

    def __init__(self):
        """Initialize the gender detector."""
        self.loader = get_loader()

    def detect(
        self,
        tagger_results: TaggerResults,
        threshold: float = 0.35,
    ) -> Optional[str]:
        """
        Detect gender from tagger results.

        Args:
            tagger_results: Combined tagger outputs
            threshold: Minimum confidence for tag matching

        Returns:
            "female", "male", or None if undetermined
        """
        female_score = 0.0
        male_score = 0.0

        # Check WD14 tags
        for tag, conf in tagger_results.wd14.items():
            if conf < threshold:
                continue
            tag_lower = tag.lower().replace(" ", "_")

            if any(ft in tag_lower for ft in self.FEMALE_TAGS):
                female_score += conf
            elif any(mt in tag_lower for mt in self.MALE_TAGS):
                male_score += conf

        # Check PixAI tags
        for tag, conf in tagger_results.pixai.items():
            if conf < threshold:
                continue
            tag_lower = tag.lower().replace(" ", "_")

            if any(ft in tag_lower for ft in self.FEMALE_TAGS):
                female_score += conf * 0.5  # Weight less than WD14
            elif any(mt in tag_lower for mt in self.MALE_TAGS):
                male_score += conf * 0.5

        # Check NudeNet gender
        if tagger_results.nudenet:
            nudenet_gender = tagger_results.nudenet.get("gender")
            if nudenet_gender == "female":
                female_score += 1.0
            elif nudenet_gender == "male":
                male_score += 1.0

        # Determine result
        if female_score > male_score and female_score > threshold:
            return "female"
        elif male_score > female_score and male_score > threshold:
            return "male"

        return None

    def get_confidence(
        self,
        tagger_results: TaggerResults,
        gender: str,
        threshold: float = 0.35,
    ) -> float:
        """
        Get confidence for a specific gender.

        Args:
            tagger_results: Combined tagger outputs
            gender: "female" or "male"
            threshold: Minimum confidence

        Returns:
            Confidence score (0-1)
        """
        target_tags = self.FEMALE_TAGS if gender == "female" else self.MALE_TAGS
        score = 0.0
        count = 0

        for tag, conf in tagger_results.wd14.items():
            if conf < threshold:
                continue
            tag_lower = tag.lower().replace(" ", "_")
            if any(tt in tag_lower for tt in target_tags):
                score += conf
                count += 1

        if tagger_results.nudenet:
            if tagger_results.nudenet.get("gender") == gender:
                score += 1.0
                count += 1

        return score / max(1, count)
