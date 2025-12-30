"""
NSFW level detection from tagger outputs.

Analyzes NudeNet output to determine content classification.
"""

from typing import Dict, Any, List
from ..types import NSFWLevel, TaggerResults
from ..templates.loader import get_loader


class NSFWDetector:
    """
    Detects NSFW level from tagger outputs.

    Uses NudeNet body part detection to classify content.
    """

    # Body parts indicating different NSFW levels
    SUGGESTIVE_PARTS = [
        "cleavage",
        "covered_breast",
        "bare_shoulders",
        "midriff",
        "thighs",
        "buttocks_covered",
    ]

    EXPLICIT_PARTS = [
        "exposed_breast",
        "exposed_nipple",
        "exposed_genitalia",
        "nude",
        "topless",
        "exposed_buttocks",
    ]

    def __init__(self):
        """Initialize the NSFW detector."""
        self.loader = get_loader()

    def detect(
        self,
        tagger_results: TaggerResults,
        threshold: float = 0.3,
    ) -> NSFWLevel:
        """
        Detect NSFW level from tagger results.

        Args:
            tagger_results: Combined tagger outputs
            threshold: Minimum confidence for detection

        Returns:
            Detected NSFWLevel
        """
        if not tagger_results.nudenet:
            return NSFWLevel.SAFE

        # Get detected parts
        parts = tagger_results.nudenet.get("parts", [])
        if isinstance(parts, dict):
            # If parts is a dict with confidence scores
            parts = [p for p, conf in parts.items() if conf >= threshold]
        elif isinstance(parts, list):
            # If parts is already a list
            parts = [p.lower() for p in parts]

        # Check for explicit content first
        for part in parts:
            part_lower = part.lower().replace(" ", "_")
            if any(ep in part_lower for ep in self.EXPLICIT_PARTS):
                return NSFWLevel.EXPLICIT

        # Check for suggestive content
        for part in parts:
            part_lower = part.lower().replace(" ", "_")
            if any(sp in part_lower for sp in self.SUGGESTIVE_PARTS):
                return NSFWLevel.SUGGESTIVE

        # Check exposure score if available
        exposure_score = tagger_results.nudenet.get("exposure_score", 0)
        if exposure_score > 0.5:
            return NSFWLevel.EXPLICIT
        elif exposure_score > 0.3:
            return NSFWLevel.SUGGESTIVE

        return NSFWLevel.SAFE

    def get_detected_parts(
        self,
        tagger_results: TaggerResults,
        threshold: float = 0.3,
    ) -> List[str]:
        """
        Get list of detected body parts.

        Args:
            tagger_results: Combined tagger outputs
            threshold: Minimum confidence

        Returns:
            List of detected part names
        """
        if not tagger_results.nudenet:
            return []

        parts = tagger_results.nudenet.get("parts", [])

        if isinstance(parts, dict):
            return [p for p, conf in parts.items() if conf >= threshold]
        elif isinstance(parts, list):
            return parts

        return []

    def get_exposure_score(self, tagger_results: TaggerResults) -> float:
        """
        Get overall exposure score.

        Args:
            tagger_results: Combined tagger outputs

        Returns:
            Exposure score (0-1)
        """
        if not tagger_results.nudenet:
            return 0.0

        return tagger_results.nudenet.get("exposure_score", 0.0)
