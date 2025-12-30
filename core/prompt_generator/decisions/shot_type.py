"""
Shot type detection from analyzer outputs.

Maps shot type analyzer results to ShotType enum.
"""

from typing import Dict, Any, Optional
from ..types import ShotType, TaggerResults
from ..templates.loader import get_loader


class ShotTypeDetector:
    """
    Detects camera shot type from analyzer outputs.

    Uses the shot_type analyzer results to determine framing.
    """

    def __init__(self):
        """Initialize the shot type detector."""
        self.loader = get_loader()

    def detect(
        self,
        tagger_results: TaggerResults,
        threshold: float = 0.4,
    ) -> ShotType:
        """
        Detect shot type from tagger results.

        Args:
            tagger_results: Combined tagger outputs
            threshold: Minimum confidence for detection

        Returns:
            Detected ShotType
        """
        shot_type_data = tagger_results.shot_type

        if not shot_type_data:
            return ShotType.UNKNOWN

        # Find highest confidence shot type
        best_shot = None
        best_conf = 0.0

        mapping = self.loader.get_shot_type_mapping()

        for shot_name, conf in shot_type_data.items():
            if conf < threshold:
                continue

            # Try to match to enum
            shot_type = self._match_shot_type(shot_name, mapping)
            if shot_type and conf > best_conf:
                best_shot = shot_type
                best_conf = conf

        return best_shot or ShotType.UNKNOWN

    def _match_shot_type(
        self,
        shot_name: str,
        mapping: Dict[str, Any],
    ) -> Optional[ShotType]:
        """Match shot name to ShotType enum using mapping."""
        shot_lower = shot_name.lower().replace(" ", "_")

        # Try direct enum match
        try:
            return ShotType[shot_lower.upper()]
        except KeyError:
            pass

        # Try alias matching
        for shot_key, config in mapping.items():
            aliases = config.get("aliases", [])
            if shot_lower in [a.lower() for a in aliases]:
                try:
                    return ShotType[shot_key]
                except KeyError:
                    pass

        # Try from_string method
        return ShotType.from_string(shot_name)

    def get_confidence(
        self,
        tagger_results: TaggerResults,
        shot_type: ShotType,
    ) -> float:
        """
        Get confidence for a specific shot type.

        Args:
            tagger_results: Combined tagger outputs
            shot_type: Shot type to get confidence for

        Returns:
            Confidence score (0-1)
        """
        shot_type_data = tagger_results.shot_type

        if not shot_type_data:
            return 0.0

        mapping = self.loader.get_shot_type_mapping()
        shot_config = mapping.get(shot_type.name, {})
        aliases = shot_config.get("aliases", [shot_type.name.lower()])

        for alias in aliases:
            alias_lower = alias.lower()
            for key, conf in shot_type_data.items():
                if key.lower() == alias_lower:
                    return conf

        return 0.0

    def is_body_visible(self, shot_type: ShotType) -> bool:
        """
        Check if body is visible for a shot type.

        Args:
            shot_type: Shot type to check

        Returns:
            True if body is visible
        """
        mapping = self.loader.get_shot_type_mapping()
        config = mapping.get(shot_type.name, {})
        body_visible = config.get("body_visible", True)

        if isinstance(body_visible, bool):
            return body_visible
        elif body_visible == "partial":
            return True  # Partial counts as visible

        return True
