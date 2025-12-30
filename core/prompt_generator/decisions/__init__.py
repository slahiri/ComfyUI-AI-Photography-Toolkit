"""
Decision engine for prompt generation.

Analyzes tagger outputs to make decisions about:
- Subject type (woman, man, landscape, etc.)
- Gender detection
- NSFW level
- Shot type
- Which template sections to activate
"""

from .engine import DecisionEngine, get_engine
from .subject import SubjectDetector
from .gender import GenderDetector
from .nsfw import NSFWDetector
from .shot_type import ShotTypeDetector
from .sections import SectionTriggerEngine

__all__ = [
    "DecisionEngine",
    "get_engine",
    "SubjectDetector",
    "GenderDetector",
    "NSFWDetector",
    "ShotTypeDetector",
    "SectionTriggerEngine",
]
