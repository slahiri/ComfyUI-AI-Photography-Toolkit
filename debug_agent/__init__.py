# -*- coding: utf-8 -*-
"""
SID Prompt Debug Agent

Agentic evaluation system for prompt quality assessment.
Uses Claude Opus 4.5 for evaluation and optionally Tavily for web search.

Author: Siddhartha Lahiri
License: MIT
"""

from .agent import PromptDebugAgent
from .knowledge import KnowledgeBase
from .system_info import get_system_info
from .storage import ResultsStorage

__all__ = [
    "PromptDebugAgent",
    "KnowledgeBase",
    "get_system_info",
    "ResultsStorage",
]
