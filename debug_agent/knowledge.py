# -*- coding: utf-8 -*-
"""
Knowledge Base Manager

Handles reading, writing, and updating the local knowledge base JSON files.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime


class KnowledgeBase:
    """
    Manages the local knowledge base for the debug agent.

    Knowledge files:
    - zimage_vocabulary.json: Z-Image specific terms and effectiveness scores
    - zimage_best_practices.json: Prompting rules and patterns
    - vision_model_prompting.json: General vision LLM prompting guidelines
    - model_specific_tips.json: Per-model recommendations
    - learned_insights.json: Accumulated insights from debug sessions
    """

    KNOWLEDGE_FILES = [
        "zimage_vocabulary",
        "zimage_best_practices",
        "vision_model_prompting",
        "model_specific_tips",
        "learned_insights",
    ]

    def __init__(self, base_path: Optional[Path] = None):
        """
        Initialize the knowledge base.

        Args:
            base_path: Path to knowledge_base directory.
                      Defaults to toolkit's knowledge_base folder.
        """
        if base_path is None:
            # Default to toolkit's knowledge_base folder
            self.base_path = Path(__file__).parent.parent / "knowledge_base"
        else:
            self.base_path = Path(base_path)

        # Ensure directory exists
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Cache for loaded knowledge
        self._cache: Dict[str, Dict] = {}

    def read(self, name: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Read a knowledge file.

        Args:
            name: Name of the knowledge file (without .json extension)
            use_cache: Whether to use cached data if available

        Returns:
            Dictionary with the knowledge data
        """
        if use_cache and name in self._cache:
            return self._cache[name]

        file_path = self.base_path / f"{name}.json"

        if not file_path.exists():
            return {}

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self._cache[name] = data
                return data
        except json.JSONDecodeError as e:
            print(f"[KnowledgeBase] Error reading {name}.json: {e}")
            return {}

    def write(self, name: str, data: Dict[str, Any]) -> bool:
        """
        Write data to a knowledge file.

        Args:
            name: Name of the knowledge file (without .json extension)
            data: Dictionary to write

        Returns:
            True if successful
        """
        file_path = self.base_path / f"{name}.json"

        try:
            # Update timestamp
            data["last_updated"] = datetime.utcnow().isoformat() + "Z"

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            # Update cache
            self._cache[name] = data
            return True
        except Exception as e:
            print(f"[KnowledgeBase] Error writing {name}.json: {e}")
            return False

    def add(self, name: str, data: Dict[str, Any], path: Optional[List[str]] = None) -> bool:
        """
        Add data to a knowledge file at a specific path.

        Args:
            name: Name of the knowledge file
            data: Data to add
            path: List of keys to navigate to (e.g., ["categories", "lighting", "terms"])

        Returns:
            True if successful
        """
        current = self.read(name, use_cache=False)

        if path:
            # Navigate to the target location
            target = current
            for key in path[:-1]:
                if key not in target:
                    target[key] = {}
                target = target[key]

            last_key = path[-1]
            if last_key not in target:
                target[last_key] = []

            # Add data
            if isinstance(target[last_key], list):
                if isinstance(data, list):
                    target[last_key].extend(data)
                else:
                    target[last_key].append(data)
            elif isinstance(target[last_key], dict):
                target[last_key].update(data)
        else:
            # Merge at root level
            self._deep_merge(current, data)

        return self.write(name, current)

    def update(self, name: str, data: Dict[str, Any]) -> bool:
        """
        Update existing data in a knowledge file.

        Args:
            name: Name of the knowledge file
            data: Data to merge/update

        Returns:
            True if successful
        """
        current = self.read(name, use_cache=False)
        self._deep_merge(current, data)
        return self.write(name, current)

    def increment_score(self, name: str, term: str, delta: float,
                       category: Optional[str] = None) -> bool:
        """
        Increment effectiveness score for a vocabulary term.

        Args:
            name: Name of the knowledge file (usually "zimage_vocabulary")
            term: The term to update
            delta: Amount to add to the score
            category: Optional category to search in

        Returns:
            True if term found and updated
        """
        data = self.read(name, use_cache=False)

        if "categories" not in data:
            return False

        categories_to_search = [category] if category else data["categories"].keys()

        for cat in categories_to_search:
            if cat not in data["categories"]:
                continue

            cat_data = data["categories"][cat]

            # Search in terms list
            if "terms" in cat_data:
                for item in cat_data["terms"]:
                    if isinstance(item, dict) and item.get("term") == term:
                        current_score = item.get("effectiveness", 5.0)
                        item["effectiveness"] = round(min(10.0, max(0.0, current_score + delta)), 2)
                        return self.write(name, data)

        return False

    def get_vocabulary_terms(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get vocabulary terms, optionally filtered by category.

        Args:
            category: Optional category to filter by

        Returns:
            List of term dictionaries
        """
        data = self.read("zimage_vocabulary")
        terms = []

        if "categories" not in data:
            return terms

        categories_to_search = [category] if category else data["categories"].keys()

        for cat in categories_to_search:
            if cat not in data["categories"]:
                continue

            cat_data = data["categories"][cat]
            if "terms" in cat_data:
                for item in cat_data["terms"]:
                    if isinstance(item, dict):
                        item_copy = item.copy()
                        item_copy["category"] = cat
                        terms.append(item_copy)

        return terms

    def get_best_practices(self) -> Dict[str, Any]:
        """
        Get Z-Image best practices.

        Returns:
            Best practices dictionary
        """
        return self.read("zimage_best_practices")

    def get_model_tips(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get tips for a specific model.

        Args:
            model_name: Name of the model

        Returns:
            Model tips dictionary or None if not found
        """
        data = self.read("model_specific_tips")

        if "vision_models" not in data:
            return None

        # Exact match
        if model_name in data["vision_models"]:
            return data["vision_models"][model_name]

        # Partial match
        model_name_lower = model_name.lower()
        for name, tips in data["vision_models"].items():
            if name.lower() in model_name_lower or model_name_lower in name.lower():
                return tips

        return None

    def add_learned_insight(self, insight: Dict[str, Any]) -> bool:
        """
        Add a learned insight from a debug session.

        Args:
            insight: Dictionary with insight data

        Returns:
            True if successful
        """
        data = self.read("learned_insights", use_cache=False)

        if "sessions" not in data:
            data["sessions"] = []

        # Add timestamp if not present
        if "timestamp" not in insight:
            insight["timestamp"] = datetime.utcnow().isoformat() + "Z"

        data["sessions"].append(insight)

        # Update statistics
        if "statistics" not in data:
            data["statistics"] = {"total_sessions": 0}
        data["statistics"]["total_sessions"] = len(data["sessions"])

        return self.write("learned_insights", data)

    def get_status(self) -> Dict[str, Any]:
        """
        Get status of all knowledge files.

        Returns:
            Status dictionary
        """
        status = {}

        for name in self.KNOWLEDGE_FILES:
            file_path = self.base_path / f"{name}.json"
            data = self.read(name)

            status[name] = {
                "exists": file_path.exists(),
                "version": data.get("version", "unknown"),
                "last_updated": data.get("last_updated", "unknown"),
            }

            # Add specific stats based on file type
            if name == "zimage_vocabulary" and "categories" in data:
                total_terms = sum(
                    len(cat.get("terms", []))
                    for cat in data["categories"].values()
                )
                status[name]["total_terms"] = total_terms

            elif name == "zimage_best_practices":
                if "must_include" in data:
                    status[name]["total_rules"] = len(data.get("must_include", {}).get("human_subjects", {}).get("elements", []))

            elif name == "model_specific_tips" and "vision_models" in data:
                status[name]["models_covered"] = len(data["vision_models"])

            elif name == "learned_insights" and "sessions" in data:
                status[name]["total_sessions"] = len(data["sessions"])

        return status

    def clear_cache(self):
        """Clear the in-memory cache."""
        self._cache.clear()

    def _deep_merge(self, base: Dict, updates: Dict) -> Dict:
        """
        Deep merge two dictionaries.

        Args:
            base: Base dictionary (modified in place)
            updates: Dictionary with updates

        Returns:
            Merged dictionary (same as base)
        """
        for key, value in updates.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
        return base
