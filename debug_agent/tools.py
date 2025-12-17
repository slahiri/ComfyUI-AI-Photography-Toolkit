# -*- coding: utf-8 -*-
"""
Agent Tool Implementations

Tools that the debug agent can invoke during evaluation.
"""

import re
from typing import Dict, Any, List, Optional
from .knowledge import KnowledgeBase


# Tool definitions for Claude's tool use
AGENT_TOOLS = [
    {
        "name": "search_web",
        "description": "Search the web for latest prompting techniques and best practices using Tavily",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query for prompting techniques"
                },
                "category": {
                    "type": "string",
                    "enum": ["zimage", "vision_model", "model_specific", "general"],
                    "description": "Category to focus the search"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "read_knowledge",
        "description": "Read from local knowledge base (vocabulary, best practices, model tips)",
        "input_schema": {
            "type": "object",
            "properties": {
                "file": {
                    "type": "string",
                    "enum": [
                        "zimage_vocabulary",
                        "zimage_best_practices",
                        "vision_model_prompting",
                        "model_specific_tips",
                        "learned_insights"
                    ],
                    "description": "Which knowledge file to read"
                },
                "section": {
                    "type": "string",
                    "description": "Optional specific section to retrieve"
                }
            },
            "required": ["file"]
        }
    },
    {
        "name": "update_knowledge",
        "description": "Update local knowledge base with new insights",
        "input_schema": {
            "type": "object",
            "properties": {
                "file": {
                    "type": "string",
                    "description": "Knowledge file to update"
                },
                "operation": {
                    "type": "string",
                    "enum": ["add", "update", "increment_score"],
                    "description": "Type of update operation"
                },
                "data": {
                    "type": "object",
                    "description": "Data to add or update"
                }
            },
            "required": ["file", "operation", "data"]
        }
    },
    {
        "name": "analyze_image",
        "description": "Request detailed analysis of source or output image",
        "input_schema": {
            "type": "object",
            "properties": {
                "image_type": {
                    "type": "string",
                    "enum": ["source", "output"],
                    "description": "Which image to analyze"
                },
                "focus_areas": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific areas to focus on (e.g., 'clothing', 'accessories', 'lighting')"
                }
            },
            "required": ["image_type"]
        }
    },
    {
        "name": "compare_images",
        "description": "Compare source and output images to find preserved/lost elements",
        "input_schema": {
            "type": "object",
            "properties": {
                "aspects": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Aspects to compare (e.g., 'subject', 'clothing', 'pose', 'background')"
                }
            }
        }
    },
    {
        "name": "evaluate_prompt",
        "description": "Evaluate the prompt against Z-Image best practices",
        "input_schema": {
            "type": "object",
            "properties": {
                "check_vocabulary": {
                    "type": "boolean",
                    "description": "Check vocabulary usage against knowledge base"
                },
                "check_structure": {
                    "type": "boolean",
                    "description": "Check prompt structure against best practices"
                },
                "check_completeness": {
                    "type": "boolean",
                    "description": "Check for required elements"
                }
            }
        }
    },
    {
        "name": "get_model_tips",
        "description": "Get tips specific to the model that generated the prompt",
        "input_schema": {
            "type": "object",
            "properties": {
                "include_optimal_settings": {
                    "type": "boolean",
                    "description": "Include optimal parameter settings"
                }
            }
        }
    },
    {
        "name": "save_results",
        "description": "Save evaluation results to disk",
        "input_schema": {
            "type": "object",
            "properties": {
                "include_images": {
                    "type": "boolean",
                    "description": "Save source and output images"
                },
                "include_knowledge_updates": {
                    "type": "boolean",
                    "description": "Save updates to knowledge base"
                }
            }
        }
    }
]


class ToolExecutor:
    """
    Executes tools for the debug agent.
    """

    def __init__(
        self,
        knowledge: KnowledgeBase,
        context: Dict[str, Any],
        tavily_client: Optional[Any] = None
    ):
        """
        Initialize the tool executor.

        Args:
            knowledge: KnowledgeBase instance
            context: Shared context with images, prompt, model config
            tavily_client: Optional Tavily client for web search
        """
        self.knowledge = knowledge
        self.context = context
        self.tavily = tavily_client

    def execute(self, tool_name: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool by name.

        Args:
            tool_name: Name of the tool to execute
            tool_input: Input parameters for the tool

        Returns:
            Tool result dictionary
        """
        tool_methods = {
            "search_web": self._search_web,
            "read_knowledge": self._read_knowledge,
            "update_knowledge": self._update_knowledge,
            "analyze_image": self._analyze_image,
            "compare_images": self._compare_images,
            "evaluate_prompt": self._evaluate_prompt,
            "get_model_tips": self._get_model_tips,
            "save_results": self._save_results,
        }

        if tool_name not in tool_methods:
            return {"error": f"Unknown tool: {tool_name}"}

        try:
            return tool_methods[tool_name](tool_input)
        except Exception as e:
            return {"error": f"Tool execution failed: {str(e)}"}

    def _search_web(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Search web using Tavily."""
        if not self.tavily:
            return {
                "status": "skipped",
                "reason": "No Tavily API key provided"
            }

        query = input["query"]
        category = input.get("category", "general")

        # Enhance query based on category
        if category == "zimage":
            query = f"Z-Image prompting {query}"
        elif category == "vision_model":
            query = f"vision model prompt engineering {query}"
        elif category == "model_specific":
            model = self.context.get("model_config", {}).get("model", "")
            query = f"{model} {query}"

        try:
            results = self.tavily.search(
                query=query,
                search_depth="advanced",
                max_results=5
            )

            return {
                "status": "success",
                "query": query,
                "results": [
                    {
                        "title": r.get("title", ""),
                        "url": r.get("url", ""),
                        "content": r.get("content", "")[:500]
                    }
                    for r in results.get("results", [])
                ]
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    def _read_knowledge(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Read from local knowledge base."""
        file = input["file"]
        section = input.get("section")

        data = self.knowledge.read(file)

        if not data:
            return {
                "status": "not_found",
                "file": file
            }

        # Return specific section if requested
        if section:
            if section in data:
                return {
                    "status": "success",
                    "file": file,
                    "section": section,
                    "data": data[section]
                }
            else:
                return {
                    "status": "section_not_found",
                    "file": file,
                    "section": section,
                    "available_sections": list(data.keys())
                }

        return {
            "status": "success",
            "file": file,
            "data": data
        }

    def _update_knowledge(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Update local knowledge base."""
        if not self.context.get("update_knowledge", False):
            return {
                "status": "skipped",
                "reason": "update_knowledge is False"
            }

        file = input["file"]
        operation = input["operation"]
        data = input["data"]

        success = False
        if operation == "add":
            success = self.knowledge.add(file, data)
        elif operation == "update":
            success = self.knowledge.update(file, data)
        elif operation == "increment_score":
            term = data.get("term", "")
            delta = data.get("delta", 0.1)
            category = data.get("category")
            success = self.knowledge.increment_score(file, term, delta, category)

        return {
            "status": "success" if success else "failed",
            "operation": operation,
            "file": file
        }

    def _analyze_image(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Return instruction for Claude to analyze the image.
        The actual images are in the conversation context.
        """
        image_type = input["image_type"]
        focus_areas = input.get("focus_areas", [])

        default_focus = [
            "subject (age, gender, appearance)",
            "clothing (garments, colors, materials, patterns)",
            "accessories (jewelry, watches, bags)",
            "hair (color, style, length)",
            "pose and expression",
            "environment/background",
            "lighting (direction, quality, color temperature)",
            "composition and framing"
        ]

        return {
            "status": "ready",
            "instruction": f"Please analyze the {image_type} image in detail",
            "focus_areas": focus_areas if focus_areas else default_focus,
            "note": f"The {image_type} image is the {'first' if image_type == 'source' else 'second'} image in the conversation"
        }

    def _compare_images(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Return instruction for Claude to compare images."""
        aspects = input.get("aspects", [])

        default_aspects = [
            "subject identity and features",
            "clothing accuracy",
            "pose preservation",
            "background fidelity",
            "lighting consistency",
            "color accuracy",
            "detail preservation"
        ]

        return {
            "status": "ready",
            "instruction": "Please compare the source (first) and output (second) images",
            "aspects_to_compare": aspects if aspects else default_aspects,
            "report_format": {
                "preserved": "Elements that match between images",
                "lost": "Elements in source but not in output",
                "added": "Elements in output but not in source",
                "changed": "Elements that differ between images"
            }
        }

    def _evaluate_prompt(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate prompt against best practices."""
        prompt = self.context.get("prompt", "")

        result = {
            "prompt_text": prompt,
            "word_count": len(prompt.split()),
            "sentence_count": len([s for s in prompt.split('.') if s.strip()]),
            "character_count": len(prompt)
        }

        # Check vocabulary
        if input.get("check_vocabulary", False):
            vocab = self.knowledge.read("zimage_vocabulary")
            result["vocabulary_check"] = self._check_vocabulary(prompt, vocab)

        # Check structure
        if input.get("check_structure", False):
            practices = self.knowledge.read("zimage_best_practices")
            result["structure_check"] = self._check_structure(prompt, practices)

        # Check completeness
        if input.get("check_completeness", False):
            practices = self.knowledge.read("zimage_best_practices")
            result["completeness_check"] = self._check_completeness(prompt, practices)

        return result

    def _check_vocabulary(self, prompt: str, vocab: Dict) -> Dict[str, Any]:
        """Check vocabulary usage in prompt."""
        prompt_lower = prompt.lower()
        terms_found = []
        terms_missing_opportunity = []

        if "categories" not in vocab:
            return {"status": "no_vocabulary_data"}

        for category, cat_data in vocab["categories"].items():
            if "terms" not in cat_data:
                continue

            for item in cat_data["terms"]:
                if isinstance(item, dict):
                    term = item.get("term", "")
                    if term.lower() in prompt_lower:
                        terms_found.append({
                            "term": term,
                            "category": category,
                            "effectiveness": item.get("effectiveness", 0)
                        })

        return {
            "terms_found": terms_found,
            "count": len(terms_found),
            "average_effectiveness": sum(t["effectiveness"] for t in terms_found) / len(terms_found) if terms_found else 0
        }

    def _check_structure(self, prompt: str, practices: Dict) -> Dict[str, Any]:
        """Check prompt structure against best practices."""
        sentences = [s.strip() for s in prompt.split('.') if s.strip()]
        words = prompt.split()

        structure_rules = practices.get("structure_rules", {})
        sentence_rules = structure_rules.get("sentence_count", {})

        checks = {
            "sentence_count": len(sentences),
            "word_count": len(words),
            "sentence_count_compliant": sentence_rules.get("min", 3) <= len(sentences) <= sentence_rules.get("max", 6),
        }

        # Check for must_avoid patterns
        must_avoid = practices.get("must_avoid", {})
        violations = []

        # Check for vague descriptors
        vague_terms = ["beautiful", "nice", "good", "pretty", "amazing", "lovely", "stunning"]
        for term in vague_terms:
            if term.lower() in prompt.lower():
                violations.append(f"vague_descriptor: {term}")

        # Check for run-on sentence
        if len(sentences) == 1 and len(words) > 50:
            violations.append("single_long_sentence")

        # Check for keyword spam (many commas with single words)
        comma_parts = prompt.split(',')
        single_word_parts = sum(1 for p in comma_parts if len(p.split()) <= 2)
        if single_word_parts > 5:
            violations.append("keyword_list_style")

        checks["violations"] = violations
        checks["compliant"] = len(violations) == 0

        return checks

    def _check_completeness(self, prompt: str, practices: Dict) -> Dict[str, Any]:
        """Check if prompt includes required elements."""
        prompt_lower = prompt.lower()
        must_include = practices.get("must_include", {})

        # Check human subject requirements
        human_checks = {}
        if "human_subjects" in must_include:
            elements = must_include["human_subjects"].get("elements", [])
            for elem in elements:
                name = elem.get("name", "")
                # Simple keyword checks
                if name == "age_gender":
                    human_checks[name] = any(w in prompt_lower for w in ["woman", "man", "girl", "boy", "young", "old", "elderly", "twenties", "thirties"])
                elif name == "hair_description":
                    human_checks[name] = any(w in prompt_lower for w in ["hair", "blonde", "brunette", "auburn", "dark", "light"])
                elif name == "skin_texture":
                    human_checks[name] = any(w in prompt_lower for w in ["skin texture", "skin pores", "photorealistic skin", "realistic skin"])
                elif name == "clothing_details":
                    human_checks[name] = any(w in prompt_lower for w in ["wearing", "dress", "shirt", "pants", "jacket", "blouse", "trousers"])
                elif name == "pose_action":
                    human_checks[name] = any(w in prompt_lower for w in ["standing", "sitting", "walking", "pose", "facing", "looking"])

        # Check all-image requirements
        all_checks = {}
        if "all_images" in must_include:
            elements = must_include["all_images"].get("elements", [])
            for elem in elements:
                name = elem.get("name", "")
                if name == "lighting":
                    all_checks[name] = any(w in prompt_lower for w in ["lighting", "light", "sunlight", "daylight", "shadows", "illuminated"])
                elif name == "background":
                    all_checks[name] = any(w in prompt_lower for w in ["background", "setting", "environment", "backdrop", "outdoor", "indoor"])
                elif name == "technical_quality":
                    all_checks[name] = any(w in prompt_lower for w in ["detailed", "photorealistic", "quality", "resolution", "sharp"])

        return {
            "human_subject_checks": human_checks,
            "all_image_checks": all_checks,
            "human_compliance": sum(human_checks.values()) / len(human_checks) if human_checks else 1.0,
            "all_compliance": sum(all_checks.values()) / len(all_checks) if all_checks else 1.0
        }

    def _get_model_tips(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Get tips for the specific model used."""
        model_config = self.context.get("model_config", {})
        model_name = model_config.get("model", "")

        tips = self.knowledge.get_model_tips(model_name)

        result = {
            "model_name": model_name,
            "provider": model_config.get("provider", "unknown"),
            "found_in_knowledge_base": tips is not None,
            "tips": tips or {}
        }

        if input.get("include_optimal_settings", False) and tips:
            optimal = tips.get("optimal_settings", {})
            current = {
                "temperature": model_config.get("temperature"),
                "quantization": model_config.get("extra_params", {}).get("quantization"),
                "repetition_penalty": model_config.get("extra_params", {}).get("repetition_penalty"),
            }

            result["optimal_settings"] = optimal
            result["current_settings"] = current

            # Compare settings
            comparisons = {}
            for key in optimal:
                if key in current and current[key] is not None:
                    comparisons[key] = {
                        "current": current[key],
                        "optimal": optimal[key],
                        "match": current[key] == optimal[key]
                    }
            result["settings_comparison"] = comparisons

        return result

    def _save_results(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Placeholder for save results - actual saving done by agent."""
        if not self.context.get("save_results", False):
            return {
                "status": "skipped",
                "reason": "save_results is False"
            }

        return {
            "status": "ready",
            "include_images": input.get("include_images", True),
            "include_knowledge_updates": input.get("include_knowledge_updates", False),
            "note": "Results will be saved after evaluation completes"
        }
