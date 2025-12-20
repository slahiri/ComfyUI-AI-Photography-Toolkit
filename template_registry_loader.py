# -*- coding: utf-8 -*-
"""
Template Registry Loader

Manages prompt templates from the template_registry folder.
Supports core (read-only), community (synced from GitHub), and custom (user-created) templates.
Templates support three model size variants: small (2B-6B), medium (7B-13B), large (13B+).

Author: Siddhartha Lahiri
License: MIT
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Literal
from datetime import datetime

# Valid model size options
MODEL_SIZES = ["small", "medium", "large"]
ModelSize = Literal["small", "medium", "large"]

# Try to import tomllib (Python 3.11+) or fall back to tomli
try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None

# For writing TOML files
try:
    import tomlkit
except ImportError:
    tomlkit = None


class TemplateRegistry:
    """
    Manages prompt templates from the template_registry folder.

    Templates are organized into:
    - core/: Built-in templates (read-only)
    - community/: Templates synced from GitHub
    - custom/: User-created templates
    """

    _instance = None
    _registry_path = None
    _registry_data = None
    _templates_cache = None

    def __new__(cls):
        """Singleton pattern to ensure single instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the registry."""
        if self._initialized:
            return

        self._registry_path = Path(__file__).parent / "template_registry"
        self._registry_data = None
        self._templates_cache = None
        self._initialized = True

    def _load_registry(self) -> Dict[str, Any]:
        """Load the registry.toml file."""
        if self._registry_data is not None:
            return self._registry_data

        registry_file = self._registry_path / "registry.toml"

        if not registry_file.exists():
            print(f"[TemplateRegistry] Registry file not found: {registry_file}")
            return {}

        if tomllib is None:
            print("[TemplateRegistry] TOML parser not available")
            return {}

        try:
            with open(registry_file, "rb") as f:
                self._registry_data = tomllib.load(f)
            return self._registry_data
        except Exception as e:
            print(f"[TemplateRegistry] Error loading registry: {e}")
            return {}

    def _load_template_file(self, relative_path: str) -> Optional[Dict[str, Any]]:
        """Load a single template TOML file."""
        template_file = self._registry_path / relative_path

        if not template_file.exists():
            print(f"[TemplateRegistry] Template file not found: {template_file}")
            return None

        if tomllib is None:
            return None

        try:
            with open(template_file, "rb") as f:
                return tomllib.load(f)
        except Exception as e:
            print(f"[TemplateRegistry] Error loading template {relative_path}: {e}")
            return None

    def reload(self) -> None:
        """Force reload of registry and templates."""
        self._registry_data = None
        self._templates_cache = None

    def get_all_templates(self) -> List[Dict[str, Any]]:
        """
        Get all available templates from all categories.
        Returns list of template metadata for dropdown population.
        """
        if self._templates_cache is not None:
            return self._templates_cache

        templates = []
        registry = self._load_registry()

        # Load core templates
        core_section = registry.get("core", {})
        for key, config in core_section.items():
            if isinstance(config, dict) and "path" in config:
                template_data = self._load_template_file(config["path"])
                if template_data:
                    metadata = template_data.get("metadata", {})
                    templates.append({
                        "key": f"core/{key}",
                        "name": metadata.get("name", key.title()),
                        "description": metadata.get("description", ""),
                        "category": "core",
                        "readonly": config.get("readonly", True),
                        "path": config["path"],
                    })

        # Load community templates
        community_section = registry.get("community", {})
        for key, config in community_section.items():
            if isinstance(config, dict) and "path" in config:
                template_data = self._load_template_file(config["path"])
                if template_data:
                    metadata = template_data.get("metadata", {})
                    templates.append({
                        "key": f"community/{key}",
                        "name": metadata.get("name", key.title()),
                        "description": metadata.get("description", ""),
                        "category": "community",
                        "readonly": config.get("readonly", False),
                        "path": config["path"],
                    })
            elif isinstance(config, str):
                # Simple path format
                template_data = self._load_template_file(config)
                if template_data:
                    metadata = template_data.get("metadata", {})
                    templates.append({
                        "key": f"community/{key}",
                        "name": metadata.get("name", key.title()),
                        "description": metadata.get("description", ""),
                        "category": "community",
                        "readonly": False,
                        "path": config,
                    })

        # Load custom templates
        custom_section = registry.get("custom", {})
        for key, config in custom_section.items():
            if isinstance(config, dict) and "path" in config:
                template_data = self._load_template_file(config["path"])
                if template_data:
                    metadata = template_data.get("metadata", {})
                    templates.append({
                        "key": f"custom/{key}",
                        "name": metadata.get("name", key.title()),
                        "description": metadata.get("description", ""),
                        "category": "custom",
                        "readonly": False,
                        "path": config["path"],
                    })
            elif isinstance(config, str):
                template_data = self._load_template_file(config)
                if template_data:
                    metadata = template_data.get("metadata", {})
                    templates.append({
                        "key": f"custom/{key}",
                        "name": metadata.get("name", key.title()),
                        "description": metadata.get("description", ""),
                        "category": "custom",
                        "readonly": False,
                        "path": config,
                    })

        # Also scan custom folder for any templates not in registry
        custom_folder = self._registry_path / "custom"
        if custom_folder.exists():
            for toml_file in custom_folder.glob("*.toml"):
                key = toml_file.stem
                full_key = f"custom/{key}"
                # Skip if already in list
                if any(t["key"] == full_key for t in templates):
                    continue

                template_data = self._load_template_file(f"custom/{toml_file.name}")
                if template_data:
                    metadata = template_data.get("metadata", {})
                    templates.append({
                        "key": full_key,
                        "name": metadata.get("name", key.title()),
                        "description": metadata.get("description", ""),
                        "category": "custom",
                        "readonly": False,
                        "path": f"custom/{toml_file.name}",
                    })

        self._templates_cache = templates
        return templates

    def get_template_names(self) -> List[str]:
        """Get list of template display names for dropdown."""
        templates = self.get_all_templates()
        return [t["name"] for t in templates]

    def _get_prompt_for_size(self, template_data: Dict[str, Any], model_size: str = "large") -> tuple:
        """
        Extract system and user prompts for the specified model size.

        Templates can have either:
        - Size-specific prompts: prompt.small, prompt.medium, prompt.large
        - Legacy single prompt: prompt.system, prompt.user

        Args:
            template_data: The loaded template TOML data
            model_size: One of "small", "medium", "large"

        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        prompt_data = template_data.get("prompt", {})

        # Check for size-specific prompts (new format)
        if model_size in prompt_data:
            size_data = prompt_data[model_size]
            return (
                size_data.get("system", ""),
                size_data.get("user", "Describe this image.")
            )

        # Fallback to legacy format (prompt.system, prompt.user)
        return (
            prompt_data.get("system", ""),
            prompt_data.get("user", "Describe this image.")
        )

    def get_template(self, name: str, model_size: str = "large") -> Optional[Dict[str, Any]]:
        """
        Get a specific template by name or key.
        Returns the full template data including system prompt for the specified model size.

        Args:
            name: Template name or key
            model_size: One of "small" (2B-6B), "medium" (7B-13B), "large" (13B+)
        """
        templates = self.get_all_templates()

        # Validate model_size
        if model_size not in MODEL_SIZES:
            model_size = "large"

        # Try to find by name first
        for t in templates:
            if t["name"] == name or t["key"] == name:
                template_data = self._load_template_file(t["path"])
                if template_data:
                    system_prompt, user_prompt = self._get_prompt_for_size(template_data, model_size)
                    # Merge metadata with prompt data
                    return {
                        **t,
                        "system": system_prompt,
                        "user": user_prompt,
                        "model_size": model_size,
                        "metadata": template_data.get("metadata", {}),
                    }

        return None

    def get_auto_template(self, detection: Dict[str, Any], model_size: str = "large") -> Optional[Dict[str, Any]]:
        """
        Get the appropriate template for Simple mode based on CV detection.

        Args:
            detection: CV analysis result with 'has_human', etc.
            model_size: One of "small" (2B-6B), "medium" (7B-13B), "large" (13B+)

        Returns:
            Template data for the detected category with appropriate model size prompt.
        """
        registry = self._load_registry()
        auto_detect = registry.get("auto_detect", {})

        # Validate model_size
        if model_size not in MODEL_SIZES:
            model_size = "large"

        # Check for human detection
        if detection.get("has_human", False):
            template_path = auto_detect.get("human", "core/portrait.toml")
        else:
            # Future: could add more detection types here
            # if detection.get("has_animal"): template_path = auto_detect.get("animal")
            # if detection.get("has_product"): template_path = auto_detect.get("product")
            template_path = auto_detect.get("default", "core/general.toml")

        # Load the template
        template_data = self._load_template_file(template_path)
        if template_data:
            metadata = template_data.get("metadata", {})
            system_prompt, user_prompt = self._get_prompt_for_size(template_data, model_size)
            return {
                "key": template_path.replace(".toml", "").replace("/", "_"),
                "name": metadata.get("name", "Auto"),
                "description": metadata.get("description", ""),
                "category": "core",
                "readonly": True,
                "path": template_path,
                "system": system_prompt,
                "user": user_prompt,
                "model_size": model_size,
                "metadata": metadata,
            }

        return None

    def is_readonly(self, name: str) -> bool:
        """Check if a template is read-only."""
        template = self.get_template(name)
        if template:
            return template.get("readonly", False)
        return True

    def save_custom_template(self, name: str, system_prompt: str = None,
                            description: str = "", tags: List[str] = None,
                            prompts: Dict[str, Dict[str, str]] = None) -> bool:
        """
        Save a new custom template.

        Args:
            name: Template display name
            system_prompt: Legacy single system prompt (used if prompts not provided)
            description: Optional description
            tags: Optional list of tags
            prompts: Dict with size variants {"small": {"system": "...", "user": "..."}, ...}

        Returns:
            True if saved successfully
        """
        if tomlkit is None:
            print("[TemplateRegistry] tomlkit not available for saving")
            return False

        # Generate safe filename from name
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name.lower())
        filename = f"{safe_name}.toml"
        filepath = self._registry_path / "custom" / filename

        # Create template content
        doc = tomlkit.document()

        # Metadata section
        metadata = tomlkit.table()
        metadata.add("name", name)
        metadata.add("description", description or f"Custom template: {name}")
        metadata.add("author", "User")
        metadata.add("version", "2.0")
        metadata.add("category", "custom")
        metadata.add("tags", tags or [])
        doc.add("metadata", metadata)

        # Prompt section - support both legacy and size-variant formats
        prompt = tomlkit.table()

        if prompts:
            # New format with size variants
            for size in MODEL_SIZES:
                size_data = prompts.get(size, {})
                size_table = tomlkit.table()
                size_table.add("system", size_data.get("system", ""))
                size_table.add("user", size_data.get("user", "Describe this image."))
                prompt.add(size, size_table)
        else:
            # Legacy format - single prompt for all sizes
            prompt.add("system", system_prompt or "")
            prompt.add("user", "Describe this image following the system instructions.")

        doc.add("prompt", prompt)

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(tomlkit.dumps(doc))

            # Clear cache to pick up new template
            self._templates_cache = None

            print(f"[TemplateRegistry] Saved custom template: {filepath}")
            return True
        except Exception as e:
            print(f"[TemplateRegistry] Error saving template: {e}")
            return False

    def delete_custom_template(self, name: str) -> bool:
        """
        Delete a custom template.

        Args:
            name: Template name or key

        Returns:
            True if deleted successfully
        """
        template = self.get_template(name)
        if not template:
            print(f"[TemplateRegistry] Template not found: {name}")
            return False

        if template.get("category") != "custom":
            print(f"[TemplateRegistry] Cannot delete non-custom template: {name}")
            return False

        filepath = self._registry_path / template["path"]
        try:
            if filepath.exists():
                filepath.unlink()
                self._templates_cache = None
                print(f"[TemplateRegistry] Deleted template: {filepath}")
                return True
        except Exception as e:
            print(f"[TemplateRegistry] Error deleting template: {e}")

        return False

    def sync_from_github(self) -> bool:
        """
        Sync community templates from GitHub repository.

        Returns:
            True if sync was successful
        """
        import requests

        registry = self._load_registry()
        reg_info = registry.get("registry", {})

        repo = reg_info.get("github_repo", "")
        branch = reg_info.get("github_branch", "main")
        path = reg_info.get("github_path", "template_registry/community")

        if not repo:
            print("[TemplateRegistry] No GitHub repo configured")
            return False

        # GitHub API URL to list directory contents
        api_url = f"https://api.github.com/repos/{repo}/contents/{path}?ref={branch}"

        try:
            response = requests.get(api_url, timeout=10)
            if response.status_code != 200:
                print(f"[TemplateRegistry] GitHub API error: {response.status_code}")
                return False

            files = response.json()
            synced_count = 0

            for file_info in files:
                if file_info.get("name", "").endswith(".toml"):
                    # Download the file
                    download_url = file_info.get("download_url")
                    if download_url:
                        file_response = requests.get(download_url, timeout=10)
                        if file_response.status_code == 200:
                            # Save to community folder
                            local_path = self._registry_path / "community" / file_info["name"]
                            with open(local_path, "w", encoding="utf-8") as f:
                                f.write(file_response.text)
                            synced_count += 1

            # Update last_sync timestamp
            # Note: This would require updating registry.toml

            self._templates_cache = None
            print(f"[TemplateRegistry] Synced {synced_count} templates from GitHub")
            return True

        except Exception as e:
            print(f"[TemplateRegistry] Sync error: {e}")
            return False


# Singleton instance
_registry = None

def get_registry() -> TemplateRegistry:
    """Get the singleton registry instance."""
    global _registry
    if _registry is None:
        _registry = TemplateRegistry()
    return _registry


# Convenience functions
def get_all_templates() -> List[Dict[str, Any]]:
    """Get all available templates."""
    return get_registry().get_all_templates()

def get_template_names() -> List[str]:
    """Get list of template display names."""
    return get_registry().get_template_names()

def get_template(name: str, model_size: str = "large") -> Optional[Dict[str, Any]]:
    """
    Get a specific template by name.

    Args:
        name: Template name or key
        model_size: One of "small" (2B-6B), "medium" (7B-13B), "large" (13B+)
    """
    return get_registry().get_template(name, model_size)

def get_auto_template(detection: Dict[str, Any], model_size: str = "large") -> Optional[Dict[str, Any]]:
    """
    Get auto-detected template for Simple mode.

    Args:
        detection: CV analysis result with 'has_human', etc.
        model_size: One of "small" (2B-6B), "medium" (7B-13B), "large" (13B+)
    """
    return get_registry().get_auto_template(detection, model_size)

def save_custom_template(name: str, system_prompt: str = None, **kwargs) -> bool:
    """Save a new custom template."""
    return get_registry().save_custom_template(name, system_prompt, **kwargs)

def delete_custom_template(name: str) -> bool:
    """Delete a custom template."""
    return get_registry().delete_custom_template(name)

def sync_from_github() -> bool:
    """Sync community templates from GitHub."""
    return get_registry().sync_from_github()

def reload_registry() -> None:
    """Force reload of registry."""
    get_registry().reload()

def get_model_sizes() -> List[str]:
    """Get list of valid model sizes for dropdown."""
    return MODEL_SIZES.copy()

def get_template_raw(name: str) -> Optional[Dict[str, Any]]:
    """
    Get a template with all size variants (for template editor).
    Returns system/user prompts for all sizes: small, medium, large.
    """
    registry = get_registry()
    templates = registry.get_all_templates()

    for t in templates:
        if t["name"] == name or t["key"] == name:
            template_data = registry._load_template_file(t["path"])
            if template_data:
                prompt_data = template_data.get("prompt", {})

                # Check if template has size variants
                has_sizes = any(size in prompt_data for size in MODEL_SIZES)

                if has_sizes:
                    # New format with size variants
                    prompts = {}
                    for size in MODEL_SIZES:
                        size_data = prompt_data.get(size, {})
                        prompts[size] = {
                            "system": size_data.get("system", ""),
                            "user": size_data.get("user", "Describe this image.")
                        }
                else:
                    # Legacy format - same prompt for all sizes
                    legacy_prompt = {
                        "system": prompt_data.get("system", ""),
                        "user": prompt_data.get("user", "Describe this image.")
                    }
                    prompts = {size: legacy_prompt.copy() for size in MODEL_SIZES}

                return {
                    **t,
                    "prompts": prompts,
                    "has_size_variants": has_sizes,
                    "metadata": template_data.get("metadata", {}),
                }

    return None
