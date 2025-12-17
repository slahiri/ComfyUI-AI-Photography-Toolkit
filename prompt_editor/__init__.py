# -*- coding: utf-8 -*-
"""
Prompt Editor - Web UI for editing TOML prompt files.

Simple file browser + TOML editor interface.
Access at: http://localhost:8188/sid/prompt-editor
"""

from .routes import setup_routes

__all__ = ["setup_routes"]
