# -*- coding: utf-8 -*-
"""
ComfyUI-AI-Photography-Toolkit

Re-architecture branch - Clean slate.

Author: Siddhartha Lahiri
License: MIT
Version: 5.0.0-dev
"""

__version__ = "5.0.0-dev"

from typing_extensions import override
from comfy_api.latest import ComfyExtension, io


class SIDPhotographyToolkitExtension(ComfyExtension):
    """
    Extension class for SID Photography Toolkit.
    """

    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        """Return list of all nodes in this extension."""
        nodes = []
        return nodes


async def comfy_entrypoint() -> SIDPhotographyToolkitExtension:
    """ComfyUI entry point."""
    print("\n[SID-Toolkit] v5.0.0-dev - Re-architecture branch")
    print("[SID-Toolkit] No nodes registered yet\n")
    return SIDPhotographyToolkitExtension()
