# -*- coding: utf-8 -*-
"""
SID Photography Toolkit - Web UI Routes

API endpoints and HTML pages for:
- Landing page (/sid)
- Template Editor (/sid/template-editor)
- Generation Results Viewer (/sid/generation-results)
- Debug Results Viewer (/sid/debug-results)
"""

import json
from pathlib import Path
from datetime import datetime
from aiohttp import web

# Get the toolkit directories
TOOLKIT_DIR = Path(__file__).parent.parent
DEBUG_RESULTS_DIR = TOOLKIT_DIR / "debug_results"
GENERATION_RESULTS_DIR = TOOLKIT_DIR / "generation_results"

def setup_routes(routes):
    """Register all routes for SID Photography Toolkit web UI."""

    # =========================================================================
    # SID Landing Page
    # =========================================================================

    @routes.get("/sid")
    async def serve_landing(request):
        """Serve the SID Toolkit landing page."""
        return web.Response(text=get_landing_html(), content_type="text/html")

    # =========================================================================
    # Config Reload API
    # =========================================================================

    @routes.post("/sid/api/reload-configs")
    async def reload_configs(request):
        """
        Reload all TOML configuration files from disk.
        This clears the config cache so changes take effect without restart.

        Usage: POST /sid/api/reload-configs
        Response: {"success": true, "message": "..."}
        """
        try:
            from ..config_loader import clear_config_cache, reload_all_configs
            reload_all_configs()
            return web.json_response({
                "success": True,
                "message": "All TOML configs reloaded from disk. Changes will take effect on next prompt generation."
            })
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    # =========================================================================
    # Debug Results Viewer Routes
    # =========================================================================

    @routes.get("/sid/debug-results")
    async def serve_debug_viewer(request):
        """Serve the debug results viewer HTML page."""
        return web.Response(text=get_debug_viewer_html(), content_type="text/html")

    def _parse_timestamp_from_folder(folder_name: str) -> str:
        """Parse timestamp from folder name like 'session_20251217_162517' or '2025-12-17_12-21-22_xxx'."""
        import re
        # Try newer format: session_YYYYMMDD_HHMMSS
        match = re.match(r'session_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})', folder_name)
        if match:
            y, mo, d, h, mi, s = match.groups()
            return f"{y}-{mo}-{d}T{h}:{mi}:{s}"
        # Try older format: YYYY-MM-DD_HH-MM-SS_xxx
        match = re.match(r'(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})-(\d{2})_', folder_name)
        if match:
            y, mo, d, h, mi, s = match.groups()
            return f"{y}-{mo}-{d}T{h}:{mi}:{s}"
        return ""

    @routes.get("/sid/debug-results/api/sessions")
    async def list_debug_sessions(request):
        """List all debug result sessions."""
        sessions = []
        if DEBUG_RESULTS_DIR.exists():
            for session_dir in sorted(DEBUG_RESULTS_DIR.iterdir(), reverse=True):
                if session_dir.is_dir() and session_dir.name != "archived":
                    eval_file = session_dir / "evaluation.json"
                    if eval_file.exists():
                        try:
                            with open(eval_file, 'r', encoding='utf-8') as f:
                                data = json.load(f)

                            # Extract summary info
                            scores = data.get("scores", {})
                            overall = scores.get("overall", {}).get("score", 0)

                            # Try different sources for GENERATION model info
                            # (the model that generated the prompt, not the evaluation model)
                            provider = "unknown"
                            model = "unknown"
                            timestamp = ""

                            # 1. Check generation_model.json (preferred - generation model)
                            gen_model_file = session_dir / "generation_model.json"
                            if gen_model_file.exists():
                                with open(gen_model_file, 'r', encoding='utf-8') as f:
                                    mc = json.load(f)
                                    provider = mc.get("provider", provider)
                                    model = mc.get("model", model)

                            # 2. Check source_metadata.json -> model_config (fallback)
                            if provider == "unknown":
                                source_meta_file = session_dir / "source_metadata.json"
                                if source_meta_file.exists():
                                    with open(source_meta_file, 'r', encoding='utf-8') as f:
                                        meta = json.load(f)
                                        mc = meta.get("model_config", {})
                                        provider = mc.get("provider", provider)
                                        model = mc.get("model", model)
                                        timestamp = meta.get("timestamp", "")

                            # 3. Check metadata.json (older format fallback)
                            if provider == "unknown":
                                metadata_file = session_dir / "metadata.json"
                                if metadata_file.exists():
                                    with open(metadata_file, 'r', encoding='utf-8') as f:
                                        meta = json.load(f)
                                        mc = meta.get("model_config", {})
                                        provider = mc.get("provider", provider)
                                        model = mc.get("model", model)
                                        timestamp = meta.get("timestamp", "")

                            # Parse timestamp from folder name if not found
                            if not timestamp:
                                timestamp = _parse_timestamp_from_folder(session_dir.name)

                            # Check for images (both .jpg and .png)
                            has_source = (session_dir / "source.jpg").exists() or (session_dir / "source.png").exists()
                            has_output = (session_dir / "output.jpg").exists() or (session_dir / "output.png").exists()

                            # Check if excluded from learning
                            excluded = False
                            excluded_file = session_dir / "excluded.json"
                            if excluded_file.exists():
                                try:
                                    with open(excluded_file, 'r', encoding='utf-8') as f:
                                        excluded_data = json.load(f)
                                        excluded = excluded_data.get("excluded", False)
                                except Exception:
                                    pass

                            sessions.append({
                                "id": session_dir.name,
                                "timestamp": timestamp,
                                "overall_score": overall,
                                "provider": provider,
                                "model": model,
                                "has_source": has_source,
                                "has_output": has_output,
                                "excluded": excluded,
                            })
                        except Exception:
                            pass
        return web.json_response({"sessions": sessions})

    @routes.post("/sid/debug-results/api/session/{session_id}/exclude")
    async def toggle_debug_excluded(request):
        """Toggle the excluded status of a debug session."""
        session_id = request.match_info["session_id"]

        # Security: validate session_id format
        if not session_id or ".." in session_id or "/" in session_id or "\\" in session_id:
            return web.json_response({"error": "Invalid session ID"}, status=400)

        session_dir = DEBUG_RESULTS_DIR / session_id
        if not session_dir.exists():
            return web.json_response({"error": "Session not found"}, status=404)

        excluded_file = session_dir / "excluded.json"

        # Read current state
        excluded = False
        if excluded_file.exists():
            try:
                with open(excluded_file, 'r', encoding='utf-8') as f:
                    excluded_data = json.load(f)
                    excluded = excluded_data.get("excluded", False)
            except Exception:
                pass

        # Toggle
        new_excluded = not excluded

        # Save new state
        with open(excluded_file, 'w', encoding='utf-8') as f:
            json.dump({"excluded": new_excluded}, f)

        return web.json_response({"success": True, "excluded": new_excluded})

    @routes.get("/sid/debug-results/api/session/{session_id}")
    async def get_debug_session(request):
        """Get details of a specific debug session."""
        session_id = request.match_info["session_id"]

        # Security: validate session_id format
        if not session_id or ".." in session_id or "/" in session_id or "\\" in session_id:
            return web.json_response({"error": "Invalid session ID"}, status=400)

        session_dir = DEBUG_RESULTS_DIR / session_id
        if not session_dir.exists():
            return web.json_response({"error": "Session not found"}, status=404)

        eval_file = session_dir / "evaluation.json"
        if not eval_file.exists():
            return web.json_response({"error": "Evaluation file not found"}, status=404)

        try:
            with open(eval_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Add image URLs (check both .jpg and .png)
            source_img = None
            output_img = None
            for ext in [".jpg", ".png"]:
                if (session_dir / f"source{ext}").exists():
                    source_img = f"/sid/debug-results/api/image/{session_id}/source{ext}"
                if (session_dir / f"output{ext}").exists():
                    output_img = f"/sid/debug-results/api/image/{session_id}/output{ext}"
            data["images"] = {"source": source_img, "output": output_img}

            # Read prompt from prompt.txt if exists
            prompt_file = session_dir / "prompt.txt"
            if prompt_file.exists():
                data["original_prompt"] = prompt_file.read_text(encoding="utf-8")

            # Read GENERATION model config (the model that generated the prompt)
            gen_model_file = session_dir / "generation_model.json"
            source_meta_file = session_dir / "source_metadata.json"
            metadata_file = session_dir / "metadata.json"

            if gen_model_file.exists():
                with open(gen_model_file, 'r', encoding='utf-8') as f:
                    data["generation_model"] = json.load(f)

            if source_meta_file.exists():
                with open(source_meta_file, 'r', encoding='utf-8') as f:
                    data["source_metadata"] = json.load(f)
                    # Also set model_config for backwards compatibility
                    if "generation_model" not in data:
                        data["generation_model"] = data["source_metadata"].get("model_config", {})

            # Load template info if available
            template_info_file = session_dir / "template_info.json"
            if template_info_file.exists():
                with open(template_info_file, 'r', encoding='utf-8') as f:
                    data["template_info"] = json.load(f)

            # Fallback to old metadata.json format
            if "generation_model" not in data and metadata_file.exists():
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                    data["generation_model"] = meta.get("model_config", {})
                    data["metadata"] = meta

            # Add parsed timestamp
            data["parsed_timestamp"] = _parse_timestamp_from_folder(session_id)

            return web.json_response(data)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    @routes.get("/sid/debug-results/api/image/{session_id}/{filename}")
    async def get_debug_image(request):
        """Serve an image from a debug session."""
        session_id = request.match_info["session_id"]
        filename = request.match_info["filename"]

        # Security: validate inputs
        if not session_id or ".." in session_id or "/" in session_id or "\\" in session_id:
            return web.json_response({"error": "Invalid session ID"}, status=400)
        allowed_files = ["source.jpg", "output.jpg", "source.png", "output.png"]
        if filename not in allowed_files:
            return web.json_response({"error": "Invalid filename"}, status=400)

        image_path = DEBUG_RESULTS_DIR / session_id / filename
        if not image_path.exists():
            return web.json_response({"error": "Image not found"}, status=404)

        return web.FileResponse(image_path)

    # =========================================================================
    # Generation Results Viewer Routes
    # =========================================================================

    @routes.get("/sid/generation-results")
    async def serve_generation_viewer(request):
        """Serve the generation results viewer HTML page."""
        return web.Response(text=get_generation_viewer_html(), content_type="text/html")

    @routes.get("/sid/generation-results/api/sessions")
    async def list_generation_sessions(request):
        """List all generation result sessions."""
        sessions = []
        if GENERATION_RESULTS_DIR.exists():
            for session_dir in sorted(GENERATION_RESULTS_DIR.iterdir(), reverse=True):
                if session_dir.is_dir() and session_dir.name.startswith("gen_"):
                    metadata_file = session_dir / "metadata.json"
                    prompt_file = session_dir / "prompt.txt"
                    excluded_file = session_dir / "excluded.json"
                    if metadata_file.exists():
                        try:
                            with open(metadata_file, 'r', encoding='utf-8') as f:
                                meta = json.load(f)

                            # Read prompt preview
                            prompt_preview = ""
                            if prompt_file.exists():
                                prompt_text = prompt_file.read_text(encoding='utf-8')
                                prompt_preview = prompt_text[:100] + "..." if len(prompt_text) > 100 else prompt_text

                            # Check if excluded from learning
                            excluded = False
                            if excluded_file.exists():
                                try:
                                    with open(excluded_file, 'r', encoding='utf-8') as f:
                                        excluded_data = json.load(f)
                                        excluded = excluded_data.get("excluded", False)
                                except Exception:
                                    pass

                            model_config = meta.get("model_config", {})
                            sessions.append({
                                "id": session_dir.name,
                                "timestamp": meta.get("timestamp", ""),
                                "prompt_style": meta.get("prompt_style", ""),
                                "template": meta.get("template", ""),
                                "provider": model_config.get("provider", "unknown"),
                                "model": model_config.get("model", "unknown"),
                                "analysis_mode": model_config.get("analysis_mode", ""),
                                "prompt_preview": prompt_preview,
                                "has_image": (session_dir / "source.jpg").exists(),
                                "timing": meta.get("timing", {}).get("total_seconds", 0),
                                "excluded": excluded,
                            })
                        except Exception:
                            pass
        return web.json_response({"sessions": sessions})

    @routes.post("/sid/generation-results/api/session/{session_id}/exclude")
    async def toggle_generation_excluded(request):
        """Toggle the excluded status of a generation session."""
        session_id = request.match_info["session_id"]

        # Security: validate session_id format
        if not session_id or ".." in session_id or "/" in session_id or "\\" in session_id:
            return web.json_response({"error": "Invalid session ID"}, status=400)

        session_dir = GENERATION_RESULTS_DIR / session_id
        if not session_dir.exists():
            return web.json_response({"error": "Session not found"}, status=404)

        excluded_file = session_dir / "excluded.json"

        # Read current state
        excluded = False
        if excluded_file.exists():
            try:
                with open(excluded_file, 'r', encoding='utf-8') as f:
                    excluded_data = json.load(f)
                    excluded = excluded_data.get("excluded", False)
            except Exception:
                pass

        # Toggle
        new_excluded = not excluded

        # Save new state
        with open(excluded_file, 'w', encoding='utf-8') as f:
            json.dump({"excluded": new_excluded}, f)

        return web.json_response({"success": True, "excluded": new_excluded})

    @routes.get("/sid/generation-results/api/session/{session_id}")
    async def get_generation_session(request):
        """Get details of a specific generation session."""
        session_id = request.match_info["session_id"]

        # Security: validate session_id format
        if not session_id or ".." in session_id or "/" in session_id or "\\" in session_id:
            return web.json_response({"error": "Invalid session ID"}, status=400)

        session_dir = GENERATION_RESULTS_DIR / session_id
        if not session_dir.exists():
            return web.json_response({"error": "Session not found"}, status=404)

        result = {"id": session_id}

        # Read metadata
        metadata_file = session_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                result["metadata"] = json.load(f)

        # Read prompt
        prompt_file = session_dir / "prompt.txt"
        if prompt_file.exists():
            result["prompt"] = prompt_file.read_text(encoding='utf-8')

        # Read negative
        negative_file = session_dir / "negative.txt"
        if negative_file.exists():
            result["negative"] = negative_file.read_text(encoding='utf-8')

        # Read caption
        caption_file = session_dir / "caption.txt"
        if caption_file.exists():
            result["caption"] = caption_file.read_text(encoding='utf-8')

        # Image URL
        if (session_dir / "source.jpg").exists():
            result["image_url"] = f"/sid/generation-results/api/image/{session_id}/source.jpg"

        return web.json_response(result)

    @routes.get("/sid/generation-results/api/image/{session_id}/{filename}")
    async def get_generation_image(request):
        """Serve an image from a generation session."""
        session_id = request.match_info["session_id"]
        filename = request.match_info["filename"]

        # Security: validate inputs
        if not session_id or ".." in session_id or "/" in session_id or "\\" in session_id:
            return web.json_response({"error": "Invalid session ID"}, status=400)
        if filename != "source.jpg":
            return web.json_response({"error": "Invalid filename"}, status=400)

        image_path = GENERATION_RESULTS_DIR / session_id / filename
        if not image_path.exists():
            return web.json_response({"error": "Image not found"}, status=404)

        return web.FileResponse(image_path)

    @routes.delete("/sid/generation-results/api/session/{session_id}")
    async def delete_generation_session(request):
        """Delete a generation session."""
        session_id = request.match_info["session_id"]

        # Security: validate session_id format
        if not session_id or ".." in session_id or "/" in session_id or "\\" in session_id:
            return web.json_response({"error": "Invalid session ID"}, status=400)

        session_dir = GENERATION_RESULTS_DIR / session_id
        if not session_dir.exists():
            return web.json_response({"error": "Session not found"}, status=404)

        try:
            import shutil
            shutil.rmtree(session_dir)
            return web.json_response({"success": True})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    # =========================================================================
    # Documentation Routes
    # =========================================================================

    @routes.get("/sid/docs/{doc_name}")
    async def serve_docs(request):
        """Serve documentation markdown files with HTML rendering."""
        doc_name = request.match_info["doc_name"]

        # Security: Only allow specific doc files
        allowed_docs = {
            "Z-IMAGE-PROMPTING-GUIDE.md": "Z-Image Prompting Guide",
            "QWEN-VL-PROMPTING-GUIDE.md": "Qwen VL Prompting Guide",
            "VISION-PROMPTING-GUIDE.md": "Vision Prompting Guide",
        }

        if doc_name not in allowed_docs:
            return web.Response(text="Document not found", status=404)

        # Get the docs directory
        import os
        docs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs")
        doc_path = os.path.join(docs_dir, doc_name)

        if not os.path.exists(doc_path):
            return web.Response(text="Document not found", status=404)

        with open(doc_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Simple HTML wrapper for markdown
        html = f'''<!DOCTYPE html>
<html>
<head>
    <title>{allowed_docs[doc_name]} - SID Photography Toolkit</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #1a1a1a;
            color: #e0e0e0;
            margin: 0;
            padding: 20px;
            line-height: 1.6;
        }}
        .container {{
            max-width: 900px;
            margin: 0 auto;
            background: #252525;
            padding: 30px 40px;
            border-radius: 8px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }}
        h1, h2, h3, h4 {{ color: #4a9eff; margin-top: 1.5em; }}
        h1 {{ border-bottom: 2px solid #3c3c3c; padding-bottom: 10px; }}
        h2 {{ border-bottom: 1px solid #3c3c3c; padding-bottom: 8px; }}
        code {{
            background: #1e1e1e;
            padding: 2px 6px;
            border-radius: 4px;
            font-family: "Consolas", "Monaco", monospace;
            font-size: 0.9em;
            color: #ff9800;
        }}
        pre {{
            background: #1e1e1e;
            padding: 15px;
            border-radius: 6px;
            overflow-x: auto;
            border: 1px solid #3c3c3c;
        }}
        pre code {{
            background: none;
            padding: 0;
            color: #e0e0e0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1em 0;
        }}
        th, td {{
            border: 1px solid #3c3c3c;
            padding: 10px;
            text-align: left;
        }}
        th {{ background: #2a2a2a; color: #4a9eff; }}
        tr:nth-child(even) {{ background: #2a2a2a; }}
        a {{ color: #4a9eff; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
        blockquote {{
            border-left: 4px solid #4a9eff;
            margin: 1em 0;
            padding: 10px 20px;
            background: #2a2a2a;
            border-radius: 0 4px 4px 0;
        }}
        hr {{ border: none; border-top: 1px solid #3c3c3c; margin: 2em 0; }}
        .back-btn {{
            display: inline-block;
            background: #3c3c3c;
            color: #e0e0e0;
            padding: 8px 16px;
            border-radius: 4px;
            text-decoration: none;
            margin-bottom: 20px;
        }}
        .back-btn:hover {{ background: #4a4a4a; text-decoration: none; }}
    </style>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
</head>
<body>
    <div class="container">
        <a href="/sid/template-editor" class="back-btn">← Back to Template Editor</a>
        <div id="content"></div>
    </div>
    <script>
        const markdown = {repr(content)};
        document.getElementById('content').innerHTML = marked.parse(markdown);
    </script>
</body>
</html>'''
        return web.Response(text=html, content_type="text/html")

    # =========================================================================
    # Template Editor Routes
    # =========================================================================

    @routes.get("/sid/template-editor")
    async def serve_template_editor(request):
        """Serve the template editor HTML page."""
        return web.Response(text=get_template_editor_html(), content_type="text/html")

    @routes.get("/sid/template-editor/api/templates")
    async def list_templates(request):
        """List all templates from the registry."""
        try:
            from ..template_registry_loader import get_all_templates, reload_registry
            reload_registry()  # Ensure fresh data
            templates = get_all_templates()
            return web.json_response({"templates": templates})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    @routes.get("/sid/template-editor/api/template/{template_name}")
    async def get_template_detail(request):
        """Get details of a specific template with all size variants."""
        template_name = request.match_info["template_name"]

        try:
            from ..template_registry_loader import get_template_raw
            template = get_template_raw(template_name)
            if not template:
                return web.json_response({"error": "Template not found"}, status=404)
            return web.json_response(template)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    @routes.post("/sid/template-editor/api/template")
    async def create_template(request):
        """Create a new custom template with size variants."""
        try:
            data = await request.json()
            name = data.get("name", "").strip()
            description = data.get("description", "")
            tags = data.get("tags", [])
            prompts = data.get("prompts", None)  # New format: {"small": {...}, "medium": {...}, "large": {...}}
            system_prompt = data.get("system_prompt", "").strip()  # Legacy fallback

            if not name:
                return web.json_response({"error": "Template name is required"}, status=400)

            # Validate prompts
            if prompts:
                # Check at least one size has content
                has_content = any(
                    prompts.get(size, {}).get("system", "").strip()
                    for size in ["small", "medium", "large"]
                )
                if not has_content:
                    return web.json_response({"error": "At least one size variant must have a system prompt"}, status=400)
            elif not system_prompt:
                return web.json_response({"error": "System prompt is required"}, status=400)

            from ..template_registry_loader import save_custom_template, reload_registry
            success = save_custom_template(name, system_prompt=system_prompt, description=description, tags=tags, prompts=prompts)

            if success:
                reload_registry()
                return web.json_response({"success": True, "message": f"Template '{name}' created"})
            else:
                return web.json_response({"error": "Failed to save template"}, status=500)
        except json.JSONDecodeError:
            return web.json_response({"error": "Invalid JSON"}, status=400)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    @routes.put("/sid/template-editor/api/template/{template_name}")
    async def update_template(request):
        """Update an existing custom template with size variants."""
        template_name = request.match_info["template_name"]

        try:
            from ..template_registry_loader import get_template, get_registry, reload_registry

            # Check if template exists and is editable
            template = get_template(template_name)
            if not template:
                return web.json_response({"error": "Template not found"}, status=404)
            if template.get("readonly", False):
                return web.json_response({"error": "Cannot edit read-only template"}, status=403)

            data = await request.json()
            description = data.get("description", "")
            tags = data.get("tags", [])
            prompts = data.get("prompts", None)  # New format with size variants
            system_prompt = data.get("system_prompt", "").strip()  # Legacy fallback

            # Validate
            if prompts:
                has_content = any(
                    prompts.get(size, {}).get("system", "").strip()
                    for size in ["small", "medium", "large"]
                )
                if not has_content:
                    return web.json_response({"error": "At least one size variant must have a system prompt"}, status=400)
            elif not system_prompt:
                return web.json_response({"error": "System prompt is required"}, status=400)

            # Get the file path and update it
            registry = get_registry()
            template_path = registry._registry_path / template["path"]

            try:
                import tomlkit
            except ImportError:
                return web.json_response({"error": "tomlkit not installed"}, status=500)

            # Read existing file and update
            with open(template_path, "rb") as f:
                try:
                    import tomllib
                except ImportError:
                    import tomli as tomllib
                doc_data = tomllib.load(f)

            # Create new document with updates
            doc = tomlkit.document()
            metadata = tomlkit.table()
            metadata.add("name", template.get("name", template_name))
            metadata.add("description", description or doc_data.get("metadata", {}).get("description", ""))
            metadata.add("author", doc_data.get("metadata", {}).get("author", "User"))
            metadata.add("version", "2.0")
            metadata.add("category", "custom")
            metadata.add("tags", tags or doc_data.get("metadata", {}).get("tags", []))
            doc.add("metadata", metadata)

            # Prompt section with size variants
            prompt = tomlkit.table()
            if prompts:
                for size in ["small", "medium", "large"]:
                    size_data = prompts.get(size, {})
                    size_table = tomlkit.table()
                    size_table.add("system", size_data.get("system", ""))
                    size_table.add("user", size_data.get("user", "Describe this image."))
                    prompt.add(size, size_table)
            else:
                # Legacy format
                prompt.add("system", system_prompt)
                prompt.add("user", doc_data.get("prompt", {}).get("user", "Describe this image following the system instructions."))
            doc.add("prompt", prompt)

            with open(template_path, "w", encoding="utf-8") as f:
                f.write(tomlkit.dumps(doc))

            reload_registry()
            return web.json_response({"success": True, "message": f"Template '{template_name}' updated"})
        except json.JSONDecodeError:
            return web.json_response({"error": "Invalid JSON"}, status=400)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    @routes.delete("/sid/template-editor/api/template/{template_name}")
    async def delete_template(request):
        """Delete a custom template."""
        template_name = request.match_info["template_name"]

        try:
            from ..template_registry_loader import delete_custom_template, reload_registry

            success = delete_custom_template(template_name)
            if success:
                reload_registry()
                return web.json_response({"success": True, "message": f"Template '{template_name}' deleted"})
            else:
                return web.json_response({"error": "Failed to delete template (may be read-only or not found)"}, status=400)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    @routes.post("/sid/template-editor/api/sync")
    async def sync_templates(request):
        """Sync community templates from GitHub."""
        try:
            from ..template_registry_loader import sync_from_github, reload_registry

            success = sync_from_github()
            if success:
                reload_registry()
                return web.json_response({"success": True, "message": "Templates synced from GitHub"})
            else:
                return web.json_response({"error": "Sync failed - check GitHub configuration"}, status=500)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    @routes.post("/sid/template-editor/api/reload")
    async def reload_templates(request):
        """Reload templates from disk."""
        try:
            from ..template_registry_loader import reload_registry
            reload_registry()
            return web.json_response({"success": True, "message": "Templates reloaded"})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    # =========================================================================
    # Model Editor Routes
    # =========================================================================

    @routes.get("/sid/model-editor")
    async def serve_model_editor(request):
        """Serve the model editor HTML page."""
        return web.Response(text=get_model_editor_html(), content_type="text/html")

    @routes.get("/sid/model-editor/api/models")
    async def list_models(request):
        """List all configured models (API and Local)."""
        try:
            # Get API models
            from ..llm_providers.sid_llm_api import MODEL_METADATA
            api_models = []
            for model_id, meta in MODEL_METADATA.items():
                api_models.append({
                    "id": model_id,
                    "provider": meta.get("provider", "unknown"),
                    "size": meta.get("size", "large"),
                    "reasoning": meta.get("reasoning", False) or meta.get("supports_reasoning", False),
                    "vision": meta.get("vision", True),
                    "max_tokens": meta.get("max_output_tokens", 4096),
                    "type": "api",
                    "readonly": True
                })

            # Get Local models
            from ..llm_providers.sid_llm_local import LOCAL_MODELS
            local_models = []
            for model_id, info in LOCAL_MODELS.items():
                local_models.append({
                    "id": model_id,
                    "name": info.name,
                    "repo_id": info.repo_id,
                    "family": info.family.value,
                    "size": info.size,
                    "reasoning": info.is_thinking,
                    "vision": info.model_type.value != "text",
                    "vram_fp16": info.vram_fp16,
                    "vram_8bit": info.vram_8bit,
                    "vram_4bit": info.vram_4bit,
                    "max_tokens": info.max_output_tokens,
                    "target_image_size": info.target_image_size,
                    "type": "local",
                    "readonly": "(Custom)" not in info.name
                })

            # Get custom API models if file exists
            custom_api_path = TOOLKIT_DIR / "custom_api_models.json"
            custom_api_models = []
            if custom_api_path.exists():
                try:
                    with open(custom_api_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    for model_id, meta in data.get("models", {}).items():
                        custom_api_models.append({
                            "id": model_id,
                            "provider": meta.get("provider", "custom"),
                            "size": meta.get("size", "large"),
                            "reasoning": meta.get("reasoning", False),
                            "vision": meta.get("vision", True),
                            "max_tokens": meta.get("max_tokens", 4096),
                            "type": "api_custom",
                            "readonly": False
                        })
                except:
                    pass

            return web.json_response({
                "api_models": api_models,
                "local_models": local_models,
                "custom_api_models": custom_api_models
            })
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    @routes.post("/sid/model-editor/api/model")
    async def add_custom_model(request):
        """Add a new custom model."""
        try:
            data = await request.json()
            model_type = data.get("model_type", "api")  # "api" or "local"

            if model_type == "api":
                # Add to custom_api_models.json
                custom_api_path = TOOLKIT_DIR / "custom_api_models.json"
                existing = {"models": {}}
                if custom_api_path.exists():
                    with open(custom_api_path, "r", encoding="utf-8") as f:
                        existing = json.load(f)

                model_id = data.get("id", "").strip()
                if not model_id:
                    return web.json_response({"error": "Model ID is required"}, status=400)

                existing["models"][model_id] = {
                    "provider": data.get("provider", "custom"),
                    "size": data.get("size", "large"),
                    "reasoning": data.get("reasoning", False),
                    "vision": data.get("vision", True),
                    "max_tokens": data.get("max_tokens", 4096)
                }

                with open(custom_api_path, "w", encoding="utf-8") as f:
                    json.dump(existing, f, indent=2)

                # Reload the API models
                from ..llm_providers.sid_llm_api import load_custom_api_models
                load_custom_api_models()

                return web.json_response({"success": True, "message": f"Added API model: {model_id}"})

            elif model_type == "local":
                # Add to custom_models.json
                custom_local_path = TOOLKIT_DIR / "custom_models.json"
                existing = {"hf_models": {}}
                if custom_local_path.exists():
                    with open(custom_local_path, "r", encoding="utf-8") as f:
                        existing = json.load(f)

                model_id = data.get("id", "").strip()
                if not model_id:
                    return web.json_response({"error": "Model ID is required"}, status=400)

                existing["hf_models"][model_id] = {
                    "repo_id": data.get("repo_id", ""),
                    "family": data.get("family", "qwenvl"),
                    "model_type": data.get("model_type_cap", "vision"),
                    "size": data.get("size", "large"),
                    "is_thinking": data.get("reasoning", False),
                    "max_output_tokens": data.get("max_tokens", 4096),
                    "target_image_size": data.get("target_image_size", 384),
                    "vram_requirement": {
                        "full": data.get("vram_fp16", 10.0),
                        "8bit": data.get("vram_8bit", 6.0),
                        "4bit": data.get("vram_4bit", 4.0)
                    },
                    "description": data.get("description", "Custom model")
                }

                with open(custom_local_path, "w", encoding="utf-8") as f:
                    json.dump(existing, f, indent=2)

                # Reload local models
                from ..llm_providers.sid_llm_local import load_custom_models
                load_custom_models()

                return web.json_response({"success": True, "message": f"Added local model: {model_id}"})

            else:
                return web.json_response({"error": "Invalid model_type"}, status=400)

        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    @routes.delete("/sid/model-editor/api/model/{model_id}")
    async def delete_custom_model(request):
        """Delete a custom model."""
        try:
            model_id = request.match_info["model_id"]
            model_type = request.query.get("type", "api")

            if model_type == "api_custom":
                custom_api_path = TOOLKIT_DIR / "custom_api_models.json"
                if custom_api_path.exists():
                    with open(custom_api_path, "r", encoding="utf-8") as f:
                        existing = json.load(f)
                    if model_id in existing.get("models", {}):
                        del existing["models"][model_id]
                        with open(custom_api_path, "w", encoding="utf-8") as f:
                            json.dump(existing, f, indent=2)
                        return web.json_response({"success": True, "message": f"Deleted: {model_id}"})
                return web.json_response({"error": "Model not found"}, status=404)

            elif model_type == "local":
                custom_local_path = TOOLKIT_DIR / "custom_models.json"
                if custom_local_path.exists():
                    with open(custom_local_path, "r", encoding="utf-8") as f:
                        existing = json.load(f)
                    if model_id in existing.get("hf_models", {}):
                        del existing["hf_models"][model_id]
                        with open(custom_local_path, "w", encoding="utf-8") as f:
                            json.dump(existing, f, indent=2)
                        return web.json_response({"success": True, "message": f"Deleted: {model_id}"})
                return web.json_response({"error": "Model not found"}, status=404)

            else:
                return web.json_response({"error": "Cannot delete built-in models"}, status=400)

        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)


def get_model_editor_html():
    """Return the HTML for the model editor UI."""
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SID Model Editor</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1e1e1e;
            color: #d4d4d4;
            height: 100vh;
            overflow: hidden;
        }
        .container { display: flex; height: 100vh; }
        .sidebar {
            width: 200px;
            background: #252526;
            border-right: 1px solid #3c3c3c;
            display: flex;
            flex-direction: column;
        }
        .sidebar-header {
            padding: 15px;
            background: #333;
            border-bottom: 1px solid #3c3c3c;
            font-weight: 600;
            font-size: 14px;
        }
        .nav-links { flex: 1; padding: 10px 0; }
        .nav-link {
            display: block;
            padding: 10px 15px;
            color: #888;
            text-decoration: none;
            font-size: 13px;
        }
        .nav-link:hover { background: #3c3c3c; color: #fff; }
        .nav-link.active { color: #4a9eff; background: #37373d; }
        .main-content { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
        .toolbar {
            padding: 10px 20px;
            background: #333;
            border-bottom: 1px solid #3c3c3c;
            display: flex;
            align-items: center;
            gap: 15px;
        }
        .toolbar h2 { flex: 1; font-size: 16px; font-weight: 500; }
        .btn {
            padding: 8px 16px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 13px;
            font-weight: 500;
        }
        .btn-primary { background: #0e639c; color: white; }
        .btn-primary:hover { background: #1177bb; }
        .btn-danger { background: #c42b1c; color: white; }
        .btn-danger:hover { background: #e03e2e; }
        .btn-secondary { background: #3c3c3c; color: #d4d4d4; }
        .btn-secondary:hover { background: #4c4c4c; }
        .tabs {
            display: flex;
            background: #2d2d2d;
            border-bottom: 1px solid #3c3c3c;
        }
        .tab {
            padding: 12px 24px;
            cursor: pointer;
            font-size: 13px;
            color: #888;
            border-bottom: 2px solid transparent;
        }
        .tab:hover { color: #d4d4d4; background: #333; }
        .tab.active { color: #4a9eff; border-bottom-color: #4a9eff; background: #1e1e1e; }
        .tab-content { flex: 1; overflow: auto; padding: 20px; }
        .model-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }
        .model-table th, .model-table td {
            padding: 10px 12px;
            text-align: left;
            border-bottom: 1px solid #3c3c3c;
        }
        .model-table th {
            background: #2d2d2d;
            color: #888;
            font-weight: 500;
            font-size: 11px;
            text-transform: uppercase;
            position: sticky;
            top: 0;
        }
        .model-table tr:hover { background: #2a2d2e; }
        .badge {
            padding: 3px 8px;
            border-radius: 3px;
            font-size: 11px;
            font-weight: 500;
        }
        .badge-small { background: #16825d; color: white; }
        .badge-medium { background: #0e639c; color: white; }
        .badge-large { background: #7c3aed; color: white; }
        .badge-yes { background: #16825d; color: white; }
        .badge-no { background: #6b7280; color: white; }
        .badge-custom { background: #f59e0b; color: black; }
        .delete-btn {
            padding: 4px 8px;
            font-size: 11px;
            background: #3c3c3c;
            border: none;
            border-radius: 3px;
            color: #d4d4d4;
            cursor: pointer;
        }
        .delete-btn:hover { background: #c42b1c; }
        .delete-btn:disabled { opacity: 0.3; cursor: not-allowed; }
        /* Modal */
        .modal-overlay {
            display: none;
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(0,0,0,0.7);
            align-items: center;
            justify-content: center;
            z-index: 1000;
        }
        .modal-overlay.show { display: flex; }
        .modal {
            background: #252526;
            border-radius: 8px;
            width: 500px;
            max-width: 90vw;
        }
        .modal-header {
            padding: 15px 20px;
            border-bottom: 1px solid #3c3c3c;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .modal-close {
            background: none;
            border: none;
            color: #888;
            font-size: 20px;
            cursor: pointer;
        }
        .modal-body { padding: 20px; }
        .modal-footer {
            padding: 15px 20px;
            border-top: 1px solid #3c3c3c;
            display: flex;
            justify-content: flex-end;
            gap: 10px;
        }
        .form-group { margin-bottom: 15px; }
        .form-group label { display: block; font-size: 12px; color: #888; margin-bottom: 5px; }
        .form-group input, .form-group select {
            width: 100%;
            padding: 8px 12px;
            background: #2d2d2d;
            border: 1px solid #3c3c3c;
            border-radius: 4px;
            color: #d4d4d4;
            font-size: 13px;
        }
        .form-row { display: flex; gap: 15px; }
        .form-row .form-group { flex: 1; }
        .checkbox-group {
            display: flex;
            gap: 20px;
            margin-top: 10px;
        }
        .checkbox-group label {
            display: flex;
            align-items: center;
            gap: 5px;
            color: #d4d4d4;
            cursor: pointer;
        }
        .status-bar {
            padding: 8px 20px;
            background: #007acc;
            font-size: 12px;
        }
        .status-bar.error { background: #c42b1c; }
        .status-bar.success { background: #16825d; }
        .model-count { font-size: 12px; color: #888; }
    </style>
</head>
<body>
    <div class="container">
        <div class="sidebar">
            <div class="sidebar-header">SID Toolkit</div>
            <div class="nav-links">
                <a href="/sid" class="nav-link">Home</a>
                <a href="/sid/template-editor" class="nav-link">Template Editor</a>
                <a href="/sid/model-editor" class="nav-link active">Model Editor</a>
                <a href="/sid/generation-results" class="nav-link">Generation Results</a>
                <a href="/sid/debug-results" class="nav-link">Debug Viewer</a>
            </div>
        </div>
        <div class="main-content">
            <div class="toolbar">
                <h2>Model Editor</h2>
                <span class="model-count" id="modelCount"></span>
                <button class="btn btn-primary" onclick="showAddModal()">+ Add Model</button>
            </div>
            <div class="tabs">
                <div class="tab active" data-tab="api" onclick="switchTab('api')">API Models</div>
                <div class="tab" data-tab="local" onclick="switchTab('local')">Local Models</div>
            </div>
            <div class="tab-content" id="tabContent">
                <div style="text-align: center; padding: 40px; color: #888;">Loading models...</div>
            </div>
            <div class="status-bar" id="statusBar">Ready</div>
        </div>
    </div>

    <!-- Add Model Modal -->
    <div class="modal-overlay" id="addModal">
        <div class="modal">
            <div class="modal-header">
                <span>Add Custom Model</span>
                <button class="modal-close" onclick="hideAddModal()">&times;</button>
            </div>
            <div class="modal-body">
                <div class="form-group">
                    <label>Model Type</label>
                    <select id="addModelType" onchange="updateAddForm()">
                        <option value="api">API Model (Anthropic, OpenAI, etc.)</option>
                        <option value="local">Local Model (HuggingFace)</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>Model ID *</label>
                    <input type="text" id="addModelId" placeholder="e.g., gpt-4-turbo or Qwen3-VL-72B">
                </div>
                <div id="apiFields">
                    <div class="form-group">
                        <label>Provider</label>
                        <select id="addProvider">
                            <option value="anthropic">Anthropic</option>
                            <option value="openai">OpenAI</option>
                            <option value="google">Google</option>
                            <option value="openrouter">OpenRouter</option>
                            <option value="custom">Custom</option>
                        </select>
                    </div>
                </div>
                <div id="localFields" style="display: none;">
                    <div class="form-group">
                        <label>HuggingFace Repo ID *</label>
                        <input type="text" id="addRepoId" placeholder="e.g., Qwen/Qwen3-VL-72B-Instruct">
                    </div>
                    <div class="form-group">
                        <label>Model Family</label>
                        <select id="addFamily">
                            <option value="qwenvl">Qwen VL</option>
                            <option value="qwen25vl">Qwen 2.5 VL</option>
                            <option value="qwen3_text">Qwen 3 Text</option>
                            <option value="llama_text">Llama Text</option>
                        </select>
                    </div>
                    <div class="form-row">
                        <div class="form-group">
                            <label>VRAM FP16 (GB)</label>
                            <input type="number" id="addVramFp16" value="20" step="0.5">
                        </div>
                        <div class="form-group">
                            <label>VRAM 8-bit (GB)</label>
                            <input type="number" id="addVram8bit" value="12" step="0.5">
                        </div>
                        <div class="form-group">
                            <label>VRAM 4-bit (GB)</label>
                            <input type="number" id="addVram4bit" value="8" step="0.5">
                        </div>
                    </div>
                    <div class="form-row">
                        <div class="form-group">
                            <label>Target Image Size (px)</label>
                            <input type="number" id="addImageSize" value="384" step="32" min="128" max="1024">
                        </div>
                    </div>
                </div>
                <div class="form-row">
                    <div class="form-group">
                        <label>Size Category</label>
                        <select id="addSize">
                            <option value="small">Small (2B-6B)</option>
                            <option value="medium">Medium (7B-13B)</option>
                            <option value="large" selected>Large (13B+)</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Max Output Tokens</label>
                        <input type="number" id="addMaxTokens" value="4096" step="512" min="256">
                    </div>
                </div>
                <div class="checkbox-group">
                    <label><input type="checkbox" id="addVision" checked> Supports Vision</label>
                    <label><input type="checkbox" id="addReasoning"> Supports Reasoning</label>
                </div>
            </div>
            <div class="modal-footer">
                <button class="btn btn-secondary" onclick="hideAddModal()">Cancel</button>
                <button class="btn btn-primary" onclick="addModel()">Add Model</button>
            </div>
        </div>
    </div>

    <script>
        let currentTab = 'api';
        let models = { api_models: [], local_models: [], custom_api_models: [] };

        loadModels();

        async function loadModels() {
            try {
                const res = await fetch('/sid/model-editor/api/models');
                models = await res.json();
                updateModelCount();
                renderTable();
            } catch (e) {
                setStatus('Error loading models: ' + e.message, 'error');
            }
        }

        function updateModelCount() {
            const apiCount = models.api_models.length + models.custom_api_models.length;
            const localCount = models.local_models.length;
            document.getElementById('modelCount').textContent = `${apiCount} API | ${localCount} Local`;
        }

        function switchTab(tab) {
            currentTab = tab;
            document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.dataset.tab === tab));
            renderTable();
        }

        function renderTable() {
            const content = document.getElementById('tabContent');

            if (currentTab === 'api') {
                const allApi = [...models.api_models, ...models.custom_api_models];
                content.innerHTML = `
                    <table class="model-table">
                        <thead>
                            <tr>
                                <th>Model ID</th>
                                <th>Provider</th>
                                <th>Size</th>
                                <th>Max Tokens</th>
                                <th>Vision</th>
                                <th>Reasoning</th>
                                <th>Type</th>
                                <th></th>
                            </tr>
                        </thead>
                        <tbody>
                            ${allApi.map(m => `
                                <tr>
                                    <td><strong>${escapeHtml(m.id)}</strong></td>
                                    <td>${m.provider}</td>
                                    <td><span class="badge badge-${m.size}">${m.size}</span></td>
                                    <td>${(m.max_tokens || 4096).toLocaleString()}</td>
                                    <td><span class="badge badge-${m.vision ? 'yes' : 'no'}">${m.vision ? 'Yes' : 'No'}</span></td>
                                    <td><span class="badge badge-${m.reasoning ? 'yes' : 'no'}">${m.reasoning ? 'Yes' : 'No'}</span></td>
                                    <td>${m.type === 'api_custom' ? '<span class="badge badge-custom">Custom</span>' : 'Built-in'}</td>
                                    <td>
                                        <button class="delete-btn" ${m.readonly ? 'disabled' : ''} onclick="deleteModel('${escapeHtml(m.id)}', '${m.type}')">Delete</button>
                                    </td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                `;
            } else {
                content.innerHTML = `
                    <table class="model-table">
                        <thead>
                            <tr>
                                <th>Model ID</th>
                                <th>Family</th>
                                <th>Size</th>
                                <th>Max Tokens</th>
                                <th>Img Size</th>
                                <th>Vision</th>
                                <th>Reasoning</th>
                                <th>VRAM (4-bit)</th>
                                <th>Type</th>
                                <th></th>
                            </tr>
                        </thead>
                        <tbody>
                            ${models.local_models.map(m => `
                                <tr>
                                    <td><strong>${escapeHtml(m.id)}</strong></td>
                                    <td>${m.family}</td>
                                    <td><span class="badge badge-${m.size}">${m.size}</span></td>
                                    <td>${(m.max_tokens || 4096).toLocaleString()}</td>
                                    <td>${m.target_image_size || 384}px</td>
                                    <td><span class="badge badge-${m.vision ? 'yes' : 'no'}">${m.vision ? 'Yes' : 'No'}</span></td>
                                    <td><span class="badge badge-${m.reasoning ? 'yes' : 'no'}">${m.reasoning ? 'Yes' : 'No'}</span></td>
                                    <td>${m.vram_4bit} GB</td>
                                    <td>${m.readonly ? 'Built-in' : '<span class="badge badge-custom">Custom</span>'}</td>
                                    <td>
                                        <button class="delete-btn" ${m.readonly ? 'disabled' : ''} onclick="deleteModel('${escapeHtml(m.id)}', 'local')">Delete</button>
                                    </td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                `;
            }
        }

        function showAddModal() {
            document.getElementById('addModal').classList.add('show');
            updateAddForm();
        }

        function hideAddModal() {
            document.getElementById('addModal').classList.remove('show');
            document.getElementById('addModelId').value = '';
            document.getElementById('addRepoId').value = '';
            document.getElementById('addMaxTokens').value = '4096';
            document.getElementById('addImageSize').value = '384';
            document.getElementById('addVramFp16').value = '20';
            document.getElementById('addVram8bit').value = '12';
            document.getElementById('addVram4bit').value = '8';
        }

        function updateAddForm() {
            const type = document.getElementById('addModelType').value;
            document.getElementById('apiFields').style.display = type === 'api' ? 'block' : 'none';
            document.getElementById('localFields').style.display = type === 'local' ? 'block' : 'none';
        }

        async function addModel() {
            const modelType = document.getElementById('addModelType').value;
            const modelId = document.getElementById('addModelId').value.trim();

            if (!modelId) {
                alert('Model ID is required');
                return;
            }

            const data = {
                model_type: modelType,
                id: modelId,
                size: document.getElementById('addSize').value,
                vision: document.getElementById('addVision').checked,
                reasoning: document.getElementById('addReasoning').checked,
                max_tokens: parseInt(document.getElementById('addMaxTokens').value) || 4096
            };

            if (modelType === 'api') {
                data.provider = document.getElementById('addProvider').value;
            } else {
                data.repo_id = document.getElementById('addRepoId').value.trim();
                data.family = document.getElementById('addFamily').value;
                data.vram_fp16 = parseFloat(document.getElementById('addVramFp16').value);
                data.vram_8bit = parseFloat(document.getElementById('addVram8bit').value);
                data.vram_4bit = parseFloat(document.getElementById('addVram4bit').value);
                data.target_image_size = parseInt(document.getElementById('addImageSize').value) || 384;
            }

            try {
                const res = await fetch('/sid/model-editor/api/model', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });
                const result = await res.json();

                if (result.error) {
                    setStatus('Error: ' + result.error, 'error');
                    return;
                }

                hideAddModal();
                await loadModels();
                setStatus('Model added successfully', 'success');
                setTimeout(() => setStatus('Ready', ''), 3000);
            } catch (e) {
                setStatus('Error: ' + e.message, 'error');
            }
        }

        async function deleteModel(modelId, type) {
            if (!confirm(`Delete model "${modelId}"?`)) return;

            try {
                const res = await fetch(`/sid/model-editor/api/model/${encodeURIComponent(modelId)}?type=${type}`, {
                    method: 'DELETE'
                });
                const result = await res.json();

                if (result.error) {
                    setStatus('Error: ' + result.error, 'error');
                    return;
                }

                await loadModels();
                setStatus('Model deleted', 'success');
                setTimeout(() => setStatus('Ready', ''), 3000);
            } catch (e) {
                setStatus('Error: ' + e.message, 'error');
            }
        }

        function setStatus(text, type) {
            const bar = document.getElementById('statusBar');
            bar.className = 'status-bar' + (type ? ' ' + type : '');
            bar.textContent = text;
        }

        function escapeHtml(text) {
            if (!text) return '';
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
    </script>
</body>
</html>'''


def get_debug_viewer_html():
    """Return the HTML for the comprehensive debug results viewer UI."""
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SID Intelligent Debug Viewer</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1e1e1e;
            color: #d4d4d4;
            height: 100vh;
            overflow: hidden;
        }
        .container { display: flex; height: 100vh; }

        /* Sidebar */
        .sidebar {
            width: 300px;
            background: #252526;
            border-right: 1px solid #3c3c3c;
            display: flex;
            flex-direction: column;
        }
        .sidebar-header {
            padding: 12px 15px;
            background: #333;
            border-bottom: 1px solid #3c3c3c;
            font-weight: 600;
            font-size: 13px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .nav-links {
            border-top: 1px solid #3c3c3c;
            padding: 8px 0;
            background: #2d2d2d;
        }
        .nav-link {
            display: block;
            padding: 6px 15px;
            color: #888;
            text-decoration: none;
            font-size: 11px;
            transition: all 0.2s;
        }
        .nav-link:hover { background: #3c3c3c; color: #fff; }
        .nav-link.active { color: #4a9eff; background: #1e1e1e; }
        .btn-sm {
            padding: 4px 10px;
            background: #0e639c;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 11px;
        }
        .btn-sm:hover { background: #1177bb; }
        .btn-green { background: #16825d; }
        .btn-green:hover { background: #1a9e6e; }
        .btn-sm:disabled { background: #3c3c3c; color: #666; cursor: not-allowed; }
        .session-list { flex: 1; overflow-y: auto; }
        .session-item {
            padding: 10px 12px;
            cursor: pointer;
            font-size: 11px;
            border-bottom: 1px solid #333;
        }
        .session-item:hover { background: #2a2d2e; }
        .session-item.active { background: #37373d; border-left: 3px solid #4a9eff; }
        .session-item.excluded { opacity: 0.5; }
        .session-header { display: flex; align-items: center; gap: 6px; margin-bottom: 3px; }
        .exclude-checkbox { width: 14px; height: 14px; cursor: pointer; accent-color: #b89500; }
        .session-id { font-weight: 600; color: #fff; flex: 1; font-size: 11px; }
        .session-meta { color: #888; font-size: 10px; }
        .session-score {
            display: inline-block;
            padding: 2px 6px;
            border-radius: 8px;
            font-size: 10px;
            font-weight: 600;
        }
        .score-good { background: #16825d; color: #fff; }
        .score-ok { background: #c49a3a; color: #000; }
        .score-bad { background: #c42b1c; color: #fff; }

        /* Main content */
        .main-content { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
        .toolbar {
            padding: 8px 15px;
            background: #333;
            border-bottom: 1px solid #3c3c3c;
            font-size: 12px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .toolbar-buttons { display: flex; gap: 8px; }
        .content-area { flex: 1; overflow-y: auto; padding: 15px; }
        .empty-state {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 100%;
            color: #888;
            gap: 10px;
        }
        .loading { display: flex; align-items: center; justify-content: center; height: 100%; color: #888; }

        /* Section styles */
        .section {
            background: #252526;
            border-radius: 6px;
            margin-bottom: 15px;
            border: 1px solid #3c3c3c;
        }
        .section-header {
            padding: 10px 15px;
            background: #333;
            font-size: 12px;
            font-weight: 600;
            border-radius: 6px 6px 0 0;
            display: flex;
            justify-content: space-between;
            align-items: center;
            cursor: pointer;
            user-select: none;
        }
        .section-header:hover { background: #3a3a3a; }
        .section-header .toggle { color: #888; font-size: 10px; }
        .section-content { padding: 12px 15px; }
        .section.collapsed .section-content { display: none; }
        .section.collapsed .section-header { border-radius: 6px; }

        /* Two-column layout */
        .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }
        @media (max-width: 1200px) { .two-col { grid-template-columns: 1fr; } }

        /* Images */
        .images-row { display: flex; gap: 15px; margin-bottom: 15px; }
        .image-card { flex: 1; background: #252526; border-radius: 6px; overflow: hidden; border: 1px solid #3c3c3c; }
        .image-card h4 { padding: 8px 12px; background: #333; font-size: 11px; font-weight: 600; }
        .image-card img { width: 100%; height: auto; display: block; }

        /* Scores grid */
        .scores-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px; }
        .score-card { background: #1e1e1e; border-radius: 6px; padding: 12px; text-align: center; }
        .score-card h5 { font-size: 10px; color: #888; margin-bottom: 5px; text-transform: uppercase; }
        .score-card .value { font-size: 24px; font-weight: 700; }
        .score-card .reason { font-size: 10px; color: #888; margin-top: 5px; line-height: 1.4; }

        /* Prompt display */
        .prompt-box {
            background: #1e1e1e;
            border-radius: 4px;
            padding: 12px;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 12px;
            line-height: 1.5;
            white-space: pre-wrap;
            word-break: break-word;
            border: 1px solid #333;
        }
        .prompt-improved { background: #1a2e1a; border-color: #4caf50; }

        /* Lists */
        .item-list { list-style: none; }
        .item-list li {
            padding: 6px 0;
            font-size: 12px;
            border-bottom: 1px solid #333;
            display: flex;
            align-items: flex-start;
            gap: 8px;
        }
        .item-list li:last-child { border-bottom: none; }
        .item-list .icon { flex-shrink: 0; }
        .tag-issue { color: #64b5f6; }
        .tag-missing { color: #ef5350; }
        .tag-hallucinated { color: #ffb74d; }
        .tag-strength { color: #81c784; }
        .tag-suggestion { color: #ba68c8; }

        /* Template suggestions */
        .suggestion-card {
            background: #1e1e1e;
            border-radius: 4px;
            padding: 10px 12px;
            margin-bottom: 8px;
            border-left: 3px solid #9c27b0;
        }
        .suggestion-card .instruction {
            font-family: 'Consolas', monospace;
            font-size: 11px;
            color: #ce93d8;
            margin-bottom: 5px;
        }
        .suggestion-card .reason { font-size: 11px; color: #888; }

        /* Metadata table */
        .meta-table { width: 100%; font-size: 11px; }
        .meta-table td { padding: 4px 8px; border-bottom: 1px solid #333; }
        .meta-table td:first-child { color: #888; width: 140px; }
        .meta-table td:last-child { color: #d4d4d4; font-family: 'Consolas', monospace; }

        /* JSON viewer */
        .json-viewer {
            background: #1e1e1e;
            border-radius: 4px;
            padding: 10px;
            font-family: 'Consolas', monospace;
            font-size: 11px;
            max-height: 300px;
            overflow: auto;
            white-space: pre-wrap;
            color: #9cdcfe;
        }

        /* Tabs */
        .tabs { display: flex; gap: 0; border-bottom: 1px solid #3c3c3c; margin-bottom: 12px; }
        .tab {
            padding: 8px 16px;
            background: transparent;
            border: none;
            color: #888;
            cursor: pointer;
            font-size: 12px;
            border-bottom: 2px solid transparent;
            margin-bottom: -1px;
        }
        .tab:hover { color: #fff; }
        .tab.active { color: #4a9eff; border-bottom-color: #4a9eff; }
        .tab-content { display: none; }
        .tab-content.active { display: block; }

        /* Copy button */
        .copy-btn {
            padding: 3px 8px;
            background: #333;
            border: 1px solid #555;
            color: #888;
            border-radius: 3px;
            cursor: pointer;
            font-size: 10px;
        }
        .copy-btn:hover { background: #444; color: #fff; }
    </style>
</head>
<body>
    <div class="container">
        <div class="sidebar">
            <div class="sidebar-header">
                <span>Debug Sessions</span>
                <button class="btn-sm" onclick="loadSessions()">Refresh</button>
            </div>
            <div class="session-list" id="sessionList">
                <div class="loading">Loading...</div>
            </div>
            <div class="nav-links">
                <a href="/sid" class="nav-link">Home</a>
                <a href="/sid/template-editor" class="nav-link">Template Editor</a>
                <a href="/sid/model-editor" class="nav-link">Model Editor</a>
                <a href="/sid/generation-results" class="nav-link">Generation Results</a>
                <a href="/sid/debug-results" class="nav-link active">Debug Viewer</a>
            </div>
        </div>
        <div class="main-content">
            <div class="toolbar">
                <span id="toolbarTitle">Select a session to view results</span>
                <div class="toolbar-buttons">
                    <button class="btn-sm btn-green" id="downloadBtn" disabled onclick="downloadPDF()">Download PDF</button>
                </div>
            </div>
            <div class="content-area" id="contentArea">
                <div class="empty-state">
                    <div style="font-size: 48px;">🔍</div>
                    <div>Select a debug session from the sidebar</div>
                    <div style="font-size: 12px; color: #666;">Intelligent debug requires reasoning-capable models</div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/html2pdf.js/0.10.1/html2pdf.bundle.min.js"></script>
    <script>
        let currentSession = null;
        let currentSessionId = null;

        loadSessions();

        async function loadSessions() {
            try {
                const res = await fetch('/sid/debug-results/api/sessions');
                const data = await res.json();
                const list = document.getElementById('sessionList');

                if (data.sessions.length === 0) {
                    list.innerHTML = '<div class="empty-state" style="padding:20px;">No debug sessions</div>';
                    return;
                }

                list.innerHTML = data.sessions.map(s => {
                    let dateStr = 'Unknown';
                    if (s.timestamp) {
                        const d = new Date(s.timestamp);
                        const today = new Date();
                        dateStr = d.toDateString() === today.toDateString()
                            ? d.toLocaleTimeString()
                            : d.toLocaleDateString() + ' ' + d.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
                    }
                    const shortModel = (s.model || '').replace('claude-', '').replace(/-20[0-9]+$/, '');
                    const scoreClass = s.overall_score >= 7 ? 'score-good' : s.overall_score >= 5 ? 'score-ok' : 'score-bad';
                    return `
                        <div class="session-item ${s.excluded ? 'excluded' : ''}" data-id="${s.id}">
                            <div class="session-header">
                                <input type="checkbox" class="exclude-checkbox" ${s.excluded ? 'checked' : ''}
                                    onclick="toggleExclude(event, '${s.id}')" title="Exclude">
                                <div class="session-id" onclick="loadSession('${s.id}')">${dateStr}</div>
                                <span class="session-score ${scoreClass}">${(s.overall_score || 0).toFixed(1)}</span>
                            </div>
                            <div class="session-meta" onclick="loadSession('${s.id}')">${s.provider || 'unknown'} / ${shortModel}</div>
                        </div>
                    `;
                }).join('');
            } catch (e) {
                document.getElementById('sessionList').innerHTML = '<div class="empty-state">Error loading</div>';
            }
        }

        async function toggleExclude(event, sessionId) {
            event.stopPropagation();
            const cb = event.target;
            try {
                const res = await fetch(`/sid/debug-results/api/session/${sessionId}/exclude`, { method: 'POST' });
                const data = await res.json();
                if (data.success) {
                    cb.checked = data.excluded;
                    cb.closest('.session-item').classList.toggle('excluded', data.excluded);
                }
            } catch (e) { cb.checked = !cb.checked; }
        }

        async function loadSession(sessionId) {
            document.querySelectorAll('.session-item').forEach(el => {
                el.classList.toggle('active', el.dataset.id === sessionId);
            });
            currentSessionId = sessionId;
            document.getElementById('toolbarTitle').textContent = `Session: ${sessionId}`;
            document.getElementById('contentArea').innerHTML = '<div class="loading">Loading...</div>';
            document.getElementById('downloadBtn').disabled = true;

            try {
                const res = await fetch(`/sid/debug-results/api/session/${sessionId}`);
                const data = await res.json();
                if (data.error) {
                    document.getElementById('contentArea').innerHTML = `<div class="empty-state">${data.error}</div>`;
                    return;
                }
                currentSession = data;
                renderSession(data);
                document.getElementById('downloadBtn').disabled = false;
            } catch (e) {
                document.getElementById('contentArea').innerHTML = '<div class="empty-state">Error loading session</div>';
            }
        }

        function toggleSection(el) {
            el.closest('.section').classList.toggle('collapsed');
            const toggle = el.querySelector('.toggle');
            if (toggle) toggle.textContent = el.closest('.section').classList.contains('collapsed') ? '▶' : '▼';
        }

        function copyText(text) {
            navigator.clipboard.writeText(text);
        }

        function copyPrompt(type) {
            const el = document.getElementById('prompt-' + type);
            if (el) navigator.clipboard.writeText(el.textContent);
        }

        function switchTab(tabGroup, tabName) {
            document.querySelectorAll(`[data-tab-group="${tabGroup}"] .tab`).forEach(t => t.classList.remove('active'));
            document.querySelectorAll(`[data-tab-group="${tabGroup}"] .tab-content`).forEach(c => c.classList.remove('active'));
            document.querySelector(`[data-tab-group="${tabGroup}"] .tab[data-tab="${tabName}"]`).classList.add('active');
            document.querySelector(`[data-tab-group="${tabGroup}"] .tab-content[data-tab="${tabName}"]`).classList.add('active');
        }

        function renderSession(data) {
            const scores = data.scores || {};
            const images = data.images || {};
            const prompt = data.original_prompt || '';
            const improvedPrompt = data.improved_prompt || '';
            const promptAnalysis = data.prompt_analysis || {};
            const templateAnalysis = data.template_analysis || {};
            const templateSuggestions = data.template_suggestions || {};
            const imageAnalysis = data.image_analysis || {};
            const sourceMetadata = data.source_metadata || {};
            const genSettings = sourceMetadata.generation_settings || {};
            const templateInfo = data.template_info || genSettings.template_info || templateAnalysis.template_used || {};
            const evaluator = data.evaluator || {};
            const timing = data.timing || {};

            let html = '';

            // === IMAGES ===
            if (images.source || images.output) {
                html += '<div class="images-row">';
                if (images.source) html += `<div class="image-card"><h4>Source Image</h4><img src="${images.source}" alt="Source"></div>`;
                if (images.output) html += `<div class="image-card"><h4>Output Image</h4><img src="${images.output}" alt="Output"></div>`;
                html += '</div>';
            }

            // === SCORES ===
            if (Object.keys(scores).length > 0) {
                html += '<div class="section"><div class="section-header" onclick="toggleSection(this)"><span>Scores</span><span class="toggle">▼</span></div><div class="section-content"><div class="scores-grid">';
                for (const [key, val] of Object.entries(scores)) {
                    const score = val.score || val || 0;
                    const color = score >= 7 ? '#4caf50' : score >= 5 ? '#ff9800' : '#f44336';
                    const reason = val.reason || '';
                    html += `<div class="score-card"><h5>${key.replace(/_/g, ' ')}</h5><div class="value" style="color:${color}">${typeof score === 'number' ? score.toFixed(1) : score}</div>${reason ? `<div class="reason">${esc(reason)}</div>` : ''}</div>`;
                }
                html += '</div></div></div>';
            }

            // === PROMPTS (Tabs) ===
            html += `<div class="section" data-tab-group="prompts">
                <div class="section-header" onclick="toggleSection(this)"><span>Prompts</span><span class="toggle">▼</span></div>
                <div class="section-content">
                    <div class="tabs">
                        <button class="tab active" data-tab="original" onclick="switchTab('prompts','original')">Original</button>
                        <button class="tab" data-tab="improved" onclick="switchTab('prompts','improved')">Improved</button>
                    </div>
                    <div class="tab-content active" data-tab="original">
                        <div style="display:flex;justify-content:flex-end;margin-bottom:5px;"><button class="copy-btn" onclick="copyPrompt('original')">Copy</button></div>
                        <div class="prompt-box" id="prompt-original">${esc(prompt) || 'No prompt available'}</div>
                    </div>
                    <div class="tab-content" data-tab="improved">
                        <div style="display:flex;justify-content:flex-end;margin-bottom:5px;"><button class="copy-btn" onclick="copyPrompt('improved')">Copy</button></div>
                        <div class="prompt-box prompt-improved" id="prompt-improved">${esc(improvedPrompt) || 'No improved prompt available'}</div>
                    </div>
                </div>
            </div>`;

            // === PROMPT ANALYSIS ===
            const issues = promptAnalysis.issues || data.issues || [];
            const strengths = promptAnalysis.strengths || [];
            const missing = promptAnalysis.missing_elements || data.missing_elements || [];
            const hallucinated = promptAnalysis.hallucinated_elements || data.hallucinated_elements || [];
            const vocabIssues = promptAnalysis.vocabulary_issues || [];

            if (issues.length || strengths.length || missing.length || hallucinated.length || vocabIssues.length) {
                html += '<div class="section"><div class="section-header" onclick="toggleSection(this)"><span>Prompt Analysis</span><span class="toggle">▼</span></div><div class="section-content">';
                html += '<div class="two-col">';

                // Left column
                html += '<div>';
                if (strengths.length) {
                    html += '<h5 style="color:#81c784;font-size:11px;margin-bottom:8px;">STRENGTHS</h5><ul class="item-list">';
                    strengths.forEach(s => html += `<li class="tag-strength"><span class="icon">✓</span>${esc(s)}</li>`);
                    html += '</ul>';
                }
                if (issues.length) {
                    html += '<h5 style="color:#64b5f6;font-size:11px;margin:12px 0 8px;">ISSUES</h5><ul class="item-list">';
                    issues.forEach(s => html += `<li class="tag-issue"><span class="icon">⚡</span>${esc(s)}</li>`);
                    html += '</ul>';
                }
                html += '</div>';

                // Right column
                html += '<div>';
                if (missing.length) {
                    html += '<h5 style="color:#ef5350;font-size:11px;margin-bottom:8px;">MISSING ELEMENTS</h5><ul class="item-list">';
                    missing.forEach(s => html += `<li class="tag-missing"><span class="icon">✗</span>${esc(s)}</li>`);
                    html += '</ul>';
                }
                if (hallucinated.length) {
                    html += '<h5 style="color:#ffb74d;font-size:11px;margin:12px 0 8px;">HALLUCINATED</h5><ul class="item-list">';
                    hallucinated.forEach(s => html += `<li class="tag-hallucinated"><span class="icon">⚠</span>${esc(s)}</li>`);
                    html += '</ul>';
                }
                if (vocabIssues.length) {
                    html += '<h5 style="color:#ce93d8;font-size:11px;margin:12px 0 8px;">VOCABULARY ISSUES</h5><ul class="item-list">';
                    vocabIssues.forEach(s => html += `<li class="tag-suggestion"><span class="icon">📝</span>${esc(s)}</li>`);
                    html += '</ul>';
                }
                html += '</div></div></div></div>';
            }

            // === TEMPLATE INFO & SUGGESTIONS ===
            if (templateInfo.name || Object.keys(templateSuggestions).length) {
                html += '<div class="section"><div class="section-header" onclick="toggleSection(this)"><span>Template Analysis & Suggestions</span><span class="toggle">▼</span></div><div class="section-content">';

                // Template info
                if (templateInfo.name) {
                    html += `<div style="background:#1e1e1e;padding:10px;border-radius:4px;margin-bottom:12px;">
                        <table class="meta-table">
                            <tr><td>Template</td><td>${esc(templateInfo.name)}</td></tr>
                            ${templateInfo.path ? `<tr><td>Path</td><td>${esc(templateInfo.path)}</td></tr>` : ''}
                            ${templateInfo.category ? `<tr><td>Category</td><td>${esc(templateInfo.category)}</td></tr>` : ''}
                            ${templateInfo.model_size_used || templateInfo.model_size ? `<tr><td>Model Size</td><td>${esc(templateInfo.model_size_used || templateInfo.model_size)}</td></tr>` : ''}
                            ${templateInfo.auto_detected ? `<tr><td>Auto-Detected</td><td>${esc(templateInfo.auto_detected)}</td></tr>` : ''}
                        </table>
                    </div>`;
                }

                // Template effectiveness issues
                const effIssues = templateAnalysis.effectiveness_issues || [];
                const missingInstr = templateAnalysis.missing_instructions || [];
                if (effIssues.length || missingInstr.length) {
                    html += '<h5 style="color:#ff9800;font-size:11px;margin-bottom:8px;">TEMPLATE ISSUES</h5><ul class="item-list">';
                    effIssues.forEach(s => html += `<li class="tag-hallucinated"><span class="icon">⚠</span>${esc(s)}</li>`);
                    missingInstr.forEach(s => html += `<li class="tag-missing"><span class="icon">✗</span>${esc(s)}</li>`);
                    html += '</ul>';
                }

                // Priority additions
                const additions = templateSuggestions.priority_additions || [];
                if (additions.length) {
                    html += '<h5 style="color:#9c27b0;font-size:11px;margin:12px 0 8px;">PRIORITY ADDITIONS</h5>';
                    additions.forEach(item => {
                        const instr = typeof item === 'string' ? item : item.instruction || JSON.stringify(item);
                        const reason = typeof item === 'object' ? item.reason : '';
                        html += `<div class="suggestion-card"><div class="instruction">+ ${esc(instr)}</div>${reason ? `<div class="reason">${esc(reason)}</div>` : ''}</div>`;
                    });
                }

                // Clarifications
                const clarifications = templateSuggestions.clarifications_needed || [];
                if (clarifications.length) {
                    html += '<h5 style="color:#2196f3;font-size:11px;margin:12px 0 8px;">CLARIFICATIONS NEEDED</h5>';
                    clarifications.forEach(item => {
                        if (typeof item === 'object') {
                            html += `<div class="suggestion-card"><div class="instruction">Current: ${esc(item.current || '')}</div><div class="instruction" style="color:#81c784;">Suggested: ${esc(item.suggested || '')}</div>${item.reason ? `<div class="reason">${esc(item.reason)}</div>` : ''}</div>`;
                        } else {
                            html += `<div class="suggestion-card"><div class="instruction">${esc(item)}</div></div>`;
                        }
                    });
                }

                // Structure improvements
                if (templateSuggestions.structure_improvements) {
                    html += `<h5 style="color:#4caf50;font-size:11px;margin:12px 0 8px;">STRUCTURE IMPROVEMENTS</h5><div style="font-size:12px;color:#aaa;line-height:1.5;">${esc(templateSuggestions.structure_improvements)}</div>`;
                }

                // Model size notes
                if (templateSuggestions.model_size_notes) {
                    html += `<h5 style="color:#ff5722;font-size:11px;margin:12px 0 8px;">MODEL SIZE NOTES</h5><div style="font-size:12px;color:#aaa;line-height:1.5;">${esc(templateSuggestions.model_size_notes)}</div>`;
                }

                html += '</div></div>';
            }

            // === IMAGE ANALYSIS (from Phase 1) ===
            const sourceAnalysis = imageAnalysis.source_analysis || {};
            const outputAnalysis = imageAnalysis.output_analysis || {};
            const comparison = imageAnalysis.comparison || {};
            if (Object.keys(sourceAnalysis).length || Object.keys(comparison).length) {
                html += '<div class="section collapsed"><div class="section-header" onclick="toggleSection(this)"><span>Image Analysis (Detailed)</span><span class="toggle">▶</span></div><div class="section-content">';
                html += '<div class="two-col">';

                // Source analysis
                if (Object.keys(sourceAnalysis).length) {
                    html += '<div><h5 style="font-size:11px;margin-bottom:8px;">SOURCE IMAGE</h5><table class="meta-table">';
                    for (const [k, v] of Object.entries(sourceAnalysis)) {
                        const val = Array.isArray(v) ? v.join(', ') : (typeof v === 'object' ? JSON.stringify(v) : v);
                        html += `<tr><td>${k.replace(/_/g, ' ')}</td><td>${esc(String(val))}</td></tr>`;
                    }
                    html += '</table></div>';
                }

                // Comparison
                if (Object.keys(comparison).length) {
                    html += '<div><h5 style="font-size:11px;margin-bottom:8px;">COMPARISON</h5><table class="meta-table">';
                    for (const [k, v] of Object.entries(comparison)) {
                        const val = Array.isArray(v) ? v.join(', ') : (typeof v === 'object' ? JSON.stringify(v) : v);
                        html += `<tr><td>${k.replace(/_/g, ' ')}</td><td>${esc(String(val))}</td></tr>`;
                    }
                    html += '</table></div>';
                }

                html += '</div></div></div>';
            }

            // === GENERATION METADATA ===
            const genModel = data.generation_model || sourceMetadata.model_config || {};
            const cvAnalysis = sourceMetadata.cv_analysis || {};
            const genTiming = sourceMetadata.timing || {};

            html += '<div class="section collapsed"><div class="section-header" onclick="toggleSection(this)"><span>Generation Metadata</span><span class="toggle">▶</span></div><div class="section-content">';
            html += '<div class="two-col">';

            // Generation settings
            html += '<div><h5 style="font-size:11px;margin-bottom:8px;">GENERATION SETTINGS</h5><table class="meta-table">';
            if (genModel.provider) html += `<tr><td>Provider</td><td>${esc(genModel.provider)}</td></tr>`;
            if (genModel.model) html += `<tr><td>Model</td><td>${esc(genModel.model)}</td></tr>`;
            if (genSettings.analysis_mode) html += `<tr><td>Mode</td><td>${esc(genSettings.analysis_mode)}</td></tr>`;
            if (genSettings.model_size) html += `<tr><td>Model Size</td><td>${esc(genSettings.model_size)}</td></tr>`;
            if (genModel.temperature) html += `<tr><td>Temperature</td><td>${genModel.temperature}</td></tr>`;
            if (genSettings.prompt_length) html += `<tr><td>Prompt Length</td><td>${genSettings.prompt_length}</td></tr>`;
            if (genSettings.seed) html += `<tr><td>Seed</td><td>${genSettings.seed}</td></tr>`;
            html += '</table></div>';

            // CV Analysis
            if (Object.keys(cvAnalysis).length) {
                html += '<div><h5 style="font-size:11px;margin-bottom:8px;">CV ANALYSIS</h5><table class="meta-table">';
                for (const [k, v] of Object.entries(cvAnalysis)) {
                    const val = typeof v === 'object' ? JSON.stringify(v) : String(v);
                    html += `<tr><td>${k.replace(/_/g, ' ')}</td><td>${esc(val)}</td></tr>`;
                }
                html += '</table></div>';
            }

            html += '</div></div></div>';

            // === EVALUATION METADATA ===
            html += '<div class="section collapsed"><div class="section-header" onclick="toggleSection(this)"><span>Evaluation Metadata</span><span class="toggle">▶</span></div><div class="section-content">';
            html += '<table class="meta-table">';
            if (evaluator.provider) html += `<tr><td>Provider</td><td>${esc(evaluator.provider)}</td></tr>`;
            if (evaluator.model) html += `<tr><td>Model</td><td>${esc(evaluator.model)}</td></tr>`;
            if (evaluator.reasoning_enabled !== undefined) html += `<tr><td>Reasoning</td><td>${evaluator.reasoning_enabled ? 'Enabled' : 'Disabled'}</td></tr>`;
            if (data.api_calls) html += `<tr><td>API Calls</td><td>${data.api_calls}</td></tr>`;
            if (timing.total_seconds) html += `<tr><td>Total Time</td><td>${timing.total_seconds.toFixed(2)}s</td></tr>`;
            html += '</table></div></div>';

            // === RAW JSON ===
            html += '<div class="section collapsed"><div class="section-header" onclick="toggleSection(this)"><span>Raw JSON Data</span><span class="toggle">▶</span></div><div class="section-content">';
            html += `<div class="json-viewer">${esc(JSON.stringify(data, null, 2))}</div>`;
            html += '</div></div>';

            document.getElementById('contentArea').innerHTML = html;
        }

        function esc(text) {
            if (!text) return '';
            const div = document.createElement('div');
            div.textContent = String(text);
            return div.innerHTML;
        }

        async function downloadPDF() {
            if (!currentSession || !currentSessionId) return;
            const btn = document.getElementById('downloadBtn');
            btn.textContent = 'Generating...';
            btn.disabled = true;
            try {
                // Expand all sections for PDF
                document.querySelectorAll('.section.collapsed').forEach(s => s.classList.remove('collapsed'));
                const content = document.getElementById('contentArea');
                const opt = {
                    margin: 10,
                    filename: `debug_${currentSessionId}.pdf`,
                    image: { type: 'jpeg', quality: 0.95 },
                    html2canvas: { scale: 2, useCORS: true, backgroundColor: '#1e1e1e' },
                    jsPDF: { unit: 'mm', format: 'a4', orientation: 'portrait' }
                };
                await html2pdf().set(opt).from(content).save();
                btn.textContent = 'Download PDF';
                btn.disabled = false;
            } catch (e) {
                console.error('PDF failed:', e);
                alert('PDF generation failed: ' + e.message);
                btn.textContent = 'Download PDF';
                btn.disabled = false;
            }
        }
    </script>
</body>
</html>'''


def get_generation_viewer_html():
    """Return the HTML for the generation results viewer UI."""
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SID Generation Results</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1e1e1e;
            color: #d4d4d4;
            height: 100vh;
            overflow: hidden;
        }
        .container { display: flex; height: 100vh; }
        .sidebar {
            width: 350px;
            background: #252526;
            border-right: 1px solid #3c3c3c;
            display: flex;
            flex-direction: column;
        }
        .sidebar-header {
            padding: 15px;
            background: #333;
            border-bottom: 1px solid #3c3c3c;
            font-weight: 600;
            font-size: 14px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .nav-links {
            border-top: 1px solid #3c3c3c;
            padding: 10px 0;
            background: #2d2d2d;
        }
        .nav-link {
            display: block;
            padding: 8px 15px;
            color: #888;
            text-decoration: none;
            font-size: 12px;
            transition: all 0.2s;
        }
        .nav-link:hover {
            background: #3c3c3c;
            color: #fff;
        }
        .nav-link.active {
            color: #4a9eff;
            background: #1e1e1e;
        }
        .btn-refresh {
            padding: 4px 10px;
            background: #0e639c;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
        }
        .btn-refresh:hover { background: #1177bb; }
        .session-list {
            flex: 1;
            overflow-y: auto;
            padding: 10px 0;
        }
        .session-item {
            padding: 12px 15px;
            cursor: pointer;
            font-size: 12px;
            border-bottom: 1px solid #3c3c3c;
        }
        .session-item:hover { background: #2a2d2e; }
        .session-item.active { background: #37373d; }
        .session-item.excluded { opacity: 0.5; }
        .session-item.excluded .session-preview { text-decoration: line-through; }
        .session-header { display: flex; align-items: center; gap: 8px; margin-bottom: 4px; }
        .session-content { cursor: pointer; }
        .exclude-checkbox {
            width: 16px;
            height: 16px;
            min-width: 16px;
            cursor: pointer;
            accent-color: #b89500;
            margin: 0;
            flex-shrink: 0;
        }
        .session-time { font-weight: 600; color: #fff; flex: 1; }
        .session-meta { color: #888; font-size: 11px; margin-bottom: 4px; }
        .session-preview { color: #aaa; font-size: 11px; font-style: italic; }
        .session-badge {
            display: inline-block;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 10px;
            margin-right: 5px;
        }
        .badge-style { background: #0e639c; color: #fff; }
        .badge-time { background: #16825d; color: #fff; }
        .main-content {
            flex: 1;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        .toolbar {
            padding: 10px 15px;
            background: #333;
            border-bottom: 1px solid #3c3c3c;
            font-size: 13px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .toolbar-buttons { display: flex; gap: 8px; }
        .btn-delete {
            padding: 4px 10px;
            background: #c42b1c;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
        }
        .btn-delete:hover { background: #e03e2e; }
        .btn-delete:disabled { background: #3c3c3c; color: #888; cursor: not-allowed; }
        .btn-copy {
            padding: 4px 10px;
            background: #16825d;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
        }
        .btn-copy:hover { background: #1a9e6e; }
        .content-area {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
        }
        .empty-state {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 100%;
            color: #888;
            gap: 10px;
        }
        .detail-grid {
            display: grid;
            grid-template-columns: 300px 1fr;
            gap: 20px;
        }
        .image-card {
            background: #252526;
            border-radius: 8px;
            overflow: hidden;
        }
        .image-card h3 {
            padding: 10px 15px;
            background: #333;
            font-size: 13px;
            font-weight: 600;
        }
        .image-card img {
            width: 100%;
            height: auto;
            display: block;
        }
        .prompt-section {
            background: #252526;
            border-radius: 8px;
            margin-bottom: 15px;
        }
        .prompt-section h3 {
            padding: 10px 15px;
            background: #333;
            font-size: 13px;
            font-weight: 600;
            border-radius: 8px 8px 0 0;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .prompt-content {
            padding: 15px;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 13px;
            line-height: 1.6;
            white-space: pre-wrap;
            word-break: break-word;
        }
        .metadata-section {
            background: #252526;
            border-radius: 8px;
        }
        .metadata-section h3 {
            padding: 10px 15px;
            background: #333;
            font-size: 13px;
            font-weight: 600;
            border-radius: 8px 8px 0 0;
        }
        .metadata-grid {
            padding: 15px;
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
        }
        .metadata-item {
            font-size: 12px;
        }
        .metadata-item label {
            color: #888;
            display: block;
            margin-bottom: 2px;
        }
        .metadata-item span {
            color: #d4d4d4;
        }
        .loading {
            display: flex;
            align-items: center;
            justify-content: center;
            height: 100%;
            color: #888;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="sidebar">
            <div class="sidebar-header">
                <span>Generation Results</span>
                <button class="btn-refresh" onclick="loadSessions()">Refresh</button>
            </div>
            <div class="session-list" id="sessionList">
                <div class="loading">Loading...</div>
            </div>
            <div class="nav-links">
                <a href="/sid" class="nav-link">🏠 Home</a>
                <a href="/sid/template-editor" class="nav-link">🎨 Template Editor</a>
                <a href="/sid/model-editor" class="nav-link">🤖 Model Editor</a>
                <a href="/sid/generation-results" class="nav-link active">🖼️ Generation Results</a>
                <a href="/sid/debug-results" class="nav-link">🔍 Debug Viewer</a>
            </div>
        </div>
        <div class="main-content">
            <div class="toolbar">
                <span id="toolbar">Select a session to view</span>
                <div class="toolbar-buttons">
                    <button class="btn-copy" id="copyBtn" disabled onclick="copyPrompt()">Copy Prompt</button>
                    <button class="btn-delete" id="deleteBtn" disabled onclick="deleteSession()">Delete</button>
                </div>
            </div>
            <div class="content-area" id="contentArea">
                <div class="empty-state">
                    <div style="font-size: 48px;">📝</div>
                    <div>Select a generation session from the sidebar</div>
                    <div style="font-size: 12px; color: #666;">Enable "Store Results" in the Prompt Generator node to save results here</div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let currentSessionId = null;
        let currentPrompt = '';

        loadSessions();

        async function loadSessions() {
            try {
                const res = await fetch('/sid/generation-results/api/sessions');
                const data = await res.json();

                const list = document.getElementById('sessionList');
                if (data.sessions.length === 0) {
                    list.innerHTML = '<div class="empty-state" style="padding: 20px;">No results yet.<br><br>Enable "Store Results" in the Prompt Generator node.</div>';
                    return;
                }

                list.innerHTML = data.sessions.map(s => {
                    let dateStr = 'Unknown';
                    if (s.timestamp) {
                        const d = new Date(s.timestamp);
                        const today = new Date();
                        if (d.toDateString() === today.toDateString()) {
                            dateStr = d.toLocaleTimeString();
                        } else {
                            dateStr = d.toLocaleDateString() + ' ' + d.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
                        }
                    }
                    const imgIcon = s.has_image ? '🖼️ ' : '';
                    const styleText = s.template || s.prompt_style || '';
                    const excludedClass = s.excluded ? 'excluded' : '';
                    const excludedChecked = s.excluded ? 'checked' : '';
                    return `
                        <div class="session-item ${excludedClass}" data-id="${s.id}">
                            <div class="session-header">
                                <input type="checkbox" class="exclude-checkbox" ${excludedChecked}
                                    onclick="toggleExclude(event, '${s.id}')"
                                    title="Exclude from learning">
                                <div class="session-time" onclick="loadSession('${s.id}')">${imgIcon}${dateStr}</div>
                            </div>
                            <div class="session-content" onclick="loadSession('${s.id}')">
                                <div class="session-meta">
                                    <span class="session-badge badge-style">${styleText}</span>
                                    <span class="session-badge badge-time">${s.timing.toFixed(1)}s</span>
                                    ${s.provider}/${s.model.split('/').pop()}
                                </div>
                                <div class="session-preview">${escapeHtml(s.prompt_preview)}</div>
                            </div>
                        </div>
                    `;
                }).join('');
            } catch (e) {
                document.getElementById('sessionList').innerHTML = '<div class="empty-state">Error loading sessions</div>';
            }
        }

        async function loadSession(sessionId) {
            document.querySelectorAll('.session-item').forEach(el => {
                el.classList.toggle('active', el.dataset.id === sessionId);
            });

            currentSessionId = sessionId;
            document.getElementById('toolbar').textContent = `Session: ${sessionId}`;
            document.getElementById('contentArea').innerHTML = '<div class="loading">Loading...</div>';
            document.getElementById('deleteBtn').disabled = true;
            document.getElementById('copyBtn').disabled = true;

            try {
                const res = await fetch(`/sid/generation-results/api/session/${sessionId}`);
                const data = await res.json();

                if (data.error) {
                    document.getElementById('contentArea').innerHTML = `<div class="empty-state">${data.error}</div>`;
                    return;
                }

                currentPrompt = data.prompt || '';
                renderSession(data);
                document.getElementById('deleteBtn').disabled = false;
                document.getElementById('copyBtn').disabled = !currentPrompt;
            } catch (e) {
                document.getElementById('contentArea').innerHTML = '<div class="empty-state">Error loading session</div>';
            }
        }

        async function toggleExclude(event, sessionId) {
            event.stopPropagation(); // Don't trigger loadSession
            const checkbox = event.target;
            const sessionItem = checkbox.closest('.session-item');

            try {
                const res = await fetch(`/sid/generation-results/api/session/${sessionId}/exclude`, {
                    method: 'POST'
                });
                const data = await res.json();

                if (data.success) {
                    checkbox.checked = data.excluded;
                    sessionItem.classList.toggle('excluded', data.excluded);
                } else {
                    // Revert checkbox on error
                    checkbox.checked = !checkbox.checked;
                }
            } catch (e) {
                // Revert checkbox on error
                checkbox.checked = !checkbox.checked;
                console.error('Failed to toggle exclude:', e);
            }
        }

        function renderSession(data) {
            const meta = data.metadata || {};
            const modelConfig = meta.model_config || {};

            let html = '<div class="detail-grid">';

            // Left column - Image
            html += '<div>';
            if (data.image_url) {
                html += `
                    <div class="image-card">
                        <h3>Source Image</h3>
                        <img src="${data.image_url}" alt="Source">
                    </div>
                `;
            }

            // Metadata below image
            html += `
                <div class="metadata-section" style="margin-top: 15px;">
                    <h3>Settings</h3>
                    <div class="metadata-grid">
                        <div class="metadata-item"><label>Style</label><span>${meta.prompt_style || '-'}</span></div>
                        <div class="metadata-item"><label>Template</label><span>${meta.template || '-'}</span></div>
                        <div class="metadata-item"><label>Provider</label><span>${modelConfig.provider || '-'}</span></div>
                        <div class="metadata-item"><label>Model</label><span>${modelConfig.model || '-'}</span></div>
                        <div class="metadata-item"><label>Mode</label><span>${modelConfig.analysis_mode || '-'}</span></div>
                        <div class="metadata-item"><label>Temperature</label><span>${modelConfig.temperature || '-'}</span></div>
                        <div class="metadata-item"><label>Seed</label><span>${meta.seed || '-'}</span></div>
                        <div class="metadata-item"><label>Length</label><span>${meta.prompt_length || '-'}</span></div>
                        <div class="metadata-item"><label>NSFW</label><span>${meta.nsfw_mode ? 'Yes' : 'No'}</span></div>
                        <div class="metadata-item"><label>Time</label><span>${meta.timing?.total_seconds?.toFixed(2) || '-'}s</span></div>
                    </div>
                </div>
            `;
            html += '</div>';

            // Right column - Prompts
            html += '<div>';
            if (data.prompt) {
                html += `
                    <div class="prompt-section">
                        <h3>Generated Prompt <button class="btn-copy" style="padding: 2px 8px; font-size: 11px;" onclick="copyPrompt()">Copy</button></h3>
                        <div class="prompt-content">${escapeHtml(data.prompt)}</div>
                    </div>
                `;
            }
            if (data.negative) {
                html += `
                    <div class="prompt-section">
                        <h3>Negative Prompt</h3>
                        <div class="prompt-content">${escapeHtml(data.negative)}</div>
                    </div>
                `;
            }
            if (data.caption) {
                html += `
                    <div class="prompt-section">
                        <h3>Caption</h3>
                        <div class="prompt-content">${escapeHtml(data.caption)}</div>
                    </div>
                `;
            }
            if (meta.prompt_enhance) {
                html += `
                    <div class="prompt-section">
                        <h3>Enhancement Keywords</h3>
                        <div class="prompt-content">${escapeHtml(meta.prompt_enhance)}</div>
                    </div>
                `;
            }
            html += '</div></div>';

            document.getElementById('contentArea').innerHTML = html;
        }

        function escapeHtml(text) {
            if (!text) return '';
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        async function copyPrompt() {
            if (!currentPrompt) return;
            try {
                await navigator.clipboard.writeText(currentPrompt);
                const btn = document.getElementById('copyBtn');
                btn.textContent = 'Copied!';
                setTimeout(() => { btn.textContent = 'Copy Prompt'; }, 1500);
            } catch (e) {
                alert('Failed to copy: ' + e.message);
            }
        }

        async function deleteSession() {
            if (!currentSessionId) return;
            if (!confirm('Delete this session? This cannot be undone.')) return;

            try {
                const res = await fetch(`/sid/generation-results/api/session/${currentSessionId}`, {
                    method: 'DELETE'
                });
                const data = await res.json();

                if (data.success) {
                    currentSessionId = null;
                    currentPrompt = '';
                    document.getElementById('toolbar').textContent = 'Select a session to view';
                    document.getElementById('contentArea').innerHTML = '<div class="empty-state"><div style="font-size: 48px;">📝</div><div>Session deleted</div></div>';
                    document.getElementById('deleteBtn').disabled = true;
                    document.getElementById('copyBtn').disabled = true;
                    loadSessions();
                } else {
                    alert('Error: ' + (data.error || 'Unknown error'));
                }
            } catch (e) {
                alert('Error: ' + e.message);
            }
        }
    </script>
</body>
</html>'''


def get_landing_html():
    """Generate the SID Toolkit landing page HTML."""
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SID Photography Toolkit</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #e4e4e4;
        }
        .container {
            text-align: center;
            padding: 40px;
        }
        h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            color: #4a9eff;
        }
        .subtitle {
            color: #888;
            margin-bottom: 50px;
            font-size: 1.1em;
        }
        .cards {
            display: flex;
            gap: 30px;
            justify-content: center;
            flex-wrap: wrap;
        }
        .card {
            background: #16213e;
            border: 1px solid #2a2a4a;
            border-radius: 12px;
            padding: 30px;
            width: 280px;
            text-decoration: none;
            color: inherit;
            transition: all 0.3s ease;
        }
        .card:hover {
            transform: translateY(-5px);
            border-color: #4a9eff;
            box-shadow: 0 10px 40px rgba(74, 158, 255, 0.2);
        }
        .card-icon {
            font-size: 48px;
            margin-bottom: 15px;
        }
        .card-title {
            font-size: 1.3em;
            font-weight: 600;
            margin-bottom: 10px;
            color: #4a9eff;
        }
        .card-desc {
            color: #888;
            font-size: 0.95em;
            line-height: 1.5;
        }
        .footer {
            margin-top: 50px;
            color: #555;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>SID Photography Toolkit</h1>
        <p class="subtitle">AI-powered prompt generation for ComfyUI</p>

        <div class="cards">
            <a href="/sid/template-editor" class="card">
                <div class="card-icon">🎨</div>
                <div class="card-title">Template Editor</div>
                <div class="card-desc">Create and manage prompt templates for Simple/Template modes</div>
            </a>

            <a href="/sid/model-editor" class="card">
                <div class="card-icon">🤖</div>
                <div class="card-title">Model Editor</div>
                <div class="card-desc">View and configure API and local LLM models</div>
            </a>

            <a href="/sid/generation-results" class="card">
                <div class="card-icon">🖼️</div>
                <div class="card-title">Generation Results</div>
                <div class="card-desc">Browse saved prompts, metadata, and source images</div>
            </a>

            <a href="/sid/debug-results" class="card">
                <div class="card-icon">🔍</div>
                <div class="card-title">Debug Viewer</div>
                <div class="card-desc">Analyze prompt quality scores and evaluation results</div>
            </a>
        </div>

        <div class="footer">
            SID Photography Toolkit v4.3.0 | <a href="https://github.com/slahiri/ComfyUI-AI-Photography-Toolkit" style="color: #4a9eff;">GitHub</a>
        </div>
    </div>
</body>
</html>'''


def get_template_editor_html():
    """Return the HTML for the template editor UI with model size tabs."""
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SID Template Editor</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1e1e1e;
            color: #d4d4d4;
            height: 100vh;
            overflow: hidden;
        }
        .container { display: flex; height: 100vh; }
        .sidebar {
            width: 280px;
            background: #252526;
            border-right: 1px solid #3c3c3c;
            display: flex;
            flex-direction: column;
        }
        .sidebar-header {
            padding: 15px;
            background: #333;
            border-bottom: 1px solid #3c3c3c;
            font-weight: 600;
            font-size: 14px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .sidebar-actions { display: flex; gap: 5px; }
        .btn-small {
            padding: 4px 8px;
            font-size: 11px;
            border: none;
            border-radius: 3px;
            cursor: pointer;
        }
        .btn-new { background: #16825d; color: white; }
        .btn-new:hover { background: #1a9e6e; }
        .btn-sync { background: #0e639c; color: white; }
        .btn-sync:hover { background: #1177bb; }
        .nav-links {
            border-top: 1px solid #3c3c3c;
            padding: 10px 0;
            background: #2d2d2d;
        }
        .nav-link {
            display: block;
            padding: 8px 15px;
            color: #888;
            text-decoration: none;
            font-size: 12px;
        }
        .nav-link:hover { background: #3c3c3c; color: #fff; }
        .nav-link.active { color: #4a9eff; background: #1e1e1e; }
        .template-list { flex: 1; overflow-y: auto; padding: 10px 0; }
        .category-header {
            padding: 8px 15px;
            font-size: 11px;
            font-weight: 600;
            color: #888;
            text-transform: uppercase;
            background: #2a2a2a;
            border-bottom: 1px solid #3c3c3c;
        }
        .template-item {
            padding: 10px 15px;
            cursor: pointer;
            font-size: 13px;
            border-bottom: 1px solid #333;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .template-item:hover { background: #2a2d2e; }
        .template-item.active { background: #37373d; color: #fff; }
        .template-name { flex: 1; }
        .template-badge { padding: 2px 6px; border-radius: 3px; font-size: 10px; font-weight: 600; }
        .badge-core { background: #0e639c; color: #fff; }
        .badge-community { background: #7c3aed; color: #fff; }
        .badge-custom { background: #16825d; color: #fff; }
        .badge-readonly { background: #6b7280; color: #fff; }
        .editor-container { flex: 1; display: flex; flex-direction: column; }
        .toolbar {
            padding: 10px 15px;
            background: #333;
            border-bottom: 1px solid #3c3c3c;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .toolbar-title { flex: 1; font-size: 13px; color: #888; }
        .toolbar-title strong { color: #d4d4d4; }
        .btn {
            padding: 6px 14px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
            font-weight: 500;
        }
        .btn-primary { background: #0e639c; color: white; }
        .btn-primary:hover { background: #1177bb; }
        .btn-primary:disabled { background: #3c3c3c; color: #888; cursor: not-allowed; }
        .btn-secondary { background: #3c3c3c; color: #d4d4d4; }
        .btn-secondary:hover { background: #4c4c4c; }
        .btn-danger { background: #c42b1c; color: white; }
        .btn-danger:hover { background: #e03e2e; }
        .btn-danger:disabled { background: #3c3c3c; color: #888; cursor: not-allowed; }
        .editor-content { flex: 1; display: flex; flex-direction: column; overflow: hidden; padding: 20px; }
        .form-group { margin-bottom: 15px; }
        .form-group label { display: block; font-size: 12px; color: #888; margin-bottom: 5px; }
        .form-group input, .form-group textarea {
            width: 100%;
            padding: 8px 12px;
            background: #2d2d2d;
            border: 1px solid #3c3c3c;
            border-radius: 4px;
            color: #d4d4d4;
            font-family: inherit;
            font-size: 13px;
        }
        .form-group input:focus, .form-group textarea:focus { outline: none; border-color: #0e639c; }
        .form-group textarea { resize: vertical; font-family: 'Consolas', 'Monaco', monospace; line-height: 1.5; }
        .system-prompt-editor { min-height: 200px; flex: 1; }
        .user-prompt-editor { min-height: 100px; flex: 1; }
        .form-row { display: flex; gap: 15px; }
        .form-row .form-group { flex: 1; }
        .status-bar {
            padding: 5px 15px;
            background: #007acc;
            font-size: 12px;
            display: flex;
            justify-content: space-between;
        }
        .status-bar.error { background: #c42b1c; }
        .status-bar.success { background: #16825d; }
        .empty-state {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 100%;
            color: #888;
            gap: 10px;
        }
        .empty-state .icon { font-size: 48px; }
        /* Size tabs */
        .size-tabs {
            display: flex;
            gap: 0;
            border-bottom: 1px solid #3c3c3c;
            margin-bottom: 15px;
        }
        .size-tab {
            padding: 10px 20px;
            background: #2d2d2d;
            border: 1px solid #3c3c3c;
            border-bottom: none;
            cursor: pointer;
            font-size: 12px;
            color: #888;
            margin-right: -1px;
        }
        .size-tab:first-child { border-radius: 4px 0 0 0; }
        .size-tab:last-child { border-radius: 0 4px 0 0; }
        .size-tab.active { background: #1e1e1e; color: #4a9eff; border-bottom: 1px solid #1e1e1e; }
        .size-tab:hover:not(.active) { background: #3c3c3c; }
        .size-panel { display: none; flex: 1; flex-direction: column; gap: 12px; }
        .size-panel.active { display: flex; }
        .prompt-container { flex: 1; display: flex; flex-direction: column; }
        .prompt-container textarea { flex: 1; min-height: 200px; }
        /* View Toggle */
        .view-toggle {
            display: flex;
            gap: 0;
            background: #2d2d2d;
            border-radius: 4px;
            overflow: hidden;
            margin-right: 10px;
        }
        .view-toggle-btn {
            padding: 6px 12px;
            border: none;
            background: transparent;
            color: #888;
            cursor: pointer;
            font-size: 12px;
            font-weight: 500;
        }
        .view-toggle-btn:hover { background: #3c3c3c; color: #d4d4d4; }
        .view-toggle-btn.active { background: #0e639c; color: white; }
        /* TOML Editor */
        .toml-editor {
            flex: 1;
            display: none;
            flex-direction: column;
        }
        .toml-editor.active { display: flex; }
        .toml-editor textarea {
            flex: 1;
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
            font-size: 13px;
            line-height: 1.5;
            padding: 15px;
            background: #1e1e1e;
            border: 1px solid #3c3c3c;
            border-radius: 4px;
            color: #d4d4d4;
            resize: none;
        }
        .toml-editor textarea:focus { outline: none; border-color: #0e639c; }
        .form-editor { display: flex; flex-direction: column; flex: 1; }
        .form-editor.hidden { display: none; }
        /* Prompt stats */
        .prompt-stats {
            display: flex;
            gap: 15px;
            padding: 6px 12px;
            background: #1e1e1e;
            border: 1px solid #3c3c3c;
            border-top: none;
            border-radius: 0 0 4px 4px;
            font-size: 11px;
            color: #888;
        }
        .prompt-stats .stat {
            display: flex;
            align-items: center;
            gap: 4px;
        }
        .prompt-stats .stat-value {
            color: #4a9eff;
            font-weight: 600;
            font-family: 'Consolas', monospace;
        }
        .prompt-stats .stat-warn { color: #ff9800; }
        .prompt-stats .stat-danger { color: #f44336; }
        /* Modal */
        .modal-overlay {
            display: none;
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(0,0,0,0.7);
            align-items: center;
            justify-content: center;
            z-index: 1000;
        }
        .modal-overlay.show { display: flex; }
        .modal {
            background: #252526;
            border-radius: 8px;
            width: 700px;
            max-width: 90vw;
            max-height: 90vh;
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }
        .modal-header {
            padding: 15px 20px;
            background: #333;
            font-weight: 600;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .modal-close { background: none; border: none; color: #888; font-size: 20px; cursor: pointer; }
        .modal-close:hover { color: #fff; }
        .modal-body { padding: 20px; overflow-y: auto; flex: 1; }
        .modal-footer {
            padding: 15px 20px;
            background: #2d2d2d;
            display: flex;
            justify-content: flex-end;
            gap: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="sidebar">
            <div class="sidebar-header">
                <span>Templates</span>
                <div class="sidebar-actions">
                    <button class="btn-small btn-new" onclick="showNewModal()" title="Create new template">+ New</button>
                    <button class="btn-small btn-sync" onclick="syncTemplates()" title="Sync from GitHub">Sync</button>
                </div>
            </div>
            <div class="template-list" id="templateList">
                <div class="empty-state" style="padding: 20px;">Loading...</div>
            </div>
            <div class="nav-links">
                <a href="/sid" class="nav-link">Home</a>
                <a href="/sid/template-editor" class="nav-link active">Template Editor</a>
                <a href="/sid/model-editor" class="nav-link">Model Editor</a>
                <a href="/sid/generation-results" class="nav-link">Generation Results</a>
                <a href="/sid/debug-results" class="nav-link">Debug Viewer</a>
            </div>
        </div>
        <div class="editor-container">
            <div class="toolbar">
                <div class="toolbar-title" id="toolbarTitle">Select a template to view or edit</div>
                <div class="view-toggle" id="viewToggle" style="display: none;">
                    <button class="view-toggle-btn active" id="formViewBtn" onclick="switchView('form')">Editor</button>
                    <button class="view-toggle-btn" id="tomlViewBtn" onclick="switchView('toml')">TOML</button>
                </div>
                <div class="toolbar-guides" style="display: flex; gap: 4px; margin-right: 8px;">
                    <button class="btn btn-secondary" onclick="openGuide('zimage')" title="Z-Image Prompting Guide">Z-Image Guide</button>
                    <button class="btn btn-secondary" onclick="openGuide('qwen')" title="Qwen VL Prompting Guide">Qwen VL Guide</button>
                </div>
                <button class="btn btn-secondary" id="duplicateBtn" disabled onclick="duplicateTemplate()">Duplicate</button>
                <button class="btn btn-danger" id="deleteBtn" disabled onclick="deleteTemplate()">Delete</button>
                <button class="btn btn-primary" id="saveBtn" disabled onclick="saveTemplate()">Save</button>
            </div>
            <div class="editor-content" id="editorContent">
                <div class="empty-state">
                    <div class="icon">🎨</div>
                    <div>Select a template from the sidebar</div>
                    <div style="font-size: 12px; color: #666;">Or create a new custom template</div>
                </div>
            </div>
            <div class="status-bar" id="statusBar">
                <span id="statusText">Ready</span>
                <span id="templateInfo"></span>
            </div>
        </div>
    </div>

    <!-- New Template Modal with Size Tabs -->
    <div class="modal-overlay" id="newModal">
        <div class="modal">
            <div class="modal-header">
                <span>Create New Template</span>
                <button class="modal-close" onclick="hideNewModal()">&times;</button>
            </div>
            <div class="modal-body">
                <div class="form-group">
                    <label>Template Name *</label>
                    <input type="text" id="newName" placeholder="My Custom Template">
                </div>
                <div class="form-row">
                    <div class="form-group">
                        <label>Description</label>
                        <input type="text" id="newDescription" placeholder="Brief description">
                    </div>
                    <div class="form-group">
                        <label>Tags (comma-separated)</label>
                        <input type="text" id="newTags" placeholder="portrait, custom">
                    </div>
                </div>
                <div class="size-tabs">
                    <div class="size-tab active" onclick="switchModalTab('small')">Small (2B-6B)</div>
                    <div class="size-tab" onclick="switchModalTab('medium')">Medium (7B-13B)</div>
                    <div class="size-tab" onclick="switchModalTab('large')">Large (13B+)</div>
                </div>
                <div id="modalPromptSmall" class="size-panel active">
                    <div class="form-group" style="flex: 1; display: flex; flex-direction: column;">
                        <label>System Prompt - Small Models</label>
                        <textarea id="newPromptSmall" class="system-prompt-editor" style="border-radius: 4px 4px 0 0;" placeholder="Concise instructions for small models..." oninput="updateModalStats('Small')"></textarea>
                        <div class="prompt-stats">
                            <div class="stat"><span>Words:</span><span class="stat-value" id="newWordsSmall">0</span></div>
                            <div class="stat"><span>Chars:</span><span class="stat-value" id="newCharsSmall">0</span></div>
                            <div class="stat"><span>Tokens:</span><span class="stat-value" id="newTokensSmall">0</span><span style="color:#666;">(est.)</span></div>
                        </div>
                    </div>
                    <div class="form-group" style="flex: 1; display: flex; flex-direction: column;">
                        <label>User Prompt - Small Models</label>
                        <textarea id="newUserSmall" class="user-prompt-editor" placeholder="Describe this image."></textarea>
                    </div>
                </div>
                <div id="modalPromptMedium" class="size-panel">
                    <div class="form-group" style="flex: 1; display: flex; flex-direction: column;">
                        <label>System Prompt - Medium Models</label>
                        <textarea id="newPromptMedium" class="system-prompt-editor" style="border-radius: 4px 4px 0 0;" placeholder="Balanced instructions for medium models..." oninput="updateModalStats('Medium')"></textarea>
                        <div class="prompt-stats">
                            <div class="stat"><span>Words:</span><span class="stat-value" id="newWordsMedium">0</span></div>
                            <div class="stat"><span>Chars:</span><span class="stat-value" id="newCharsSmall">0</span></div>
                            <div class="stat"><span>Tokens:</span><span class="stat-value" id="newTokensMedium">0</span><span style="color:#666;">(est.)</span></div>
                        </div>
                    </div>
                    <div class="form-group" style="flex: 1; display: flex; flex-direction: column;">
                        <label>User Prompt - Medium Models</label>
                        <textarea id="newUserMedium" class="user-prompt-editor" placeholder="Describe this image."></textarea>
                    </div>
                </div>
                <div id="modalPromptLarge" class="size-panel">
                    <div class="form-group" style="flex: 1; display: flex; flex-direction: column;">
                        <label>System Prompt - Large Models</label>
                        <textarea id="newPromptLarge" class="system-prompt-editor" style="border-radius: 4px 4px 0 0;" placeholder="Detailed instructions for large models..." oninput="updateModalStats('Large')"></textarea>
                        <div class="prompt-stats">
                            <div class="stat"><span>Words:</span><span class="stat-value" id="newWordsLarge">0</span></div>
                            <div class="stat"><span>Chars:</span><span class="stat-value" id="newCharsLarge">0</span></div>
                            <div class="stat"><span>Tokens:</span><span class="stat-value" id="newTokensLarge">0</span><span style="color:#666;">(est.)</span></div>
                        </div>
                    </div>
                    <div class="form-group" style="flex: 1; display: flex; flex-direction: column;">
                        <label>User Prompt - Large Models</label>
                        <textarea id="newUserLarge" class="user-prompt-editor" placeholder="Describe this image."></textarea>
                    </div>
                </div>
            </div>
            <div class="modal-footer">
                <button class="btn btn-secondary" onclick="hideNewModal()">Cancel</button>
                <button class="btn btn-primary" onclick="createTemplate()">Create</button>
            </div>
        </div>
    </div>

    <script>
        let templates = [];
        let currentTemplate = null;
        let isModified = false;

        // Load templates on page load
        loadTemplates();

        function openGuide(guide) {
            const guides = {
                'zimage': '/sid/docs/Z-IMAGE-PROMPTING-GUIDE.md',
                'qwen': '/sid/docs/QWEN-VL-PROMPTING-GUIDE.md'
            };
            const url = guides[guide];
            if (url) {
                window.open(url, '_blank');
            }
        }

        async function loadTemplates() {
            try {
                const res = await fetch('/sid/template-editor/api/templates');
                const data = await res.json();

                if (data.error) {
                    setStatus('Error: ' + data.error, 'error');
                    return;
                }

                templates = data.templates || [];
                renderTemplateList();
            } catch (e) {
                setStatus('Error loading templates: ' + e.message, 'error');
            }
        }

        // Tab switching for modal
        function switchModalTab(size) {
            document.querySelectorAll('#newModal .size-tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('#newModal .size-panel').forEach(p => p.classList.remove('active'));
            document.querySelector(`#newModal .size-tab[onclick*="${size}"]`).classList.add('active');
            document.getElementById(`modalPrompt${size.charAt(0).toUpperCase() + size.slice(1)}`).classList.add('active');
        }

        // Tab switching for editor
        let currentEditorSize = 'small';
        function switchEditorTab(size) {
            currentEditorSize = size;
            document.querySelectorAll('#editorContent .size-tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('#editorContent .size-panel').forEach(p => p.classList.remove('active'));
            document.querySelector(`#editorContent .size-tab[onclick*="${size}"]`).classList.add('active');
            document.getElementById(`editorPrompt${size.charAt(0).toUpperCase() + size.slice(1)}`).classList.add('active');
        }

        function renderTemplateList() {
            const list = document.getElementById('templateList');

            if (templates.length === 0) {
                list.innerHTML = '<div class="empty-state" style="padding: 20px;">No templates found</div>';
                return;
            }

            // Group by category
            const groups = { core: [], community: [], custom: [] };
            templates.forEach(t => {
                const cat = t.category || 'custom';
                if (groups[cat]) groups[cat].push(t);
            });

            let html = '';
            for (const [category, items] of Object.entries(groups)) {
                if (items.length === 0) continue;

                html += `<div class="category-header">${category} (${items.length})</div>`;
                items.forEach(t => {
                    const badgeClass = t.readonly ? 'badge-readonly' : `badge-${category}`;
                    const badgeText = t.readonly ? 'Read-only' : category;
                    html += `
                        <div class="template-item" data-key="${t.key}" onclick="loadTemplate('${t.key}')">
                            <span class="template-name">${escapeHtml(t.name)}</span>
                            <span class="template-badge ${badgeClass}">${badgeText}</span>
                        </div>
                    `;
                });
            }

            list.innerHTML = html;

            // Re-highlight current template if exists
            if (currentTemplate) {
                document.querySelectorAll('.template-item').forEach(el => {
                    el.classList.toggle('active', el.dataset.key === currentTemplate.key);
                });
            }
        }

        async function loadTemplate(key) {
            if (isModified && !confirm('You have unsaved changes. Discard them?')) {
                return;
            }

            setStatus('Loading...', '');

            try {
                const res = await fetch(`/sid/template-editor/api/template/${encodeURIComponent(key)}`);
                const data = await res.json();

                if (data.error) {
                    setStatus('Error: ' + data.error, 'error');
                    return;
                }

                currentTemplate = data;
                renderEditor(data);
                isModified = false;
                updateButtons();

                // Highlight in list
                document.querySelectorAll('.template-item').forEach(el => {
                    el.classList.toggle('active', el.dataset.key === key);
                });

                setStatus(`Loaded: ${data.name}`, 'success');
                setTimeout(() => setStatus('Ready', ''), 2000);
            } catch (e) {
                setStatus('Error: ' + e.message, 'error');
            }
        }

        let currentView = 'form';  // 'form' or 'toml'

        function renderEditor(template) {
            const isReadonly = template.readonly;
            const content = document.getElementById('editorContent');

            // Get prompts for each size (template.prompts contains small/medium/large)
            const prompts = template.prompts || {};
            const smallSystem = prompts.small?.system || template.system || '';
            const mediumSystem = prompts.medium?.system || template.system || '';
            const largeSystem = prompts.large?.system || template.system || '';
            const smallUser = prompts.small?.user || template.user || 'Describe this image.';
            const mediumUser = prompts.medium?.user || template.user || 'Describe this image.';
            const largeUser = prompts.large?.user || template.user || 'Describe this image.';

            // Generate TOML representation
            const tomlContent = templateToToml(template, smallSystem, mediumSystem, largeSystem, smallUser, mediumUser, largeUser);

            content.innerHTML = `
                <!-- Form Editor View -->
                <div class="form-editor" id="formEditorView">
                    <div class="form-row">
                        <div class="form-group">
                            <label>Name</label>
                            <input type="text" id="editName" value="${escapeHtml(template.name)}" ${isReadonly ? 'readonly' : ''}>
                        </div>
                        <div class="form-group">
                            <label>Category</label>
                            <input type="text" value="${template.category}" readonly>
                        </div>
                    </div>
                    <div class="form-group">
                        <label>Description</label>
                        <input type="text" id="editDescription" value="${escapeHtml(template.description || '')}" ${isReadonly ? 'readonly' : ''}>
                    </div>
                    <div class="form-group">
                        <label>Tags</label>
                        <input type="text" id="editTags" value="${(template.metadata?.tags || []).join(', ')}" ${isReadonly ? 'readonly' : ''}>
                    </div>
                    <div class="size-tabs">
                        <div class="size-tab active" onclick="switchEditorTab('small')">Small (2B-6B)</div>
                        <div class="size-tab" onclick="switchEditorTab('medium')">Medium (7B-13B)</div>
                        <div class="size-tab" onclick="switchEditorTab('large')">Large (13B+)</div>
                    </div>
                    <div id="editorPromptSmall" class="size-panel active">
                        <div class="form-group" style="flex: 1; display: flex; flex-direction: column;">
                            <label>System Prompt - Small Models (2B-6B)</label>
                            <textarea id="editSystemSmall" class="system-prompt-editor" style="border-radius: 4px 4px 0 0;" ${isReadonly ? 'readonly' : ''} oninput="updateStats('small')">${escapeHtml(smallSystem)}</textarea>
                            <div class="prompt-stats" id="statsSmall">
                                <div class="stat"><span>Words:</span><span class="stat-value" id="wordsSmall">0</span></div>
                                <div class="stat"><span>Chars:</span><span class="stat-value" id="charsSmall">0</span></div>
                                <div class="stat"><span>Tokens:</span><span class="stat-value" id="tokensSmall">0</span><span style="color:#666;">(est.)</span></div>
                            </div>
                        </div>
                        <div class="form-group" style="flex: 1; display: flex; flex-direction: column;">
                            <label>User Prompt - Small Models</label>
                            <textarea id="editUserSmall" class="user-prompt-editor" ${isReadonly ? 'readonly' : ''} placeholder="Describe this image.">${escapeHtml(smallUser)}</textarea>
                        </div>
                    </div>
                    <div id="editorPromptMedium" class="size-panel">
                        <div class="form-group" style="flex: 1; display: flex; flex-direction: column;">
                            <label>System Prompt - Medium Models (7B-13B)</label>
                            <textarea id="editSystemMedium" class="system-prompt-editor" style="border-radius: 4px 4px 0 0;" ${isReadonly ? 'readonly' : ''} oninput="updateStats('medium')">${escapeHtml(mediumSystem)}</textarea>
                            <div class="prompt-stats" id="statsMedium">
                                <div class="stat"><span>Words:</span><span class="stat-value" id="wordsMedium">0</span></div>
                                <div class="stat"><span>Chars:</span><span class="stat-value" id="charsMedium">0</span></div>
                                <div class="stat"><span>Tokens:</span><span class="stat-value" id="tokensMedium">0</span><span style="color:#666;">(est.)</span></div>
                            </div>
                        </div>
                        <div class="form-group" style="flex: 1; display: flex; flex-direction: column;">
                            <label>User Prompt - Medium Models</label>
                            <textarea id="editUserMedium" class="user-prompt-editor" ${isReadonly ? 'readonly' : ''} placeholder="Describe this image.">${escapeHtml(mediumUser)}</textarea>
                        </div>
                    </div>
                    <div id="editorPromptLarge" class="size-panel">
                        <div class="form-group" style="flex: 1; display: flex; flex-direction: column;">
                            <label>System Prompt - Large Models (13B+)</label>
                            <textarea id="editSystemLarge" class="system-prompt-editor" style="border-radius: 4px 4px 0 0;" ${isReadonly ? 'readonly' : ''} oninput="updateStats('large')">${escapeHtml(largeSystem)}</textarea>
                            <div class="prompt-stats" id="statsLarge">
                                <div class="stat"><span>Words:</span><span class="stat-value" id="wordsLarge">0</span></div>
                                <div class="stat"><span>Chars:</span><span class="stat-value" id="charsLarge">0</span></div>
                                <div class="stat"><span>Tokens:</span><span class="stat-value" id="tokensLarge">0</span><span style="color:#666;">(est.)</span></div>
                            </div>
                        </div>
                        <div class="form-group" style="flex: 1; display: flex; flex-direction: column;">
                            <label>User Prompt - Large Models</label>
                            <textarea id="editUserLarge" class="user-prompt-editor" ${isReadonly ? 'readonly' : ''} placeholder="Describe this image.">${escapeHtml(largeUser)}</textarea>
                        </div>
                    </div>
                </div>

                <!-- TOML Editor View -->
                <div class="toml-editor" id="tomlEditorView">
                    <div class="form-group" style="flex: 1; display: flex; flex-direction: column;">
                        <label>TOML Source ${isReadonly ? '(Read-only)' : ''}</label>
                        <textarea id="editToml" style="flex: 1;" ${isReadonly ? 'readonly' : ''}>${escapeHtml(tomlContent)}</textarea>
                    </div>
                </div>
            `;

            // Show view toggle
            document.getElementById('viewToggle').style.display = 'flex';

            // Reset to form view
            currentView = 'form';
            document.getElementById('formViewBtn').classList.add('active');
            document.getElementById('tomlViewBtn').classList.remove('active');

            document.getElementById('toolbarTitle').innerHTML = `<strong>${escapeHtml(template.name)}</strong> ${isReadonly ? '(Read-only)' : ''}`;
            document.getElementById('templateInfo').textContent = `Category: ${template.category}`;

            // Track changes
            if (!isReadonly) {
                ['editName', 'editDescription', 'editTags', 'editSystemSmall', 'editSystemMedium', 'editSystemLarge', 'editUserSmall', 'editUserMedium', 'editUserLarge', 'editToml'].forEach(id => {
                    const el = document.getElementById(id);
                    if (el) {
                        el.addEventListener('input', () => {
                            isModified = true;
                            updateButtons();
                        });
                    }
                });
            }

            // Initialize stats for all size variants
            updateStats('small');
            updateStats('medium');
            updateStats('large');
        }

        // Estimate token count (approximation: ~4 chars per token for English text)
        function estimateTokens(text) {
            if (!text) return 0;
            // More accurate estimation considering:
            // - Words are roughly 1.3 tokens on average
            // - Punctuation and special chars add tokens
            // - Code/structured text has more tokens per char
            const words = text.trim().split(/\\s+/).filter(w => w.length > 0).length;
            const chars = text.length;
            // Blend word-based and char-based estimates
            const wordEstimate = Math.ceil(words * 1.3);
            const charEstimate = Math.ceil(chars / 4);
            return Math.max(wordEstimate, charEstimate);
        }

        function updateStats(size) {
            const sizeCapitalized = size.charAt(0).toUpperCase() + size.slice(1);
            const textarea = document.getElementById('editSystem' + sizeCapitalized);
            if (!textarea) return;

            const text = textarea.value || '';
            const chars = text.length;
            const words = text.trim() ? text.trim().split(/\\s+/).filter(w => w.length > 0).length : 0;
            const tokens = estimateTokens(text);

            // Update display
            const wordsEl = document.getElementById('words' + sizeCapitalized);
            const charsEl = document.getElementById('chars' + sizeCapitalized);
            const tokensEl = document.getElementById('tokens' + sizeCapitalized);

            if (wordsEl) wordsEl.textContent = words.toLocaleString();
            if (charsEl) charsEl.textContent = chars.toLocaleString();
            if (tokensEl) {
                tokensEl.textContent = tokens.toLocaleString();
                // Color code tokens based on typical limits
                tokensEl.classList.remove('stat-warn', 'stat-danger');
                if (tokens > 2000) tokensEl.classList.add('stat-danger');
                else if (tokens > 1000) tokensEl.classList.add('stat-warn');
            }
        }

        function updateModalStats(size) {
            const textarea = document.getElementById('newPrompt' + size);
            if (!textarea) return;

            const text = textarea.value || '';
            const chars = text.length;
            const words = text.trim() ? text.trim().split(/\\s+/).filter(w => w.length > 0).length : 0;
            const tokens = estimateTokens(text);

            const wordsEl = document.getElementById('newWords' + size);
            const charsEl = document.getElementById('newChars' + size);
            const tokensEl = document.getElementById('newTokens' + size);

            if (wordsEl) wordsEl.textContent = words.toLocaleString();
            if (charsEl) charsEl.textContent = chars.toLocaleString();
            if (tokensEl) {
                tokensEl.textContent = tokens.toLocaleString();
                tokensEl.classList.remove('stat-warn', 'stat-danger');
                if (tokens > 2000) tokensEl.classList.add('stat-danger');
                else if (tokens > 1000) tokensEl.classList.add('stat-warn');
            }
        }

        function switchView(view) {
            currentView = view;
            const formView = document.getElementById('formEditorView');
            const tomlView = document.getElementById('tomlEditorView');
            const formBtn = document.getElementById('formViewBtn');
            const tomlBtn = document.getElementById('tomlViewBtn');

            if (view === 'form') {
                formView.classList.remove('hidden');
                tomlView.classList.remove('active');
                formBtn.classList.add('active');
                tomlBtn.classList.remove('active');
                // Sync TOML changes back to form if TOML was edited
                syncTomlToForm();
            } else {
                formView.classList.add('hidden');
                tomlView.classList.add('active');
                formBtn.classList.remove('active');
                tomlBtn.classList.add('active');
                // Sync form changes to TOML
                syncFormToToml();
            }
        }

        function templateToToml(template, smallSystem, mediumSystem, largeSystem, smallUser, mediumUser, largeUser) {
            const tags = template.metadata?.tags || [];
            const tagsStr = tags.length > 0 ? `[${tags.map(t => `"${t}"`).join(', ')}]` : '[]';

            // Default user prompts if not provided
            smallUser = smallUser || 'Describe this image.';
            mediumUser = mediumUser || 'Describe this image.';
            largeUser = largeUser || 'Describe this image.';

            return `# Template: ${template.name}
# Category: ${template.category}

[metadata]
name = "${template.name}"
description = "${template.description || ''}"
author = "${template.metadata?.author || 'Custom'}"
version = "${template.metadata?.version || '1.0'}"
category = "${template.category}"
tags = ${tagsStr}

[prompt.small]
system = """
${smallSystem}
"""
user = "${smallUser}"

[prompt.medium]
system = """
${mediumSystem}
"""
user = "${mediumUser}"

[prompt.large]
system = """
${largeSystem}
"""
user = "${largeUser}"
`;
        }

        function syncFormToToml() {
            if (!currentTemplate) return;
            const name = document.getElementById('editName')?.value || currentTemplate.name;
            const description = document.getElementById('editDescription')?.value || '';
            const tagsStr = document.getElementById('editTags')?.value || '';
            const tags = tagsStr ? tagsStr.split(',').map(t => t.trim()).filter(t => t) : [];
            const smallSystem = document.getElementById('editSystemSmall')?.value || '';
            const mediumSystem = document.getElementById('editSystemMedium')?.value || '';
            const largeSystem = document.getElementById('editSystemLarge')?.value || '';
            const smallUser = document.getElementById('editUserSmall')?.value || 'Describe this image.';
            const mediumUser = document.getElementById('editUserMedium')?.value || 'Describe this image.';
            const largeUser = document.getElementById('editUserLarge')?.value || 'Describe this image.';

            const tempTemplate = {
                ...currentTemplate,
                name,
                description,
                metadata: { ...currentTemplate.metadata, tags }
            };
            const toml = templateToToml(tempTemplate, smallSystem, mediumSystem, largeSystem, smallUser, mediumUser, largeUser);
            const tomlEl = document.getElementById('editToml');
            if (tomlEl) tomlEl.value = toml;
        }

        function syncTomlToForm() {
            // Parse TOML and update form fields
            const tomlEl = document.getElementById('editToml');
            if (!tomlEl) return;

            const toml = tomlEl.value;

            // Simple TOML parser for our format
            try {
                // Extract metadata
                const nameMatch = toml.match(/^name\\s*=\\s*"([^"]*)"$/m);
                const descMatch = toml.match(/^description\\s*=\\s*"([^"]*)"$/m);
                const tagsMatch = toml.match(/^tags\\s*=\\s*\\[([^\\]]*)\\]/m);

                if (nameMatch) {
                    const el = document.getElementById('editName');
                    if (el && !el.readOnly) el.value = nameMatch[1];
                }
                if (descMatch) {
                    const el = document.getElementById('editDescription');
                    if (el && !el.readOnly) el.value = descMatch[1];
                }
                if (tagsMatch) {
                    const tags = tagsMatch[1].match(/"([^"]*)"/g)?.map(t => t.replace(/"/g, '')) || [];
                    const el = document.getElementById('editTags');
                    if (el && !el.readOnly) el.value = tags.join(', ');
                }

                // Extract system prompts using multiline regex
                const smallMatch = toml.match(/\\[prompt\\.small\\][\\s\\S]*?system\\s*=\\s*"""([\\s\\S]*?)"""/);
                const mediumMatch = toml.match(/\\[prompt\\.medium\\][\\s\\S]*?system\\s*=\\s*"""([\\s\\S]*?)"""/);
                const largeMatch = toml.match(/\\[prompt\\.large\\][\\s\\S]*?system\\s*=\\s*"""([\\s\\S]*?)"""/);

                // Extract user prompts
                const smallUserMatch = toml.match(/\\[prompt\\.small\\][\\s\\S]*?user\\s*=\\s*"([^"]*)"/);
                const mediumUserMatch = toml.match(/\\[prompt\\.medium\\][\\s\\S]*?user\\s*=\\s*"([^"]*)"/);
                const largeUserMatch = toml.match(/\\[prompt\\.large\\][\\s\\S]*?user\\s*=\\s*"([^"]*)"/);

                if (smallMatch) {
                    const el = document.getElementById('editSystemSmall');
                    if (el && !el.readOnly) el.value = smallMatch[1].trim();
                }
                if (mediumMatch) {
                    const el = document.getElementById('editSystemMedium');
                    if (el && !el.readOnly) el.value = mediumMatch[1].trim();
                }
                if (largeMatch) {
                    const el = document.getElementById('editSystemLarge');
                    if (el && !el.readOnly) el.value = largeMatch[1].trim();
                }

                // Update user prompts
                if (smallUserMatch) {
                    const el = document.getElementById('editUserSmall');
                    if (el && !el.readOnly) el.value = smallUserMatch[1];
                }
                if (mediumUserMatch) {
                    const el = document.getElementById('editUserMedium');
                    if (el && !el.readOnly) el.value = mediumUserMatch[1];
                }
                if (largeUserMatch) {
                    const el = document.getElementById('editUserLarge');
                    if (el && !el.readOnly) el.value = largeUserMatch[1];
                }
            } catch (e) {
                console.warn('TOML parse warning:', e);
            }
        }

        function updateButtons() {
            const isReadonly = currentTemplate?.readonly;
            document.getElementById('saveBtn').disabled = !currentTemplate || isReadonly || !isModified;
            document.getElementById('deleteBtn').disabled = !currentTemplate || isReadonly;
            document.getElementById('duplicateBtn').disabled = !currentTemplate;
        }

        async function saveTemplate() {
            if (!currentTemplate || currentTemplate.readonly) return;

            const description = document.getElementById('editDescription').value.trim();
            const tagsStr = document.getElementById('editTags').value;
            const tags = tagsStr ? tagsStr.split(',').map(t => t.trim()).filter(t => t) : [];

            // Get all 3 size system prompts
            const systemSmall = document.getElementById('editSystemSmall').value.trim();
            const systemMedium = document.getElementById('editSystemMedium').value.trim();
            const systemLarge = document.getElementById('editSystemLarge').value.trim();

            // Get all 3 size user prompts
            const userSmall = document.getElementById('editUserSmall')?.value.trim() || 'Describe this image.';
            const userMedium = document.getElementById('editUserMedium')?.value.trim() || 'Describe this image.';
            const userLarge = document.getElementById('editUserLarge')?.value.trim() || 'Describe this image.';

            if (!systemSmall && !systemMedium && !systemLarge) {
                setStatus('At least one system prompt is required', 'error');
                return;
            }

            setStatus('Saving...', '');

            try {
                const res = await fetch(`/sid/template-editor/api/template/${encodeURIComponent(currentTemplate.key)}`, {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        description,
                        tags,
                        prompts: {
                            small: { system: systemSmall, user: userSmall },
                            medium: { system: systemMedium, user: userMedium },
                            large: { system: systemLarge, user: userLarge }
                        }
                    })
                });
                const data = await res.json();

                if (data.error) {
                    setStatus('Error: ' + data.error, 'error');
                    return;
                }

                isModified = false;
                updateButtons();
                setStatus('Saved successfully', 'success');
                setTimeout(() => setStatus('Ready', ''), 2000);
            } catch (e) {
                setStatus('Error: ' + e.message, 'error');
            }
        }

        async function deleteTemplate() {
            if (!currentTemplate || currentTemplate.readonly) return;

            if (!confirm(`Delete template "${currentTemplate.name}"? This cannot be undone.`)) return;

            setStatus('Deleting...', '');

            try {
                const res = await fetch(`/sid/template-editor/api/template/${encodeURIComponent(currentTemplate.key)}`, {
                    method: 'DELETE'
                });
                const data = await res.json();

                if (data.error) {
                    setStatus('Error: ' + data.error, 'error');
                    return;
                }

                currentTemplate = null;
                isModified = false;
                document.getElementById('editorContent').innerHTML = `
                    <div class="empty-state">
                        <div class="icon">🎨</div>
                        <div>Template deleted</div>
                    </div>
                `;
                document.getElementById('toolbarTitle').textContent = 'Select a template';
                updateButtons();
                loadTemplates();
                setStatus('Template deleted', 'success');
                setTimeout(() => setStatus('Ready', ''), 2000);
            } catch (e) {
                setStatus('Error: ' + e.message, 'error');
            }
        }

        function duplicateTemplate() {
            if (!currentTemplate) return;

            // Pre-fill modal with current template data
            document.getElementById('newName').value = currentTemplate.name + ' (Copy)';
            document.getElementById('newDescription').value = currentTemplate.description || '';
            document.getElementById('newTags').value = (currentTemplate.metadata?.tags || []).join(', ');

            // Copy all 3 size prompts
            const prompts = currentTemplate.prompts || {};
            document.getElementById('newPromptSmall').value = prompts.small?.system || currentTemplate.system || '';
            document.getElementById('newPromptMedium').value = prompts.medium?.system || currentTemplate.system || '';
            document.getElementById('newPromptLarge').value = prompts.large?.system || currentTemplate.system || '';

            // Reset to first tab
            switchModalTab('small');
            showNewModal();
        }

        function showNewModal() {
            document.getElementById('newModal').classList.add('show');
        }

        function hideNewModal() {
            document.getElementById('newModal').classList.remove('show');
            // Clear form
            document.getElementById('newName').value = '';
            document.getElementById('newDescription').value = '';
            document.getElementById('newTags').value = '';
            document.getElementById('newPromptSmall').value = '';
            document.getElementById('newPromptMedium').value = '';
            document.getElementById('newPromptLarge').value = '';
            document.getElementById('newUserSmall').value = '';
            document.getElementById('newUserMedium').value = '';
            document.getElementById('newUserLarge').value = '';
            // Reset to first tab
            switchModalTab('small');
        }

        async function createTemplate() {
            const name = document.getElementById('newName').value.trim();
            const description = document.getElementById('newDescription').value.trim();
            const tagsStr = document.getElementById('newTags').value;
            const tags = tagsStr ? tagsStr.split(',').map(t => t.trim()).filter(t => t) : [];

            // Get all 3 size prompts (system and user)
            const promptSmall = document.getElementById('newPromptSmall').value.trim();
            const promptMedium = document.getElementById('newPromptMedium').value.trim();
            const promptLarge = document.getElementById('newPromptLarge').value.trim();
            const userSmall = document.getElementById('newUserSmall').value.trim() || 'Describe this image.';
            const userMedium = document.getElementById('newUserMedium').value.trim() || 'Describe this image.';
            const userLarge = document.getElementById('newUserLarge').value.trim() || 'Describe this image.';

            if (!name) {
                alert('Template name is required');
                return;
            }
            if (!promptSmall && !promptMedium && !promptLarge) {
                alert('At least one system prompt is required');
                return;
            }

            setStatus('Creating...', '');

            try {
                const res = await fetch('/sid/template-editor/api/template', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        name,
                        description,
                        tags,
                        prompts: {
                            small: { system: promptSmall, user: userSmall },
                            medium: { system: promptMedium, user: userMedium },
                            large: { system: promptLarge, user: userLarge }
                        }
                    })
                });
                const data = await res.json();

                if (data.error) {
                    setStatus('Error: ' + data.error, 'error');
                    return;
                }

                hideNewModal();
                await loadTemplates();
                setStatus('Template created', 'success');

                // Try to load the new template
                const newKey = `custom/${name.toLowerCase().replace(/[^a-z0-9_-]/g, '_')}`;
                loadTemplate(newKey);
            } catch (e) {
                setStatus('Error: ' + e.message, 'error');
            }
        }

        async function syncTemplates() {
            if (!confirm('Sync community templates from GitHub?')) return;

            setStatus('Syncing from GitHub...', '');

            try {
                const res = await fetch('/sid/template-editor/api/sync', { method: 'POST' });
                const data = await res.json();

                if (data.error) {
                    setStatus('Sync failed: ' + data.error, 'error');
                    return;
                }

                await loadTemplates();
                setStatus('Sync complete', 'success');
                setTimeout(() => setStatus('Ready', ''), 2000);
            } catch (e) {
                setStatus('Sync error: ' + e.message, 'error');
            }
        }

        function setStatus(text, type) {
            const bar = document.getElementById('statusBar');
            const textEl = document.getElementById('statusText');
            bar.className = 'status-bar' + (type ? ' ' + type : '');
            textEl.textContent = text;
        }

        function escapeHtml(text) {
            if (!text) return '';
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        // Warn before leaving with unsaved changes
        window.addEventListener('beforeunload', (e) => {
            if (isModified) {
                e.preventDefault();
                e.returnValue = '';
            }
        });
    </script>
</body>
</html>'''
