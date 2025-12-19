# -*- coding: utf-8 -*-
"""
Prompt Editor Routes

API endpoints for the prompt editor web UI.
"""

import json
from pathlib import Path
from datetime import datetime
from aiohttp import web

# Get the config directories
TOOLKIT_DIR = Path(__file__).parent.parent
CONFIG_DIR = TOOLKIT_DIR / "config"
PROMPTS_DIR = CONFIG_DIR / "prompts"
BACKUP_DIR = CONFIG_DIR / "prompts_backup"
DEBUG_RESULTS_DIR = TOOLKIT_DIR / "debug_results"
GENERATION_RESULTS_DIR = TOOLKIT_DIR / "generation_results"

# TOML files to show in editor (from config/ root)
CONFIG_TOML_FILES = ["templates.toml", "components.toml", "providers.toml", "filters.toml", "settings.toml"]


def setup_routes(routes):
    """Register all prompt editor routes."""

    # =========================================================================
    # SID Landing Page
    # =========================================================================

    @routes.get("/sid")
    async def serve_landing(request):
        """Serve the SID Toolkit landing page."""
        return web.Response(text=get_landing_html(), content_type="text/html")

    @routes.get("/sid/prompt-editor")
    async def serve_editor(request):
        """Serve the prompt editor HTML page."""
        return web.Response(text=get_editor_html(), content_type="text/html")

    @routes.get("/sid/prompt-editor/api/files")
    async def list_files(request):
        """List all TOML files (config/ root + config/prompts/)."""
        files = []

        # First: config/*.toml (components, providers, filters)
        if CONFIG_DIR.exists():
            for name in CONFIG_TOML_FILES:
                f = CONFIG_DIR / name
                if f.exists():
                    files.append({
                        "name": f"[config] {f.name}",
                        "filename": f.name,
                        "dir": "config",
                        "path": str(f.relative_to(TOOLKIT_DIR)),
                        "size": f.stat().st_size,
                        "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat()
                    })

        # Then: config/prompts/*.toml
        if PROMPTS_DIR.exists():
            for f in sorted(PROMPTS_DIR.glob("*.toml")):
                files.append({
                    "name": f.name,
                    "filename": f.name,
                    "dir": "prompts",
                    "path": str(f.relative_to(TOOLKIT_DIR)),
                    "size": f.stat().st_size,
                    "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat()
                })

        return web.json_response({"files": files})

    def _resolve_filepath(filename: str, directory: str = None):
        """Resolve filepath based on directory hint or search both."""
        if not filename.endswith(".toml"):
            return None

        # If directory specified, use it
        if directory == "config":
            return CONFIG_DIR / filename
        elif directory == "prompts":
            return PROMPTS_DIR / filename

        # Search in both directories
        if filename in CONFIG_TOML_FILES:
            return CONFIG_DIR / filename
        return PROMPTS_DIR / filename

    @routes.get("/sid/prompt-editor/api/file/{filename}")
    async def get_file(request):
        """Get contents of a specific TOML file."""
        filename = request.match_info["filename"]
        directory = request.query.get("dir", None)

        # Security: only allow .toml files
        if not filename.endswith(".toml"):
            return web.json_response({"error": "Only .toml files allowed"}, status=400)

        filepath = _resolve_filepath(filename, directory)
        if not filepath or not filepath.exists():
            return web.json_response({"error": "File not found"}, status=404)

        try:
            content = filepath.read_text(encoding="utf-8")
            return web.json_response({
                "filename": filename,
                "content": content,
                "size": len(content)
            })
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    @routes.put("/sid/prompt-editor/api/file/{filename}")
    async def save_file(request):
        """Save contents to a TOML file (with backup)."""
        filename = request.match_info["filename"]

        # Security: only allow .toml files
        if not filename.endswith(".toml"):
            return web.json_response({"error": "Only .toml files allowed"}, status=400)

        try:
            data = await request.json()
            content = data.get("content", "")
            directory = data.get("dir", None)

            filepath = _resolve_filepath(filename, directory)
            if not filepath:
                return web.json_response({"error": "Invalid file"}, status=400)

            # Validate TOML syntax using tomlkit (preserves comments)
            try:
                import tomlkit
                tomlkit.parse(content)
            except Exception as e:
                return web.json_response({
                    "error": f"Invalid TOML syntax: {e}"
                }, status=400)

            # Create backup
            if filepath.exists():
                BACKUP_DIR.mkdir(exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = BACKUP_DIR / f"{filename}.{timestamp}.bak"
                backup_path.write_text(filepath.read_text(encoding="utf-8"), encoding="utf-8")

            # Save file
            filepath.write_text(content, encoding="utf-8")

            return web.json_response({
                "success": True,
                "filename": filename,
                "size": len(content)
            })
        except json.JSONDecodeError:
            return web.json_response({"error": "Invalid JSON in request"}, status=400)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    @routes.post("/sid/prompt-editor/api/reset/{filename}")
    async def reset_file(request):
        """Reset file to git HEAD version."""
        import subprocess

        filename = request.match_info["filename"]
        directory = request.query.get("dir", None)

        if not filename.endswith(".toml"):
            return web.json_response({"error": "Only .toml files allowed"}, status=400)

        filepath = _resolve_filepath(filename, directory)
        if not filepath:
            return web.json_response({"error": "Invalid file"}, status=400)

        # Determine relative path for git
        if filename in CONFIG_TOML_FILES:
            relative_path = f"config/{filename}"
        else:
            relative_path = f"config/prompts/{filename}"

        try:
            # Get content from git HEAD
            result = subprocess.run(
                ["git", "show", f"HEAD:{relative_path}"],
                cwd=TOOLKIT_DIR,
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode != 0:
                return web.json_response({
                    "error": "Could not get file from git (file may not be tracked)"
                }, status=400)

            original_content = result.stdout

            # Backup current version
            if filepath.exists():
                BACKUP_DIR.mkdir(exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = BACKUP_DIR / f"{filename}.{timestamp}.before_reset.bak"
                backup_path.write_text(filepath.read_text(encoding="utf-8"), encoding="utf-8")

            # Write original content
            filepath.write_text(original_content, encoding="utf-8")

            return web.json_response({
                "success": True,
                "filename": filename,
                "content": original_content
            })
        except subprocess.TimeoutExpired:
            return web.json_response({"error": "Git command timed out"}, status=500)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

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


def get_editor_html():
    """Return the HTML for the prompt editor UI."""
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SID Prompt Editor</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1e1e1e;
            color: #d4d4d4;
            height: 100vh;
            overflow: hidden;
        }
        .container {
            display: flex;
            height: 100vh;
        }
        /* Sidebar */
        .sidebar {
            width: 250px;
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
        .file-list {
            flex: 1;
            overflow-y: auto;
            padding: 10px 0;
        }
        .file-item {
            padding: 8px 15px;
            cursor: pointer;
            font-size: 13px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .file-item:hover { background: #2a2d2e; }
        .file-item.active { background: #37373d; color: #fff; }
        .file-item.modified::after {
            content: "●";
            color: #e2c08d;
            margin-left: auto;
        }
        .file-icon { opacity: 0.7; }
        /* Editor */
        .editor-container {
            flex: 1;
            display: flex;
            flex-direction: column;
        }
        .toolbar {
            padding: 10px 15px;
            background: #333;
            border-bottom: 1px solid #3c3c3c;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .toolbar-title {
            flex: 1;
            font-size: 13px;
            color: #888;
        }
        .toolbar-title strong { color: #d4d4d4; }
        .btn {
            padding: 6px 14px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
            font-weight: 500;
        }
        .btn-primary {
            background: #0e639c;
            color: white;
        }
        .btn-primary:hover { background: #1177bb; }
        .btn-primary:disabled {
            background: #3c3c3c;
            color: #888;
            cursor: not-allowed;
        }
        .btn-secondary {
            background: #3c3c3c;
            color: #d4d4d4;
        }
        .btn-secondary:hover { background: #4c4c4c; }
        .btn-warning {
            background: #b89500;
            color: white;
        }
        .btn-warning:hover { background: #d4aa00; }
        .editor-wrapper {
            flex: 1;
            overflow: hidden;
        }
        #editor {
            width: 100%;
            height: 100%;
        }
        /* Status bar */
        .status-bar {
            padding: 5px 15px;
            background: #007acc;
            font-size: 12px;
            display: flex;
            justify-content: space-between;
        }
        .status-bar.error { background: #c42b1c; }
        .status-bar.success { background: #16825d; }
        /* Loading */
        .loading {
            display: flex;
            align-items: center;
            justify-content: center;
            height: 100%;
            color: #888;
        }
        /* Empty state */
        .empty-state {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 100%;
            color: #888;
            gap: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="sidebar">
            <div class="sidebar-header">📁 Prompt Files</div>
            <div class="file-list" id="fileList">
                <div class="loading">Loading...</div>
            </div>
            <div class="nav-links">
                <a href="/sid" class="nav-link">🏠 Home</a>
                <a href="/sid/prompt-editor" class="nav-link active">📝 Prompt Editor</a>
                <a href="/sid/generation-results" class="nav-link">🖼️ Generation Results</a>
                <a href="/sid/debug-results" class="nav-link">🔍 Debug Viewer</a>
            </div>
        </div>
        <div class="editor-container">
            <div class="toolbar">
                <div class="toolbar-title" id="toolbarTitle">Select a file to edit</div>
                <button class="btn btn-warning" id="reloadBtn" onclick="reloadConfigs()" title="Reload all TOML configs from disk (no restart needed)">🔄 Reload Configs</button>
                <button class="btn btn-secondary" id="resetBtn" disabled onclick="resetFile()">Reset to Default</button>
                <button class="btn btn-primary" id="saveBtn" disabled onclick="saveFile()">Save (Ctrl+S)</button>
            </div>
            <div class="editor-wrapper">
                <div id="editor"></div>
            </div>
            <div class="status-bar" id="statusBar">
                <span id="statusText">Ready</span>
                <span id="cursorPos"></span>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/monaco-editor@0.45.0/min/vs/loader.js"></script>
    <script>
        let editor = null;
        let currentFile = null;
        let currentDir = null;
        let originalContent = '';
        let isModified = false;

        // Initialize Monaco Editor
        require.config({ paths: { vs: 'https://cdn.jsdelivr.net/npm/monaco-editor@0.45.0/min/vs' } });
        require(['vs/editor/editor.main'], function () {
            // Register TOML language
            monaco.languages.register({ id: 'toml' });
            monaco.languages.setMonarchTokensProvider('toml', {
                tokenizer: {
                    root: [
                        [/#.*$/, 'comment'],
                        [/\\[[^\\]]*\\]/, 'keyword'],
                        [/[a-zA-Z_][a-zA-Z0-9_]*(?=\\s*=)/, 'variable'],
                        [/"([^"\\\\]|\\\\.)*"/, 'string'],
                        [/'([^'\\\\]|\\\\.)*'/, 'string'],
                        [/"""[\\s\\S]*?"""/, 'string'],
                        [/\'\'\'[\\s\\S]*?\'\'\'/, 'string'],
                        [/true|false/, 'keyword'],
                        [/\\d+\\.\\d+/, 'number'],
                        [/\\d+/, 'number'],
                    ]
                }
            });

            editor = monaco.editor.create(document.getElementById('editor'), {
                value: '# Select a file from the sidebar to edit',
                language: 'toml',
                theme: 'vs-dark',
                fontSize: 14,
                minimap: { enabled: true },
                wordWrap: 'on',
                automaticLayout: true,
                scrollBeyondLastLine: false,
                renderWhitespace: 'selection',
            });

            // Track changes
            editor.onDidChangeModelContent(() => {
                if (currentFile) {
                    const modified = editor.getValue() !== originalContent;
                    setModified(modified);
                }
            });

            // Cursor position
            editor.onDidChangeCursorPosition((e) => {
                document.getElementById('cursorPos').textContent =
                    `Ln ${e.position.lineNumber}, Col ${e.position.column}`;
            });

            // Ctrl+S to save
            editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.KeyS, saveFile);

            // Load file list
            loadFileList();
        });

        async function loadFileList() {
            try {
                const res = await fetch('/sid/prompt-editor/api/files');
                const data = await res.json();

                const list = document.getElementById('fileList');
                if (data.files.length === 0) {
                    list.innerHTML = '<div class="empty-state">No .toml files found</div>';
                    return;
                }

                list.innerHTML = data.files.map(f => `
                    <div class="file-item" data-file="${f.filename}" data-dir="${f.dir}" onclick="loadFile('${f.filename}', '${f.dir}')">
                        <span class="file-icon">📄</span>
                        <span>${f.name}</span>
                    </div>
                `).join('');
            } catch (e) {
                document.getElementById('fileList').innerHTML =
                    '<div class="empty-state">Error loading files</div>';
            }
        }

        async function loadFile(filename, dir) {
            if (isModified && !confirm('You have unsaved changes. Discard them?')) {
                return;
            }

            setStatus('Loading...', '');

            try {
                const res = await fetch(`/sid/prompt-editor/api/file/${filename}?dir=${dir}`);
                const data = await res.json();

                if (data.error) {
                    setStatus(data.error, 'error');
                    return;
                }

                currentFile = filename;
                currentDir = dir;
                originalContent = data.content;
                editor.setValue(data.content);
                setModified(false);

                // Update UI
                const displayName = dir === 'config' ? `[config] ${filename}` : filename;
                document.getElementById('toolbarTitle').innerHTML =
                    `Editing: <strong>${displayName}</strong>`;
                document.getElementById('saveBtn').disabled = true;
                document.getElementById('resetBtn').disabled = false;

                // Highlight active file
                document.querySelectorAll('.file-item').forEach(el => {
                    el.classList.toggle('active', el.dataset.file === filename && el.dataset.dir === dir);
                });

                setStatus(`Loaded ${filename}`, 'success');
                setTimeout(() => setStatus('Ready', ''), 2000);
            } catch (e) {
                setStatus('Error loading file: ' + e.message, 'error');
            }
        }

        async function saveFile() {
            if (!currentFile || !isModified) return;

            setStatus('Saving...', '');

            try {
                const res = await fetch(`/sid/prompt-editor/api/file/${currentFile}`, {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ content: editor.getValue(), dir: currentDir })
                });
                const data = await res.json();

                if (data.error) {
                    setStatus('Error: ' + data.error, 'error');
                    return;
                }

                originalContent = editor.getValue();
                setModified(false);
                setStatus(`Saved ${currentFile}`, 'success');
                setTimeout(() => setStatus('Ready', ''), 2000);
            } catch (e) {
                setStatus('Error saving: ' + e.message, 'error');
            }
        }

        async function reloadConfigs() {
            setStatus('Reloading configs...', '');
            try {
                const res = await fetch('/sid/api/reload-configs', { method: 'POST' });
                const data = await res.json();
                if (data.error) {
                    setStatus('Error: ' + data.error, 'error');
                    return;
                }
                setStatus(data.message, 'success');
                setTimeout(() => setStatus('Ready', ''), 3000);
            } catch (e) {
                setStatus('Error reloading: ' + e.message, 'error');
            }
        }

        async function resetFile() {
            if (!currentFile) return;

            if (!confirm(`Reset ${currentFile} to the original git version?\\n\\nYour current changes will be backed up.`)) {
                return;
            }

            setStatus('Resetting...', '');

            try {
                const res = await fetch(`/sid/prompt-editor/api/reset/${currentFile}?dir=${currentDir}`, {
                    method: 'POST'
                });
                const data = await res.json();

                if (data.error) {
                    setStatus('Error: ' + data.error, 'error');
                    return;
                }

                originalContent = data.content;
                editor.setValue(data.content);
                setModified(false);
                setStatus(`Reset ${currentFile} to default`, 'success');
                setTimeout(() => setStatus('Ready', ''), 2000);
            } catch (e) {
                setStatus('Error resetting: ' + e.message, 'error');
            }
        }

        function setModified(modified) {
            isModified = modified;
            document.getElementById('saveBtn').disabled = !modified;

            // Update file item indicator
            document.querySelectorAll('.file-item').forEach(el => {
                if (el.dataset.file === currentFile) {
                    el.classList.toggle('modified', modified);
                }
            });
        }

        function setStatus(text, type) {
            const bar = document.getElementById('statusBar');
            const textEl = document.getElementById('statusText');
            bar.className = 'status-bar' + (type ? ' ' + type : '');
            textEl.textContent = text;
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


def get_debug_viewer_html():
    """Return the HTML for the debug results viewer UI."""
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SID Debug Results Viewer</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1e1e1e;
            color: #d4d4d4;
            height: 100vh;
            overflow: hidden;
        }
        .container {
            display: flex;
            height: 100vh;
        }
        /* Sidebar */
        .sidebar {
            width: 320px;
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
        .btn-download {
            padding: 4px 10px;
            background: #16825d;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
            margin-left: 8px;
        }
        .btn-download:hover { background: #1a9e6e; }
        .btn-download:disabled { background: #3c3c3c; color: #888; cursor: not-allowed; }
        .toolbar-buttons { display: flex; gap: 8px; }
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
        .session-item.excluded .session-meta { text-decoration: line-through; }
        .session-header { display: flex; align-items: center; gap: 8px; margin-bottom: 4px; }
        .exclude-checkbox {
            width: 16px;
            height: 16px;
            min-width: 16px;
            cursor: pointer;
            accent-color: #b89500;
            margin: 0;
            flex-shrink: 0;
        }
        .session-id { font-weight: 600; color: #fff; flex: 1; cursor: pointer; }
        .session-meta { color: #888; font-size: 11px; cursor: pointer; }
        .session-score {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 11px;
            font-weight: 600;
            margin-top: 4px;
        }
        .score-good { background: #16825d; color: #fff; }
        .score-ok { background: #c49a3a; color: #000; }
        .score-bad { background: #c42b1c; color: #fff; }
        /* Main content */
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
        /* Images section */
        .images-section {
            display: flex;
            gap: 20px;
            margin-bottom: 20px;
        }
        .image-card {
            flex: 1;
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
        /* Scores section */
        .scores-section {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        .score-card {
            background: #252526;
            border-radius: 8px;
            padding: 15px;
        }
        .score-card h4 {
            font-size: 12px;
            color: #888;
            margin-bottom: 8px;
            text-transform: uppercase;
        }
        .score-value {
            font-size: 28px;
            font-weight: 700;
            margin-bottom: 8px;
        }
        .score-reason {
            font-size: 12px;
            color: #aaa;
            line-height: 1.4;
        }
        /* Prompt section */
        .prompt-section {
            background: #252526;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        .prompt-section h3 {
            padding: 10px 15px;
            background: #333;
            font-size: 13px;
            font-weight: 600;
            border-radius: 8px 8px 0 0;
        }
        .prompt-content {
            padding: 15px;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 13px;
            line-height: 1.6;
            white-space: pre-wrap;
            word-break: break-word;
        }
        .improved-prompt {
            background: #1a3a1a;
            border-left: 3px solid #4caf50;
        }
        /* Analysis section */
        .analysis-section {
            background: #252526;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        .analysis-section h3 {
            padding: 10px 15px;
            background: #333;
            font-size: 13px;
            font-weight: 600;
            border-radius: 8px 8px 0 0;
        }
        .analysis-content {
            padding: 15px;
        }
        .analysis-list {
            list-style: none;
            padding: 0;
        }
        .analysis-list li {
            padding: 6px 0;
            font-size: 13px;
            border-bottom: 1px solid #3c3c3c;
        }
        .analysis-list li:last-child { border-bottom: none; }
        .tag-missing { color: #f44336; }
        .tag-hallucinated { color: #ff9800; }
        .tag-detected { color: #4caf50; }
        .tag-issue { color: #2196f3; }
        /* Recommendations */
        .recommendations-list {
            list-style: decimal;
            padding-left: 20px;
        }
        .recommendations-list li {
            padding: 8px 0;
            font-size: 13px;
            line-height: 1.5;
        }
        /* Loading */
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
                <span>Debug Sessions</span>
                <button class="btn-refresh" onclick="loadSessions()">Refresh</button>
            </div>
            <div class="session-list" id="sessionList">
                <div class="loading">Loading...</div>
            </div>
            <div class="nav-links">
                <a href="/sid" class="nav-link">🏠 Home</a>
                <a href="/sid/prompt-editor" class="nav-link">📝 Prompt Editor</a>
                <a href="/sid/generation-results" class="nav-link">🖼️ Generation Results</a>
                <a href="/sid/debug-results" class="nav-link active">🔍 Debug Viewer</a>
            </div>
        </div>
        <div class="main-content">
            <div class="toolbar">
                <span id="toolbar">Select a session to view results</span>
                <div class="toolbar-buttons">
                    <button class="btn-download" id="downloadBtn" disabled onclick="downloadPDF()">Download PDF</button>
                </div>
            </div>
            <div class="content-area" id="contentArea">
                <div class="empty-state">
                    <div style="font-size: 48px;">📊</div>
                    <div>Select a debug session from the sidebar</div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/html2pdf.js/0.10.1/html2pdf.bundle.min.js"></script>
    <script>
        let currentSession = null;
        let currentSessionId = null;

        // Load sessions on page load
        loadSessions();

        async function loadSessions() {
            try {
                const res = await fetch('/sid/debug-results/api/sessions');
                const data = await res.json();

                const list = document.getElementById('sessionList');
                if (data.sessions.length === 0) {
                    list.innerHTML = '<div class="empty-state">No debug sessions found</div>';
                    return;
                }

                list.innerHTML = data.sessions.map(s => {
                    // Format timestamp nicely
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
                    // Shorten model name
                    const shortModel = s.model.replace('claude-', '').replace('-20250929', '').replace('-20241022', '');
                    const scoreClass = s.overall_score >= 7 ? 'score-good' : s.overall_score >= 5 ? 'score-ok' : 'score-bad';
                    const imgIndicator = (s.has_source || s.has_output) ? '🖼️ ' : '';
                    const excludedClass = s.excluded ? 'excluded' : '';
                    const excludedChecked = s.excluded ? 'checked' : '';
                    return `
                        <div class="session-item ${excludedClass}" data-id="${s.id}">
                            <div class="session-header">
                                <input type="checkbox" class="exclude-checkbox" ${excludedChecked}
                                    onclick="toggleExclude(event, '${s.id}')"
                                    title="Exclude from learning">
                                <div class="session-id" onclick="loadSession('${s.id}')">${imgIndicator}${dateStr}</div>
                                <span class="session-score ${scoreClass}">${s.overall_score.toFixed(1)}</span>
                            </div>
                            <div class="session-meta" onclick="loadSession('${s.id}')">${s.provider} / ${shortModel}</div>
                        </div>
                    `;
                }).join('');
            } catch (e) {
                document.getElementById('sessionList').innerHTML =
                    '<div class="empty-state">Error loading sessions</div>';
            }
        }

        async function toggleExclude(event, sessionId) {
            event.stopPropagation(); // Don't trigger loadSession
            const checkbox = event.target;
            const sessionItem = checkbox.closest('.session-item');

            try {
                const res = await fetch(`/sid/debug-results/api/session/${sessionId}/exclude`, {
                    method: 'POST'
                });
                const data = await res.json();

                if (data.success) {
                    checkbox.checked = data.excluded;
                    sessionItem.classList.toggle('excluded', data.excluded);
                } else {
                    checkbox.checked = !checkbox.checked;
                }
            } catch (e) {
                checkbox.checked = !checkbox.checked;
                console.error('Failed to toggle exclude:', e);
            }
        }

        async function loadSession(sessionId) {
            // Update active state
            document.querySelectorAll('.session-item').forEach(el => {
                el.classList.toggle('active', el.dataset.id === sessionId);
            });

            currentSessionId = sessionId;
            document.getElementById('toolbar').textContent = `Session: ${sessionId}`;
            document.getElementById('contentArea').innerHTML = '<div class="loading">Loading...</div>';
            document.getElementById('downloadBtn').disabled = true;

            try {
                const res = await fetch(`/sid/debug-results/api/session/${sessionId}`);
                const data = await res.json();

                if (data.error) {
                    document.getElementById('contentArea').innerHTML =
                        `<div class="empty-state">${data.error}</div>`;
                    return;
                }

                currentSession = data;
                renderSession(data);
                document.getElementById('downloadBtn').disabled = false;
            } catch (e) {
                document.getElementById('contentArea').innerHTML =
                    '<div class="empty-state">Error loading session</div>';
                document.getElementById('downloadBtn').disabled = true;
            }
        }

        function renderSession(data) {
            const scores = data.scores || {};
            const images = data.images || {};
            // Try multiple sources for original prompt
            const prompt = data.original_prompt || data.inputs?.prompt?.content || '';
            const improvedPrompt = data.improved_prompt || '';
            // Fields can be at top level or under analysis (handle both)
            const issues = data.issues || data.analysis?.issues || [];
            const missingElements = data.missing_elements || data.analysis?.missing_elements || [];
            const hallucinatedElements = data.hallucinated_elements || data.analysis?.hallucinated_elements || [];
            const recommendations = data.recommendations || data.analysis?.recommendations || [];

            let html = '';

            // Images
            html += '<div class="images-section">';
            if (images.source) {
                html += `
                    <div class="image-card">
                        <h3>Source Image</h3>
                        <img src="${images.source}" alt="Source">
                    </div>
                `;
            }
            if (images.output) {
                html += `
                    <div class="image-card">
                        <h3>Output Image</h3>
                        <img src="${images.output}" alt="Output">
                    </div>
                `;
            }
            html += '</div>';

            // Scores
            html += '<div class="scores-section">';
            for (const [key, value] of Object.entries(scores)) {
                const score = value.score || 0;
                const color = score >= 7 ? '#4caf50' : score >= 5 ? '#ff9800' : '#f44336';
                html += `
                    <div class="score-card">
                        <h4>${key.replace(/_/g, ' ')}</h4>
                        <div class="score-value" style="color: ${color}">${score.toFixed(1)}</div>
                        <div class="score-reason">${value.reason || ''}</div>
                    </div>
                `;
            }
            html += '</div>';

            // Original Prompt
            if (prompt) {
                html += `
                    <div class="prompt-section">
                        <h3>Original Prompt</h3>
                        <div class="prompt-content">${escapeHtml(prompt)}</div>
                    </div>
                `;
            }

            // Improved Prompt
            if (improvedPrompt) {
                html += `
                    <div class="prompt-section">
                        <h3>Improved Prompt (Suggested)</h3>
                        <div class="prompt-content improved-prompt">${escapeHtml(improvedPrompt)}</div>
                    </div>
                `;
            }

            // Issues
            if (issues.length > 0) {
                html += `
                    <div class="analysis-section">
                        <h3>Issues Found</h3>
                        <div class="analysis-content">
                            <ul class="analysis-list">
                                ${issues.map(e => `<li class="tag-issue">⚡ ${escapeHtml(e)}</li>`).join('')}
                            </ul>
                        </div>
                    </div>
                `;
            }

            // Missing Elements
            if (missingElements.length > 0) {
                html += `
                    <div class="analysis-section">
                        <h3>Missing Elements</h3>
                        <div class="analysis-content">
                            <ul class="analysis-list">
                                ${missingElements.map(e => `<li class="tag-missing">❌ ${escapeHtml(e)}</li>`).join('')}
                            </ul>
                        </div>
                    </div>
                `;
            }

            // Hallucinated Elements
            if (hallucinatedElements.length > 0) {
                html += `
                    <div class="analysis-section">
                        <h3>Hallucinated Elements</h3>
                        <div class="analysis-content">
                            <ul class="analysis-list">
                                ${hallucinatedElements.map(e => `<li class="tag-hallucinated">⚠️ ${escapeHtml(e)}</li>`).join('')}
                            </ul>
                        </div>
                    </div>
                `;
            }

            // Recommendations
            if (recommendations.length > 0) {
                html += `
                    <div class="analysis-section">
                        <h3>Recommendations</h3>
                        <div class="analysis-content">
                            <ol class="recommendations-list">
                                ${recommendations.map(r => `<li>${escapeHtml(r)}</li>`).join('')}
                            </ol>
                        </div>
                    </div>
                `;
            }

            // Generation Model (the model that generated the prompt)
            const genModel = data.generation_model || {};
            if (genModel.provider || genModel.model) {
                html += `
                    <div class="analysis-section">
                        <h3>Generation Model</h3>
                        <div class="analysis-content" style="font-size: 12px; color: #888;">
                            ${genModel.provider ? `<div>Provider: ${escapeHtml(genModel.provider)}</div>` : ''}
                            ${genModel.model ? `<div>Model: ${escapeHtml(genModel.model)}</div>` : ''}
                            ${genModel.analysis_mode ? `<div>Mode: ${escapeHtml(genModel.analysis_mode)}</div>` : ''}
                            ${genModel.temperature ? `<div>Temperature: ${genModel.temperature}</div>` : ''}
                        </div>
                    </div>
                `;
            }

            // Evaluation Metadata (the model that evaluated the prompt)
            const evaluator = data.evaluator || {};
            const timing = data.timing || {};
            const apiCalls = data.api_calls;
            if (evaluator.model || timing.total_seconds || apiCalls) {
                html += `
                    <div class="analysis-section">
                        <h3>Evaluation Model</h3>
                        <div class="analysis-content" style="font-size: 12px; color: #888;">
                            ${evaluator.provider ? `<div>Provider: ${escapeHtml(evaluator.provider)}</div>` : ''}
                            ${evaluator.model ? `<div>Model: ${escapeHtml(evaluator.model)}</div>` : ''}
                            ${apiCalls ? `<div>API Calls: ${apiCalls}</div>` : ''}
                            ${timing.total_seconds ? `<div>Time: ${timing.total_seconds.toFixed(2)}s</div>` : ''}
                        </div>
                    </div>
                `;
            }

            document.getElementById('contentArea').innerHTML = html;
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        async function downloadPDF() {
            if (!currentSession || !currentSessionId) return;

            const btn = document.getElementById('downloadBtn');
            btn.textContent = 'Generating...';
            btn.disabled = true;

            try {
                const content = document.getElementById('contentArea');

                // Configure PDF options
                const opt = {
                    margin: 10,
                    filename: `debug_${currentSessionId}.pdf`,
                    image: { type: 'jpeg', quality: 0.95 },
                    html2canvas: {
                        scale: 2,
                        useCORS: true,
                        backgroundColor: '#1e1e1e'
                    },
                    jsPDF: { unit: 'mm', format: 'a4', orientation: 'portrait' }
                };

                // Generate and download PDF
                await html2pdf().set(opt).from(content).save();

                btn.textContent = 'Download PDF';
                btn.disabled = false;
            } catch (e) {
                console.error('PDF generation failed:', e);
                alert('Failed to generate PDF: ' + e.message);
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
                <a href="/sid/prompt-editor" class="nav-link">📝 Prompt Editor</a>
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
            <a href="/sid/prompt-editor" class="card">
                <div class="card-icon">📝</div>
                <div class="card-title">Prompt Editor</div>
                <div class="card-desc">Edit TOML configuration files for prompts, templates, and settings</div>
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
