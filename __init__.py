# -*- coding: utf-8 -*-
"""
ComfyUI-AI-Photography-Toolkit

LLM Provider nodes for ComfyUI - Cloud and Local model support.
All nodes are prefixed with SID_ for easy identification.

Author: Siddhartha Lahiri
Email: siddhartha.lahiri@gmail.com
License: MIT
Version: 4.3.0
"""

__version__ = "4.3.0"

# Tell ComfyUI to load web extensions from ./web folder
WEB_DIRECTORY = "./web"

import sys
import subprocess
import importlib.util
from pathlib import Path

# =============================================================================
# Load Settings (including debug_mode)
# =============================================================================

DEBUG_MODE = False
_settings = {}

def load_settings():
    """Load settings from config/settings.toml"""
    global DEBUG_MODE, _settings

    settings_path = Path(__file__).parent / "config" / "settings.toml"

    if settings_path.exists():
        try:
            # Python 3.11+ has tomllib built-in
            import tomllib
            with open(settings_path, "rb") as f:
                _settings = tomllib.load(f)
        except ImportError:
            # Fallback for Python < 3.11
            try:
                import tomli
                with open(settings_path, "rb") as f:
                    _settings = tomli.load(f)
            except ImportError:
                # Manual parse for debug_mode only
                with open(settings_path, "r") as f:
                    content = f.read()
                    if "debug_mode = true" in content.lower():
                        _settings = {"development": {"debug_mode": True}}

        DEBUG_MODE = _settings.get("development", {}).get("debug_mode", False)

load_settings()

# Track installation status for welcome message
_dependency_status = {}


def check_and_install_dependencies():
    """
    Check and install required dependencies. Track status for logging.
    """
    global _dependency_status

    # Required dependencies (auto-installed)
    dependencies = {
        "anthropic": {
            "import_name": "anthropic",
            "pip_spec": "anthropic>=0.39.0",
            "description": "Anthropic Claude API SDK",
        },
        "openai": {
            "import_name": "openai",
            "pip_spec": "openai>=1.0.0",
            "description": "OpenAI API SDK (also used for Grok, LM Studio, etc.)",
        },
        "pyyaml": {
            "import_name": "yaml",
            "pip_spec": "pyyaml>=6.0",
            "description": "YAML configuration parser",
        },
        "requests": {
            "import_name": "requests",
            "pip_spec": "requests>=2.28.0",
            "description": "HTTP library (for Ollama API)",
        },
        "typing_extensions": {
            "import_name": "typing_extensions",
            "pip_spec": "typing_extensions>=4.0.0",
            "description": "Extended typing support",
        },
        "transformers": {
            "import_name": "transformers",
            "pip_spec": "transformers>=4.57.0",
            "description": "HuggingFace Transformers (for local models)",
        },
        "accelerate": {
            "import_name": "accelerate",
            "pip_spec": "accelerate>=0.33.0",
            "description": "HuggingFace Accelerate (for model loading)",
        },
        "psutil": {
            "import_name": "psutil",
            "pip_spec": "psutil>=5.9.0",
            "description": "System utilities (for memory detection)",
        },
        "ultralytics": {
            "import_name": "ultralytics",
            "pip_spec": "ultralytics>=8.0.0",
            "description": "YOLO models (for human detection)",
        },
        # MediaPipe removed - has protobuf compatibility issues on Windows
        # YOLO provides sufficient human detection
        "opencv": {
            "import_name": "cv2",
            "pip_spec": "opencv-python>=4.8.0",
            "description": "OpenCV (for image processing)",
        },
        "sklearn": {
            "import_name": "sklearn",
            "pip_spec": "scikit-learn>=1.3.0",
            "description": "Scikit-learn (for color clustering)",
        },
        "aiohttp": {
            "import_name": "aiohttp",
            "pip_spec": "aiohttp>=3.8.0",
            "description": "Async HTTP (for web routes)",
        },
        "tomlkit": {
            "import_name": "tomlkit",
            "pip_spec": "tomlkit>=0.12.0",
            "description": "TOML parser (preserves comments)",
        },
    }

    python_exe = sys.executable

    # Check and install required dependencies
    for pkg_name, pkg_info in dependencies.items():
        import_name = pkg_info["import_name"]
        pip_spec = pkg_info["pip_spec"]

        if importlib.util.find_spec(import_name) is not None:
            _dependency_status[pkg_name] = "installed"
        else:
            # Try to install
            try:
                subprocess.check_call(
                    [python_exe, "-m", "pip", "install", pip_spec],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                _dependency_status[pkg_name] = "just_installed"
            except subprocess.CalledProcessError:
                _dependency_status[pkg_name] = "failed"


# Run dependency check on module load
check_and_install_dependencies()

# Track model download status
_model_status = {}


def download_cv_models():
    """
    Pre-download CV models (YOLO, MediaPipe) at init time.
    This ensures models are ready when user needs them.
    """
    global _model_status

    print("\n[SID-Toolkit] Checking CV models...")

    # Download YOLOv8n model
    try:
        from ultralytics import YOLO
        import os

        # YOLO downloads to ~/.cache/ultralytics or current dir
        # Check if model exists, if not it will download
        print("[SID-Toolkit] Loading YOLO model (yolov8n.pt)...")
        model = YOLO("yolov8n.pt")
        _model_status["yolo"] = "ready"
        print("[SID-Toolkit] YOLO model ready")

        # Clean up to free memory
        del model
    except ImportError:
        _model_status["yolo"] = "missing_ultralytics"
        print("[SID-Toolkit] WARNING: ultralytics not installed, YOLO unavailable")
    except Exception as e:
        _model_status["yolo"] = f"error: {e}"
        print(f"[SID-Toolkit] WARNING: YOLO model download failed: {e}")

    # MediaPipe disabled due to protobuf compatibility issues on Windows
    # YOLO provides sufficient person detection
    _model_status["mediapipe"] = "disabled"

    print("[SID-Toolkit] CV model check complete\n")


# Download CV models at init
download_cv_models()

from typing_extensions import override
from comfy_api.latest import ComfyExtension, io

# Import LLM provider nodes
from .llm_providers.sid_llm_api import SID_LLM_API
from .llm_providers.sid_llm_local import SID_LLM_Local, SID_LLM_Local_API

# Import Z-Image Prompt Generator V2 (CV-based detection + simplified interface)
from .sid_prompt_generator_v2 import SID_ZImagePromptGeneratorV2 as SID_ZImagePromptGenerator
from .sid_prompt_generator_v2 import SID_PromptTemplate

# Import Debug Agent (always available)
try:
    from .sid_prompt_debug import SID_PromptDebugAgent
except ImportError as e:
    SID_PromptDebugAgent = None
    print(f"[SID-Toolkit] Debug Agent not available: {e}")

# Import PromptServer for API routes
try:
    from server import PromptServer
    from aiohttp import web
    _server_available = True
except ImportError as e:
    _server_available = False
    print(f"[SID-Toolkit] Server not available: {e}")


class SIDPhotographyToolkitExtension(ComfyExtension):
    """
    Extension class for SID Photography Toolkit.
    Registers SID_ prefixed LLM nodes with ComfyUI.
    """

    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        """
        Return list of all nodes in this extension.
        """
        nodes = [
            SID_LLM_API,              # Cloud APIs (Anthropic, OpenAI, Gemini, Groq, etc.)
            SID_LLM_Local,            # Local transformers models (QwenVL, Florence, etc.)
            SID_LLM_Local_API,        # Local API providers (Ollama, LM Studio, OpenAI Compatible)
            SID_ZImagePromptGenerator,  # Z-Image prompt generator
            SID_PromptTemplate,       # Prompt template selector
        ]

        # Add debug node if available (debug_mode = true)
        if SID_PromptDebugAgent is not None:
            nodes.append(SID_PromptDebugAgent)

        return nodes


def print_welcome_message():
    """
    Print welcome message with dependency status and available nodes.
    """
    global _dependency_status, _model_status

    # Header
    print("")
    print("=" * 65)
    print("  SID Photography Toolkit for ComfyUI")
    print(f"  Version: {__version__}")
    print("  Author: Siddhartha Lahiri")
    print("=" * 65)

    # CV Models status
    print("")
    print("  CV Models:")
    yolo_status = _model_status.get("yolo", "unknown")
    print(f"    [{'OK' if yolo_status == 'ready' else 'X'}] YOLO v8-nano (human detection)")

    # Dependencies status
    print("")
    print("  Dependencies:")

    # Required dependencies
    required_pkgs = ["anthropic", "openai", "pyyaml", "requests", "typing_extensions",
                     "transformers", "accelerate", "psutil", "ultralytics",
                     "opencv", "sklearn", "aiohttp", "tomlkit"]
    pkg_descriptions = {
        "anthropic": "Anthropic Claude API",
        "openai": "OpenAI/Grok/LM Studio API",
        "pyyaml": "YAML config parser",
        "requests": "HTTP library (Ollama)",
        "typing_extensions": "Type hints",
        "transformers": "HuggingFace Transformers",
        "accelerate": "HuggingFace Accelerate",
        "psutil": "System memory detection",
        "ultralytics": "YOLO (human detection)",
        "opencv": "OpenCV (image processing)",
        "sklearn": "Scikit-learn (color analysis)",
        "aiohttp": "Async HTTP (web routes)",
        "tomlkit": "TOML parser (prompt editor)",
    }

    for pkg in required_pkgs:
        status = _dependency_status.get(pkg, "unknown")
        desc = pkg_descriptions.get(pkg, pkg)
        if status == "installed":
            print(f"    [OK] {desc}")
        elif status == "just_installed":
            print(f"    [INSTALLED] {desc} (just installed)")
        elif status == "failed":
            print(f"    [FAILED] {desc} - please install manually")
        else:
            print(f"    [?] {desc}")

    # Available nodes
    print("")
    print("-" * 65)
    print("  Available Nodes:")
    print("")
    print("  LLM Providers:")
    print("    - SID_LLM_API       Cloud APIs (Anthropic, OpenAI, Gemini, Grok)")
    print("    - SID_LLM_Local     Local models (Florence-2, Moondream, SmolVLM, Phi-3.5, QwenVL)")
    print("    - SID_LLM_Local_API Local API providers (Ollama, LM Studio, OpenAI Compatible)")
    print("")
    print("  Prompt Tools:")
    print("    - SID_ZImagePromptGenerator  Z-Image prompts (CV detection + Vision LLM)")
    print("    - SID_PromptTemplate         Select and edit prompt templates")

    # Debug/Testing tools
    if SID_PromptDebugAgent is not None:
        print("")
        print("  Testing Tools:")
        print("    - SID_PromptDebugAgent  Evaluate prompt quality with Claude Opus 4.5")
        print("                            Compares source/output, scores against best practices")

    # Web Tools
    if _server_available:
        print("")
        print("  Web Tools:")
        print("    - Generation Results: http://localhost:8188/sid/generation-results")
        print("                          Browse saved prompts and metadata")
        print("    - Debug Viewer:       http://localhost:8188/sid/debug-results")
        print("                          Browse prompt evaluation results")

    print("")
    print("  Standalone CLI (batch processing for training):")
    print("    python prompt_generator_core.py --batch ./images/ --output prompts.jsonl")
    print("")
    print("=" * 65)
    print("")


def setup_api_routes(routes):
    """Setup API routes for SID Toolkit."""
    from .prompt_generator_core import get_template_by_name, get_template_names
    import os

    base_dir = Path(__file__).parent

    @routes.get("/sid/api/templates")
    async def get_templates(request):
        """Get list of all template names."""
        try:
            names = get_template_names() or []
            return web.json_response({"templates": names})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    @routes.get("/sid/api/template/{name}")
    async def get_template(request):
        """Get template content by name."""
        try:
            name = request.match_info.get("name", "")
            template = get_template_by_name(name)
            if template:
                return web.json_response({
                    "name": template.get("name", name),
                    "system": template.get("system", ""),
                    "description": template.get("description", "")
                })
            else:
                return web.json_response({"error": f"Template not found: {name}"}, status=404)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    # =========================================================================
    # Generation Results Viewer
    # =========================================================================

    @routes.get("/sid/generation-results")
    async def generation_results_page(request):
        """Serve generation results browser page."""
        results_dir = base_dir / "generation_results"
        sessions = []

        if results_dir.exists():
            for session_dir in sorted(results_dir.iterdir(), reverse=True):
                if session_dir.is_dir():
                    meta_file = session_dir / "metadata.json"
                    prompt_file = session_dir / "prompt.txt"
                    metadata = {}
                    prompt_preview = ""

                    if meta_file.exists():
                        try:
                            import json
                            metadata = json.loads(meta_file.read_text(encoding="utf-8"))
                        except:
                            pass

                    if prompt_file.exists():
                        try:
                            prompt_preview = prompt_file.read_text(encoding="utf-8")[:200] + "..."
                        except:
                            pass

                    sessions.append({
                        "id": session_dir.name,
                        "timestamp": metadata.get("timestamp", session_dir.name),
                        "model": metadata.get("model_config", {}).get("model", "unknown"),
                        "provider": metadata.get("model_config", {}).get("provider", "unknown"),
                        "prompt_preview": prompt_preview,
                    })

        html = _generate_results_html("Generation Results", sessions, "generation_results")
        return web.Response(text=html, content_type="text/html")

    @routes.get("/sid/generation-results/{session_id}")
    async def generation_result_detail(request):
        """Serve single generation result detail page."""
        session_id = request.match_info.get("session_id", "")
        session_dir = base_dir / "generation_results" / session_id

        if not session_dir.exists():
            return web.Response(text="Session not found", status=404)

        html = _generate_detail_html(session_dir, "generation")
        return web.Response(text=html, content_type="text/html")

    @routes.get("/sid/generation-results/{session_id}/{filename}")
    async def generation_result_file(request):
        """Serve files from generation results."""
        session_id = request.match_info.get("session_id", "")
        filename = request.match_info.get("filename", "")
        file_path = base_dir / "generation_results" / session_id / filename

        if not file_path.exists() or not file_path.is_file():
            return web.Response(text="File not found", status=404)

        # Determine content type
        ext = file_path.suffix.lower()
        content_types = {
            ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
            ".json": "application/json", ".txt": "text/plain"
        }
        content_type = content_types.get(ext, "application/octet-stream")

        return web.FileResponse(file_path, headers={"Content-Type": content_type})

    # =========================================================================
    # Debug Results Viewer
    # =========================================================================

    @routes.get("/sid/debug-results")
    async def debug_results_page(request):
        """Serve debug results browser page."""
        results_dir = base_dir / "debug_results"
        sessions = []

        if results_dir.exists():
            for session_dir in sorted(results_dir.iterdir(), reverse=True):
                if session_dir.is_dir():
                    eval_file = session_dir / "evaluation.json"
                    gen_model_file = session_dir / "generation_model.json"
                    evaluation = {}
                    gen_model = {}

                    if eval_file.exists():
                        try:
                            import json
                            evaluation = json.loads(eval_file.read_text(encoding="utf-8"))
                        except:
                            pass

                    if gen_model_file.exists():
                        try:
                            import json
                            gen_model = json.loads(gen_model_file.read_text(encoding="utf-8"))
                        except:
                            pass

                    # Extract score
                    score = "N/A"
                    scores = evaluation.get("scores", {})
                    if "overall" in scores:
                        score = scores["overall"].get("score", "N/A")

                    sessions.append({
                        "id": session_dir.name,
                        "timestamp": session_dir.name.replace("session_", ""),
                        "model": gen_model.get("model", "unknown"),
                        "provider": gen_model.get("provider", "unknown"),
                        "score": score,
                    })

        html = _generate_debug_results_html(sessions)
        return web.Response(text=html, content_type="text/html")

    @routes.get("/sid/debug-results/{session_id}")
    async def debug_result_detail(request):
        """Serve single debug result detail page."""
        session_id = request.match_info.get("session_id", "")
        session_dir = base_dir / "debug_results" / session_id

        if not session_dir.exists():
            return web.Response(text="Session not found", status=404)

        html = _generate_detail_html(session_dir, "debug")
        return web.Response(text=html, content_type="text/html")

    @routes.get("/sid/debug-results/{session_id}/{filename}")
    async def debug_result_file(request):
        """Serve files from debug results."""
        session_id = request.match_info.get("session_id", "")
        filename = request.match_info.get("filename", "")
        file_path = base_dir / "debug_results" / session_id / filename

        if not file_path.exists() or not file_path.is_file():
            return web.Response(text="File not found", status=404)

        ext = file_path.suffix.lower()
        content_types = {
            ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
            ".json": "application/json", ".txt": "text/plain"
        }
        content_type = content_types.get(ext, "application/octet-stream")

        return web.FileResponse(file_path, headers={"Content-Type": content_type})


def _generate_results_html(title, sessions, results_type):
    """Generate HTML for results list page."""
    rows = ""
    for s in sessions:
        rows += f"""
        <tr onclick="window.location='/sid/{results_type}/{s['id']}'" style="cursor:pointer;">
            <td>{s['timestamp']}</td>
            <td>{s['provider']}</td>
            <td>{s['model']}</td>
            <td style="max-width:400px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{s.get('prompt_preview', '')}</td>
        </tr>"""

    return f"""<!DOCTYPE html>
<html><head><title>{title} - SID Toolkit</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 20px; background: #1a1a2e; color: #eee; }}
h1 {{ color: #4a9eff; }}
table {{ width: 100%; border-collapse: collapse; background: #16213e; }}
th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #2a2a4a; }}
th {{ background: #0f3460; color: #4a9eff; }}
tr:hover {{ background: #1f4068; }}
a {{ color: #4a9eff; }}
.empty {{ text-align: center; padding: 40px; color: #888; }}
</style></head><body>
<h1>{title}</h1>
<p><a href="/sid/generation-results">Generation Results</a> | <a href="/sid/debug-results">Debug Results</a></p>
{'<table><tr><th>Timestamp</th><th>Provider</th><th>Model</th><th>Prompt Preview</th></tr>' + rows + '</table>' if sessions else '<div class="empty">No results yet. Run the prompt generator with "Store Results" enabled.</div>'}
</body></html>"""


def _generate_debug_results_html(sessions):
    """Generate HTML for debug results list page."""
    rows = ""
    for s in sessions:
        score_color = "#4caf50" if isinstance(s['score'], (int, float)) and s['score'] >= 7 else "#ff9800" if isinstance(s['score'], (int, float)) and s['score'] >= 5 else "#f44336"
        rows += f"""
        <tr onclick="window.location='/sid/debug-results/{s['id']}'" style="cursor:pointer;">
            <td>{s['timestamp']}</td>
            <td>{s['provider']}</td>
            <td>{s['model']}</td>
            <td style="color:{score_color};font-weight:bold;">{s['score']}</td>
        </tr>"""

    return f"""<!DOCTYPE html>
<html><head><title>Debug Results - SID Toolkit</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 20px; background: #1a1a2e; color: #eee; }}
h1 {{ color: #4a9eff; }}
table {{ width: 100%; border-collapse: collapse; background: #16213e; }}
th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #2a2a4a; }}
th {{ background: #0f3460; color: #4a9eff; }}
tr:hover {{ background: #1f4068; }}
a {{ color: #4a9eff; }}
.empty {{ text-align: center; padding: 40px; color: #888; }}
</style></head><body>
<h1>Debug Results</h1>
<p><a href="/sid/generation-results">Generation Results</a> | <a href="/sid/debug-results">Debug Results</a></p>
{'<table><tr><th>Timestamp</th><th>Provider</th><th>Model</th><th>Score</th></tr>' + rows + '</table>' if sessions else '<div class="empty">No debug results yet. Run the debug agent to evaluate prompts.</div>'}
</body></html>"""


def _generate_detail_html(session_dir, result_type):
    """Generate HTML for detail page."""
    import json

    files = {}
    for f in session_dir.iterdir():
        if f.is_file():
            files[f.name] = f

    # Read text files
    prompt = files.get("prompt.txt", None)
    prompt_text = prompt.read_text(encoding="utf-8") if prompt else ""

    # Read JSON files
    json_sections = ""
    json_files = ["metadata.json", "source_metadata.json", "evaluation.json", "generation_model.json", "evaluation_model.json", "model_config.json"]
    for jf in json_files:
        if jf in files:
            try:
                content = json.loads(files[jf].read_text(encoding="utf-8"))
                json_sections += f"<h3>{jf}</h3><pre>{json.dumps(content, indent=2)}</pre>"
            except:
                pass

    # Images
    images_html = ""
    for img_name in ["source.jpg", "output.jpg"]:
        if img_name in files:
            images_html += f'<div style="display:inline-block;margin:10px;"><h4>{img_name}</h4><img src="{img_name}" style="max-width:400px;max-height:400px;border:1px solid #444;"></div>'

    back_link = f"/sid/{result_type}-results" if result_type == "generation" else "/sid/debug-results"

    return f"""<!DOCTYPE html>
<html><head><title>{session_dir.name} - SID Toolkit</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 20px; background: #1a1a2e; color: #eee; }}
h1, h2, h3, h4 {{ color: #4a9eff; }}
pre {{ background: #16213e; padding: 15px; border-radius: 5px; overflow-x: auto; white-space: pre-wrap; word-wrap: break-word; }}
a {{ color: #4a9eff; }}
.prompt {{ background: #0f3460; padding: 20px; border-radius: 5px; white-space: pre-wrap; line-height: 1.6; }}
.images {{ margin: 20px 0; }}
</style></head><body>
<p><a href="{back_link}">&larr; Back to list</a></p>
<h1>{session_dir.name}</h1>
<div class="images">{images_html}</div>
<h2>Prompt</h2>
<div class="prompt">{prompt_text}</div>
{json_sections}
</body></html>"""


async def comfy_entrypoint() -> SIDPhotographyToolkitExtension:
    """
    ComfyUI calls this function to load the extension and its nodes.
    This is the entry point for the custom node package.
    """
    # Register API routes
    if _server_available:
        try:
            setup_api_routes(PromptServer.instance.routes)
            print("[SID-Toolkit] API routes registered")
        except Exception as e:
            print(f"[SID-Toolkit] Failed to register API routes: {e}")

    print_welcome_message()
    return SIDPhotographyToolkitExtension()
