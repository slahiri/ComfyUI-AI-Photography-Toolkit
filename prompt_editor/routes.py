# -*- coding: utf-8 -*-
"""
Prompt Editor Routes

API endpoints for the prompt editor web UI.
"""

import json
from pathlib import Path
from datetime import datetime
from aiohttp import web

# Get the config/prompts directory
TOOLKIT_DIR = Path(__file__).parent.parent
PROMPTS_DIR = TOOLKIT_DIR / "config" / "prompts"
BACKUP_DIR = TOOLKIT_DIR / "config" / "prompts_backup"


def setup_routes(routes):
    """Register all prompt editor routes."""

    @routes.get("/sid/prompt-editor")
    async def serve_editor(request):
        """Serve the prompt editor HTML page."""
        return web.Response(text=get_editor_html(), content_type="text/html")

    @routes.get("/sid/prompt-editor/api/files")
    async def list_files(request):
        """List all TOML files in the prompts directory."""
        files = []
        if PROMPTS_DIR.exists():
            for f in sorted(PROMPTS_DIR.glob("*.toml")):
                files.append({
                    "name": f.name,
                    "path": str(f.relative_to(TOOLKIT_DIR)),
                    "size": f.stat().st_size,
                    "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat()
                })
        return web.json_response({"files": files})

    @routes.get("/sid/prompt-editor/api/file/{filename}")
    async def get_file(request):
        """Get contents of a specific TOML file."""
        filename = request.match_info["filename"]

        # Security: only allow .toml files in prompts directory
        if not filename.endswith(".toml"):
            return web.json_response({"error": "Only .toml files allowed"}, status=400)

        filepath = PROMPTS_DIR / filename
        if not filepath.exists():
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

        filepath = PROMPTS_DIR / filename

        try:
            data = await request.json()
            content = data.get("content", "")

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

        if not filename.endswith(".toml"):
            return web.json_response({"error": "Only .toml files allowed"}, status=400)

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
            filepath = PROMPTS_DIR / filename
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
        </div>
        <div class="editor-container">
            <div class="toolbar">
                <div class="toolbar-title" id="toolbarTitle">Select a file to edit</div>
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
                    <div class="file-item" data-file="${f.name}" onclick="loadFile('${f.name}')">
                        <span class="file-icon">📄</span>
                        <span>${f.name}</span>
                    </div>
                `).join('');
            } catch (e) {
                document.getElementById('fileList').innerHTML =
                    '<div class="empty-state">Error loading files</div>';
            }
        }

        async function loadFile(filename) {
            if (isModified && !confirm('You have unsaved changes. Discard them?')) {
                return;
            }

            setStatus('Loading...', '');

            try {
                const res = await fetch(`/sid/prompt-editor/api/file/${filename}`);
                const data = await res.json();

                if (data.error) {
                    setStatus(data.error, 'error');
                    return;
                }

                currentFile = filename;
                originalContent = data.content;
                editor.setValue(data.content);
                setModified(false);

                // Update UI
                document.getElementById('toolbarTitle').innerHTML =
                    `Editing: <strong>${filename}</strong>`;
                document.getElementById('saveBtn').disabled = true;
                document.getElementById('resetBtn').disabled = false;

                // Highlight active file
                document.querySelectorAll('.file-item').forEach(el => {
                    el.classList.toggle('active', el.dataset.file === filename);
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
                    body: JSON.stringify({ content: editor.getValue() })
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

        async function resetFile() {
            if (!currentFile) return;

            if (!confirm(`Reset ${currentFile} to the original git version?\\n\\nYour current changes will be backed up.`)) {
                return;
            }

            setStatus('Resetting...', '');

            try {
                const res = await fetch(`/sid/prompt-editor/api/reset/${currentFile}`, {
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
