# -*- coding: utf-8 -*-
"""
Results Storage Manager

Handles saving debug evaluation results to disk.
"""

import json
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
from PIL import Image
import io
import base64


class ResultsStorage:
    """
    Manages storage of debug evaluation results.

    Directory structure:
    debug_results/
    └── YYYY-MM-DD_HH-MM-SS_<session_id>/
        ├── source.jpg
        ├── output.jpg
        ├── prompt.txt
        ├── evaluation.json
        └── metadata.json
    """

    def __init__(self, base_path: Optional[Path] = None):
        """
        Initialize the results storage.

        Args:
            base_path: Path to debug_results directory.
                      Defaults to toolkit's debug_results folder.
        """
        if base_path is None:
            self.base_path = Path(__file__).parent.parent / "debug_results"
        else:
            self.base_path = Path(base_path)

        # Ensure directory exists
        self.base_path.mkdir(parents=True, exist_ok=True)

    def create_session(self) -> tuple[str, Path]:
        """
        Create a new session directory.

        Returns:
            Tuple of (session_id, session_path)
        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        session_id = str(uuid.uuid4())[:8]
        folder_name = f"{timestamp}_{session_id}"

        session_path = self.base_path / folder_name
        session_path.mkdir(parents=True, exist_ok=True)

        return session_id, session_path

    def save_image(self, session_path: Path, image_data: Any,
                   filename: str = "image.jpg") -> Dict[str, Any]:
        """
        Save an image to the session directory.

        Args:
            session_path: Path to session directory
            image_data: Image data (PIL Image, base64 string, or tensor)
            filename: Output filename

        Returns:
            Dictionary with image metadata
        """
        file_path = session_path / filename

        # Convert to PIL Image if needed
        if isinstance(image_data, str):
            # Base64 string
            image_bytes = base64.b64decode(image_data)
            pil_image = Image.open(io.BytesIO(image_bytes))
        elif hasattr(image_data, 'cpu'):
            # PyTorch tensor
            import numpy as np
            if len(image_data.shape) == 4:
                image_data = image_data[0]
            np_image = (image_data.cpu().numpy() * 255).astype(np.uint8)
            pil_image = Image.fromarray(np_image)
        elif isinstance(image_data, Image.Image):
            pil_image = image_data
        else:
            raise ValueError(f"Unsupported image type: {type(image_data)}")

        # Save image
        pil_image.save(file_path, "JPEG", quality=95)

        # Get file size
        file_size_kb = file_path.stat().st_size / 1024

        return {
            "path": str(file_path),
            "relative_path": filename,
            "dimensions": list(pil_image.size),
            "format": "JPEG",
            "file_size_kb": round(file_size_kb, 1)
        }

    def save_prompt(self, session_path: Path, prompt: str) -> Dict[str, Any]:
        """
        Save the prompt text to the session directory.

        Args:
            session_path: Path to session directory
            prompt: Prompt text

        Returns:
            Dictionary with prompt metadata
        """
        file_path = session_path / "prompt.txt"

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(prompt)

        return {
            "path": str(file_path),
            "word_count": len(prompt.split()),
            "character_count": len(prompt),
            "content": prompt
        }

    def save_metadata(self, session_path: Path, metadata: Dict[str, Any]) -> bool:
        """
        Save session metadata.

        Args:
            session_path: Path to session directory
            metadata: Metadata dictionary

        Returns:
            True if successful
        """
        file_path = session_path / "metadata.json"

        # Add timestamp if not present
        if "timestamp" not in metadata:
            metadata["timestamp"] = datetime.utcnow().isoformat() + "Z"

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"[Storage] Error saving metadata: {e}")
            return False

    def save_evaluation(self, session_path: Path, evaluation: Dict[str, Any]) -> bool:
        """
        Save the full evaluation results.

        Args:
            session_path: Path to session directory
            evaluation: Full evaluation dictionary

        Returns:
            True if successful
        """
        file_path = session_path / "evaluation.json"

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(evaluation, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"[Storage] Error saving evaluation: {e}")
            return False

    def save_full_session(
        self,
        source_image: Any,
        output_image: Any,
        prompt: str,
        model_config: Dict[str, Any],
        evaluation: Dict[str, Any],
        generator_settings: Optional[Dict[str, Any]] = None,
        system_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Save a complete debug session.

        Args:
            source_image: Source image data
            output_image: Output image data
            prompt: Generated prompt
            model_config: LLM model configuration
            evaluation: Full evaluation results
            generator_settings: Optional generator settings
            system_info: Optional system information

        Returns:
            Dictionary with session info and paths
        """
        session_id, session_path = self.create_session()

        # Save images
        source_info = self.save_image(session_path, source_image, "source.jpg")
        output_info = self.save_image(session_path, output_image, "output.jpg")

        # Save prompt
        prompt_info = self.save_prompt(session_path, prompt)

        # Build metadata
        metadata = {
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "model_config": model_config,
        }

        if generator_settings:
            metadata["generator_settings"] = generator_settings

        if system_info:
            metadata["system_info"] = system_info

        # Save metadata
        self.save_metadata(session_path, metadata)

        # Add input info to evaluation
        evaluation["inputs"] = {
            "source_image": source_info,
            "output_image": output_info,
            "prompt": prompt_info,
        }
        evaluation["debug_session"] = {
            "id": session_id,
            "timestamp": metadata["timestamp"],
            "path": str(session_path),
        }

        # Save evaluation
        self.save_evaluation(session_path, evaluation)

        return {
            "session_id": session_id,
            "path": str(session_path),
            "files": {
                "source": source_info,
                "output": output_info,
                "prompt": prompt_info,
                "metadata": str(session_path / "metadata.json"),
                "evaluation": str(session_path / "evaluation.json"),
            }
        }

    def list_sessions(self, limit: int = 50) -> list[Dict[str, Any]]:
        """
        List recent debug sessions.

        Args:
            limit: Maximum number of sessions to return

        Returns:
            List of session info dictionaries
        """
        sessions = []

        if not self.base_path.exists():
            return sessions

        # Get all session directories
        session_dirs = sorted(
            [d for d in self.base_path.iterdir() if d.is_dir()],
            reverse=True  # Most recent first
        )[:limit]

        for session_dir in session_dirs:
            session_info = {
                "path": str(session_dir),
                "name": session_dir.name,
            }

            # Try to load metadata
            metadata_path = session_dir / "metadata.json"
            if metadata_path.exists():
                try:
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)
                        session_info["timestamp"] = metadata.get("timestamp")
                        session_info["model"] = metadata.get("model_config", {}).get("model")
                except:
                    pass

            # Try to get score from evaluation
            eval_path = session_dir / "evaluation.json"
            if eval_path.exists():
                try:
                    with open(eval_path, "r") as f:
                        evaluation = json.load(f)
                        session_info["score"] = evaluation.get("evaluation", {}).get("scores", {}).get("overall", {}).get("score")
                except:
                    pass

            sessions.append(session_info)

        return sessions

    def load_session(self, session_path: str) -> Optional[Dict[str, Any]]:
        """
        Load a previous session's data.

        Args:
            session_path: Path to session directory

        Returns:
            Session data dictionary or None if not found
        """
        path = Path(session_path)

        if not path.exists():
            return None

        session_data = {"path": str(path)}

        # Load metadata
        metadata_path = path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                session_data["metadata"] = json.load(f)

        # Load evaluation
        eval_path = path / "evaluation.json"
        if eval_path.exists():
            with open(eval_path, "r") as f:
                session_data["evaluation"] = json.load(f)

        # Load prompt
        prompt_path = path / "prompt.txt"
        if prompt_path.exists():
            with open(prompt_path, "r") as f:
                session_data["prompt"] = f.read()

        # Check for images
        session_data["has_source"] = (path / "source.jpg").exists()
        session_data["has_output"] = (path / "output.jpg").exists()

        return session_data
