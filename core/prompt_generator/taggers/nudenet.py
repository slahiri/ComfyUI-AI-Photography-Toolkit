"""
NudeNet tagger implementation.

Uses NudeNet for body part detection and NSFW classification.
"""

import os
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

# Lazy imports for heavy dependencies
_detector = None


def _get_model_path() -> str:
    """Get the path to cache NudeNet model files."""
    cache_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", ".cache", "nudenet")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _load_detector():
    """Load NudeNet detector."""
    global _detector

    if _detector is not None:
        return _detector

    try:
        from nudenet import NudeDetector
        _detector = NudeDetector()
        print("[NudeNet] Loaded detector")
        return _detector
    except ImportError:
        print("[NudeNet] nudenet not installed. Install with: pip install nudenet")
        return None


def _image_to_path_or_array(image: Any) -> Any:
    """Convert image to format NudeNet expects."""
    # Convert ComfyUI tensor to numpy
    if hasattr(image, 'cpu'):
        img_np = image[0].cpu().numpy()
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
        else:
            img_np = img_np.astype(np.uint8)
        return img_np
    elif isinstance(image, Image.Image):
        return np.array(image.convert("RGB"))
    else:
        return np.array(image[0])


def run_nudenet(
    image: Any,
    threshold: float = 0.3,
) -> Dict[str, Any]:
    """
    Run NudeNet on an image.

    Args:
        image: Image tensor from ComfyUI or PIL Image
        threshold: Detection confidence threshold

    Returns:
        Dict with:
            - gender: "female" or "male" or None
            - parts: Dict[str, float] of detected parts with confidences
            - exposure_score: float 0-1 indicating exposure level
            - detections: List of raw detection dicts
    """
    try:
        detector = _load_detector()
        if detector is None:
            return {}

        # Convert image
        img_np = _image_to_path_or_array(image)

        # Save temp image (NudeNet works best with file paths)
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            temp_path = f.name
            Image.fromarray(img_np).save(temp_path, "JPEG")

        try:
            # Run detection
            detections = detector.detect(temp_path)
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_path)
            except:
                pass

        if not detections:
            return {
                "gender": None,
                "parts": {},
                "exposure_score": 0.0,
                "detections": [],
            }

        # Parse detections
        parts: Dict[str, float] = {}
        exposure_parts = []

        # Define exposure levels for different body parts
        exposure_weights = {
            "EXPOSED_BREAST_F": 1.0,
            "EXPOSED_BREAST_M": 0.3,
            "EXPOSED_GENITALIA_F": 1.0,
            "EXPOSED_GENITALIA_M": 1.0,
            "EXPOSED_BUTTOCKS": 0.8,
            "EXPOSED_ANUS": 1.0,
            "COVERED_BREAST_F": 0.2,
            "COVERED_GENITALIA_F": 0.2,
            "COVERED_GENITALIA_M": 0.1,
            "COVERED_BUTTOCKS": 0.1,
            "FACE_F": 0.0,
            "FACE_M": 0.0,
            "BELLY": 0.1,
            "FEET": 0.0,
            "ARMPITS": 0.05,
        }

        # Track gender
        has_female = False
        has_male = False

        for det in detections:
            class_name = det.get("class", "")
            score = det.get("score", 0)

            if score < threshold:
                continue

            # Track parts
            parts[class_name] = max(parts.get(class_name, 0), score)

            # Track exposure
            weight = exposure_weights.get(class_name, 0)
            if weight > 0:
                exposure_parts.append(score * weight)

            # Track gender
            if "_F" in class_name or "FEMALE" in class_name.upper():
                has_female = True
            elif "_M" in class_name or "MALE" in class_name.upper():
                has_male = True

        # Determine gender
        gender = None
        if has_female and not has_male:
            gender = "female"
        elif has_male and not has_female:
            gender = "male"

        # Calculate exposure score (max of weighted exposures, capped at 1.0)
        exposure_score = min(1.0, max(exposure_parts) if exposure_parts else 0.0)

        result = {
            "gender": gender,
            "parts": parts,
            "exposure_score": exposure_score,
            "detections": detections,
        }

        print(f"[NudeNet] Detected {len(parts)} parts, exposure={exposure_score:.2f}, gender={gender}")
        return result

    except ImportError as e:
        print(f"[NudeNet] Missing dependency: {e}")
        return {}
    except Exception as e:
        print(f"[NudeNet] Error: {e}")
        import traceback
        traceback.print_exc()
        return {}


def unload_detector():
    """Unload NudeNet detector from memory."""
    global _detector
    _detector = None
    print("[NudeNet] Detector unloaded")
