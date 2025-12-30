"""
Shot type analyzer implementation.

Uses a classification model or heuristics to determine camera framing.
"""

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image

# Lazy imports
_model = None
_labels = None


# Shot type labels matching the ShotType enum
SHOT_TYPE_LABELS = [
    "ECU",   # Extreme close-up (face detail)
    "CU",    # Close-up (head and shoulders)
    "MCU",   # Medium close-up (chest up)
    "MS",    # Medium shot (waist up)
    "MFS",   # Medium full shot (cowboy shot, thighs up)
    "FS",    # Full shot (full body)
    "MLS",   # Medium long shot (full body + environment)
    "LS",    # Long shot (environment prominent)
    "ELS",   # Extreme long shot (environment dominant)
]


def _get_model_path() -> str:
    """Get the path to cache shot type model files."""
    cache_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", ".cache", "shot_type")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _load_model():
    """Load shot type classification model."""
    global _model, _labels

    if _model is not None:
        return _model, _labels

    # For now, we'll use a heuristic-based approach with face detection
    # until we have a trained shot type classifier
    _labels = SHOT_TYPE_LABELS

    try:
        # Try to load a pre-trained model if available
        import timm
        import torch

        # Check for custom trained model first
        custom_model_path = os.path.join(_get_model_path(), "shot_type_model.pth")
        if os.path.exists(custom_model_path):
            _model = torch.load(custom_model_path)
            print("[ShotType] Loaded custom model")
            return _model, _labels

        # Use MobileNetV3 as base for shot type classification
        _model = timm.create_model("mobilenetv3_small_100", pretrained=True, num_classes=len(SHOT_TYPE_LABELS))
        _model.eval()
        print("[ShotType] Using MobileNetV3 base (untrained for shot type)")
        return _model, _labels

    except ImportError:
        print("[ShotType] Using heuristic-based detection (no timm/torch)")
        _model = "heuristic"
        return _model, _labels


def _estimate_shot_type_heuristic(image: Image.Image) -> Dict[str, float]:
    """
    Estimate shot type using face detection and aspect ratio heuristics.

    This is a fallback when no trained model is available.
    """
    try:
        # Try face detection
        face_ratio = _detect_face_ratio(image)

        if face_ratio is None:
            # No face detected - likely landscape or object shot
            return {
                "FS": 0.5,
                "MLS": 0.3,
                "LS": 0.2,
            }

        # Use face size relative to image to estimate shot type
        # face_ratio = face_area / image_area
        if face_ratio > 0.3:
            return {"ECU": 0.8, "CU": 0.2}
        elif face_ratio > 0.15:
            return {"CU": 0.7, "MCU": 0.3}
        elif face_ratio > 0.08:
            return {"MCU": 0.6, "MS": 0.3, "CU": 0.1}
        elif face_ratio > 0.04:
            return {"MS": 0.6, "MFS": 0.3, "MCU": 0.1}
        elif face_ratio > 0.02:
            return {"MFS": 0.5, "FS": 0.4, "MS": 0.1}
        elif face_ratio > 0.01:
            return {"FS": 0.6, "MLS": 0.3, "MFS": 0.1}
        elif face_ratio > 0.005:
            return {"MLS": 0.5, "LS": 0.4, "FS": 0.1}
        else:
            return {"LS": 0.5, "ELS": 0.4, "MLS": 0.1}

    except Exception as e:
        print(f"[ShotType] Heuristic error: {e}")
        return {"MS": 0.5}  # Default to medium shot


def _detect_face_ratio(image: Image.Image) -> Optional[float]:
    """
    Detect faces and return the ratio of largest face area to image area.

    Returns None if no face detected.
    """
    try:
        # Try using OpenCV face detection
        import cv2
        import numpy as np

        img_np = np.array(image.convert("RGB"))
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        # Load Haar cascade
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(cascade_path)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        if len(faces) == 0:
            return None

        # Find largest face
        image_area = image.width * image.height
        max_face_area = 0

        for (x, y, w, h) in faces:
            face_area = w * h
            max_face_area = max(max_face_area, face_area)

        return max_face_area / image_area

    except ImportError:
        print("[ShotType] OpenCV not available for face detection")
        return None
    except Exception as e:
        print(f"[ShotType] Face detection error: {e}")
        return None


def run_shot_type(
    image: Any,
    use_heuristic: bool = True,
) -> Dict[str, float]:
    """
    Analyze shot type/camera framing.

    Args:
        image: Image tensor from ComfyUI or PIL Image
        use_heuristic: Use heuristic fallback if model unavailable

    Returns:
        Dict mapping shot type labels to confidence scores
    """
    try:
        # Convert ComfyUI tensor to PIL
        if hasattr(image, 'cpu'):
            img_np = image[0].cpu().numpy()
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)
            else:
                img_np = img_np.astype(np.uint8)
            pil_image = Image.fromarray(img_np)
        elif isinstance(image, Image.Image):
            pil_image = image
        else:
            pil_image = Image.fromarray(np.array(image[0]))

        # Load model
        model, labels = _load_model()

        # Use heuristic if model is not trained
        if model == "heuristic" or use_heuristic:
            result = _estimate_shot_type_heuristic(pil_image)
            best_shot = max(result, key=result.get)
            print(f"[ShotType] Estimated: {best_shot} ({result[best_shot]:.2f})")
            return result

        # If we have a trained model, use it
        import torch
        from torchvision import transforms

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        input_tensor = transform(pil_image.convert("RGB")).unsqueeze(0)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)[0]

        result = {}
        for i, label in enumerate(labels):
            result[label] = float(probs[i])

        best_shot = max(result, key=result.get)
        print(f"[ShotType] Detected: {best_shot} ({result[best_shot]:.2f})")
        return result

    except Exception as e:
        print(f"[ShotType] Error: {e}")
        import traceback
        traceback.print_exc()
        return {"MS": 0.5}  # Default to medium shot


def unload_model():
    """Unload shot type model from memory."""
    global _model, _labels
    _model = None
    _labels = None
    print("[ShotType] Model unloaded")
