"""
CLIP Camera analyzer implementation.

Uses CLIP to classify camera angles, DOF, and perspective.
"""

import os
from typing import Any, Dict, List

import numpy as np
from PIL import Image

_model = None
_processor = None


# Camera angle prompts for CLIP
CAMERA_ANGLE_PROMPTS = [
    "eye level shot",
    "low angle shot looking up",
    "high angle shot looking down",
    "bird's eye view from above",
    "worm's eye view from below",
    "dutch angle tilted shot",
    "over the shoulder shot",
    "point of view shot",
]

# DOF prompts
DOF_PROMPTS = [
    "shallow depth of field with blurred background",
    "deep depth of field with everything in focus",
    "bokeh effect with circular blur",
    "tilt-shift miniature effect",
]

# Perspective prompts
PERSPECTIVE_PROMPTS = [
    "wide angle perspective with distortion",
    "telephoto compressed perspective",
    "normal perspective natural look",
    "fisheye distorted perspective",
    "macro close-up perspective",
]


def _get_model_path() -> str:
    """Get the path to cache CLIP model files."""
    cache_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", ".cache", "clip")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _load_model():
    """Load CLIP model."""
    global _model, _processor

    if _model is not None:
        return _model, _processor

    try:
        import torch
        from transformers import CLIPProcessor, CLIPModel

        model_id = "openai/clip-vit-base-patch32"
        cache_dir = _get_model_path()

        _processor = CLIPProcessor.from_pretrained(model_id, cache_dir=cache_dir)
        _model = CLIPModel.from_pretrained(model_id, cache_dir=cache_dir)
        _model.eval()

        if torch.cuda.is_available():
            _model = _model.cuda()

        print("[CLIP Camera] Loaded CLIP model")
        return _model, _processor

    except ImportError as e:
        print(f"[CLIP Camera] Missing dependency: {e}")
        print("[CLIP Camera] Install with: pip install transformers")
        _model = "not_available"
        _processor = None
        return _model, _processor
    except Exception as e:
        print(f"[CLIP Camera] Error loading model: {e}")
        _model = "not_available"
        _processor = None
        return _model, _processor


def _classify_with_clip(
    model: Any,
    processor: Any,
    pil_image: Image.Image,
    prompts: List[str],
) -> Dict[str, float]:
    """
    Classify image against a list of prompts using CLIP.
    """
    import torch

    # Prepare inputs
    inputs = processor(
        text=prompts,
        images=pil_image,
        return_tensors="pt",
        padding=True,
    )

    if torch.cuda.is_available() and next(model.parameters()).is_cuda:
        inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)[0]

    result = {}
    for prompt, prob in zip(prompts, probs.cpu().numpy()):
        result[prompt] = float(prob)

    return result


def _image_to_pil(image: Any) -> Image.Image:
    """Convert image to PIL."""
    if hasattr(image, 'cpu'):
        img_np = image[0].cpu().numpy()
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
        else:
            img_np = img_np.astype(np.uint8)
        return Image.fromarray(img_np)
    elif isinstance(image, Image.Image):
        return image
    elif isinstance(image, np.ndarray):
        if image.ndim == 4:
            image = image[0]
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=2)
        return Image.fromarray(image.astype(np.uint8))
    else:
        return Image.fromarray(np.array(image[0]))


def run_clip_camera(image: Any) -> Dict[str, Any]:
    """
    Analyze camera angles, DOF, and perspective using CLIP.

    Args:
        image: Image tensor from ComfyUI or PIL Image

    Returns:
        Dict with camera analysis:
            - camera_angle: Dict of angle types with scores
            - dof: Dict of DOF types with scores
            - perspective: Dict of perspective types with scores
            - best_angle: Top camera angle
            - best_dof: Top DOF type
            - best_perspective: Top perspective type
    """
    try:
        model, processor = _load_model()

        if model == "not_available" or processor is None:
            return _heuristic_camera_analysis(image)

        pil_image = _image_to_pil(image).convert("RGB")

        # Classify camera angles
        angle_scores = _classify_with_clip(model, processor, pil_image, CAMERA_ANGLE_PROMPTS)

        # Classify DOF
        dof_scores = _classify_with_clip(model, processor, pil_image, DOF_PROMPTS)

        # Classify perspective
        perspective_scores = _classify_with_clip(model, processor, pil_image, PERSPECTIVE_PROMPTS)

        # Get best matches
        best_angle = max(angle_scores, key=angle_scores.get)
        best_dof = max(dof_scores, key=dof_scores.get)
        best_perspective = max(perspective_scores, key=perspective_scores.get)

        result = {
            "camera_angle": angle_scores,
            "dof": dof_scores,
            "perspective": perspective_scores,
            "best_angle": best_angle,
            "best_angle_score": angle_scores[best_angle],
            "best_dof": best_dof,
            "best_dof_score": dof_scores[best_dof],
            "best_perspective": best_perspective,
            "best_perspective_score": perspective_scores[best_perspective],
        }

        print(f"[CLIP Camera] Angle: {best_angle}, DOF: {best_dof}")
        return result

    except Exception as e:
        print(f"[CLIP Camera] Error: {e}")
        import traceback
        traceback.print_exc()
        return _heuristic_camera_analysis(image)


def _heuristic_camera_analysis(image: Any) -> Dict[str, Any]:
    """
    Heuristic camera analysis when CLIP is not available.
    """
    # Return default values
    return {
        "camera_angle": {"eye level shot": 0.5},
        "dof": {"normal depth of field": 0.5},
        "perspective": {"normal perspective natural look": 0.5},
        "best_angle": "eye level shot",
        "best_angle_score": 0.5,
        "best_dof": "normal depth of field",
        "best_dof_score": 0.5,
        "best_perspective": "normal perspective natural look",
        "best_perspective_score": 0.5,
    }


def unload_model():
    """Unload CLIP model from memory."""
    global _model, _processor
    if _model is not None and _model != "not_available":
        del _model
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    _model = None
    _processor = None
    print("[CLIP Camera] Model unloaded")
