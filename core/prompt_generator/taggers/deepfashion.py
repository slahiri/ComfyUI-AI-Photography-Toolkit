"""
DeepFashion analyzer implementation.

Analyzes clothing attributes: fabric, pattern, texture, style.
"""

import os
from typing import Any, Dict, List

import numpy as np
from PIL import Image

_model = None
_labels = None


# Fashion attributes to detect
FABRIC_TYPES = [
    "cotton", "silk", "wool", "leather", "denim",
    "linen", "velvet", "satin", "lace", "chiffon",
    "polyester", "knit", "jersey", "tweed", "corduroy",
]

PATTERN_TYPES = [
    "solid", "striped", "plaid", "floral", "polka dot",
    "geometric", "abstract", "paisley", "animal print",
    "camouflage", "checkered", "herringbone", "houndstooth",
]

TEXTURE_TYPES = [
    "smooth", "rough", "shiny", "matte", "textured",
    "ribbed", "quilted", "embroidered", "sequined", "beaded",
]

STYLE_TYPES = [
    "casual", "formal", "sporty", "bohemian", "vintage",
    "minimalist", "streetwear", "elegant", "punk", "preppy",
]


def _get_model_path() -> str:
    """Get the path to cache DeepFashion model files."""
    cache_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", ".cache", "deepfashion")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _image_to_array(image: Any) -> np.ndarray:
    """Convert image to numpy array."""
    if hasattr(image, 'cpu'):
        img_np = image[0].cpu().numpy()
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
        else:
            img_np = img_np.astype(np.uint8)
        return img_np
    elif isinstance(image, Image.Image):
        return np.array(image.convert("RGB"))
    elif isinstance(image, np.ndarray):
        if image.ndim == 4:
            image = image[0]
        if image.ndim == 2:
            return np.stack([image] * 3, axis=2)
        return image
    else:
        return np.array(image)


def _load_clip_model():
    """Load CLIP model for fashion attribute classification."""
    global _model, _labels

    if _model is not None:
        return _model, _labels

    try:
        import torch
        from transformers import CLIPProcessor, CLIPModel

        model_id = "openai/clip-vit-base-patch32"
        cache_dir = _get_model_path()

        processor = CLIPProcessor.from_pretrained(model_id, cache_dir=cache_dir)
        model = CLIPModel.from_pretrained(model_id, cache_dir=cache_dir)
        model.eval()

        if torch.cuda.is_available():
            model = model.cuda()

        _model = (model, processor)
        _labels = {
            "fabric": FABRIC_TYPES,
            "pattern": PATTERN_TYPES,
            "texture": TEXTURE_TYPES,
            "style": STYLE_TYPES,
        }

        print("[DeepFashion] Loaded CLIP model for fashion analysis")
        return _model, _labels

    except ImportError as e:
        print(f"[DeepFashion] Missing dependency: {e}")
        _model = "not_available"
        _labels = None
        return _model, _labels


def _classify_attribute(
    model: Any,
    processor: Any,
    pil_image: Image.Image,
    attribute_name: str,
    options: List[str],
) -> Dict[str, float]:
    """
    Classify an attribute using CLIP zero-shot classification.
    """
    import torch

    # Create prompts for clothing attribute
    prompts = [f"clothing made of {opt}" if attribute_name == "fabric"
               else f"clothing with {opt} pattern" if attribute_name == "pattern"
               else f"clothing with {opt} texture" if attribute_name == "texture"
               else f"{opt} style clothing"
               for opt in options]

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
        logits = outputs.logits_per_image
        probs = logits.softmax(dim=1)[0]

    result = {}
    for opt, prob in zip(options, probs.cpu().numpy()):
        result[opt] = float(prob)

    return result


def _analyze_texture_heuristic(img: np.ndarray) -> Dict[str, float]:
    """
    Analyze texture using image statistics.
    """
    gray = np.mean(img, axis=2)

    # Texture metrics
    try:
        import cv2

        # Edge density (textured surfaces have more edges)
        edges = cv2.Canny(gray.astype(np.uint8), 50, 150)
        edge_density = np.sum(edges > 0) / edges.size

        # Local variance (texture indicator)
        local_var = cv2.blur(gray ** 2, (11, 11)) - cv2.blur(gray, (11, 11)) ** 2
        texture_score = np.mean(local_var) / 1000

    except ImportError:
        # Fallback
        gx = np.diff(gray, axis=1)
        gy = np.diff(gray, axis=0)
        edge_density = np.mean(np.abs(gx)) + np.mean(np.abs(gy))
        edge_density = min(1.0, edge_density / 50)
        texture_score = np.var(gray) / 1000

    # Classify based on metrics
    result = {
        "smooth": max(0, 1 - edge_density * 2),
        "textured": min(1, edge_density * 2),
        "rough": min(1, texture_score),
    }

    # Check for shine/matte
    highlights = np.sum(gray > 240) / gray.size
    result["shiny"] = min(1.0, highlights * 10)
    result["matte"] = max(0, 1 - result["shiny"])

    return result


def run_deepfashion(image: Any) -> Dict[str, Any]:
    """
    Analyze clothing/fashion attributes.

    Args:
        image: Image tensor from ComfyUI or PIL Image

    Returns:
        Dict with fashion analysis:
            - fabric: Dict of fabric type scores
            - pattern: Dict of pattern type scores
            - texture: Dict of texture type scores
            - style: Dict of style type scores
            - best_fabric, best_pattern, etc.
    """
    try:
        img = _image_to_array(image)
        pil_image = Image.fromarray(img).convert("RGB")

        model_data, labels = _load_clip_model()

        if model_data == "not_available" or labels is None:
            # Use heuristic analysis only
            result = {
                "fabric": {"cotton": 0.5},
                "pattern": {"solid": 0.5},
                "texture": _analyze_texture_heuristic(img),
                "style": {"casual": 0.5},
            }
        else:
            model, processor = model_data

            result = {}
            for attr_name, options in labels.items():
                scores = _classify_attribute(model, processor, pil_image, attr_name, options)
                result[attr_name] = scores

        # Add best matches
        for attr_name in ["fabric", "pattern", "texture", "style"]:
            if attr_name in result:
                scores = result[attr_name]
                best = max(scores, key=scores.get)
                result[f"best_{attr_name}"] = best
                result[f"best_{attr_name}_score"] = scores[best]

        print(f"[DeepFashion] Fabric: {result.get('best_fabric', 'unknown')}, Pattern: {result.get('best_pattern', 'unknown')}")
        return result

    except Exception as e:
        print(f"[DeepFashion] Error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "fabric": {},
            "pattern": {},
            "texture": {},
            "style": {},
        }


def unload_model():
    """Unload DeepFashion model from memory."""
    global _model, _labels
    if _model is not None and _model != "not_available":
        del _model
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    _model = None
    _labels = None
    print("[DeepFashion] Model unloaded")
