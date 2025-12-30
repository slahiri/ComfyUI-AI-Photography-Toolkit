"""
Landmarks detector using CLIP.

Detects famous buildings, monuments, and landmarks.
"""

import os
from typing import Any, Dict, List

import numpy as np
from PIL import Image

_model = None
_processor = None


# Landmark prompts for CLIP classification
LANDMARK_PROMPTS = [
    # Famous Buildings
    "Eiffel Tower in Paris",
    "Statue of Liberty in New York",
    "Big Ben clock tower in London",
    "Colosseum ancient Roman amphitheater",
    "Taj Mahal white marble mausoleum",
    "Great Wall of China",
    "Sydney Opera House",
    "Golden Gate Bridge San Francisco",
    "Empire State Building New York",
    "Burj Khalifa Dubai skyscraper",
    "Leaning Tower of Pisa",
    "Christ the Redeemer statue Rio",
    "Machu Picchu ancient ruins",
    "Pyramids of Giza Egypt",
    "Stonehenge ancient monument",
    "Tower Bridge London",
    "Arc de Triomphe Paris",
    "Notre Dame Cathedral",
    "St. Basil's Cathedral Moscow",
    "Brandenburg Gate Berlin",
    # Generic architecture
    "famous landmark building",
    "historic monument",
    "ancient ruins archaeological site",
    "modern skyscraper cityscape",
    "historic castle fortress",
    "cathedral church architecture",
    "palace royal residence",
    "bridge famous architectural",
    # No landmark
    "ordinary building no landmark",
    "residential house neighborhood",
    "generic street scene",
]

# Categories for grouping
LANDMARK_CATEGORIES = {
    "famous_landmarks": [
        "Eiffel Tower", "Statue of Liberty", "Big Ben", "Colosseum",
        "Taj Mahal", "Great Wall", "Sydney Opera House", "Golden Gate Bridge",
        "Empire State Building", "Burj Khalifa", "Leaning Tower of Pisa",
        "Christ the Redeemer", "Machu Picchu", "Pyramids of Giza",
        "Stonehenge", "Tower Bridge", "Arc de Triomphe", "Notre Dame",
        "St. Basil's Cathedral", "Brandenburg Gate",
    ],
    "architecture_types": [
        "landmark", "monument", "ruins", "skyscraper", "castle",
        "cathedral", "palace", "bridge",
    ],
}


def _get_model_path() -> str:
    """Get the path to cache CLIP model files."""
    cache_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", ".cache", "clip")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _load_model():
    """Load CLIP model (shared with clip_camera)."""
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

        print("[Landmarks] Loaded CLIP model")
        return _model, _processor

    except ImportError as e:
        print(f"[Landmarks] Missing dependency: {e}")
        _model = "not_available"
        _processor = None
        return _model, _processor
    except Exception as e:
        print(f"[Landmarks] Error loading model: {e}")
        _model = "not_available"
        _processor = None
        return _model, _processor


def _classify_with_clip(
    model: Any,
    processor: Any,
    pil_image: Image.Image,
    prompts: List[str],
) -> Dict[str, float]:
    """Classify image against prompts using CLIP."""
    import torch

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


def run_landmarks(image: Any, threshold: float = 0.15) -> Dict[str, Any]:
    """
    Detect landmarks in image using CLIP.

    Args:
        image: Image tensor from ComfyUI or PIL Image
        threshold: Minimum confidence for landmark detection

    Returns:
        Dict with landmark detection results:
            - detected: bool - whether a landmark was detected
            - landmarks: Dict of landmark names with scores
            - best_landmark: Top detected landmark
            - best_score: Confidence of top landmark
            - is_famous: Whether it's a world-famous landmark
            - architecture_type: Type of architecture if detected
    """
    try:
        model, processor = _load_model()

        if model == "not_available" or processor is None:
            return {"detected": False, "error": "CLIP model not available"}

        pil_image = _image_to_pil(image).convert("RGB")

        # Classify against landmark prompts
        scores = _classify_with_clip(model, processor, pil_image, LANDMARK_PROMPTS)

        # Filter by threshold and exclude "no landmark" prompts
        no_landmark_prompts = ["ordinary building", "residential house", "generic street"]
        landmark_scores = {}

        for prompt, score in scores.items():
            is_negative = any(neg in prompt.lower() for neg in no_landmark_prompts)
            if not is_negative and score >= threshold:
                # Clean up prompt for display
                landmark_scores[prompt] = score

        # Check if "no landmark" has highest score
        no_landmark_score = max(
            scores.get(p, 0) for p in LANDMARK_PROMPTS
            if any(neg in p.lower() for neg in no_landmark_prompts)
        )

        # Get best landmark
        if landmark_scores:
            best_landmark = max(landmark_scores, key=landmark_scores.get)
            best_score = landmark_scores[best_landmark]
        else:
            best_landmark = None
            best_score = 0

        # Determine if it's a famous landmark
        is_famous = False
        if best_landmark:
            for famous in LANDMARK_CATEGORIES["famous_landmarks"]:
                if famous.lower() in best_landmark.lower():
                    is_famous = True
                    break

        # Determine architecture type
        architecture_type = None
        if best_landmark:
            for arch_type in LANDMARK_CATEGORIES["architecture_types"]:
                if arch_type.lower() in best_landmark.lower():
                    architecture_type = arch_type
                    break

        # Detection is valid if landmark score beats no-landmark score
        detected = best_score > no_landmark_score and best_score >= threshold

        result = {
            "detected": detected,
            "landmarks": landmark_scores,
            "best_landmark": best_landmark if detected else None,
            "best_score": best_score if detected else 0,
            "is_famous": is_famous and detected,
            "architecture_type": architecture_type,
            "no_landmark_score": no_landmark_score,
        }

        if detected:
            print(f"[Landmarks] Detected: {best_landmark} ({best_score:.2f})")
        else:
            print(f"[Landmarks] No landmark detected")

        return result

    except Exception as e:
        print(f"[Landmarks] Error: {e}")
        import traceback
        traceback.print_exc()
        return {"detected": False, "error": str(e)}


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
    print("[Landmarks] Model unloaded")
