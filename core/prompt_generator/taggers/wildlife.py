"""
Wildlife analyzer implementation.

Detects animals and wildlife in images.
"""

import os
from typing import Any, Dict, List

import numpy as np
from PIL import Image

_model = None


# Common wildlife/animal categories
WILDLIFE_CATEGORIES = [
    # Mammals
    "dog", "cat", "horse", "cow", "sheep", "pig", "goat",
    "lion", "tiger", "bear", "wolf", "fox", "deer", "elephant",
    "giraffe", "zebra", "monkey", "gorilla", "panda", "koala",
    "rabbit", "squirrel", "mouse", "rat", "hedgehog",
    # Birds
    "bird", "eagle", "owl", "parrot", "penguin", "flamingo",
    "duck", "chicken", "peacock", "crow", "sparrow", "pigeon",
    # Reptiles
    "snake", "lizard", "turtle", "crocodile", "alligator",
    # Marine
    "fish", "shark", "whale", "dolphin", "octopus", "jellyfish",
    # Insects
    "butterfly", "bee", "spider", "ant",
]


def _get_model_path() -> str:
    """Get the path to cache wildlife model files."""
    cache_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", ".cache", "wildlife")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


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


def _load_model():
    """Load wildlife detection model."""
    global _model

    if _model is not None:
        return _model

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
        print("[Wildlife] Loaded CLIP model for wildlife detection")
        return _model

    except ImportError as e:
        print(f"[Wildlife] Missing dependency: {e}")
        _model = "not_available"
        return _model


def _detect_with_clip(
    model: Any,
    processor: Any,
    pil_image: Image.Image,
    categories: List[str],
    threshold: float = 0.3,
) -> Dict[str, float]:
    """
    Detect categories using CLIP zero-shot classification.
    """
    import torch

    # Create prompts
    prompts = [f"a photo of a {cat}" for cat in categories]

    # Add negative prompt
    prompts.append("a photo with no animals")

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

    # Get scores for each category
    result = {}
    probs_np = probs.cpu().numpy()

    for i, cat in enumerate(categories):
        score = float(probs_np[i])
        if score >= threshold:
            result[cat] = score

    # Check if "no animals" has highest score
    no_animals_score = float(probs_np[-1])

    return result, no_animals_score


def run_wildlife(image: Any, threshold: float = 0.15) -> Dict[str, Any]:
    """
    Detect wildlife/animals in image.

    Args:
        image: Image tensor from ComfyUI or PIL Image
        threshold: Detection confidence threshold

    Returns:
        Dict with wildlife detection:
            - detected: bool
            - animals: Dict of detected animals with scores
            - primary_animal: Most confident detection
            - count_estimate: Estimated number of animals
    """
    try:
        pil_image = _image_to_pil(image).convert("RGB")

        model_data = _load_model()

        if model_data == "not_available":
            return {
                "detected": False,
                "animals": {},
                "error": "Model not available",
            }

        model, processor = model_data

        # Detect animals
        animals, no_animals_score = _detect_with_clip(
            model, processor, pil_image, WILDLIFE_CATEGORIES, threshold
        )

        # Determine if animals are present
        if not animals or no_animals_score > max(animals.values(), default=0):
            return {
                "detected": False,
                "animals": {},
                "no_animals_confidence": no_animals_score,
            }

        # Get primary animal
        primary_animal = max(animals, key=animals.get)

        # Categorize by type
        categories = {
            "mammals": ["dog", "cat", "horse", "cow", "sheep", "pig", "lion", "tiger", "bear", "wolf", "fox", "deer", "elephant", "giraffe", "zebra", "monkey", "gorilla", "panda", "koala", "rabbit", "squirrel"],
            "birds": ["bird", "eagle", "owl", "parrot", "penguin", "flamingo", "duck", "chicken", "peacock"],
            "reptiles": ["snake", "lizard", "turtle", "crocodile", "alligator"],
            "marine": ["fish", "shark", "whale", "dolphin", "octopus", "jellyfish"],
            "insects": ["butterfly", "bee", "spider", "ant"],
        }

        animal_category = "other"
        for cat_name, cat_animals in categories.items():
            if primary_animal in cat_animals:
                animal_category = cat_name
                break

        result = {
            "detected": True,
            "animals": animals,
            "primary_animal": primary_animal,
            "primary_confidence": animals[primary_animal],
            "animal_category": animal_category,
            "animal_count": len(animals),
        }

        print(f"[Wildlife] Detected: {primary_animal} ({animals[primary_animal]:.2f})")
        return result

    except Exception as e:
        print(f"[Wildlife] Error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "detected": False,
            "animals": {},
            "error": str(e),
        }


def unload_model():
    """Unload wildlife model from memory."""
    global _model
    if _model is not None and _model != "not_available":
        del _model
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    _model = None
    print("[Wildlife] Model unloaded")
