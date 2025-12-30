"""
Sky & Weather detector using CLIP.

Detects sky conditions, weather, time of day, and atmospheric effects.
"""

import os
from typing import Any, Dict, List

import numpy as np
from PIL import Image

_model = None
_processor = None


# Sky condition prompts
SKY_PROMPTS = [
    "clear blue sky sunny day",
    "cloudy overcast sky",
    "partly cloudy sky with white clouds",
    "dramatic storm clouds dark sky",
    "sunset orange red sky",
    "sunrise pink golden sky",
    "night sky with stars",
    "twilight dusk blue hour sky",
    "hazy foggy misty sky",
    "rainbow in the sky after rain",
]

# Weather prompts
WEATHER_PROMPTS = [
    "sunny clear weather",
    "rainy weather with rain drops",
    "snowy weather with snow falling",
    "foggy misty weather",
    "stormy weather with lightning",
    "windy weather",
    "overcast cloudy weather",
    "partly cloudy fair weather",
]

# Time of day prompts
TIME_OF_DAY_PROMPTS = [
    "bright midday noon sunlight",
    "golden hour warm afternoon light",
    "sunset evening warm light",
    "sunrise early morning light",
    "blue hour twilight",
    "night time darkness",
    "overcast flat daylight",
]

# Atmospheric effects prompts
ATMOSPHERE_PROMPTS = [
    "clear atmosphere high visibility",
    "hazy atmosphere reduced visibility",
    "foggy dense atmosphere",
    "misty soft atmosphere",
    "dusty sandy atmosphere",
    "smoky atmosphere",
    "rainy wet atmosphere",
    "snowy winter atmosphere",
]

# Cloud type prompts
CLOUD_PROMPTS = [
    "cumulus fluffy white clouds",
    "stratus flat gray cloud layer",
    "cirrus wispy high clouds",
    "cumulonimbus thunderstorm clouds",
    "altocumulus mid-level clouds",
    "no clouds clear sky",
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

        print("[Sky & Weather] Loaded CLIP model")
        return _model, _processor

    except ImportError as e:
        print(f"[Sky & Weather] Missing dependency: {e}")
        _model = "not_available"
        _processor = None
        return _model, _processor
    except Exception as e:
        print(f"[Sky & Weather] Error loading model: {e}")
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


def _get_best_match(scores: Dict[str, float]) -> tuple:
    """Get best matching prompt and score."""
    if not scores:
        return None, 0
    best = max(scores, key=scores.get)
    return best, scores[best]


def _simplify_prompt(prompt: str) -> str:
    """Extract key descriptor from prompt."""
    # Return first few words as the key description
    words = prompt.split()[:2]
    return " ".join(words)


def run_sky_weather(image: Any) -> Dict[str, Any]:
    """
    Analyze sky and weather conditions using CLIP.

    Args:
        image: Image tensor from ComfyUI or PIL Image

    Returns:
        Dict with sky/weather analysis:
            - sky_visible: bool - whether sky is visible
            - sky_condition: str - best matching sky condition
            - weather: str - best matching weather
            - time_of_day: str - best matching time of day
            - atmosphere: str - atmospheric condition
            - cloud_type: str - type of clouds if present
            - scores: Dict with all category scores
    """
    try:
        model, processor = _load_model()

        if model == "not_available" or processor is None:
            return _heuristic_sky_analysis(image)

        pil_image = _image_to_pil(image).convert("RGB")

        # Classify each category
        sky_scores = _classify_with_clip(model, processor, pil_image, SKY_PROMPTS)
        weather_scores = _classify_with_clip(model, processor, pil_image, WEATHER_PROMPTS)
        time_scores = _classify_with_clip(model, processor, pil_image, TIME_OF_DAY_PROMPTS)
        atmosphere_scores = _classify_with_clip(model, processor, pil_image, ATMOSPHERE_PROMPTS)
        cloud_scores = _classify_with_clip(model, processor, pil_image, CLOUD_PROMPTS)

        # Get best matches
        best_sky, sky_score = _get_best_match(sky_scores)
        best_weather, weather_score = _get_best_match(weather_scores)
        best_time, time_score = _get_best_match(time_scores)
        best_atmosphere, atmosphere_score = _get_best_match(atmosphere_scores)
        best_cloud, cloud_score = _get_best_match(cloud_scores)

        # Determine if sky is visible (check if sky-related scores are significant)
        sky_visible = sky_score > 0.1 and "night" not in best_sky.lower() or sky_score > 0.15

        # Determine if it's golden hour / blue hour
        is_golden_hour = "golden hour" in best_time.lower() or "sunset" in best_sky.lower() or "sunrise" in best_sky.lower()
        is_blue_hour = "blue hour" in best_time.lower() or "twilight" in best_sky.lower()

        # Determine if dramatic lighting
        is_dramatic = (
            "storm" in best_sky.lower() or
            "dramatic" in best_sky.lower() or
            "sunset" in best_sky.lower() or
            "sunrise" in best_sky.lower()
        )

        result = {
            "sky_visible": sky_visible,
            "sky_condition": _simplify_prompt(best_sky),
            "sky_score": sky_score,
            "weather": _simplify_prompt(best_weather),
            "weather_score": weather_score,
            "time_of_day": _simplify_prompt(best_time),
            "time_score": time_score,
            "atmosphere": _simplify_prompt(best_atmosphere),
            "atmosphere_score": atmosphere_score,
            "cloud_type": _simplify_prompt(best_cloud) if "no clouds" not in best_cloud.lower() else None,
            "cloud_score": cloud_score,
            "is_golden_hour": is_golden_hour,
            "is_blue_hour": is_blue_hour,
            "is_dramatic": is_dramatic,
            "is_night": "night" in best_sky.lower() or "night" in best_time.lower(),
            "scores": {
                "sky": sky_scores,
                "weather": weather_scores,
                "time_of_day": time_scores,
                "atmosphere": atmosphere_scores,
                "clouds": cloud_scores,
            }
        }

        print(f"[Sky & Weather] Sky: {result['sky_condition']}, Weather: {result['weather']}, Time: {result['time_of_day']}")
        return result

    except Exception as e:
        print(f"[Sky & Weather] Error: {e}")
        import traceback
        traceback.print_exc()
        return _heuristic_sky_analysis(image)


def _heuristic_sky_analysis(image: Any) -> Dict[str, Any]:
    """
    Heuristic sky analysis based on image brightness/color.
    """
    try:
        pil_image = _image_to_pil(image).convert("RGB")
        img_np = np.array(pil_image)

        # Analyze top portion of image (likely sky)
        h = img_np.shape[0]
        top_region = img_np[:h//3, :, :]

        # Average color of top region
        avg_color = np.mean(top_region, axis=(0, 1))
        brightness = np.mean(avg_color)

        # Blue ratio (for sky detection)
        blue_ratio = avg_color[2] / (np.mean(avg_color) + 1e-6)

        # Determine conditions based on heuristics
        if brightness > 200:
            sky_condition = "bright"
            weather = "sunny"
            time_of_day = "midday"
        elif brightness > 150:
            sky_condition = "clear"
            weather = "fair"
            time_of_day = "daytime"
        elif brightness > 100:
            sky_condition = "overcast"
            weather = "cloudy"
            time_of_day = "daytime"
        elif brightness > 50:
            sky_condition = "dark"
            weather = "overcast"
            time_of_day = "evening"
        else:
            sky_condition = "night"
            weather = "clear"
            time_of_day = "night"

        # Check for warm tones (sunset/sunrise)
        if avg_color[0] > avg_color[2] and brightness > 100:
            sky_condition = "sunset"
            time_of_day = "golden hour"

        return {
            "sky_visible": blue_ratio > 0.9 or brightness > 100,
            "sky_condition": sky_condition,
            "sky_score": 0.5,
            "weather": weather,
            "weather_score": 0.5,
            "time_of_day": time_of_day,
            "time_score": 0.5,
            "atmosphere": "normal",
            "atmosphere_score": 0.5,
            "cloud_type": None,
            "cloud_score": 0.5,
            "is_golden_hour": "golden" in time_of_day,
            "is_blue_hour": False,
            "is_dramatic": "sunset" in sky_condition,
            "is_night": "night" in sky_condition,
            "scores": {},
        }

    except Exception:
        return {
            "sky_visible": False,
            "sky_condition": "unknown",
            "weather": "unknown",
            "time_of_day": "unknown",
            "atmosphere": "unknown",
            "cloud_type": None,
            "is_golden_hour": False,
            "is_blue_hour": False,
            "is_dramatic": False,
            "is_night": False,
            "scores": {},
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
    print("[Sky & Weather] Model unloaded")
