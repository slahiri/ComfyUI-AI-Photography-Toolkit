"""
Lighting analyzer implementation.

Analyzes lighting direction, quality, shadows, and characteristics.
"""

from typing import Any, Dict, Tuple

import numpy as np
from PIL import Image


def _image_to_array(image: Any) -> np.ndarray:
    """Convert image to numpy array (RGB, 0-255, shape HxWx3)."""
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
        if image.ndim == 2:
            return np.stack([image, image, image], axis=2)
        elif image.ndim == 3:
            if image.shape[2] == 4:
                return image[:, :, :3]
            return image
        elif image.ndim == 4:
            return image[0]
        return image
    else:
        arr = np.array(image)
        if arr.ndim == 4:
            arr = arr[0]
        if arr.ndim == 2:
            return np.stack([arr, arr, arr], axis=2)
        return arr


def _analyze_light_direction(img: np.ndarray) -> Dict[str, float]:
    """
    Estimate primary light direction from brightness gradients.
    """
    h, w = img.shape[:2]
    gray = np.mean(img, axis=2)

    # Divide image into regions
    regions = {
        "top": gray[:h//3, :],
        "bottom": gray[2*h//3:, :],
        "left": gray[:, :w//3],
        "right": gray[:, 2*w//3:],
        "center": gray[h//3:2*h//3, w//3:2*w//3],
    }

    brightness = {k: np.mean(v) for k, v in regions.items()}

    # Normalize to sum to 1
    total = sum(brightness.values())
    if total > 0:
        normalized = {k: v / total for k, v in brightness.items()}
    else:
        normalized = {k: 0.2 for k in brightness}

    # Determine primary direction
    directions = ["top", "bottom", "left", "right"]
    dir_brightness = {d: brightness[d] for d in directions}
    primary_direction = max(dir_brightness, key=dir_brightness.get)

    # Calculate direction scores
    h_gradient = (brightness["right"] - brightness["left"]) / 255
    v_gradient = (brightness["bottom"] - brightness["top"]) / 255

    return {
        "top_light": normalized["top"],
        "bottom_light": normalized["bottom"],
        "left_light": normalized["left"],
        "right_light": normalized["right"],
        "center_light": normalized["center"],
        "primary_direction": primary_direction,
        "horizontal_gradient": h_gradient,
        "vertical_gradient": v_gradient,
    }


def _analyze_light_quality(img: np.ndarray) -> Dict[str, float]:
    """
    Analyze light quality (hard vs soft).
    """
    gray = np.mean(img, axis=2)

    # Calculate local contrast as indicator of hard/soft light
    # Hard light = high local contrast, Soft light = low local contrast
    try:
        import cv2
        kernel_size = 21
        local_mean = cv2.blur(gray, (kernel_size, kernel_size))
        local_contrast = np.std(gray - local_mean)
    except ImportError:
        # Fallback
        local_contrast = np.std(gray)

    # Normalize (0 = soft, 1 = hard)
    hardness = min(1.0, local_contrast / 50)

    # Calculate histogram spread as indicator of contrast
    hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 256))
    hist_nonzero = np.where(hist > 0)[0]

    if len(hist_nonzero) > 0:
        tonal_range = (hist_nonzero[-1] - hist_nonzero[0]) / 255
    else:
        tonal_range = 0.5

    return {
        "hard_light": hardness,
        "soft_light": 1 - hardness,
        "tonal_range": tonal_range,
        "contrast_level": min(1.0, np.std(gray) / 80),
    }


def _analyze_shadows(img: np.ndarray) -> Dict[str, float]:
    """
    Analyze shadow characteristics.
    """
    gray = np.mean(img, axis=2)

    # Shadow detection (dark pixels)
    shadow_threshold = 50
    shadow_mask = gray < shadow_threshold
    shadow_ratio = np.sum(shadow_mask) / gray.size

    # Deep shadow (very dark)
    deep_shadow_threshold = 25
    deep_shadow_mask = gray < deep_shadow_threshold
    deep_shadow_ratio = np.sum(deep_shadow_mask) / gray.size

    # Highlight detection (bright pixels)
    highlight_threshold = 200
    highlight_mask = gray > highlight_threshold
    highlight_ratio = np.sum(highlight_mask) / gray.size

    # Shadow edge sharpness (indicates hard vs soft shadows)
    try:
        import cv2
        edges = cv2.Canny(shadow_mask.astype(np.uint8) * 255, 100, 200)
        shadow_edge_density = np.sum(edges > 0) / edges.size
    except ImportError:
        shadow_edge_density = 0.5

    return {
        "shadow_ratio": shadow_ratio,
        "deep_shadow_ratio": deep_shadow_ratio,
        "highlight_ratio": highlight_ratio,
        "shadow_contrast": shadow_edge_density,
        "has_strong_shadows": shadow_ratio > 0.2,
        "has_highlights": highlight_ratio > 0.1,
    }


def _analyze_color_temperature(img: np.ndarray) -> Dict[str, float]:
    """
    Analyze color temperature of the light.
    """
    mean_r = np.mean(img[:, :, 0])
    mean_g = np.mean(img[:, :, 1])
    mean_b = np.mean(img[:, :, 2])

    # Temperature bias
    temp_bias = (mean_r - mean_b) / 255

    # Categorize
    if temp_bias > 0.15:
        temp_category = "warm"
    elif temp_bias < -0.15:
        temp_category = "cool"
    else:
        temp_category = "neutral"

    # Golden hour detection (warm with moderate brightness)
    brightness = (mean_r + mean_g + mean_b) / 3 / 255
    golden_hour_score = 0
    if temp_bias > 0.1 and 0.3 < brightness < 0.7:
        golden_hour_score = min(1.0, temp_bias * 2)

    # Blue hour detection (cool with low brightness)
    blue_hour_score = 0
    if temp_bias < -0.05 and brightness < 0.4:
        blue_hour_score = min(1.0, abs(temp_bias) * 2)

    return {
        "temperature_bias": temp_bias,
        "temperature_category": temp_category,
        "warm_tones": max(0, temp_bias),
        "cool_tones": max(0, -temp_bias),
        "golden_hour": golden_hour_score,
        "blue_hour": blue_hour_score,
    }


def _classify_lighting_type(
    direction: Dict[str, float],
    quality: Dict[str, float],
    shadows: Dict[str, float],
) -> Dict[str, float]:
    """
    Classify lighting into common types.
    """
    types = {}

    # Flat lighting (even distribution, low contrast)
    if quality["contrast_level"] < 0.3 and abs(direction["horizontal_gradient"]) < 0.1:
        types["flat_lighting"] = 0.8
    else:
        types["flat_lighting"] = 0.2

    # Rembrandt lighting (strong side light with triangle shadow)
    if abs(direction["horizontal_gradient"]) > 0.2 and shadows["shadow_ratio"] > 0.15:
        types["rembrandt_lighting"] = 0.7
    else:
        types["rembrandt_lighting"] = 0.2

    # Split lighting (half face lit, half in shadow)
    if abs(direction["horizontal_gradient"]) > 0.3:
        types["split_lighting"] = 0.7
    else:
        types["split_lighting"] = 0.2

    # Butterfly/Paramount lighting (top-down, nose shadow)
    if direction["vertical_gradient"] < -0.1 and direction["top_light"] > direction["bottom_light"]:
        types["butterfly_lighting"] = 0.6
    else:
        types["butterfly_lighting"] = 0.2

    # Rim/Back lighting (bright edges, dark center)
    center_brightness = direction["center_light"]
    edge_brightness = (direction["left_light"] + direction["right_light"]) / 2
    if center_brightness < edge_brightness:
        types["rim_lighting"] = 0.6
    else:
        types["rim_lighting"] = 0.2

    # Natural/Available light (moderate contrast, some directionality)
    if 0.3 < quality["contrast_level"] < 0.6:
        types["natural_light"] = 0.6
    else:
        types["natural_light"] = 0.3

    return types


def run_lighting(image: Any) -> Dict[str, Any]:
    """
    Analyze lighting characteristics.

    Args:
        image: Image tensor from ComfyUI or PIL Image

    Returns:
        Dict with lighting analysis:
            - direction: Light direction analysis
            - quality: Hard/soft light metrics
            - shadows: Shadow analysis
            - temperature: Color temperature
            - lighting_types: Classification scores
    """
    try:
        img = _image_to_array(image)

        # Analyze components
        direction = _analyze_light_direction(img)
        quality = _analyze_light_quality(img)
        shadows = _analyze_shadows(img)
        temperature = _analyze_color_temperature(img)
        lighting_types = _classify_lighting_type(direction, quality, shadows)

        # Compile result
        result = {
            # Direction
            "primary_direction": direction["primary_direction"],
            "top_light": direction["top_light"],
            "side_light": max(direction["left_light"], direction["right_light"]),

            # Quality
            "hard_light": quality["hard_light"],
            "soft_light": quality["soft_light"],
            "contrast": quality["contrast_level"],

            # Shadows
            "shadow_intensity": shadows["shadow_ratio"],
            "has_strong_shadows": shadows["has_strong_shadows"],

            # Temperature
            "warm_tones": temperature["warm_tones"],
            "cool_tones": temperature["cool_tones"],
            "golden_hour": temperature["golden_hour"],
            "blue_hour": temperature["blue_hour"],

            # Types
            "lighting_types": lighting_types,
        }

        # Best lighting type
        best_type = max(lighting_types, key=lighting_types.get)
        result["best_lighting_type"] = best_type

        print(f"[Lighting] Type: {best_type}, Direction: {direction['primary_direction']}")
        return result

    except Exception as e:
        print(f"[Lighting] Error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "primary_direction": "front",
            "hard_light": 0.5,
            "soft_light": 0.5,
        }
