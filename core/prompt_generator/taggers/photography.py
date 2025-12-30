"""
Photography analyzer implementation.

Analyzes lighting, color temperature, and photographic attributes.
"""

import os
from typing import Any, Dict, Tuple

import numpy as np
from PIL import Image


def _image_to_array(image: Any) -> np.ndarray:
    """Convert image to numpy array (RGB, 0-255, shape HxWx3)."""
    if hasattr(image, 'cpu'):
        # PyTorch tensor
        img_np = image[0].cpu().numpy()
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
        else:
            img_np = img_np.astype(np.uint8)
        return img_np
    elif isinstance(image, Image.Image):
        return np.array(image.convert("RGB"))
    elif isinstance(image, np.ndarray):
        # Already numpy array
        if image.ndim == 2:
            # Grayscale - convert to RGB
            return np.stack([image, image, image], axis=2)
        elif image.ndim == 3:
            if image.shape[2] == 4:
                # RGBA - drop alpha
                return image[:, :, :3]
            return image
        elif image.ndim == 4:
            # Batch - take first
            return image[0]
        return image
    else:
        # Try to convert
        arr = np.array(image)
        if arr.ndim == 4:
            arr = arr[0]
        if arr.ndim == 2:
            return np.stack([arr, arr, arr], axis=2)
        return arr


def _analyze_brightness(img: np.ndarray) -> Tuple[float, str]:
    """
    Analyze image brightness.

    Returns:
        Tuple of (brightness_value, key_type)
        key_type is "low_key", "mid_key", or "high_key"
    """
    # Convert to grayscale
    gray = np.mean(img, axis=2)

    # Calculate mean brightness (0-1)
    brightness = np.mean(gray) / 255.0

    # Determine key type
    if brightness < 0.35:
        key_type = "low_key"
    elif brightness > 0.65:
        key_type = "high_key"
    else:
        key_type = "mid_key"

    return brightness, key_type


def _analyze_contrast(img: np.ndarray) -> float:
    """
    Analyze image contrast.

    Returns:
        Contrast value (0-1)
    """
    gray = np.mean(img, axis=2)

    # Standard deviation as contrast measure
    std = np.std(gray)
    contrast = min(1.0, std / 80.0)  # Normalize to 0-1

    return contrast


def _analyze_color_temperature(img: np.ndarray) -> Tuple[float, str]:
    """
    Analyze color temperature.

    Returns:
        Tuple of (temperature_bias, temperature_name)
        Negative = cool, positive = warm
    """
    # Calculate mean R, G, B
    mean_r = np.mean(img[:, :, 0])
    mean_g = np.mean(img[:, :, 1])
    mean_b = np.mean(img[:, :, 2])

    # Temperature bias: positive = warm (more red), negative = cool (more blue)
    temp_bias = (mean_r - mean_b) / 255.0

    if temp_bias > 0.1:
        temp_name = "warm"
    elif temp_bias < -0.1:
        temp_name = "cool"
    else:
        temp_name = "neutral"

    return temp_bias, temp_name


def _analyze_saturation(img: np.ndarray) -> float:
    """
    Analyze overall color saturation.

    Returns:
        Saturation value (0-1)
    """
    # Convert to HSV-like saturation
    max_channel = np.max(img, axis=2).astype(float)
    min_channel = np.min(img, axis=2).astype(float)

    # Avoid division by zero
    denominator = np.maximum(max_channel, 1)
    saturation_map = (max_channel - min_channel) / denominator

    # Mean saturation
    return float(np.mean(saturation_map))


def _analyze_lighting_direction(img: np.ndarray) -> Dict[str, float]:
    """
    Estimate lighting direction from brightness distribution.

    Returns:
        Dict with directional weights
    """
    gray = np.mean(img, axis=2)
    h, w = gray.shape

    # Split into regions
    top = np.mean(gray[:h//3, :])
    bottom = np.mean(gray[2*h//3:, :])
    left = np.mean(gray[:, :w//3])
    right = np.mean(gray[:, 2*w//3:])
    center = np.mean(gray[h//3:2*h//3, w//3:2*w//3])

    # Normalize
    total = top + bottom + left + right + center
    if total == 0:
        total = 1

    return {
        "top_light": top / total,
        "bottom_light": bottom / total,
        "left_light": left / total,
        "right_light": right / total,
        "center_light": center / total,
    }


def _analyze_shadow_intensity(img: np.ndarray) -> float:
    """
    Analyze shadow intensity.

    Returns:
        Shadow intensity (0-1), higher = more shadows
    """
    gray = np.mean(img, axis=2)

    # Count dark pixels
    dark_threshold = 50  # Pixels below this are considered shadow
    shadow_pixels = np.sum(gray < dark_threshold)
    total_pixels = gray.size

    return shadow_pixels / total_pixels


def _is_dramatic_lighting(
    contrast: float,
    shadow_intensity: float,
    key_type: str,
) -> float:
    """
    Determine if lighting is dramatic.

    Returns:
        Dramatic score (0-1)
    """
    score = 0.0

    if key_type == "low_key":
        score += 0.3

    if contrast > 0.5:
        score += 0.3

    if shadow_intensity > 0.2:
        score += 0.2

    return min(1.0, score)


def run_photography(image: Any) -> Dict[str, float]:
    """
    Analyze photographic attributes of an image.

    Args:
        image: Image tensor from ComfyUI or PIL Image

    Returns:
        Dict with photography analysis results:
            - brightness: 0-1
            - low_key: 0-1
            - high_key: 0-1
            - contrast: 0-1
            - warm_tones: 0-1
            - cool_tones: 0-1
            - saturation: 0-1
            - dramatic: 0-1
            - soft: 0-1
            - shadow_intensity: 0-1
    """
    try:
        img = _image_to_array(image)

        # Brightness analysis
        brightness, key_type = _analyze_brightness(img)

        # Contrast analysis
        contrast = _analyze_contrast(img)

        # Color temperature
        temp_bias, temp_name = _analyze_color_temperature(img)

        # Saturation
        saturation = _analyze_saturation(img)

        # Shadow analysis
        shadow_intensity = _analyze_shadow_intensity(img)

        # Dramatic lighting
        dramatic = _is_dramatic_lighting(contrast, shadow_intensity, key_type)

        # Soft lighting (opposite of dramatic)
        soft = max(0, 1.0 - dramatic - 0.3) if key_type == "high_key" else 0

        result = {
            "brightness": brightness,
            "low_key": 1.0 if key_type == "low_key" else 0.0,
            "high_key": 1.0 if key_type == "high_key" else 0.0,
            "contrast": contrast,
            "warm_tones": max(0, temp_bias),
            "cool_tones": max(0, -temp_bias),
            "saturation": saturation,
            "dramatic": dramatic,
            "soft": soft,
            "shadow_intensity": shadow_intensity,
        }

        # Log summary
        print(f"[Photography] Key: {key_type}, Contrast: {contrast:.2f}, Temp: {temp_name}")

        return result

    except Exception as e:
        print(f"[Photography] Error: {e}")
        import traceback
        traceback.print_exc()
        return {}
