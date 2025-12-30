"""
IQA (Image Quality Assessment) analyzer.

Provides aesthetic and technical quality scores.
Uses NIMA/MUSIQ models when available, falls back to heuristics.
"""

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image

# Lazy imports
_aesthetic_model = None
_technical_model = None


def _get_model_path() -> str:
    """Get the path to cache IQA model files."""
    cache_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", ".cache", "iqa")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


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


def _load_models():
    """Load IQA models."""
    global _aesthetic_model, _technical_model

    if _aesthetic_model is not None:
        return _aesthetic_model, _technical_model

    try:
        # Try to use pyiqa for quality assessment
        import pyiqa
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # MUSIQ for aesthetic quality
        try:
            _aesthetic_model = pyiqa.create_metric("musiq", device=device)
            print("[IQA] Loaded MUSIQ aesthetic model")
        except Exception as e:
            print(f"[IQA] Could not load MUSIQ: {e}")
            _aesthetic_model = "heuristic"

        # BRISQUE for technical quality
        try:
            _technical_model = pyiqa.create_metric("brisque", device=device)
            print("[IQA] Loaded BRISQUE technical model")
        except Exception as e:
            print(f"[IQA] Could not load BRISQUE: {e}")
            _technical_model = "heuristic"

        return _aesthetic_model, _technical_model

    except ImportError:
        print("[IQA] pyiqa not installed. Using heuristic analysis.")
        print("[IQA] Install with: pip install pyiqa")
        _aesthetic_model = "heuristic"
        _technical_model = "heuristic"
        return _aesthetic_model, _technical_model


def _heuristic_aesthetic_score(img: np.ndarray) -> float:
    """
    Estimate aesthetic quality using image statistics.

    Returns score 0-10.
    """
    score = 5.0  # Start at middle

    # Color variety (more colors = potentially better)
    unique_colors = len(np.unique(img.reshape(-1, 3), axis=0))
    total_pixels = img.shape[0] * img.shape[1]
    color_ratio = min(1.0, unique_colors / (total_pixels * 0.1))
    score += color_ratio * 1.0

    # Contrast (good contrast is positive)
    gray = np.mean(img, axis=2)
    contrast = np.std(gray) / 128.0  # Normalize by half of 255
    if 0.2 < contrast < 0.6:
        score += 1.0
    elif contrast < 0.1:
        score -= 1.0

    # Saturation (moderate saturation is good)
    max_channel = np.max(img, axis=2).astype(float)
    min_channel = np.min(img, axis=2).astype(float)
    saturation = np.mean((max_channel - min_channel) / (max_channel + 1))
    if 0.2 < saturation < 0.6:
        score += 0.5

    # Brightness (not too dark, not too bright)
    brightness = np.mean(img) / 255.0
    if 0.3 < brightness < 0.7:
        score += 0.5
    elif brightness < 0.1 or brightness > 0.9:
        score -= 0.5

    # Rule of thirds (check for interest in thirds intersections)
    h, w = img.shape[:2]
    third_h, third_w = h // 3, w // 3

    # Check variance in thirds regions
    roi_variance = []
    for y in [third_h, 2*third_h]:
        for x in [third_w, 2*third_w]:
            roi = img[max(0, y-20):min(h, y+20), max(0, x-20):min(w, x+20)]
            if roi.size > 0:
                roi_variance.append(np.var(roi))

    if roi_variance and max(roi_variance) > 1000:
        score += 0.5

    return max(0, min(10, score))


def _heuristic_technical_score(img: np.ndarray) -> float:
    """
    Estimate technical quality using image statistics.

    Returns score 0-10 (higher = better quality).
    """
    score = 5.0  # Start at middle

    # Sharpness estimation using Laplacian variance
    try:
        import cv2
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Higher variance = sharper
        if laplacian_var > 500:
            score += 2.0
        elif laplacian_var > 100:
            score += 1.0
        elif laplacian_var < 50:
            score -= 1.0
    except ImportError:
        # Fallback sharpness estimation
        gray = np.mean(img, axis=2)
        # Simple gradient magnitude
        gx = np.diff(gray, axis=1)
        gy = np.diff(gray, axis=0)
        gradient_mag = np.sqrt(gx[:, :-1]**2 + gy[:-1, :]**2)
        sharpness = np.mean(gradient_mag)

        if sharpness > 20:
            score += 1.5
        elif sharpness > 10:
            score += 0.5
        elif sharpness < 5:
            score -= 1.0

    # Noise estimation (high frequency content)
    # Use difference between original and blurred
    try:
        import cv2
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        noise = np.std(img.astype(float) - blurred.astype(float))

        if noise < 5:
            score += 1.0  # Low noise
        elif noise > 20:
            score -= 1.0  # High noise
    except ImportError:
        pass

    # Resolution bonus
    h, w = img.shape[:2]
    pixels = h * w
    if pixels > 2000000:  # 2MP+
        score += 0.5
    elif pixels < 500000:  # Less than 0.5MP
        score -= 0.5

    # Color banding detection (indicates compression artifacts)
    unique_per_channel = sum(
        len(np.unique(img[:, :, c])) for c in range(3)
    )
    if unique_per_channel < 300:  # Very few unique colors = banding
        score -= 0.5

    return max(0, min(10, score))


def run_iqa(image: Any) -> Dict[str, float]:
    """
    Analyze image quality (aesthetic and technical).

    Args:
        image: Image tensor from ComfyUI or PIL Image

    Returns:
        Dict with:
            - aesthetic_score: 0-10
            - technical_score: 0-10
            - sharpness: 0-1
            - noise_level: 0-1
    """
    try:
        img = _image_to_array(image)
        pil_image = Image.fromarray(img)

        aesthetic_model, technical_model = _load_models()

        result = {}

        # Aesthetic score
        if aesthetic_model == "heuristic":
            result["aesthetic_score"] = _heuristic_aesthetic_score(img)
        else:
            try:
                import torch
                from torchvision import transforms

                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                ])

                input_tensor = transform(pil_image).unsqueeze(0)
                if torch.cuda.is_available():
                    input_tensor = input_tensor.cuda()

                with torch.no_grad():
                    score = aesthetic_model(input_tensor)
                    # MUSIQ outputs scores roughly 0-100, normalize to 0-10
                    result["aesthetic_score"] = float(score.item()) / 10.0
            except Exception as e:
                print(f"[IQA] Aesthetic model error: {e}")
                result["aesthetic_score"] = _heuristic_aesthetic_score(img)

        # Technical score
        if technical_model == "heuristic":
            result["technical_score"] = _heuristic_technical_score(img)
        else:
            try:
                import torch
                from torchvision import transforms

                transform = transforms.Compose([
                    transforms.ToTensor(),
                ])

                input_tensor = transform(pil_image).unsqueeze(0)
                if torch.cuda.is_available():
                    input_tensor = input_tensor.cuda()

                with torch.no_grad():
                    score = technical_model(input_tensor)
                    # BRISQUE: lower is better, 0-100 range
                    # Invert and normalize to 0-10
                    brisque_score = float(score.item())
                    result["technical_score"] = max(0, 10 - brisque_score / 10)
            except Exception as e:
                print(f"[IQA] Technical model error: {e}")
                result["technical_score"] = _heuristic_technical_score(img)

        # Additional metrics
        result["sharpness"] = _estimate_sharpness(img)
        result["noise_level"] = _estimate_noise(img)

        avg_score = (result["aesthetic_score"] + result["technical_score"]) / 2
        print(f"[IQA] Aesthetic: {result['aesthetic_score']:.1f}, Technical: {result['technical_score']:.1f}")

        return result

    except Exception as e:
        print(f"[IQA] Error: {e}")
        import traceback
        traceback.print_exc()
        return {"aesthetic_score": 5.0, "technical_score": 5.0}


def _estimate_sharpness(img: np.ndarray) -> float:
    """Estimate image sharpness (0-1)."""
    try:
        import cv2
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        # Normalize: 1000+ is very sharp
        return min(1.0, laplacian_var / 1000.0)
    except ImportError:
        gray = np.mean(img, axis=2)
        gx = np.diff(gray, axis=1)
        gy = np.diff(gray, axis=0)
        gradient_mag = np.mean(np.abs(gx[:, :-1]) + np.abs(gy[:-1, :]))
        return min(1.0, gradient_mag / 40.0)


def _estimate_noise(img: np.ndarray) -> float:
    """Estimate noise level (0-1, higher = more noise)."""
    try:
        import cv2
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        diff = img.astype(float) - blurred.astype(float)
        noise = np.std(diff)
        # Normalize: 30+ is very noisy
        return min(1.0, noise / 30.0)
    except ImportError:
        # Simple noise estimation
        h, w = img.shape[:2]
        if h > 10 and w > 10:
            center = img[h//4:3*h//4, w//4:3*w//4]
            local_std = np.std(center)
            return min(1.0, local_std / 50.0)
        return 0.0


def unload_models():
    """Unload IQA models from memory."""
    global _aesthetic_model, _technical_model
    _aesthetic_model = None
    _technical_model = None
    print("[IQA] Models unloaded")
