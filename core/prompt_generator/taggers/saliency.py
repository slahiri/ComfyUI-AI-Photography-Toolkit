"""
Saliency analyzer implementation.

Detects salient regions and subject attention using gradient-based
or DINOv2 methods.
"""

import os
from typing import Any, Dict, Tuple

import numpy as np
from PIL import Image

_model = None


def _get_model_path() -> str:
    """Get the path to cache saliency model files."""
    cache_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", ".cache", "saliency")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


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


def _load_dino_model():
    """Load DINOv2 model for saliency detection."""
    global _model

    if _model is not None:
        return _model

    try:
        import torch

        # Try to load DINOv2
        _model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        _model.eval()

        if torch.cuda.is_available():
            _model = _model.cuda()

        print("[Saliency] Loaded DINOv2 model")
        return _model

    except Exception as e:
        print(f"[Saliency] Could not load DINOv2: {e}")
        print("[Saliency] Using gradient-based fallback")
        _model = "gradient_fallback"
        return _model


def _gradient_saliency(img: np.ndarray) -> np.ndarray:
    """
    Compute saliency map using image gradients.

    Simple but effective fallback when no model available.
    """
    try:
        import cv2

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Compute gradients
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        # Gradient magnitude
        magnitude = np.sqrt(gx**2 + gy**2)

        # Normalize
        magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)

        # Apply Gaussian blur to smooth
        saliency = cv2.GaussianBlur(magnitude, (21, 21), 0)

        return saliency

    except ImportError:
        # Fallback without OpenCV
        gray = np.mean(img, axis=2)
        gx = np.diff(gray, axis=1, prepend=gray[:, :1])
        gy = np.diff(gray, axis=0, prepend=gray[:1, :])
        magnitude = np.sqrt(gx**2 + gy**2)
        magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)
        return magnitude


def _dino_saliency(model: Any, img: np.ndarray) -> np.ndarray:
    """
    Compute saliency using DINOv2 attention maps.
    """
    import torch
    from torchvision import transforms

    pil_image = Image.fromarray(img)

    transform = transforms.Compose([
        transforms.Resize((518, 518)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = transform(pil_image).unsqueeze(0)

    if torch.cuda.is_available() and next(model.parameters()).is_cuda:
        input_tensor = input_tensor.cuda()

    with torch.no_grad():
        # Get attention from the last layer
        features = model.get_intermediate_layers(input_tensor, n=1)[0]

        # Reshape to spatial dimensions
        h = w = int(features.shape[1] ** 0.5)
        attention = features[:, :, 0].reshape(1, h, w)

        # Normalize
        attention = attention - attention.min()
        attention = attention / (attention.max() + 1e-8)

        # Resize to original image size
        saliency = torch.nn.functional.interpolate(
            attention.unsqueeze(0),
            size=(img.shape[0], img.shape[1]),
            mode='bilinear',
            align_corners=False
        )[0, 0]

    return saliency.cpu().numpy()


def _analyze_saliency_map(saliency: np.ndarray, h: int, w: int) -> Dict[str, Any]:
    """
    Analyze saliency map to extract attention information.
    """
    # Find centroid of salient region
    y_indices, x_indices = np.indices(saliency.shape)
    total_saliency = np.sum(saliency) + 1e-8

    centroid_y = np.sum(y_indices * saliency) / total_saliency
    centroid_x = np.sum(x_indices * saliency) / total_saliency

    # Normalize to 0-1
    centroid_y_norm = centroid_y / h
    centroid_x_norm = centroid_x / w

    # Find peak saliency location
    peak_idx = np.unravel_index(np.argmax(saliency), saliency.shape)
    peak_y_norm = peak_idx[0] / h
    peak_x_norm = peak_idx[1] / w

    # Calculate spread (standard deviation)
    spread_y = np.sqrt(np.sum(((y_indices - centroid_y) ** 2) * saliency) / total_saliency) / h
    spread_x = np.sqrt(np.sum(((x_indices - centroid_x) ** 2) * saliency) / total_saliency) / w

    # Determine position description
    if centroid_x_norm < 0.33:
        h_pos = "left"
    elif centroid_x_norm > 0.67:
        h_pos = "right"
    else:
        h_pos = "center"

    if centroid_y_norm < 0.33:
        v_pos = "top"
    elif centroid_y_norm > 0.67:
        v_pos = "bottom"
    else:
        v_pos = "middle"

    position = f"{v_pos}_{h_pos}"

    # Check if subject is at rule of thirds intersection
    thirds_h = [0.33, 0.67]
    thirds_w = [0.33, 0.67]

    at_thirds = False
    for th in thirds_h:
        for tw in thirds_w:
            if abs(centroid_y_norm - th) < 0.1 and abs(centroid_x_norm - tw) < 0.1:
                at_thirds = True
                break

    return {
        "centroid_x": centroid_x_norm,
        "centroid_y": centroid_y_norm,
        "peak_x": peak_x_norm,
        "peak_y": peak_y_norm,
        "spread_x": spread_x,
        "spread_y": spread_y,
        "position": position,
        "at_rule_of_thirds": at_thirds,
        "focus_concentration": 1 - (spread_x + spread_y) / 2,  # Higher = more focused
    }


def run_saliency(image: Any) -> Dict[str, Any]:
    """
    Analyze image saliency/attention.

    Args:
        image: Image tensor from ComfyUI or PIL Image

    Returns:
        Dict with saliency analysis:
            - centroid_x, centroid_y: Center of attention (0-1)
            - peak_x, peak_y: Most salient point (0-1)
            - position: Descriptive position ("top_left", "center", etc.)
            - at_rule_of_thirds: bool
            - focus_concentration: 0-1 score
    """
    try:
        img = _image_to_array(image)
        h, w = img.shape[:2]

        model = _load_dino_model()

        if model == "gradient_fallback":
            saliency_map = _gradient_saliency(img)
        else:
            try:
                saliency_map = _dino_saliency(model, img)
            except Exception as e:
                print(f"[Saliency] DINOv2 failed, using gradient: {e}")
                saliency_map = _gradient_saliency(img)

        result = _analyze_saliency_map(saliency_map, h, w)
        result["method"] = "dino" if model != "gradient_fallback" else "gradient"

        print(f"[Saliency] Subject at {result['position']}, concentration: {result['focus_concentration']:.2f}")
        return result

    except Exception as e:
        print(f"[Saliency] Error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "centroid_x": 0.5,
            "centroid_y": 0.5,
            "position": "center",
            "focus_concentration": 0.5,
        }


def unload_model():
    """Unload saliency model from memory."""
    global _model
    if _model is not None and _model != "gradient_fallback":
        del _model
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    _model = None
    print("[Saliency] Model unloaded")
