"""
PixAI tagger implementation.

Uses WD14 SwinV2 variant as PixAI's model isn't publicly available.
Provides anime-style tagging similar to the original PixAI tagger.
"""

import os
from typing import Any, Dict

import numpy as np
from PIL import Image

_model = None
_labels = None


def _get_model_path() -> str:
    """Get the path to cache model files."""
    cache_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", ".cache", "pixai")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _load_model():
    """Load WD14 SwinV2 model (PixAI alternative)."""
    global _model, _labels

    if _model is not None:
        return _model, _labels

    try:
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download

        # Use WD14 SwinV2 variant - better for anime/illustration
        model_id = "SmilingWolf/wd-swinv2-tagger-v3"
        cache_dir = _get_model_path()

        try:
            model_path = hf_hub_download(
                repo_id=model_id,
                filename="model.onnx",
                cache_dir=cache_dir,
            )
            labels_path = hf_hub_download(
                repo_id=model_id,
                filename="selected_tags.csv",
                cache_dir=cache_dir,
            )
        except Exception as e:
            print(f"[PixAI] SwinV2 model not available: {e}")
            print("[PixAI] Falling back to WD14 ViT")
            _model = "fallback_to_wd14"
            _labels = []
            return _model, _labels

        # Load labels from CSV
        import csv
        _labels = []
        with open(labels_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                if len(row) >= 2:
                    _labels.append(row[1])  # Tag name is second column

        # Load ONNX model
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        _model = ort.InferenceSession(model_path, providers=providers)

        print(f"[PixAI] Loaded SwinV2 model with {len(_labels)} tags")
        return _model, _labels

    except ImportError as e:
        print(f"[PixAI] Missing dependency: {e}")
        _model = "fallback_to_wd14"
        _labels = []
        return _model, _labels


def _preprocess_image(image: Image.Image, target_size: int = 448) -> np.ndarray:
    """Preprocess image for the model."""
    image = image.convert("RGB")

    # Resize to fit target size
    w, h = image.size
    scale = target_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Pad to square
    padded = Image.new("RGB", (target_size, target_size), (255, 255, 255))
    offset_x = (target_size - new_w) // 2
    offset_y = (target_size - new_h) // 2
    padded.paste(image, (offset_x, offset_y))

    # Convert to numpy and normalize
    img_np = np.array(padded).astype(np.float32)
    img_np = img_np[:, :, ::-1]  # RGB to BGR
    img_np = np.expand_dims(img_np, axis=0)

    return img_np


def run_pixai(
    image: Any,
    threshold: float = 0.35,
    max_tags: int = 50,
) -> Dict[str, float]:
    """
    Run PixAI (SwinV2) tagger on an image.

    Args:
        image: Image tensor from ComfyUI or PIL Image
        threshold: Tag confidence threshold
        max_tags: Maximum number of tags to return

    Returns:
        Dict mapping tag names to confidence scores
    """
    try:
        model, labels = _load_model()

        if model == "fallback_to_wd14" or not labels:
            # Fall back to WD14 ViT
            from .wd14 import run_wd14
            return run_wd14(image, threshold=threshold, max_tags=max_tags)

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
        elif isinstance(image, np.ndarray):
            if image.ndim == 4:
                image = image[0]
            pil_image = Image.fromarray(image.astype(np.uint8))
        else:
            pil_image = Image.fromarray(np.array(image[0]))

        # Preprocess
        input_data = _preprocess_image(pil_image)

        # Get input name
        input_name = model.get_inputs()[0].name

        # Run inference
        outputs = model.run(None, {input_name: input_data})
        probs = outputs[0][0]

        # Parse results
        results = {}
        for i, prob in enumerate(probs):
            if i >= len(labels):
                break

            tag = labels[i]
            conf = float(prob)

            # Skip malformed tags
            if not tag or "?" in tag or len(tag) < 2 or tag.startswith(" "):
                continue

            if conf >= threshold:
                results[tag] = conf

        # Sort by confidence and limit
        sorted_results = dict(
            sorted(results.items(), key=lambda x: x[1], reverse=True)[:max_tags]
        )

        print(f"[PixAI] Found {len(sorted_results)} tags above threshold {threshold}")
        return sorted_results

    except Exception as e:
        print(f"[PixAI] Error: {e}")
        # Fall back to WD14
        from .wd14 import run_wd14
        return run_wd14(image, threshold=threshold, max_tags=max_tags)


def unload_model():
    """Unload model from memory."""
    global _model, _labels
    _model = None
    _labels = None
    print("[PixAI] Model unloaded")
