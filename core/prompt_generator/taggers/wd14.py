"""
WD14 tagger implementation.

Uses SmilingWolf's WD14 ViT model for anime/photo tagging.
"""

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image

# Lazy imports for heavy dependencies
_model = None
_labels = None
_model_lock = None


def _get_model_path() -> str:
    """Get the path to cache WD14 model files."""
    cache_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", ".cache", "wd14")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _load_model():
    """Load WD14 model and labels."""
    global _model, _labels

    if _model is not None:
        return _model, _labels

    import onnxruntime as ort
    from huggingface_hub import hf_hub_download

    model_id = "SmilingWolf/wd-v1-4-vit-tagger-v2"
    cache_dir = _get_model_path()

    # Download model files
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

    # Load labels
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

    print(f"[WD14] Loaded model with {len(_labels)} tags")
    return _model, _labels


def _preprocess_image(image: Image.Image, target_size: int = 448) -> np.ndarray:
    """Preprocess image for WD14 model."""
    # Resize with aspect ratio preservation then center crop/pad
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


def run_wd14(
    image: Any,
    threshold: float = 0.35,
    character_threshold: float = 0.85,
    max_tags: int = 50,
) -> Dict[str, float]:
    """
    Run WD14 tagger on an image.

    Args:
        image: Image tensor from ComfyUI or PIL Image
        threshold: General tag confidence threshold
        character_threshold: Character tag threshold (higher for better precision)
        max_tags: Maximum number of tags to return

    Returns:
        Dict mapping tag names to confidence scores
    """
    try:
        model, labels = _load_model()

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

            # Skip low confidence
            if conf < threshold:
                continue

            # Character tags (starting with a name) need higher threshold
            if tag[0].isupper() and conf < character_threshold:
                continue

            results[tag] = conf

        # Sort by confidence and limit
        sorted_results = dict(
            sorted(results.items(), key=lambda x: x[1], reverse=True)[:max_tags]
        )

        print(f"[WD14] Found {len(sorted_results)} tags above threshold {threshold}")
        return sorted_results

    except ImportError as e:
        print(f"[WD14] Missing dependency: {e}")
        print("[WD14] Install with: pip install onnxruntime huggingface_hub")
        return {}
    except Exception as e:
        print(f"[WD14] Error: {e}")
        import traceback
        traceback.print_exc()
        return {}


def unload_model():
    """Unload WD14 model from memory."""
    global _model, _labels
    _model = None
    _labels = None
    print("[WD14] Model unloaded")
