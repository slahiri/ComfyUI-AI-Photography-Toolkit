"""
JoyTag tagger implementation.

Uses JoyTag model for general image tagging.
"""

import os
from typing import Any, Dict

import numpy as np
from PIL import Image

_model = None
_labels = None


def _get_model_path() -> str:
    """Get the path to cache JoyTag model files."""
    cache_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", ".cache", "joytag")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _load_model():
    """Load JoyTag model and labels."""
    global _model, _labels

    if _model is not None:
        return _model, _labels

    try:
        import torch
        from huggingface_hub import hf_hub_download

        model_id = "fancyfeast/joytag"
        cache_dir = _get_model_path()

        try:
            # Download model files
            model_path = hf_hub_download(
                repo_id=model_id,
                filename="model.safetensors",
                cache_dir=cache_dir,
            )
            config_path = hf_hub_download(
                repo_id=model_id,
                filename="config.json",
                cache_dir=cache_dir,
            )
            tags_path = hf_hub_download(
                repo_id=model_id,
                filename="top_tags.txt",
                cache_dir=cache_dir,
            )
        except Exception as e:
            print(f"[JoyTag] Model download failed: {e}")
            _model = "fallback_to_wd14"
            _labels = []
            return _model, _labels

        # Load labels
        _labels = []
        with open(tags_path, "r", encoding="utf-8") as f:
            for line in f:
                tag = line.strip()
                if tag:
                    _labels.append(tag)

        # Load model using transformers
        try:
            from transformers import AutoModel, AutoProcessor

            _model = AutoModel.from_pretrained(
                model_id,
                cache_dir=cache_dir,
                trust_remote_code=True,
            )
            _model.eval()

            if torch.cuda.is_available():
                _model = _model.cuda()

            print(f"[JoyTag] Loaded model with {len(_labels)} tags")

        except Exception as e:
            print(f"[JoyTag] Failed to load model: {e}")
            _model = "fallback_to_wd14"

        return _model, _labels

    except ImportError as e:
        print(f"[JoyTag] Missing dependency: {e}")
        _model = "fallback_to_wd14"
        _labels = []
        return _model, _labels


def run_joytag(
    image: Any,
    threshold: float = 0.4,
    max_tags: int = 50,
) -> Dict[str, float]:
    """
    Run JoyTag tagger on an image.

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
            # Fall back to WD14
            from .wd14 import run_wd14
            return run_wd14(image, threshold=threshold, max_tags=max_tags)

        import torch
        from torchvision import transforms

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

        pil_image = pil_image.convert("RGB")

        # Preprocess
        transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        input_tensor = transform(pil_image).unsqueeze(0)

        if torch.cuda.is_available() and next(model.parameters()).is_cuda:
            input_tensor = input_tensor.cuda()

        # Run inference
        with torch.no_grad():
            outputs = model(input_tensor)
            if hasattr(outputs, 'logits'):
                probs = torch.sigmoid(outputs.logits)[0]
            else:
                probs = torch.sigmoid(outputs)[0]

        # Parse results
        results = {}
        probs_np = probs.cpu().numpy()

        for i, prob in enumerate(probs_np):
            if i >= len(labels):
                break

            tag = labels[i]
            conf = float(prob)

            if conf >= threshold:
                results[tag] = conf

        # Sort by confidence and limit
        sorted_results = dict(
            sorted(results.items(), key=lambda x: x[1], reverse=True)[:max_tags]
        )

        print(f"[JoyTag] Found {len(sorted_results)} tags above threshold {threshold}")
        return sorted_results

    except Exception as e:
        print(f"[JoyTag] Error: {e}")
        import traceback
        traceback.print_exc()
        # Fall back to WD14
        from .wd14 import run_wd14
        return run_wd14(image, threshold=threshold, max_tags=max_tags)


def unload_model():
    """Unload JoyTag model from memory."""
    global _model, _labels
    if _model is not None and _model != "fallback_to_wd14":
        del _model
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    _model = None
    _labels = None
    print("[JoyTag] Model unloaded")
