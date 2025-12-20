"""SID Caption node for ComfyUI."""

import numpy as np
from PIL import Image

from ..core.models import ModelFactory
from ..core.models.base import CaptionMode
from ..core.output import clean_caption


def _tensor_to_pil(tensor) -> Image.Image:
    """Convert ComfyUI image tensor to PIL Image."""
    # ComfyUI tensors are (B, H, W, C) in 0-1 range
    if len(tensor.shape) == 4:
        tensor = tensor[0]  # Take first image from batch

    # Convert to numpy and scale to 0-255
    np_image = (tensor.cpu().numpy() * 255).astype(np.uint8)

    return Image.fromarray(np_image, mode="RGB")


class SID_Caption:
    """
    Simple image captioning node.

    Generates captions optimized for image generation prompts.
    Uses Florence-2 PromptGen for fast, high-quality results.
    """

    # Available caption modes
    MODES = ["detailed", "short", "tags", "analyze"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (cls.MODES, {"default": "detailed"}),
            },
            "optional": {
                "clean_output": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("caption",)
    FUNCTION = "generate_caption"
    CATEGORY = "SID Photography Toolkit"

    def generate_caption(
        self,
        image,
        mode: str = "detailed",
        clean_output: bool = True,
    ) -> tuple[str]:
        """
        Generate caption for input image.

        Args:
            image: ComfyUI image tensor
            mode: Caption mode (detailed/short/tags/analyze)
            clean_output: Whether to clean the output

        Returns:
            Tuple containing caption string
        """
        # Convert tensor to PIL
        pil_image = _tensor_to_pil(image)

        # Get model (cached)
        model = ModelFactory.get("florence")

        # Map string mode to enum
        caption_mode = CaptionMode(mode)

        # Generate caption
        caption = model.generate(pil_image, mode=caption_mode)

        # Clean if requested
        if clean_output:
            caption = clean_caption(caption)

        return (caption,)
