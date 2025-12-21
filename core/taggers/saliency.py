"""Saliency detection tagger - Subject position analysis."""

from PIL import Image
import numpy as np

from .base import BaseTagger, TagItem, TaggerResult
from .utils import download_hf_model


class SaliencyTagger(BaseTagger):
    """
    Saliency detection for subject position analysis.

    Uses DINO/DINOv2 to find salient regions and calculates
    subject position relative to composition grids:
    - Rule of thirds intersections
    - Center position
    - Golden ratio points
    - Edge positions

    Generates tags describing subject placement.
    """

    TAGGER_NAME = "saliency"
    MODEL_ID = "facebook/dinov2-base"

    def __init__(
        self,
        use_dino: bool = True,
        fallback_to_cv: bool = True,
        **kwargs
    ):
        """
        Initialize saliency tagger.

        Args:
            use_dino: Use DINOv2 for saliency (more accurate)
            fallback_to_cv: Use CV saliency if DINO unavailable
        """
        super().__init__(**kwargs)
        self.use_dino = use_dino
        self.fallback_to_cv = fallback_to_cv
        self._model = None
        self._processor = None
        self._device = None

    def load(self) -> None:
        """Load saliency model."""
        if self._is_loaded:
            return

        if self.use_dino:
            try:
                import torch
                from transformers import AutoImageProcessor, AutoModel

                self._device = "cuda" if torch.cuda.is_available() else "cpu"

                print("[SID-Saliency] Loading DINOv2...")
                model_path = download_hf_model(self.MODEL_ID, "saliency")

                self._processor = AutoImageProcessor.from_pretrained(model_path)
                self._model = AutoModel.from_pretrained(model_path)
                self._model.to(self._device)
                self._model.eval()
                print("[SID-Saliency] DINOv2 loaded")

            except Exception as e:
                print(f"[SID-Saliency] DINOv2 not available: {e}")
                if self.fallback_to_cv:
                    print("[SID-Saliency] Using CV saliency fallback")
                self._model = None

        self._is_loaded = True

    def _get_dino_saliency(self, image: Image.Image) -> np.ndarray:
        """Get saliency map using DINOv2 attention."""
        import torch

        rgb_image = image.convert("RGB")

        # Process image
        inputs = self._processor(images=rgb_image, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        # Get attention maps
        with torch.no_grad():
            outputs = self._model(**inputs, output_attentions=True)

            # Use last layer attention
            attentions = outputs.attentions[-1]  # [batch, heads, tokens, tokens]

            # Average across heads and get CLS token attention
            # CLS token attends to patch tokens
            cls_attention = attentions[0, :, 0, 1:].mean(dim=0)  # [num_patches]

            # Reshape to spatial grid
            num_patches = cls_attention.shape[0]
            grid_size = int(np.sqrt(num_patches))
            attention_map = cls_attention.reshape(grid_size, grid_size)

            # Resize to image size
            attention_np = attention_map.cpu().numpy()
            attention_resized = np.array(
                Image.fromarray(attention_np).resize(
                    (image.width, image.height),
                    Image.BILINEAR
                )
            )

            # Normalize
            attention_resized = (attention_resized - attention_resized.min())
            if attention_resized.max() > 0:
                attention_resized = attention_resized / attention_resized.max()

        return attention_resized

    def _get_cv_saliency(self, image: Image.Image) -> np.ndarray:
        """Get saliency map using OpenCV."""
        import cv2

        img_array = np.array(image.convert("RGB"))
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # Try spectral residual saliency
        saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
        success, saliency_map = saliency.computeSaliency(img_bgr)

        if success:
            return saliency_map.astype(np.float32)

        # Fallback to edge-based saliency
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        blurred = cv2.GaussianBlur(edges.astype(np.float32), (31, 31), 0)
        if blurred.max() > 0:
            blurred = blurred / blurred.max()
        return blurred

    def _analyze_subject_position(
        self,
        saliency_map: np.ndarray,
        image_size: tuple
    ) -> list[TagItem]:
        """Analyze subject position from saliency map."""
        tags = []
        h, w = saliency_map.shape

        # Find centroid of salient regions
        threshold = np.percentile(saliency_map, 90)  # Top 10% salient
        salient_mask = saliency_map >= threshold

        # Get centroid
        y_indices, x_indices = np.where(salient_mask)
        if len(x_indices) == 0:
            return [TagItem(text="unclear subject", confidence=0.5, category="saliency:position")]

        centroid_x = np.mean(x_indices)
        centroid_y = np.mean(y_indices)

        # Normalize to 0-1
        norm_x = centroid_x / w
        norm_y = centroid_y / h

        # Rule of thirds positions
        thirds_x = [1/3, 2/3]
        thirds_y = [1/3, 2/3]

        # Check thirds alignment
        near_thirds_x = any(abs(norm_x - t) < 0.1 for t in thirds_x)
        near_thirds_y = any(abs(norm_y - t) < 0.1 for t in thirds_y)

        if near_thirds_x and near_thirds_y:
            tags.append(TagItem(
                text="subject at thirds intersection",
                confidence=0.85,
                category="saliency:position"
            ))
            tags.append(TagItem(
                text="rule of thirds composition",
                confidence=0.8,
                category="saliency:composition"
            ))
        elif near_thirds_x:
            tags.append(TagItem(
                text="subject on vertical third",
                confidence=0.75,
                category="saliency:position"
            ))
        elif near_thirds_y:
            tags.append(TagItem(
                text="subject on horizontal third",
                confidence=0.75,
                category="saliency:position"
            ))

        # Check center position
        center_dist = np.sqrt((norm_x - 0.5)**2 + (norm_y - 0.5)**2)
        if center_dist < 0.15:
            tags.append(TagItem(
                text="centered subject",
                confidence=0.85,
                category="saliency:position"
            ))
            tags.append(TagItem(
                text="central composition",
                confidence=0.8,
                category="saliency:composition"
            ))

        # Check golden ratio positions (0.382, 0.618)
        golden = [0.382, 0.618]
        near_golden_x = any(abs(norm_x - g) < 0.08 for g in golden)
        near_golden_y = any(abs(norm_y - g) < 0.08 for g in golden)

        if near_golden_x and near_golden_y:
            tags.append(TagItem(
                text="subject at golden ratio",
                confidence=0.75,
                category="saliency:position"
            ))

        # Describe general position
        if norm_x < 0.33:
            h_pos = "left"
        elif norm_x > 0.67:
            h_pos = "right"
        else:
            h_pos = "center"

        if norm_y < 0.33:
            v_pos = "upper"
        elif norm_y > 0.67:
            v_pos = "lower"
        else:
            v_pos = "middle"

        if h_pos != "center" or v_pos != "middle":
            position_text = f"subject in {v_pos} {h_pos}"
            if h_pos == "center":
                position_text = f"subject in {v_pos} area"
            elif v_pos == "middle":
                position_text = f"subject on {h_pos} side"

            tags.append(TagItem(
                text=position_text,
                confidence=0.7,
                category="saliency:position"
            ))

        # Check if subject fills frame
        salient_area = np.sum(salient_mask) / (h * w)
        if salient_area > 0.4:
            tags.append(TagItem(
                text="subject fills frame",
                confidence=0.75,
                category="saliency:framing"
            ))
        elif salient_area < 0.1:
            tags.append(TagItem(
                text="small subject in frame",
                confidence=0.7,
                category="saliency:framing"
            ))
            tags.append(TagItem(
                text="environmental context",
                confidence=0.65,
                category="saliency:style"
            ))

        # Check for multiple salient regions
        from scipy import ndimage
        labeled, num_features = ndimage.label(salient_mask)
        if num_features > 1:
            # Get sizes of each region
            region_sizes = ndimage.sum(salient_mask, labeled, range(1, num_features + 1))
            significant_regions = sum(1 for s in region_sizes if s > (h * w * 0.02))

            if significant_regions >= 2:
                tags.append(TagItem(
                    text="multiple subjects",
                    confidence=0.7,
                    category="saliency:subjects"
                ))

        return tags

    def tag(self, image: Image.Image) -> TaggerResult:
        """
        Analyze subject position using saliency.

        Args:
            image: PIL Image to analyze

        Returns:
            TaggerResult with subject position tags
        """
        if not self._is_loaded:
            self.load()

        tags = []
        metadata = {}

        # Get saliency map
        if self._model is not None:
            try:
                saliency_map = self._get_dino_saliency(image)
                metadata["saliency_method"] = "dinov2"
            except Exception as e:
                print(f"[SID-Saliency] DINO failed: {e}")
                if self.fallback_to_cv:
                    saliency_map = self._get_cv_saliency(image)
                    metadata["saliency_method"] = "cv-spectral"
                else:
                    return TaggerResult(tags=[], metadata={"error": str(e)})
        elif self.fallback_to_cv:
            try:
                saliency_map = self._get_cv_saliency(image)
                metadata["saliency_method"] = "cv-spectral"
            except Exception as e:
                return TaggerResult(tags=[], metadata={"error": str(e)})
        else:
            return TaggerResult(tags=[], metadata={"error": "No saliency method available"})

        # Analyze subject position
        position_tags = self._analyze_subject_position(saliency_map, image.size)
        tags.extend(position_tags)

        return TaggerResult(
            tags=tags,
            metadata={
                "model": "saliency-analysis",
                **metadata,
            }
        )

    def unload(self) -> None:
        """Release model."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        self._device = None
        super().unload()
