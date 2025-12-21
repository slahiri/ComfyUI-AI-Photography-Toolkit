# SID Tag Filters Reference

This document catalogs all available tag filters for the SID Tagging Pipeline.
Each filter can be enabled/disabled and configured via the `SID_TagConfigurator` node.

---

## Table of Contents

1. [General Tagging](#1-general-tagging)
2. [Aesthetic & Quality Assessment](#2-aesthetic--quality-assessment)
3. [Scene Classification](#3-scene-classification)
4. [Clothing & Fashion](#4-clothing--fashion)
5. [Pose Estimation](#5-pose-estimation)
6. [Depth Estimation](#6-depth-estimation)
7. [Blur & Sharpness Detection](#7-blur--sharpness-detection)
8. [NSFW / Content Safety](#8-nsfw--content-safety)
9. [Color Analysis](#9-color-analysis)
10. [Composition Analysis](#10-composition-analysis)
11. [Face & Person Analysis](#11-face--person-analysis)
12. [Object Detection](#12-object-detection)
13. [Lighting Estimation](#13-lighting-estimation)

---

## 1. General Tagging

### 1.1 WD14 Tagger (SmilingWolf)

**Purpose:** General-purpose image tagging using Danbooru/anime-style tags. Works well for both anime and realistic images.

| Attribute | Details |
|-----------|---------|
| **Type** | Multi-label Classification |
| **Output** | Booru-style tags (character, general, rating) |
| **VRAM** | ~2GB |
| **Speed** | Fast (~50ms) |

**Available Model Variants:**

| Model | HuggingFace ID | Size | Notes |
|-------|----------------|------|-------|
| SwinV2 v3 | `SmilingWolf/wd-swinv2-tagger-v3` | ~350MB | **Recommended** - Best accuracy |
| ViT v3 | `SmilingWolf/wd-vit-tagger-v3` | ~350MB | Good balance |
| ConvNext v3 | `SmilingWolf/wd-convnext-tagger-v3` | ~350MB | Fast inference |
| EVA02-Large v3 | `SmilingWolf/wd-eva02-large-tagger-v3` | ~1GB | Highest accuracy |
| MOAT v2 | `SmilingWolf/wd-v1-4-moat-tagger-v2` | ~300MB | Legacy, still good |

**Configuration Parameters:**

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `model_variant` | enum | `swinv2-v3` | See above | Which WD14 model to use |
| `general_threshold` | float | 0.35 | 0.1-0.9 | Confidence threshold for general tags |
| `character_threshold` | float | 0.85 | 0.1-0.9 | Confidence threshold for character tags |
| `replace_underscore` | bool | True | - | Replace `_` with space in tags |
| `trailing_comma` | bool | True | - | Add trailing comma for prompt compatibility |
| `exclude_tags` | string | "" | - | Comma-separated tags to exclude |

**Output Tags Example:**
```
1girl, solo, long hair, blonde hair, blue eyes, dress, standing, outdoors, sky, clouds
```

**Implementation Notes:**
- Uses ONNX runtime for fast inference
- Outputs three categories: general, character, rating
- Rating tags: `safe`, `questionable`, `explicit`

---

## 2. Aesthetic & Quality Assessment

### 2.1 NIMA (Neural Image Assessment)

**Purpose:** Predict aesthetic quality score (1-10) based on human ratings.

| Attribute | Details |
|-----------|---------|
| **Type** | Regression / Distribution Prediction |
| **HuggingFace** | `idealo/image-quality-assessment` |
| **Training Data** | AVA Dataset (255K images rated by photographers) |
| **VRAM** | ~50MB |
| **Speed** | Very Fast (~10ms) |

**Configuration Parameters:**

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `output_distribution` | bool | False | - | Output full score distribution vs mean |

**Output:**
- Aesthetic score: 1.0 - 10.0
- Converts to tags: `masterpiece`, `high quality`, `average quality`, `low quality`

**Score to Tag Mapping:**
| Score Range | Tags Generated |
|-------------|----------------|
| 7.0+ | `masterpiece, high quality, professional` |
| 5.5 - 7.0 | `good quality` |
| 4.0 - 5.5 | `average quality` |
| < 4.0 | `low quality` |

---

### 2.2 MUSIQ (Multi-Scale Image Quality)

**Purpose:** No-reference image quality assessment that handles any resolution.

| Attribute | Details |
|-----------|---------|
| **Type** | Quality Score Prediction |
| **Source** | `google-research/musiq` |
| **VRAM** | ~100MB |
| **Speed** | Fast (~20ms) |

**Key Features:**
- Handles arbitrary image resolutions (no resize needed)
- Multi-scale analysis
- Trained on multiple IQA datasets

---

### 2.3 TOPIQ

**Purpose:** Perceptual quality and distortion detection.

| Attribute | Details |
|-----------|---------|
| **Type** | No-Reference IQA |
| **Source** | `chaofengc/IQA-PyTorch` |
| **VRAM** | ~100MB |
| **Speed** | Fast |

---

### 2.4 BRISQUE

**Purpose:** Blind/Referenceless Image Spatial Quality Evaluator.

| Attribute | Details |
|-----------|---------|
| **Type** | No-Reference IQA |
| **Source** | OpenCV / scikit-image |
| **VRAM** | 0 (CPU-based) |
| **Speed** | Very Fast (~10ms) |

**Output:** Quality score (lower = better quality)

---

## 3. Scene Classification

### 3.1 Places365

**Purpose:** Classify scene type from 365 categories + indoor/outdoor detection.

| Attribute | Details |
|-----------|---------|
| **Type** | Multi-class Classification |
| **Developer** | MIT CSAIL |
| **Categories** | 365 scene types |
| **Attributes** | 102 scene attributes |

**Available Model Variants:**

| Model | HuggingFace ID | Size | Speed | Accuracy |
|-------|----------------|------|-------|----------|
| ResNet18 | `CSAILVision/places365` | ~45MB | Very Fast (~2ms) | Good |
| ResNet50 | `CSAILVision/places365` | ~100MB | Fast (~5ms) | Better |
| DenseNet161 | `CSAILVision/places365` | ~115MB | Medium (~15ms) | Best |

**Configuration Parameters:**

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `model_variant` | enum | `resnet18` | resnet18/resnet50/densenet161 | Model size/accuracy tradeoff |
| `top_k` | int | 5 | 1-10 | Number of top scene predictions |
| `include_attributes` | bool | True | - | Include 102 scene attributes |
| `include_io` | bool | True | - | Include indoor/outdoor prediction |

**Output Tags Example:**
```
outdoor, beach, natural light, open area, sunny, vacation
```

**102 Scene Attributes Include:**
- Lighting: `natural light`, `artificial light`, `sunny`, `cloudy`
- Space: `open area`, `enclosed area`, `cluttered`, `sparse`
- Materials: `vegetation`, `water`, `man-made`, `natural`
- And 90+ more...

---

## 4. Clothing & Fashion

### 4.1 FashionCLIP

**Purpose:** Zero-shot fashion classification and text-image matching.

| Attribute | Details |
|-----------|---------|
| **Type** | Zero-shot Classification / Embeddings |
| **HuggingFace** | `patrickjohncyh/fashion-clip` |
| **Architecture** | ViT-B/32 + Text Transformer |
| **Parameters** | ~150M |
| **Training Data** | Farfetch: 800K+ products, 3K+ brands |
| **VRAM** | ~600MB |
| **Speed** | Medium (~50ms) |

**Configuration Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `candidate_labels` | list | [predefined] | Fashion categories to detect |
| `top_k` | int | 5 | Number of top predictions |

**Default Candidate Labels:**
```python
[
    "dress", "shirt", "pants", "skirt", "jacket", "coat",
    "sweater", "blouse", "jeans", "shorts", "suit", "t-shirt",
    "hoodie", "cardigan", "blazer", "vest", "jumpsuit", "romper",
    "casual wear", "formal wear", "sportswear", "streetwear",
    "vintage style", "modern style", "minimalist", "bohemian"
]
```

---

### 4.2 YOLOS-Fashionpedia

**Purpose:** Detect and localize 46 clothing categories with bounding boxes.

| Attribute | Details |
|-----------|---------|
| **Type** | Object Detection |
| **HuggingFace** | `valentinafeve/yolos-fashionpedia` |
| **Architecture** | YOLOS (Vision Transformer) |
| **Training Data** | Fashionpedia: 46,781 images, 342,182 boxes |
| **Size** | ~123MB |
| **VRAM** | ~500MB |
| **Speed** | Medium (~100ms) |

**46 Supported Categories:**

| Category | Items |
|----------|-------|
| **Apparel (27)** | shirt, blouse, top, t-shirt, sweatshirt, sweater, cardigan, jacket, vest, pants, shorts, skirt, coat, dress, jumpsuit, cape, glasses, hat, headband, tie, glove, watch, belt, leg warmer, tights, stockings, sock |
| **Accessories (8)** | shoe, bag, wallet, scarf, umbrella |
| **Parts (7)** | hood, collar, lapel, epaulette, sleeve, pocket, neckline |
| **Details (4)** | buckle, zipper, applique, bead, bow, flower, fringe, ribbon, rivet, ruffle, sequin, tassel |

**Configuration Parameters:**

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `confidence_threshold` | float | 0.5 | 0.1-0.9 | Detection confidence threshold |
| `include_boxes` | bool | False | - | Include bounding box coordinates in metadata |

---

### 4.3 YOLOv8n Clothing Detection

**Purpose:** Fast, lightweight clothing detection with 4 basic classes.

| Attribute | Details |
|-----------|---------|
| **Type** | Object Detection |
| **HuggingFace** | `kesimeg/yolov8n-clothing-detection` |
| **Architecture** | YOLOv8 Nano |
| **Size** | ~6MB |
| **VRAM** | ~20MB |
| **Speed** | Very Fast (~5ms) |

**4 Detection Classes:**
1. Clothing
2. Shoes
3. Bags
4. Accessories

**Configuration Parameters:**

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `confidence_threshold` | float | 0.25 | 0.1-0.9 | Detection threshold |
| `iou_threshold` | float | 0.45 | 0.1-0.9 | NMS IoU threshold |

---

### 4.4 SegFormer Clothes (segformer_b2_clothes)

**Purpose:** Pixel-level clothing segmentation.

| Attribute | Details |
|-----------|---------|
| **Type** | Semantic Segmentation |
| **HuggingFace** | `mattmdjaga/segformer_b2_clothes` |
| **Architecture** | SegFormer-B2 |
| **Training Data** | ATR Dataset |
| **Size** | ~109MB |
| **VRAM** | ~400MB |
| **Speed** | Medium (~80ms) |

**18 Segmentation Classes:**
```
Background, Hat, Hair, Sunglasses, Upper-clothes, Skirt, Pants, Dress,
Belt, Left-shoe, Right-shoe, Face, Left-leg, Right-leg, Left-arm,
Right-arm, Bag, Scarf
```

**Configuration Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_area_percent` | float | 1.0 | Minimum segment area (% of image) to include |
| `output_mask` | bool | False | Include segmentation mask in metadata |

---

### 4.5 Wargon Clothing Classifier

**Purpose:** Classify clothing with secondhand/vintage attributes.

| Attribute | Details |
|-----------|---------|
| **Type** | Image Classification |
| **HuggingFace** | `wargoninnovation/wargon-clothing-classifier` |
| **Architecture** | ViT-Base/16 |
| **Parameters** | ~86M |
| **Accuracy** | 73% |
| **VRAM** | ~350MB |
| **Speed** | Fast (~30ms) |

**Special Features:**
- Trained on secondhand clothing
- Includes condition ratings (damage, stains, pilling: 1-5)

---

## 5. Pose Estimation

### 5.1 MediaPipe Pose

**Purpose:** Real-time 2D pose estimation.

| Attribute | Details |
|-----------|---------|
| **Type** | 2D Keypoint Detection |
| **Developer** | Google |
| **Keypoints** | 33 body landmarks |
| **VRAM** | ~100MB |
| **Speed** | Very Fast (<1ms on GPU) |

**Configuration Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_detection_confidence` | float | 0.5 | Minimum detection confidence |
| `min_tracking_confidence` | float | 0.5 | Minimum tracking confidence |

**Output Tags:**
- Pose descriptions: `standing`, `sitting`, `arms raised`, `facing camera`, `profile view`

---

### 5.2 Sapiens Pose

**Purpose:** High-detail pose estimation with 308 keypoints.

| Attribute | Details |
|-----------|---------|
| **Type** | 2D Keypoint Detection |
| **Developer** | Meta/Facebook AI |
| **Keypoints** | 308 (body + face + hands + feet) |

**Model Variants:**

| Model | Parameters | VRAM | Speed |
|-------|------------|------|-------|
| Sapiens-0.3B | 300M | ~1.2GB | Medium |
| Sapiens-0.6B | 600M | ~2.4GB | Slow |
| Sapiens-1B | 1B | ~4GB | Very Slow |
| Sapiens-2B | 2B | ~8GB | Very Slow |

**Configuration Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_size` | enum | `0.3b` | Model size variant |
| `include_face` | bool | True | Include 243 facial keypoints |
| `include_hands` | bool | True | Include 40 hand keypoints |

---

### 5.3 MotionBERT

**Purpose:** 3D pose estimation and motion analysis.

| Attribute | Details |
|-----------|---------|
| **Type** | 3D Pose / Motion |
| **HuggingFace** | `walterzhu/MotionBERT` |
| **Architecture** | DSTformer |
| **Keypoints** | 17 (H36M format) |
| **Max Frames** | 243 |

**Model Variants:**

| Model | Size | Use Case |
|-------|------|----------|
| MotionBERT | 162MB | Full performance |
| MotionBERT-Lite | 61MB | Lower compute |

---

## 6. Depth Estimation

### 6.1 MiDaS

**Purpose:** Monocular depth estimation.

| Attribute | Details |
|-----------|---------|
| **Type** | Relative Depth Estimation |
| **Developer** | Intel ISL |
| **Output** | Relative depth map |

**Model Variants:**

| Model | HuggingFace ID | Size | Speed | Quality |
|-------|----------------|------|-------|---------|
| Small | `Intel/dpt-small` | ~50MB | Very Fast (~5ms) | Acceptable |
| Hybrid | `Intel/dpt-hybrid-midas` | ~500MB | Fast (~50ms) | Good |
| Large | `Intel/dpt-large` | ~1.3GB | Medium (~100ms) | Very Good |
| BEiT-L-512 | `Intel/dpt-beit-large-512` | ~2GB | Slow (~200ms) | Best |
| Swin2-T-256 | `Intel/dpt-swinv2-tiny-256` | ~100MB | Fast (~15ms) | Good |

**Configuration Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_variant` | enum | `small` | Model size/quality tradeoff |
| `output_depth_map` | bool | False | Include depth map in metadata |

**Output Tags:**
- `shallow depth of field`, `deep depth of field`
- `close-up`, `distant subject`
- `foreground focus`, `background blur`

---

### 6.2 ZoeDepth

**Purpose:** Metric depth estimation (actual distance in meters).

| Attribute | Details |
|-----------|---------|
| **Type** | Metric Depth Estimation |
| **HuggingFace** | `isl-org/ZoeDepth` |
| **Output** | Depth in meters |
| **VRAM** | ~1GB |
| **Speed** | Medium |

---

## 7. Blur & Sharpness Detection

### 7.1 Laplacian Variance (No ML)

**Purpose:** Instant sharpness detection using image gradients.

| Attribute | Details |
|-----------|---------|
| **Type** | Signal Processing |
| **Library** | OpenCV |
| **VRAM** | 0 (CPU only) |
| **Speed** | Instant (<1ms) |

**Output:**
- Variance score (higher = sharper)
- Tags: `sharp`, `slightly blurry`, `blurry`, `very blurry`

**Score Thresholds:**
| Score | Tag |
|-------|-----|
| > 500 | `sharp, in focus` |
| 100-500 | `slightly soft` |
| 50-100 | `blurry` |
| < 50 | `very blurry, out of focus` |

---

### 7.2 ViT-Base-Blur

**Purpose:** ML-based blur classification.

| Attribute | Details |
|-----------|---------|
| **Type** | Binary Classification |
| **HuggingFace** | `WT-MM/vit-base-blur` |
| **VRAM** | ~350MB |
| **Speed** | Fast (~20ms) |

---

## 8. NSFW / Content Safety

### 8.1 Falconsai NSFW Detector

**Purpose:** Binary safe/nsfw classification.

| Attribute | Details |
|-----------|---------|
| **Type** | Binary Classification |
| **HuggingFace** | `Falconsai/nsfw_image_detection` |
| **Accuracy** | 98% |
| **VRAM** | ~500MB |
| **Speed** | Fast (~30ms) |

**Output Tags:**
- `safe` or `nsfw`

---

### 8.2 Image-Guard-2.0

**Purpose:** Multi-class content safety with style detection.

| Attribute | Details |
|-----------|---------|
| **Type** | Multi-class Classification |
| **HuggingFace** | `prithivMLmods/Image-Guard-2.0` |
| **Classes** | safe, nsfw-anime, nsfw-realistic |
| **Accuracy** | 99% |

---

### 8.3 FocalNet NSFW

**Purpose:** Three-class content rating.

| Attribute | Details |
|-----------|---------|
| **Type** | Multi-class Classification |
| **HuggingFace** | `MichalMlodawski/nsfw-image-detection-large` |
| **Classes** | safe, questionable, unsafe |

---

## 9. Color Analysis

### 9.1 ColorThief (No ML)

**Purpose:** Extract dominant colors from image.

| Attribute | Details |
|-----------|---------|
| **Type** | Color Quantization |
| **Library** | `colorthief` |
| **VRAM** | 0 (CPU only) |
| **Speed** | Instant (<5ms) |

**Configuration Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `color_count` | int | 6 | Number of palette colors |
| `quality` | int | 1 | Sampling quality (1=best) |

**Output Tags:**
- Color names: `red`, `blue`, `warm tones`, `cool tones`
- `monochromatic`, `colorful`, `muted colors`, `vibrant colors`
- `high contrast`, `low contrast`

---

### 9.2 Color Temperature (No ML)

**Purpose:** Estimate warm/cool color balance.

| Attribute | Details |
|-----------|---------|
| **Type** | Color Analysis |
| **Library** | OpenCV/NumPy |
| **VRAM** | 0 |
| **Speed** | Instant |

**Output Tags:**
- `warm lighting`, `cool lighting`, `neutral lighting`
- `golden hour`, `blue hour`

---

## 10. Composition Analysis

### 10.1 SAMP-Net / CADB Models

**Purpose:** Analyze image composition patterns.

| Attribute | Details |
|-----------|---------|
| **Type** | Composition Classification |
| **Source** | `bcmi/Image-Composition-Assessment` |

**13 Composition Classes:**
1. Center
2. Rule of Thirds
3. Golden Ratio
4. Triangle
5. Horizontal
6. Vertical
7. Diagonal
8. Symmetric
9. Curved
10. Radial
11. Vanishing Point
12. Pattern
13. Fill the Frame

**Output Tags:**
- `rule of thirds`, `centered composition`, `symmetrical`
- `diagonal lines`, `leading lines`, `vanishing point`

---

## 11. Face & Person Analysis

### 11.1 InsightFace

**Purpose:** Face detection with age, gender, expression.

| Attribute | Details |
|-----------|---------|
| **Type** | Face Analysis |
| **Source** | `deepinsight/insightface` |
| **Detects** | Age, gender, expression, landmarks |
| **VRAM** | ~500MB |
| **Speed** | Fast (~50ms) |

**Output Tags:**
- `young woman`, `elderly man`, `child`
- `smiling`, `serious`, `neutral expression`
- `looking at camera`, `profile`

---

### 11.2 MediaPipe Face

**Purpose:** Face mesh with 468 landmarks.

| Attribute | Details |
|-----------|---------|
| **Type** | Face Landmark Detection |
| **Developer** | Google |
| **Landmarks** | 468 points |
| **VRAM** | ~50MB |
| **Speed** | Very Fast |

---

### 11.3 RetinaFace

**Purpose:** Robust face detection.

| Attribute | Details |
|-----------|---------|
| **Type** | Face Detection |
| **HuggingFace** | `biubug6/Pytorch_Retinaface` |
| **Output** | Bounding boxes + 5 landmarks |

---

## 12. Object Detection

### 12.1 YOLOv8

**Purpose:** General object detection (80 COCO classes).

| Attribute | Details |
|-----------|---------|
| **Type** | Object Detection |
| **Developer** | Ultralytics |
| **Classes** | 80 (COCO) |

**Model Variants:**

| Model | Size | Speed | mAP |
|-------|------|-------|-----|
| YOLOv8n | 6MB | 1-2ms | 37.3 |
| YOLOv8s | 22MB | 3-5ms | 44.9 |
| YOLOv8m | 50MB | 5-10ms | 50.2 |
| YOLOv8l | 84MB | 10-15ms | 52.9 |
| YOLOv8x | 131MB | 15-25ms | 53.9 |

**Configuration Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_size` | enum | `n` | n/s/m/l/x |
| `confidence_threshold` | float | 0.25 | Detection threshold |
| `classes` | list | all | Specific classes to detect |

---

### 12.2 RT-DETR

**Purpose:** Real-time transformer-based detection.

| Attribute | Details |
|-----------|---------|
| **Type** | Object Detection |
| **Developer** | Baidu/Ultralytics |
| **Speed** | ~10ms |

---

## 13. Lighting Estimation

### 13.1 Intrinsic Image Decomposition

**Purpose:** Estimate lighting direction and intensity.

**Output Tags:**
- `front lit`, `back lit`, `side lit`
- `hard lighting`, `soft lighting`, `diffused light`
- `high key`, `low key`
- `natural light`, `artificial light`, `studio lighting`

---

### 13.2 DiffusionLight

**Purpose:** Generate HDR environment map from single image.

| Attribute | Details |
|-----------|---------|
| **Type** | Light Estimation |
| **Source** | `DiffusionLight/DiffusionLight` |
| **Output** | HDR environment map |

---

## Implementation Priority

### Phase 1: Core Taggers (Essential)
1. **WD14 Tagger** - General tags, works for everything
2. **NIMA Aesthetics** - Quality scoring
3. **Places365 Scene** - Scene classification
4. **Falconsai NSFW** - Content safety

### Phase 2: Visual Analysis (Recommended)
5. **Laplacian Blur** - Sharpness detection (instant, no model)
6. **ColorThief** - Color analysis (instant, no model)
7. **MiDaS Depth** - Depth estimation

### Phase 3: Fashion & Clothing
8. **SegFormer Clothes** - Clothing segmentation
9. **FashionCLIP** - Fashion classification
10. **YOLOS-Fashionpedia** - Detailed fashion detection

### Phase 4: People & Pose
11. **MediaPipe Pose** - Pose detection
12. **InsightFace** - Face analysis

### Phase 5: Advanced
13. **Composition Analysis** - SAMP-Net
14. **YOLOv8 Objects** - General objects
15. **Sapiens Pose** - Detailed pose (optional, heavy)

---

## Quick Reference: Speed vs Accuracy

| Speed Tier | Models | Use Case |
|------------|--------|----------|
| **Instant** (<5ms) | Laplacian, ColorThief, Color Temp | Always enable |
| **Very Fast** (5-20ms) | WD14, Places365-R18, NIMA, YOLOv8n | Default pipeline |
| **Fast** (20-50ms) | Falconsai, MediaPipe, MiDaS-Small | Recommended |
| **Medium** (50-100ms) | FashionCLIP, SegFormer, InsightFace | Fashion-focused |
| **Slow** (100ms+) | YOLOS-Fashion, Sapiens, MiDaS-Large | Quality over speed |

---

## Memory Budget Guide

| VRAM Available | Recommended Taggers |
|----------------|---------------------|
| **4GB** | WD14 + NIMA + Places365-R18 + Laplacian + ColorThief |
| **8GB** | Above + Falconsai + MediaPipe + MiDaS-Small |
| **12GB** | Above + SegFormer + FashionCLIP |
| **16GB+** | All taggers available |

---

## Output Format

All taggers output to a unified `TagResult` structure:

```python
@dataclass
class TagItem:
    text: str           # The tag text
    confidence: float   # 0.0 - 1.0
    category: str       # Tagger category name

@dataclass
class TagResult:
    tags: list[TagItem]         # All detected tags
    tags_string: str            # Comma-separated tags
    metadata: dict              # Scores, maps, boxes, etc.
```

Example aggregated output:
```
1girl, blonde hair, dress, standing, outdoor, beach, sunset,
high quality, sharp, warm tones, rule of thirds, safe
```
