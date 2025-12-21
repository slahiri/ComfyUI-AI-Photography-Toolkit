"""SID Tagger Configuration node for ComfyUI."""


class SID_TaggerConfig:
    """
    Configure image tagging models for analysis.

    Enables/disables various taggers and sets their parameters.
    Output connects to SID Z-Image Prompt Generator's tag_config input.
    """

    CATEGORY = "SID Photography Toolkit"
    RETURN_TYPES = ("TAG_CONFIG",)
    RETURN_NAMES = ("tag_config",)
    FUNCTION = "configure"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # WD14 (always enabled)
                "wd14_model": ([
                    "wd-swinv2-tagger-v3",
                    "wd-convnext-tagger-v3",
                    "wd-vit-tagger-v3",
                    "wd-eva02-large-tagger-v3",
                    "wd-v1-4-moat-tagger-v2",
                    "wd-v1-4-swinv2-tagger-v2",
                    "wd-v1-4-convnext-tagger-v2",
                    "wd-v1-4-vit-tagger-v2",
                ], {
                    "default": "wd-swinv2-tagger-v3",
                    "tooltip": "WD14 model variant (always enabled)"
                }),
                "wd14_general_threshold": ("FLOAT", {
                    "default": 0.25,
                    "min": 0.05,
                    "max": 0.9,
                    "step": 0.05,
                    "tooltip": "Lower = more tags. Confidence threshold for general tags"
                }),
                "wd14_character_threshold": ("FLOAT", {
                    "default": 0.85,
                    "min": 0.1,
                    "max": 0.9,
                    "step": 0.05,
                    "tooltip": "Confidence threshold for character tags"
                }),
                "wd14_exclude_tags": ("STRING", {
                    "default": "",
                    "tooltip": "Comma-separated tags to exclude"
                }),
                "wd14_replace_underscore": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Replace underscores with spaces"
                }),
                # JoyTag
                "joytag_enabled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "JoyTag - Photo-friendly Danbooru-style tags (5000+)"
                }),
                "joytag_threshold": ("FLOAT", {
                    "default": 0.4,
                    "min": 0.1,
                    "max": 0.9,
                    "step": 0.05,
                    "tooltip": "Confidence threshold for JoyTag"
                }),
                # NudeNet
                "nudenet_enabled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "NudeNet v2 - NSFW detection (16 classes)"
                }),
                "nudenet_threshold": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.1,
                    "max": 0.9,
                    "step": 0.05,
                    "tooltip": "Detection confidence threshold"
                }),
                "nudenet_include_covered": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Include covered body parts (bra, underwear visible)"
                }),
                # Photography (instant)
                "photography_enabled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Instant analysis: color, contrast, sharpness"
                }),
                # IQA - Image Quality Assessment
                "iqa_enabled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "IQA-PyTorch: NIMA aesthetic, MUSIQ technical, BRISQUE clarity"
                }),
                "iqa_use_nima": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "NIMA aesthetic quality scoring (1-10)"
                }),
                "iqa_use_musiq": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "MUSIQ technical quality assessment"
                }),
                "iqa_use_brisque": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "BRISQUE blur/noise detection"
                }),
                # Composition Analysis
                "composition_enabled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "CADB 13-class composition + element detection"
                }),
                "composition_threshold": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.1,
                    "max": 0.9,
                    "step": 0.05,
                    "tooltip": "Confidence threshold for composition types"
                }),
                # Saliency / Subject Position
                "saliency_enabled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "DINOv2 saliency for subject position (thirds, golden ratio)"
                }),
                # Fashion
                "yolov8_clothing_enabled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "YOLOv8n Clothing detection (~15ms)"
                }),
                "yolov8_threshold": ("FLOAT", {
                    "default": 0.25,
                    "min": 0.05,
                    "max": 0.9,
                    "step": 0.05,
                    "tooltip": "Lower = more detections"
                }),
                "yolos_fashion_enabled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "YOLOS-Fashionpedia 27 categories (~30ms)"
                }),
                "yolos_threshold": ("FLOAT", {
                    "default": 0.25,
                    "min": 0.05,
                    "max": 0.9,
                    "step": 0.05,
                    "tooltip": "Lower = more detections"
                }),
                "fashion_clip_enabled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "FashionCLIP zero-shot style (~50ms)"
                }),
                "fashion_clip_threshold": ("FLOAT", {
                    "default": 0.15,
                    "min": 0.05,
                    "max": 0.9,
                    "step": 0.05,
                    "tooltip": "Lower = more style tags (0.15 for ethnic wear)"
                }),
                "fashion_clip_top_k": ("INT", {
                    "default": 20,
                    "min": 1,
                    "max": 50,
                    "step": 1,
                    "tooltip": "Maximum style tags to return"
                }),
                "segformer_clothes_enabled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "SegFormer semantic segmentation (~60ms)"
                }),
                "segformer_min_area": ("FLOAT", {
                    "default": 0.02,
                    "min": 0.005,
                    "max": 0.2,
                    "step": 0.005,
                    "tooltip": "Minimum area % to generate tag (0.02 = 2%)"
                }),
                "wargon_classifier_enabled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Wargon ViT style classifier (~40ms)"
                }),
                "wargon_threshold": ("FLOAT", {
                    "default": 0.2,
                    "min": 0.05,
                    "max": 0.9,
                    "step": 0.05,
                    "tooltip": "Lower = more style tags"
                }),
                "wargon_top_k": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 15,
                    "step": 1,
                    "tooltip": "Maximum style tags"
                }),
                # Pose
                "mediapipe_pose_enabled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "MediaPipe Pose 33 keypoints (~20ms)"
                }),
                "mediapipe_confidence": ("FLOAT", {
                    "default": 0.4,
                    "min": 0.1,
                    "max": 0.9,
                    "step": 0.05,
                    "tooltip": "Minimum detection confidence"
                }),
                # Settings
                "max_tags": ("INT", {
                    "default": 150,
                    "min": 1,
                    "max": 500,
                    "step": 5,
                    "tooltip": "Maximum total tags to output"
                }),
                "verbose": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable verbose logging"
                }),
            },
        }

    def configure(
        self,
        # WD14 (always enabled)
        wd14_model: str = "wd-swinv2-tagger-v3",
        wd14_general_threshold: float = 0.25,
        wd14_character_threshold: float = 0.85,
        wd14_exclude_tags: str = "",
        wd14_replace_underscore: bool = True,
        # JoyTag
        joytag_enabled: bool = True,
        joytag_threshold: float = 0.4,
        # NudeNet
        nudenet_enabled: bool = True,
        nudenet_threshold: float = 0.5,
        nudenet_include_covered: bool = True,
        # Photography
        photography_enabled: bool = True,
        # IQA
        iqa_enabled: bool = True,
        iqa_use_nima: bool = True,
        iqa_use_musiq: bool = True,
        iqa_use_brisque: bool = True,
        # Composition
        composition_enabled: bool = True,
        composition_threshold: float = 0.3,
        # Saliency
        saliency_enabled: bool = True,
        # Fashion (all enabled by default for dense output)
        yolov8_clothing_enabled: bool = True,
        yolov8_threshold: float = 0.25,
        yolos_fashion_enabled: bool = True,
        yolos_threshold: float = 0.25,
        fashion_clip_enabled: bool = True,
        fashion_clip_threshold: float = 0.15,
        fashion_clip_top_k: int = 20,
        segformer_clothes_enabled: bool = True,
        segformer_min_area: float = 0.02,
        wargon_classifier_enabled: bool = True,
        wargon_threshold: float = 0.2,
        wargon_top_k: int = 5,
        # Pose
        mediapipe_pose_enabled: bool = True,
        mediapipe_confidence: float = 0.4,
        # Settings
        max_tags: int = 150,
        verbose: bool = False,
    ) -> tuple[dict]:
        """Create tag configuration dictionary."""
        config = {
            "max_tags": max_tags,
            "verbose": verbose,
            "taggers": {},
        }

        # WD14 configuration (always enabled)
        config["taggers"]["wd14"] = {
            "enabled": True,
            "model": wd14_model,
            "general_threshold": wd14_general_threshold,
            "character_threshold": wd14_character_threshold,
            "exclude_tags": wd14_exclude_tags,
            "replace_underscore": wd14_replace_underscore,
        }

        # JoyTag (photo-friendly Danbooru tags)
        if joytag_enabled:
            config["taggers"]["joytag"] = {
                "enabled": True,
                "threshold": joytag_threshold,
            }

        # NudeNet (NSFW detection)
        if nudenet_enabled:
            config["taggers"]["nudenet"] = {
                "enabled": True,
                "threshold": nudenet_threshold,
                "include_covered": nudenet_include_covered,
                "include_faces": True,
            }

        # Photography analysis (instant)
        if photography_enabled:
            config["taggers"]["photography"] = {"enabled": True}

        # IQA - Image Quality Assessment
        if iqa_enabled:
            config["taggers"]["iqa"] = {
                "enabled": True,
                "use_nima": iqa_use_nima,
                "use_musiq": iqa_use_musiq,
                "use_brisque": iqa_use_brisque,
                "use_clipiqa": False,  # Slower, disabled by default
            }

        # Composition Analysis (CADB + elements)
        if composition_enabled:
            config["taggers"]["composition"] = {
                "enabled": True,
                "threshold": composition_threshold,
                "detect_elements": True,
            }

        # Saliency / Subject Position
        if saliency_enabled:
            config["taggers"]["saliency"] = {
                "enabled": True,
                "use_dino": True,
                "fallback_to_cv": True,
            }

        # Fashion models
        fashion_models = {}
        fashion_params = {}

        if yolov8_clothing_enabled:
            fashion_models["yolov8"] = True
            fashion_params["yolov8_threshold"] = yolov8_threshold

        if yolos_fashion_enabled:
            fashion_models["yolos"] = True
            fashion_params["yolos_threshold"] = yolos_threshold

        if fashion_clip_enabled:
            fashion_models["clip"] = True
            fashion_params["clip_threshold"] = fashion_clip_threshold
            fashion_params["clip_top_k"] = fashion_clip_top_k

        if segformer_clothes_enabled:
            fashion_models["segformer"] = True
            fashion_params["segformer_min_area"] = segformer_min_area

        if wargon_classifier_enabled:
            fashion_models["wargon"] = True
            fashion_params["wargon_threshold"] = wargon_threshold
            fashion_params["wargon_top_k"] = wargon_top_k

        if fashion_models:
            config["taggers"]["fashion"] = {
                "enabled": True,
                "models": fashion_models,
                "params": fashion_params,
            }

        # Pose models
        pose_models = {}
        pose_params = {}

        if mediapipe_pose_enabled:
            pose_models["mediapipe"] = True
            pose_params["mediapipe_confidence"] = mediapipe_confidence

        if pose_models:
            config["taggers"]["pose"] = {
                "enabled": True,
                "models": pose_models,
                "params": pose_params,
            }

        return (config,)
