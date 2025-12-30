"""
Tagger runner for executing image analysis.

Provides a unified interface for running all taggers and analyzers.
Integrates WD14, NudeNet, Places365, and other analysis models.
"""

import time
from typing import Any, Dict, Optional, List, Callable
from ..types import TaggerResults

# Import tagger implementations
from .wd14 import run_wd14, unload_model as unload_wd14
from .pixai import run_pixai, unload_model as unload_pixai
from .joytag import run_joytag, unload_model as unload_joytag
from .nudenet import run_nudenet, unload_detector as unload_nudenet
from .pose import run_pose, unload_model as unload_pose

# Import analyzer implementations
from .shot_type import run_shot_type, unload_model as unload_shot_type
from .places365 import run_places365, unload_model as unload_places365
from .photography import run_photography
from .iqa import run_iqa, unload_models as unload_iqa
from .composition import run_composition
from .saliency import run_saliency, unload_model as unload_saliency
from .clip_camera import run_clip_camera, unload_model as unload_clip_camera
from .lighting import run_lighting
from .fashion_color import run_fashion_color
from .deepfashion import run_deepfashion, unload_model as unload_deepfashion
from .wildlife import run_wildlife, unload_model as unload_wildlife
from .landmarks import run_landmarks, unload_model as unload_landmarks
from .sky_weather import run_sky_weather, unload_model as unload_sky_weather


class TaggerRunner:
    """
    Runs image taggers and analyzers to extract metadata.

    Integrates real tagger implementations for comprehensive image analysis.
    """

    # List of available taggers
    TAGGERS = ["wd14", "pixai", "joytag", "nudenet", "pose"]

    # List of available analyzers
    ANALYZERS = [
        "photography", "iqa", "composition", "shot_type",
        "places365", "clip_camera", "saliency", "lighting",
        "fashion_color", "deepfashion", "wildlife", "landmarks", "sky_weather"
    ]

    # Map tagger names to their runner functions
    TAGGER_FUNCTIONS: Dict[str, Callable] = {
        "wd14": run_wd14,
        "pixai": run_pixai,
        "joytag": run_joytag,
        "nudenet": run_nudenet,
        "pose": run_pose,
    }

    # Map analyzer names to their runner functions
    ANALYZER_FUNCTIONS: Dict[str, Callable] = {
        "shot_type": run_shot_type,
        "places365": run_places365,
        "photography": run_photography,
        "iqa": run_iqa,
        "composition": run_composition,
        "saliency": run_saliency,
        "clip_camera": run_clip_camera,
        "lighting": run_lighting,
        "fashion_color": run_fashion_color,
        "deepfashion": run_deepfashion,
        "wildlife": run_wildlife,
        "landmarks": run_landmarks,
        "sky_weather": run_sky_weather,
    }

    def __init__(self):
        """Initialize the tagger runner."""
        self._loaded_taggers: Dict[str, Any] = {}
        self._loaded_analyzers: Dict[str, Any] = {}

    def run_all(
        self,
        image: Any,
        enabled_taggers: Optional[List[str]] = None,
        enabled_analyzers: Optional[List[str]] = None,
        threshold: float = 0.65,
    ) -> TaggerResults:
        """
        Run all enabled taggers and analyzers on an image.

        Args:
            image: Image tensor from ComfyUI
            enabled_taggers: List of taggers to run (None = all)
            enabled_analyzers: List of analyzers to run (None = all)
            threshold: Minimum confidence threshold for tags

        Returns:
            TaggerResults with all outputs
        """
        results = TaggerResults()
        start_time = time.time()

        # Track execution status
        succeeded = []
        failed = []
        skipped = []

        # Determine which to run
        taggers_to_run = enabled_taggers or self.TAGGERS
        analyzers_to_run = enabled_analyzers or self.ANALYZERS

        # Run taggers
        for tagger_name in taggers_to_run:
            try:
                output = self._run_tagger(tagger_name, image, threshold)
                setattr(results, tagger_name, output)
                if output:  # Non-empty result
                    succeeded.append(f"tagger:{tagger_name}")
                else:
                    skipped.append(f"tagger:{tagger_name}")
            except Exception as e:
                failed.append(f"tagger:{tagger_name}")
                print(f"[TaggerRunner] Error running {tagger_name}: {e}")

        # Run analyzers
        for analyzer_name in analyzers_to_run:
            try:
                output = self._run_analyzer(analyzer_name, image)
                setattr(results, analyzer_name, output)
                if output:  # Non-empty result
                    succeeded.append(f"analyzer:{analyzer_name}")
                else:
                    skipped.append(f"analyzer:{analyzer_name}")
            except Exception as e:
                failed.append(f"analyzer:{analyzer_name}")
                print(f"[TaggerRunner] Error running {analyzer_name}: {e}")

        total_time = time.time() - start_time

        # Print summary with detected tags
        self._print_summary(succeeded, failed, skipped, total_time, results, threshold)

        return results

    def _print_summary(
        self,
        succeeded: List[str],
        failed: List[str],
        skipped: List[str],
        total_time: float,
        results: Optional[TaggerResults] = None,
        threshold: float = 0.65,
    ):
        """Print execution summary to console."""
        print("\n" + "=" * 60)
        print("TAGGER/ANALYZER EXECUTION SUMMARY")
        print("=" * 60)

        # Succeeded
        if succeeded:
            print(f"\n[OK] SUCCEEDED ({len(succeeded)}):")
            for item in succeeded:
                category, name = item.split(":")
                print(f"     {name}")

        # Failed
        if failed:
            print(f"\n[X] FAILED ({len(failed)}):")
            for item in failed:
                category, name = item.split(":")
                print(f"     {name}")

        # Skipped (no output / not available)
        if skipped:
            print(f"\n[--] SKIPPED/NO OUTPUT ({len(skipped)}):")
            for item in skipped:
                category, name = item.split(":")
                print(f"     {name} (model not available or no detections)")

        # Summary stats
        total = len(succeeded) + len(failed) + len(skipped)
        print(f"\n" + "-" * 60)
        print(f"Total: {total} | OK: {len(succeeded)} | Failed: {len(failed)} | Skipped: {len(skipped)}")
        print(f"Time: {total_time:.2f}s")
        print("=" * 60)

        # Print detected tags with confidence
        if results:
            self._print_detected_tags(results, threshold)

    def _print_detected_tags(self, results: TaggerResults, threshold: float = 0.65):
        """Print detected tags with confidence scores."""
        print("\n" + "=" * 60)
        print(f"DETECTED TAGS (threshold >= {threshold:.2f})")
        print("=" * 60)

        # Print WD14 tags
        if results.wd14:
            print(f"\n[WD14] Top tags ({len(results.wd14)} total):")
            sorted_tags = sorted(results.wd14.items(), key=lambda x: x[1], reverse=True)
            for tag, conf in sorted_tags[:15]:
                print(f"  {conf:.2f}  {tag}")

        # Print PixAI tags
        if results.pixai:
            print(f"\n[PIXAI] Top tags ({len(results.pixai)} total):")
            sorted_tags = sorted(results.pixai.items(), key=lambda x: x[1], reverse=True)
            for tag, conf in sorted_tags[:15]:
                print(f"  {conf:.2f}  {tag}")

        # Print JoyTag tags
        if results.joytag:
            print(f"\n[JOYTAG] Top tags ({len(results.joytag)} total):")
            sorted_tags = sorted(results.joytag.items(), key=lambda x: x[1], reverse=True)
            for tag, conf in sorted_tags[:15]:
                print(f"  {conf:.2f}  {tag}")

        # Print shot type
        if results.shot_type:
            print(f"\n[SHOT TYPE]:")
            for shot, conf in sorted(results.shot_type.items(), key=lambda x: x[1], reverse=True):
                print(f"  {conf:.2f}  {shot}")

        # Print places365 scenes
        if results.places365:
            print(f"\n[PLACES365] Scenes:")
            for scene, conf in sorted(results.places365.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"  {conf:.2f}  {scene}")

        # Print NudeNet detections
        if results.nudenet:
            detections = results.nudenet.get("detections", [])
            if detections:
                print(f"\n[NUDENET] Detections ({len(detections)}):")
                for det in detections[:10]:
                    # NudeNet uses "class" and "score" keys
                    label = det.get("class", det.get("label", "unknown"))
                    conf = det.get("score", det.get("confidence", 0))
                    print(f"  {conf:.2f}  {label}")

        # Print Pose detection
        if results.pose:
            print(f"\n[POSE]:")
            detected = results.pose.get("detected", False)
            model = results.pose.get("model", "unknown")
            print(f"  Detected: {detected}, Model: {model}")
            if detected:
                pose_type = results.pose.get("pose_type", "unknown")
                pose_conf = results.pose.get("pose_confidence", 0)
                print(f"  Pose: {pose_type} ({pose_conf:.0%})")

                # Show visible keypoints
                visible = results.pose.get("visible_keypoints", 0)
                total = results.pose.get("keypoint_count", 0)
                if total > 0:
                    print(f"  Keypoints: {visible}/{total} visible")

                # Show body part visibility
                parts = []
                if results.pose.get("shoulders_visible"):
                    parts.append("shoulders")
                if results.pose.get("hips_visible"):
                    parts.append("hips")
                if results.pose.get("knees_visible"):
                    parts.append("knees")
                if results.pose.get("ankles_visible"):
                    parts.append("ankles")
                if parts:
                    print(f"  Visible: {', '.join(parts)}")

                # Show arm position
                arms_pos = results.pose.get("arms_position", "neutral")
                if results.pose.get("arms_raised"):
                    print(f"  Arms: raised")
                elif arms_pos != "neutral":
                    print(f"  Arms: {arms_pos}")

                if results.pose.get("hand_near_face"):
                    print(f"  Hand near face: Yes")

        # Print Photography analysis
        if results.photography:
            print(f"\n[PHOTOGRAPHY]:")
            for key, value in results.photography.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.2f}")
                else:
                    print(f"  {key}: {value}")

        # Print IQA scores
        if results.iqa:
            print(f"\n[IQA] Image Quality:")
            for key, value in results.iqa.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.2f}")
                else:
                    print(f"  {key}: {value}")

        # Print Composition analysis
        if results.composition:
            print(f"\n[COMPOSITION]:")
            for key, value in results.composition.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.2f}")
                elif isinstance(value, bool):
                    print(f"  {key}: {'Yes' if value else 'No'}")
                elif isinstance(value, dict):
                    print(f"  {key}:")
                    for k, v in value.items():
                        if isinstance(v, float):
                            print(f"    {k}: {v:.2f}")
                        else:
                            print(f"    {k}: {v}")
                else:
                    print(f"  {key}: {value}")

        # Print Saliency analysis
        if results.saliency:
            print(f"\n[SALIENCY]:")
            for key, value in results.saliency.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.2f}")
                elif isinstance(value, (list, tuple)) and len(value) == 2:
                    print(f"  {key}: ({value[0]:.2f}, {value[1]:.2f})")
                else:
                    print(f"  {key}: {value}")

        # Print CLIP Camera/Scene analysis
        if results.clip_camera:
            print(f"\n[CLIP CAMERA]:")
            for key, value in results.clip_camera.items():
                if isinstance(value, dict):
                    print(f"  {key}:")
                    sorted_items = sorted(value.items(), key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0, reverse=True)
                    for k, v in sorted_items[:5]:
                        if isinstance(v, float):
                            print(f"    {v:.2f}  {k}")
                        else:
                            print(f"    {k}: {v}")
                elif isinstance(value, float):
                    print(f"  {key}: {value:.2f}")
                else:
                    print(f"  {key}: {value}")

        # Print Lighting analysis
        if results.lighting:
            print(f"\n[LIGHTING]:")
            for key, value in results.lighting.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.2f}")
                elif isinstance(value, bool):
                    print(f"  {key}: {'Yes' if value else 'No'}")
                elif isinstance(value, dict):
                    print(f"  {key}:")
                    for k, v in value.items():
                        if isinstance(v, float):
                            print(f"    {k}: {v:.2f}")
                        else:
                            print(f"    {k}: {v}")
                else:
                    print(f"  {key}: {value}")

        # Print Fashion Color analysis
        if results.fashion_color:
            print(f"\n[FASHION COLOR] Dominant colors:")
            for i, color in enumerate(results.fashion_color[:5]):
                name = color.get("name", "unknown")
                hex_code = color.get("hex", "#000000")
                pct = color.get("percentage", 0)
                print(f"  {pct:.1f}%  {name} ({hex_code})")

        # Print DeepFashion analysis
        if results.deepfashion:
            print(f"\n[DEEPFASHION]:")
            for key, value in results.deepfashion.items():
                if isinstance(value, dict):
                    print(f"  {key}:")
                    sorted_items = sorted(value.items(), key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0, reverse=True)
                    for k, v in sorted_items[:5]:
                        if isinstance(v, float):
                            print(f"    {v:.2f}  {k}")
                        else:
                            print(f"    {k}: {v}")
                elif isinstance(value, float):
                    print(f"  {key}: {value:.2f}")
                else:
                    print(f"  {key}: {value}")

        # Print Wildlife detection
        if results.wildlife:
            print(f"\n[WILDLIFE]:")
            animals = results.wildlife.get("animals", {})
            if animals:
                print(f"  Animals detected:")
                for animal, conf in sorted(animals.items(), key=lambda x: x[1], reverse=True)[:5]:
                    print(f"    {conf:.2f}  {animal}")
            else:
                print(f"  No animals detected")

        # Print Landmarks detection
        if results.landmarks:
            print(f"\n[LANDMARKS]:")
            detected = results.landmarks.get("detected", False)
            if detected:
                best = results.landmarks.get("best_landmark", "unknown")
                score = results.landmarks.get("best_score", 0)
                is_famous = results.landmarks.get("is_famous", False)
                print(f"  Detected: {best} ({score:.2f})")
                print(f"  Famous landmark: {'Yes' if is_famous else 'No'}")
            else:
                print(f"  No landmarks detected")

        # Print Sky & Weather analysis
        if results.sky_weather:
            print(f"\n[SKY & WEATHER]:")
            sky_visible = results.sky_weather.get("sky_visible", False)
            if sky_visible:
                sky = results.sky_weather.get("sky_condition", "unknown")
                weather = results.sky_weather.get("weather", "unknown")
                time_of_day = results.sky_weather.get("time_of_day", "unknown")
                print(f"  Sky: {sky}")
                print(f"  Weather: {weather}")
                print(f"  Time of day: {time_of_day}")
                if results.sky_weather.get("is_golden_hour"):
                    print(f"  Golden hour: Yes")
                if results.sky_weather.get("is_dramatic"):
                    print(f"  Dramatic lighting: Yes")
            else:
                print(f"  Sky not visible")

        print("\n" + "=" * 60 + "\n")

    def _run_tagger(self, tagger_name: str, image: Any, threshold: float = 0.65) -> Dict[str, Any]:
        """
        Run a specific tagger.

        Args:
            tagger_name: Name of tagger to run
            image: Image tensor
            threshold: Minimum confidence threshold for tags

        Returns:
            Tagger output dict
        """
        tagger_func = self.TAGGER_FUNCTIONS.get(tagger_name)

        if tagger_func is None:
            # Tagger not implemented yet
            print(f"[TaggerRunner] {tagger_name} tagger not implemented")
            return {}

        try:
            start = time.time()
            # Pass threshold to taggers that support it (wd14, pixai, joytag)
            if tagger_name in ["wd14", "pixai", "joytag"]:
                print(f"[TaggerRunner] Running {tagger_name} with threshold={threshold:.2f}")
                result = tagger_func(image, threshold=threshold)
            else:
                result = tagger_func(image)
            elapsed = time.time() - start
            print(f"[TaggerRunner] {tagger_name} completed in {elapsed:.2f}s")
            return result
        except Exception as e:
            print(f"[TaggerRunner] Error running {tagger_name}: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def _run_analyzer(self, analyzer_name: str, image: Any) -> Dict[str, Any]:
        """
        Run a specific analyzer.

        Args:
            analyzer_name: Name of analyzer to run
            image: Image tensor

        Returns:
            Analyzer output dict
        """
        analyzer_func = self.ANALYZER_FUNCTIONS.get(analyzer_name)

        if analyzer_func is None:
            # Analyzer not implemented yet
            print(f"[TaggerRunner] {analyzer_name} analyzer not implemented")
            return {}

        try:
            start = time.time()
            result = analyzer_func(image)
            elapsed = time.time() - start
            print(f"[TaggerRunner] {analyzer_name} completed in {elapsed:.2f}s")
            return result
        except Exception as e:
            print(f"[TaggerRunner] Error running {analyzer_name}: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def run_minimal(self, image: Any, threshold: float = 0.65) -> TaggerResults:
        """
        Run minimal set of taggers for fast analysis.

        Only runs WD14 and shot_type for basic decisions.

        Args:
            image: Image tensor
            threshold: Minimum confidence threshold for tags

        Returns:
            TaggerResults with minimal outputs
        """
        return self.run_all(
            image,
            enabled_taggers=["wd14"],
            enabled_analyzers=["shot_type", "places365"],
            threshold=threshold,
        )

    def run_standard(self, image: Any, threshold: float = 0.65) -> TaggerResults:
        """
        Run standard set of taggers.

        Args:
            image: Image tensor
            threshold: Minimum confidence threshold for tags

        Returns:
            TaggerResults with standard outputs
        """
        return self.run_all(
            image,
            enabled_taggers=["wd14", "pixai", "nudenet"],
            enabled_analyzers=["shot_type", "places365", "photography", "iqa"],
            threshold=threshold,
        )

    def run_full(self, image: Any, threshold: float = 0.65) -> TaggerResults:
        """
        Run all available taggers and analyzers.

        Args:
            image: Image tensor
            threshold: Minimum confidence threshold for tags

        Returns:
            TaggerResults with all outputs
        """
        return self.run_all(image, threshold=threshold)

    def unload_all(self):
        """Unload all models from memory to free VRAM."""
        print("[TaggerRunner] Unloading all models...")

        # Unload taggers
        unload_funcs = [
            unload_wd14,
            unload_pixai,
            unload_joytag,
            unload_nudenet,
            unload_pose,
            # Analyzers
            unload_shot_type,
            unload_places365,
            unload_iqa,
            unload_saliency,
            unload_clip_camera,
            unload_deepfashion,
            unload_wildlife,
            unload_landmarks,
            unload_sky_weather,
        ]

        for unload_func in unload_funcs:
            try:
                unload_func()
            except Exception:
                pass

        print("[TaggerRunner] All models unloaded")


# Global singleton
_runner: Optional[TaggerRunner] = None


def get_runner() -> TaggerRunner:
    """Get the global TaggerRunner instance."""
    global _runner
    if _runner is None:
        _runner = TaggerRunner()
    return _runner
