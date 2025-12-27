"""BBox Analyzer - Extract geometric insights from Florence detection data.

Analyzes bounding boxes from Florence object detection and region captions
to derive precise composition, framing, and scene metrics.

No model inference required - pure geometric analysis of existing bbox data.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import math


class ShotType(Enum):
    """Shot types based on subject framing."""
    EXTREME_CLOSE_UP = "extreme close-up"  # Face fills frame, may be cropped
    CLOSE_UP = "close-up"                   # Head and shoulders
    MEDIUM_CLOSE_UP = "medium close-up"     # Chest up
    MEDIUM_SHOT = "medium shot"             # Waist up
    COWBOY_SHOT = "cowboy shot"             # Mid-thigh up
    MEDIUM_FULL = "medium full shot"        # Knees up
    FULL_SHOT = "full shot"                 # Full body with some headroom
    WIDE_SHOT = "wide shot"                 # Full body, significant environment
    EXTREME_WIDE = "extreme wide shot"      # Subject small in frame


class Headroom(Enum):
    """Space above subject's head."""
    CROPPED = "cropped"           # Head cut off
    TIGHT = "tight"               # Minimal space
    BALANCED = "balanced"         # Good headroom
    EXCESSIVE = "excessive"       # Too much space


class SubjectPosition(Enum):
    """Subject position in frame."""
    LEFT_THIRD = "left third"
    CENTER = "center"
    RIGHT_THIRD = "right third"
    TOP_THIRD = "top third"
    BOTTOM_THIRD = "bottom third"


@dataclass
class BBoxAnalysisResult:
    """Result of bbox geometric analysis."""
    # Shot type
    shot_type: str = "unknown"
    shot_type_confidence: float = 0.0
    body_coverage: float = 0.0  # % of frame occupied by subject

    # Subject position
    subject_center_x: float = 0.5  # 0-1, left to right
    subject_center_y: float = 0.5  # 0-1, top to bottom
    subject_position: str = "center"

    # Composition scores
    rule_of_thirds_score: float = 0.0  # 0-1, alignment to power points
    golden_ratio_score: float = 0.0    # 0-1, phi spiral alignment
    symmetry_score: float = 0.0        # 0-1, horizontal symmetry

    # Framing
    headroom: str = "balanced"
    headroom_ratio: float = 0.0  # Space above head / head height
    lead_room: str = "balanced"  # Space in direction of gaze

    # Scene metrics
    subject_count: int = 1
    object_count: int = 0
    object_density: float = 0.0  # Objects per frame area
    negative_space: float = 0.0  # % of frame without objects

    # Depth
    depth_layers: List[str] = field(default_factory=list)
    foreground_objects: List[str] = field(default_factory=list)
    background_objects: List[str] = field(default_factory=list)

    # Fashion-specific (if applicable)
    garment_coverage: float = 0.0
    visible_regions: List[str] = field(default_factory=list)

    # Raw data
    primary_subject_bbox: Optional[List[float]] = None
    face_bbox: Optional[List[float]] = None


# Golden ratio constant
PHI = 1.618033988749895

# Shot type thresholds based on face height ratio to image height
SHOT_TYPE_THRESHOLDS = [
    (0.70, ShotType.EXTREME_CLOSE_UP),  # Face > 70% of image height
    (0.45, ShotType.CLOSE_UP),           # Face 45-70%
    (0.30, ShotType.MEDIUM_CLOSE_UP),    # Face 30-45%
    (0.20, ShotType.MEDIUM_SHOT),        # Face 20-30%
    (0.15, ShotType.COWBOY_SHOT),        # Face 15-20%
    (0.10, ShotType.MEDIUM_FULL),        # Face 10-15%
    (0.05, ShotType.FULL_SHOT),          # Face 5-10%
    (0.02, ShotType.WIDE_SHOT),          # Face 2-5%
    (0.00, ShotType.EXTREME_WIDE),       # Face < 2%
]


class BBoxAnalyzer:
    """Analyze bounding boxes for geometric insights."""

    def __init__(self):
        # Subject-related labels (prioritized for main subject detection)
        self.subject_labels = {
            "woman", "man", "person", "human", "girl", "boy", "child",
            "model", "figure", "people", "crowd", "group",
        }

        # Face-related labels
        self.face_labels = {
            "face", "human face", "head", "portrait",
        }

        # Body-related labels
        self.body_labels = {
            "body", "human body", "full body", "torso",
        }

        # Clothing labels
        self.clothing_labels = {
            "dress", "fancy dress", "shirt", "pants", "skirt", "jacket",
            "coat", "suit", "blouse", "sweater", "top", "bottom",
            "miniskirt", "jeans", "shorts",
        }

        # Background/environment labels
        self.background_labels = {
            "background", "wall", "floor", "sky", "tree", "building",
            "room", "window", "door", "furniture", "table", "chair",
        }

    def analyze(
        self,
        florence_objects: Dict,
        florence_regions: Dict = None,
        image_width: int = 1,
        image_height: int = 1,
    ) -> BBoxAnalysisResult:
        """
        Analyze bbox data from Florence detection.

        Args:
            florence_objects: Dict from florence_objects metadata
                Format: {"label": {"count": N, "bboxes": [[x1,y1,x2,y2], ...]}}
            florence_regions: Dict from florence_region_captions (optional)
            image_width: Image width in pixels
            image_height: Image height in pixels

        Returns:
            BBoxAnalysisResult with all geometric metrics
        """
        result = BBoxAnalysisResult()

        if not florence_objects:
            return result

        # Extract all bboxes with labels
        all_bboxes = []
        for label, data in florence_objects.items():
            if isinstance(data, dict) and "bboxes" in data:
                for bbox in data["bboxes"]:
                    all_bboxes.append((label.lower(), bbox))

        if not all_bboxes:
            return result

        # Find primary subject (person/body)
        subject_bbox = self._find_primary_subject(all_bboxes)
        face_bbox = self._find_face(all_bboxes)

        if subject_bbox:
            result.primary_subject_bbox = subject_bbox
        if face_bbox:
            result.face_bbox = face_bbox

        # Analyze shot type
        if face_bbox:
            result.shot_type, result.shot_type_confidence = self._detect_shot_type(
                face_bbox, image_width, image_height
            )
        elif subject_bbox:
            # Estimate from body if no face
            result.shot_type, result.shot_type_confidence = self._detect_shot_type_from_body(
                subject_bbox, image_width, image_height
            )

        # Calculate body coverage
        if subject_bbox:
            result.body_coverage = self._calculate_coverage(
                subject_bbox, image_width, image_height
            )

        # Subject position
        primary_bbox = subject_bbox or face_bbox
        if primary_bbox:
            cx, cy = self._get_bbox_center(primary_bbox, image_width, image_height)
            result.subject_center_x = cx
            result.subject_center_y = cy
            result.subject_position = self._classify_position(cx, cy)

        # Composition scores
        if primary_bbox:
            result.rule_of_thirds_score = self._score_rule_of_thirds(
                primary_bbox, image_width, image_height
            )
            result.golden_ratio_score = self._score_golden_ratio(
                primary_bbox, image_width, image_height
            )
            result.symmetry_score = self._score_symmetry(
                primary_bbox, image_width, image_height
            )

        # Headroom analysis
        if face_bbox and subject_bbox:
            result.headroom, result.headroom_ratio = self._analyze_headroom(
                face_bbox, subject_bbox, image_height
            )
        elif face_bbox:
            result.headroom, result.headroom_ratio = self._analyze_headroom_face_only(
                face_bbox, image_height
            )

        # Scene metrics
        result.subject_count = self._count_subjects(all_bboxes)
        result.object_count = len(all_bboxes)
        result.object_density = len(all_bboxes) / (image_width * image_height) * 1000000
        result.negative_space = self._calculate_negative_space(
            all_bboxes, image_width, image_height
        )

        # Depth layers
        result.depth_layers, result.foreground_objects, result.background_objects = \
            self._analyze_depth_layers(all_bboxes, image_width, image_height)

        # Clothing coverage
        clothing_bboxes = [(l, b) for l, b in all_bboxes if l in self.clothing_labels]
        if clothing_bboxes and subject_bbox:
            result.garment_coverage = self._calculate_garment_coverage(
                clothing_bboxes, subject_bbox
            )
            result.visible_regions = [l for l, _ in clothing_bboxes]

        return result

    def _find_primary_subject(self, bboxes: List[Tuple[str, List]]) -> Optional[List]:
        """Find the primary subject (largest person/body bbox)."""
        subject_bboxes = []
        for label, bbox in bboxes:
            if any(s in label for s in self.subject_labels):
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                subject_bboxes.append((area, bbox))

        if subject_bboxes:
            subject_bboxes.sort(key=lambda x: x[0], reverse=True)
            return subject_bboxes[0][1]
        return None

    def _find_face(self, bboxes: List[Tuple[str, List]]) -> Optional[List]:
        """Find face bbox."""
        for label, bbox in bboxes:
            if any(f in label for f in self.face_labels):
                return bbox
        return None

    def _detect_shot_type(
        self,
        face_bbox: List,
        img_w: int,
        img_h: int,
    ) -> Tuple[str, float]:
        """Detect shot type from face bbox."""
        face_height = face_bbox[3] - face_bbox[1]
        ratio = face_height / img_h

        for threshold, shot_type in SHOT_TYPE_THRESHOLDS:
            if ratio >= threshold:
                # Calculate confidence based on how centered in the threshold range
                confidence = min(1.0, 0.7 + ratio * 0.3)
                return shot_type.value, confidence

        return ShotType.EXTREME_WIDE.value, 0.5

    def _detect_shot_type_from_body(
        self,
        body_bbox: List,
        img_w: int,
        img_h: int,
    ) -> Tuple[str, float]:
        """Estimate shot type from body bbox when face not available."""
        body_height = body_bbox[3] - body_bbox[1]
        body_coverage = body_height / img_h

        # Body-based thresholds (body typically 7-8x face height)
        if body_coverage > 0.95:
            return ShotType.CLOSE_UP.value, 0.6
        elif body_coverage > 0.85:
            return ShotType.MEDIUM_CLOSE_UP.value, 0.7
        elif body_coverage > 0.70:
            return ShotType.MEDIUM_SHOT.value, 0.75
        elif body_coverage > 0.55:
            return ShotType.COWBOY_SHOT.value, 0.8
        elif body_coverage > 0.40:
            return ShotType.MEDIUM_FULL.value, 0.75
        elif body_coverage > 0.25:
            return ShotType.FULL_SHOT.value, 0.7
        else:
            return ShotType.WIDE_SHOT.value, 0.6

    def _calculate_coverage(self, bbox: List, img_w: int, img_h: int) -> float:
        """Calculate what % of frame the bbox occupies."""
        bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        img_area = img_w * img_h
        return bbox_area / img_area if img_area > 0 else 0

    def _get_bbox_center(
        self,
        bbox: List,
        img_w: int,
        img_h: int,
    ) -> Tuple[float, float]:
        """Get normalized center point of bbox (0-1)."""
        cx = ((bbox[0] + bbox[2]) / 2) / img_w
        cy = ((bbox[1] + bbox[3]) / 2) / img_h
        return cx, cy

    def _classify_position(self, cx: float, cy: float) -> str:
        """Classify subject position in frame."""
        positions = []

        # Horizontal position
        if cx < 0.33:
            positions.append("left third")
        elif cx > 0.67:
            positions.append("right third")
        else:
            positions.append("center")

        # Vertical position (less commonly used but useful)
        if cy < 0.33:
            positions.append("upper")
        elif cy > 0.67:
            positions.append("lower")

        return ", ".join(positions) if len(positions) > 1 else positions[0]

    def _score_rule_of_thirds(
        self,
        bbox: List,
        img_w: int,
        img_h: int,
    ) -> float:
        """Score alignment to rule of thirds power points."""
        cx, cy = self._get_bbox_center(bbox, img_w, img_h)

        # Rule of thirds power points
        thirds_x = [1/3, 2/3]
        thirds_y = [1/3, 2/3]

        # Find closest power point
        min_dist = float('inf')
        for tx in thirds_x:
            for ty in thirds_y:
                dist = math.sqrt((cx - tx)**2 + (cy - ty)**2)
                min_dist = min(min_dist, dist)

        # Max possible distance from any power point is ~0.47
        # Score inversely proportional to distance
        score = max(0, 1 - (min_dist / 0.47))
        return round(score, 3)

    def _score_golden_ratio(
        self,
        bbox: List,
        img_w: int,
        img_h: int,
    ) -> float:
        """Score alignment to golden ratio points."""
        cx, cy = self._get_bbox_center(bbox, img_w, img_h)

        # Golden ratio points (1/phi and 1-1/phi)
        phi_inv = 1 / PHI  # ~0.618
        golden_x = [1 - phi_inv, phi_inv]  # ~0.382, ~0.618
        golden_y = [1 - phi_inv, phi_inv]

        min_dist = float('inf')
        for gx in golden_x:
            for gy in golden_y:
                dist = math.sqrt((cx - gx)**2 + (cy - gy)**2)
                min_dist = min(min_dist, dist)

        score = max(0, 1 - (min_dist / 0.5))
        return round(score, 3)

    def _score_symmetry(
        self,
        bbox: List,
        img_w: int,
        img_h: int,
    ) -> float:
        """Score horizontal symmetry (subject centered)."""
        cx, _ = self._get_bbox_center(bbox, img_w, img_h)

        # Distance from center (0.5)
        dist_from_center = abs(cx - 0.5)

        # Perfect center = 1.0, edges = 0.0
        score = max(0, 1 - (dist_from_center * 2))
        return round(score, 3)

    def _analyze_headroom(
        self,
        face_bbox: List,
        body_bbox: List,
        img_h: int,
    ) -> Tuple[str, float]:
        """Analyze headroom from face and body bbox."""
        face_top = face_bbox[1]
        face_height = face_bbox[3] - face_bbox[1]

        # Headroom is space above face
        headroom_pixels = face_top
        headroom_ratio = headroom_pixels / face_height if face_height > 0 else 0

        if face_top < 0:
            return Headroom.CROPPED.value, headroom_ratio
        elif headroom_ratio < 0.3:
            return Headroom.TIGHT.value, headroom_ratio
        elif headroom_ratio < 1.5:
            return Headroom.BALANCED.value, headroom_ratio
        else:
            return Headroom.EXCESSIVE.value, headroom_ratio

    def _analyze_headroom_face_only(
        self,
        face_bbox: List,
        img_h: int,
    ) -> Tuple[str, float]:
        """Analyze headroom from face bbox only."""
        face_top = face_bbox[1]
        face_height = face_bbox[3] - face_bbox[1]

        headroom_ratio = face_top / face_height if face_height > 0 else 0

        if face_top < 0:
            return Headroom.CROPPED.value, headroom_ratio
        elif headroom_ratio < 0.3:
            return Headroom.TIGHT.value, headroom_ratio
        elif headroom_ratio < 1.5:
            return Headroom.BALANCED.value, headroom_ratio
        else:
            return Headroom.EXCESSIVE.value, headroom_ratio

    def _count_subjects(self, bboxes: List[Tuple[str, List]]) -> int:
        """Count number of subjects (people/faces)."""
        count = 0
        for label, _ in bboxes:
            if any(f in label for f in self.face_labels):
                count += 1
            elif any(s in label for s in ["woman", "man", "person", "girl", "boy"]):
                count += 1
        return max(1, count)  # At least 1 if we have any detection

    def _calculate_negative_space(
        self,
        bboxes: List[Tuple[str, List]],
        img_w: int,
        img_h: int,
    ) -> float:
        """Calculate percentage of frame without objects."""
        if not bboxes:
            return 1.0

        # Simple approximation: sum of bbox areas / image area
        # (doesn't account for overlap, but good enough)
        total_bbox_area = sum(
            (b[2] - b[0]) * (b[3] - b[1])
            for _, b in bboxes
        )
        img_area = img_w * img_h

        coverage = total_bbox_area / img_area if img_area > 0 else 0
        return round(max(0, 1 - coverage), 3)

    def _analyze_depth_layers(
        self,
        bboxes: List[Tuple[str, List]],
        img_w: int,
        img_h: int,
    ) -> Tuple[List[str], List[str], List[str]]:
        """Analyze depth layers based on bbox size and position."""
        layers = set()
        foreground = []
        background = []

        for label, bbox in bboxes:
            coverage = self._calculate_coverage(bbox, img_w, img_h)

            # Large objects = foreground, small = background
            if coverage > 0.3:
                foreground.append(label)
                layers.add("foreground")
            elif coverage > 0.1:
                layers.add("midground")
            else:
                background.append(label)
                layers.add("background")

            # Subject always counts as main layer
            if any(s in label for s in self.subject_labels):
                layers.add("subject")

        return sorted(layers), foreground, background

    def _calculate_garment_coverage(
        self,
        clothing_bboxes: List[Tuple[str, List]],
        subject_bbox: List,
    ) -> float:
        """Calculate how much of the subject is covered by detected garments."""
        subject_area = (subject_bbox[2] - subject_bbox[0]) * (subject_bbox[3] - subject_bbox[1])
        if subject_area <= 0:
            return 0

        clothing_area = sum(
            (b[2] - b[0]) * (b[3] - b[1])
            for _, b in clothing_bboxes
        )

        return round(min(1.0, clothing_area / subject_area), 3)

    def to_dict(self, result: BBoxAnalysisResult) -> Dict:
        """Convert result to JSON-serializable dict."""
        return {
            "shot_type": result.shot_type,
            "shot_type_confidence": round(result.shot_type_confidence, 3),
            "body_coverage": round(result.body_coverage, 3),
            "subject_position": {
                "x": round(result.subject_center_x, 3),
                "y": round(result.subject_center_y, 3),
                "description": result.subject_position,
            },
            "composition": {
                "rule_of_thirds": result.rule_of_thirds_score,
                "golden_ratio": result.golden_ratio_score,
                "symmetry": result.symmetry_score,
            },
            "framing": {
                "headroom": result.headroom,
                "headroom_ratio": round(result.headroom_ratio, 3),
            },
            "scene": {
                "subject_count": result.subject_count,
                "object_count": result.object_count,
                "negative_space": result.negative_space,
                "depth_layers": result.depth_layers,
            },
            "fashion": {
                "garment_coverage": result.garment_coverage,
                "visible_regions": result.visible_regions,
            } if result.visible_regions else None,
        }
