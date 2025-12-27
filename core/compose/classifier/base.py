"""Base data structures for token classification.

This module defines the classification result structure and
the classified image container.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime

from .categories import (
    CanonicalCategory,
    SubjectDetailType,
    CATEGORY_ORDER,
    SUBJECT_DETAIL_ORDER,
)
from ..tokenizer.base import ImageToken


@dataclass
class TokenClassification:
    """Classification result for a single token.

    A token may belong to multiple categories (e.g., "looking at viewer"
    is both Action/Pose and Subject Details/eyes). The primary category
    is used for assembly ordering.

    Attributes:
        token: The original ImageToken being classified
        primary_category: Main category assignment
        secondary_categories: Additional category assignments
        subcategory: For SUBJECT_DETAILS, the specific subcategory
        confidence: Classification confidence (0.0-1.0)
        classifier_layer: Which layer made the classification (1-5)
        metadata: Additional classification context
    """
    token: ImageToken
    primary_category: CanonicalCategory
    secondary_categories: List[CanonicalCategory] = field(default_factory=list)
    subcategory: Optional[SubjectDetailType] = None
    confidence: float = 1.0
    classifier_layer: int = 0  # 1=deterministic, 2=source, 3=dictionary, 4=embedding, 5=uncategorized
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def all_categories(self) -> List[CanonicalCategory]:
        """Get all categories (primary + secondary)."""
        return [self.primary_category] + self.secondary_categories

    @property
    def is_subject_detail(self) -> bool:
        """Check if this is a subject detail token."""
        return self.primary_category == CanonicalCategory.SUBJECT_DETAILS

    @property
    def is_filtered(self) -> bool:
        """Check if this token was filtered (META pattern)."""
        return self.primary_category == CanonicalCategory.FILTERED_META

    @property
    def is_uncategorized(self) -> bool:
        """Check if this token couldn't be categorized."""
        return self.primary_category == CanonicalCategory.UNCATEGORIZED

    def __repr__(self):
        cat = self.primary_category.value
        if self.subcategory:
            cat = f"{cat}/{self.subcategory.value}"
        return f"TokenClassification({self.token.text!r} → {cat})"


@dataclass
class ClassifiedImage:
    """Collection of classified tokens for a single image.

    Organizes tokens by category for efficient prompt assembly.
    """
    # Tokens organized by category
    categories: Dict[CanonicalCategory, List[TokenClassification]] = field(default_factory=dict)

    # Subject details organized by subcategory
    subject_details: Dict[SubjectDetailType, List[TokenClassification]] = field(default_factory=dict)

    # All classifications (for iteration)
    all_classifications: List[TokenClassification] = field(default_factory=list)

    # Filtered tokens (META patterns)
    filtered: List[TokenClassification] = field(default_factory=list)

    # Metadata
    source_file: Optional[str] = None
    extraction_timestamp: Optional[datetime] = None
    total_tokens_input: int = 0
    image_info: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize category dictionaries."""
        # Ensure all categories have empty lists
        for cat in CanonicalCategory:
            if cat not in self.categories:
                self.categories[cat] = []

        # Ensure all subcategories have empty lists
        for subcat in SubjectDetailType:
            if subcat not in self.subject_details:
                self.subject_details[subcat] = []

    def add(self, classification: TokenClassification) -> None:
        """Add a classification to the appropriate category."""
        self.all_classifications.append(classification)

        # Handle filtered tokens
        if classification.is_filtered:
            self.filtered.append(classification)
            return

        # Add to main category
        cat = classification.primary_category
        if cat not in self.categories:
            self.categories[cat] = []
        self.categories[cat].append(classification)

        # Add to subcategory if subject detail
        if classification.is_subject_detail and classification.subcategory:
            subcat = classification.subcategory
            if subcat not in self.subject_details:
                self.subject_details[subcat] = []
            self.subject_details[subcat].append(classification)

    def get_category(self, category: CanonicalCategory) -> List[TokenClassification]:
        """Get all classifications for a category."""
        return self.categories.get(category, [])

    def get_subject_detail(self, subcategory: SubjectDetailType) -> List[TokenClassification]:
        """Get all classifications for a subject detail subcategory."""
        return self.subject_details.get(subcategory, [])

    def get_tokens_by_category(self, category: CanonicalCategory) -> List[ImageToken]:
        """Get just the tokens (not full classifications) for a category."""
        return [c.token for c in self.get_category(category)]

    def get_texts_by_category(self, category: CanonicalCategory) -> List[str]:
        """Get just the text values for a category."""
        return [c.token.text for c in self.get_category(category)]

    def iter_by_order(self):
        """Iterate categories in canonical output order."""
        for cat in CATEGORY_ORDER:
            classifications = self.get_category(cat)
            if classifications:
                yield cat, classifications

    def iter_subject_details_by_order(self):
        """Iterate subject details in canonical order."""
        for subcat in SUBJECT_DETAIL_ORDER:
            classifications = self.get_subject_detail(subcat)
            if classifications:
                yield subcat, classifications

    @property
    def has_subject(self) -> bool:
        """Check if any subject tokens exist."""
        return bool(self.get_category(CanonicalCategory.SUBJECT))

    @property
    def subject_count(self) -> int:
        """Get estimated subject count from subject tokens."""
        subjects = self.get_texts_by_category(CanonicalCategory.SUBJECT)
        # Look for count indicators
        for text in subjects:
            if "2girl" in text or "2boy" in text or "couple" in text:
                return 2
            if "3girl" in text or "3boy" in text or "group" in text:
                return 3
            if "multiple" in text or "crowd" in text:
                return 4
        return 1  # Default to solo

    @property
    def subject_gender(self) -> Optional[str]:
        """Infer subject gender from subject tokens."""
        subjects = self.get_texts_by_category(CanonicalCategory.SUBJECT)
        for text in subjects:
            if "girl" in text or "woman" in text or "female" in text or "lady" in text:
                return "female"
            if "boy" in text or "man" in text or "male" in text or "gentleman" in text:
                return "male"
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for JSON serialization."""
        result = {
            "total_tokens": len(self.all_classifications),
            "filtered_count": len(self.filtered),
            "categories": {},
        }

        for cat in CATEGORY_ORDER:
            classifications = self.get_category(cat)
            if classifications:
                result["categories"][cat.value] = [
                    {
                        "text": c.token.text,
                        "confidence": c.token.confidence,
                        "source": c.token.source.value,
                        "subcategory": c.subcategory.value if c.subcategory else None,
                    }
                    for c in classifications
                ]

        if self.filtered:
            result["filtered"] = [c.token.text for c in self.filtered]

        return result

    def __len__(self):
        return len(self.all_classifications)

    def __repr__(self):
        counts = {cat.value: len(cls) for cat, cls in self.categories.items() if cls}
        return f"ClassifiedImage({counts})"
