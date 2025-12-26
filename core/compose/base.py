"""Base classes for prompt composition."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any
import re

import yaml


class ComposeMode(Enum):
    """Prompt composition mode."""
    STANDARD = "standard"
    ENHANCE_AI = "enhance_ai"  # Formerly LLM


# Philosophical/meta phrases to filter out (VLM/Florence artifacts)
# These are common preambles and commentary that don't belong in generation prompts
# Patterns that trigger ENTIRE SENTENCE removal if found ANYWHERE in sentence
FILTER_PHRASES = [
    # Meta descriptions about the image
    r'the image (?:captures?|shows?|presents?|depicts?|features?|conveys?|is)',
    r'this (?:image|photo|photograph|picture) (?:captures?|shows?|presents?|depicts?)',
    r'in this (?:image|photo|photograph|picture)',
    # Abstract interpretations
    r'conveying a sense of',
    r'creating (?:a|an) (?:sense|feeling|atmosphere) of',
    r'adding (?:a touch|an element) of',
    r'suggesting (?:a|an) (?:sense|feeling)',
    r'evoking (?:a|an)',
    r'implying (?:a|an)',
    # Meta commentary - these should trigger removal anywhere in sentence
    r'the overall (?:mood|tone|atmosphere|feeling|composition|image)',
    r'the (?:viewer|audience|observer) (?:is|can|may)',
    r'which (?:adds?|creates?|gives?|lends?|helps?|keeps?)',
    r'(?:seems?|appears?) to (?:be|suggest|indicate)',
    r'(?:enhancing|complementing|accentuating|highlighting) the',
    r'indicative of',
    r'reminiscent of',
    r'possibly (?:indicating|suggesting)',
    r'perhaps (?:indicating|suggesting)',
    # Filler/padding phrases
    r'it is (?:worth|important to) (?:noting|mentioning)',
    r'one can (?:see|observe|notice)',
    r'as (?:seen|shown|depicted) in',
    # Quality/technical meta commentary
    r'high[- ]?quality,? (?:professional|studio)',
    r'well[- ]?(?:lit|composed|balanced)',
    r'no visible (?:grain|noise|distraction)',
    r'professional (?:lighting|photograph|feel)',
    # Mood/atmosphere meta
    r'sense of (?:tension|contrast|depth|calm|control|balance|intimacy)',
    r'adds to the (?:sense|mood|atmosphere|overall)',
    r'keeps the focus on',
    r'helping to (?:keep|create|add)',
    # Both/and constructions that are often filler
    r'in a manner that is both',
    r'which is both',
    r'that is both \w+ and \w+',
]


@dataclass
class ComposeConfig:
    """Configuration for prompt composition."""
    min_confidence: float = 0.1
    caption_source: str = "Auto"  # Auto, Synthesis (VLM), Analysis (Florence), Tags Only
    temperature: float = 0.1


@dataclass
class TagStats:
    """Statistics for a single tag."""
    tag: str
    confidence: float
    source: str  # wd14, joytag, pixai, florence
    category: Optional[str] = None
    status: str = "pending"  # pending, included, excluded, skipped
    reason: str = ""  # Why it was excluded/skipped


@dataclass
class SourceStats:
    """Statistics for a tagging source."""
    name: str
    total_tags: int = 0
    included_tags: int = 0
    excluded_tags: int = 0
    skipped_tags: int = 0
    top_included: List[str] = field(default_factory=list)
    top_excluded: List[str] = field(default_factory=list)


@dataclass
class ComposeAnalytics:
    """Comprehensive analytics for prompt composition."""
    # Per-tag tracking
    all_tags: List[TagStats] = field(default_factory=list)

    # Per-source statistics
    sources: Dict[str, SourceStats] = field(default_factory=dict)

    # Category usage
    categories_used: Dict[str, List[str]] = field(default_factory=dict)
    categories_empty: List[str] = field(default_factory=list)

    # Summary stats
    total_input_tags: int = 0
    total_included: int = 0
    total_excluded: int = 0
    total_skipped: int = 0

    # Florence usage
    florence_used: bool = False
    florence_sections_used: List[str] = field(default_factory=list)

    def add_tag(self, tag: str, confidence: float, source: str,
                category: Optional[str] = None, status: str = "pending", reason: str = ""):
        """Add a tag to tracking."""
        self.all_tags.append(TagStats(
            tag=tag, confidence=confidence, source=source,
            category=category, status=status, reason=reason
        ))

    def finalize(self):
        """Calculate final statistics."""
        self.total_input_tags = len(self.all_tags)
        self.total_included = sum(1 for t in self.all_tags if t.status == "included")
        self.total_excluded = sum(1 for t in self.all_tags if t.status == "excluded")
        self.total_skipped = sum(1 for t in self.all_tags if t.status == "skipped")

        # Per-source stats
        source_tags = {}
        for tag in self.all_tags:
            if tag.source not in source_tags:
                source_tags[tag.source] = []
            source_tags[tag.source].append(tag)

        for source, tags in source_tags.items():
            included = [t for t in tags if t.status == "included"]
            excluded = [t for t in tags if t.status == "excluded"]
            skipped = [t for t in tags if t.status == "skipped"]

            self.sources[source] = SourceStats(
                name=source,
                total_tags=len(tags),
                included_tags=len(included),
                excluded_tags=len(excluded),
                skipped_tags=len(skipped),
                top_included=[t.tag for t in sorted(included, key=lambda x: x.confidence, reverse=True)[:5]],
                top_excluded=[t.tag for t in sorted(excluded, key=lambda x: x.confidence, reverse=True)[:5]],
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "summary": {
                "total_input": self.total_input_tags,
                "included": self.total_included,
                "excluded": self.total_excluded,
                "skipped": self.total_skipped,
                "inclusion_rate": f"{self.total_included / max(1, self.total_input_tags) * 100:.1f}%",
            },
            "by_source": {
                name: {
                    "total": s.total_tags,
                    "included": s.included_tags,
                    "excluded": s.excluded_tags,
                    "skipped": s.skipped_tags,
                    "top_included": s.top_included,
                    "top_excluded": s.top_excluded,
                } for name, s in self.sources.items()
            },
            "categories": {
                "used": {k: v for k, v in self.categories_used.items() if v},
                "empty": self.categories_empty,
            },
            "florence": {
                "used": self.florence_used,
                "sections": self.florence_sections_used,
            }
        }


@dataclass
class ComposeResult:
    """Result from prompt composition."""
    prompt: str
    mode: ComposeMode
    style: str
    word_count: int = 0
    categories_used: List[str] = field(default_factory=list)
    analytics: Optional[ComposeAnalytics] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.word_count = len(self.prompt.split())

    def get_analytics_summary(self) -> str:
        """Get human-readable analytics summary."""
        if not self.analytics:
            return "No analytics available"

        lines = [
            f"Tags: {self.analytics.total_included}/{self.analytics.total_input_tags} included",
            f"Sources: {', '.join(self.analytics.sources.keys())}",
            f"Categories: {len([c for c in self.analytics.categories_used.values() if c])} used",
        ]
        if self.analytics.florence_used:
            lines.append(f"Florence: {', '.join(self.analytics.florence_sections_used)}")

        return " | ".join(lines)


class StyleConfig:
    """Loaded style configuration from YAML."""

    _cache: Dict[str, 'StyleConfig'] = {}
    _styles_data: Optional[Dict] = None

    def __init__(self, style_name: str):
        self.style_name = style_name
        self._load_styles()

        if style_name not in self._styles_data.get('styles', {}):
            raise ValueError(f"Unknown style: {style_name}. Available: {list(self._styles_data.get('styles', {}).keys())}")

        self._style = self._styles_data['styles'][style_name]
        self._defaults = self._styles_data.get('defaults', {})
        self._exclude_tags = set(self._styles_data.get('exclude_tags', []))

    @classmethod
    def _load_styles(cls):
        """Load styles YAML file."""
        if cls._styles_data is not None:
            return

        config_path = Path(__file__).parent.parent / "config" / "compose_styles.yaml"
        with open(config_path, 'r', encoding='utf-8') as f:
            cls._styles_data = yaml.safe_load(f)

    @classmethod
    def get(cls, style_name: str) -> 'StyleConfig':
        """Get cached style config."""
        if style_name not in cls._cache:
            cls._cache[style_name] = cls(style_name)
        return cls._cache[style_name]

    @classmethod
    def available_styles(cls) -> List[str]:
        """Get list of available style names."""
        cls._load_styles()
        return list(cls._styles_data.get('styles', {}).keys())

    @property
    def name(self) -> str:
        return self._style.get('name', self.style_name)

    @property
    def description(self) -> str:
        return self._style.get('description', '')

    @property
    def syntax(self) -> str:
        return self._style.get('syntax', 'sentences')

    @property
    def supports_negative(self) -> bool:
        return self._style.get('supports_negative', False)

    @property
    def supports_weights(self) -> bool:
        return self._style.get('supports_weights', False)

    @property
    def max_tokens(self) -> int:
        return self._style.get('max_tokens', 1024)

    @property
    def structure(self) -> List[str]:
        return self._style.get('structure', [])

    @property
    def categories(self) -> Dict[str, Dict]:
        return self._style.get('categories', {})

    @property
    def templates(self) -> Dict[str, str]:
        return self._style.get('templates', {})

    @property
    def exclude_tags(self) -> set:
        return self._exclude_tags

    @property
    def llm_system_prompt(self) -> str:
        return self._style.get('llm_system_prompt', '')

    @property
    def llm_user_prompt(self) -> str:
        return self._style.get('llm_user_prompt', '')

    def get_default(self, key: str, fallback: Any = None) -> Any:
        return self._defaults.get(key, fallback)


class BaseComposeGenerator(ABC):
    """Base class for prompt composition generators."""

    def __init__(self, style: str = "z-image", config: Optional[ComposeConfig] = None,
                 caption_source: str = "Auto"):
        self.style_config = StyleConfig.get(style)
        self.caption_source = caption_source
        self.config = config or ComposeConfig(
            min_confidence=self.style_config.get_default('min_confidence', 0.4),
            caption_source=caption_source,
        )
        self._compile_patterns()
        # Compile filter patterns for philosophical content removal
        self._filter_patterns = [re.compile(p, re.IGNORECASE) for p in FILTER_PHRASES]

    def filter_philosophical(self, text: str) -> str:
        """Remove philosophical/meta sentences from caption text.

        Filters out VLM/Florence artifacts like 'The image captures...',
        'conveying a sense of...', etc. that don't belong in generation prompts.
        """
        if not text:
            return text

        sentences = re.split(r'(?<=[.!?])\s+', text)
        filtered = []

        for sentence in sentences:
            if not sentence.strip():
                continue

            should_filter = False
            for pattern in self._filter_patterns:
                if pattern.search(sentence):
                    should_filter = True
                    break

            if not should_filter:
                filtered.append(sentence.strip())

        return ' '.join(filtered)

    def _compile_patterns(self):
        """Compile regex patterns for tag classification."""
        self._patterns = {}
        for category, info in self.style_config.categories.items():
            keywords = info.get('keywords', [])
            if keywords:
                # Build regex pattern from keywords with word boundaries
                # This prevents "boy" from matching "cowboy"
                pattern = '|'.join(r'\b' + re.escape(kw) + r'\b' for kw in keywords)
                self._patterns[category] = {
                    'regex': re.compile(pattern, re.IGNORECASE),
                    'priority': info.get('priority', 99),
                    'max_items': info.get('max_items', 3),
                }

    def extract_tags_with_tracking(self, metadata: Dict, analytics: ComposeAnalytics) -> Dict[str, float]:
        """Extract and merge tags from all tagging sources with full tracking."""
        tags = {}
        tag_sources = {}  # Track which source each tag came from
        exclude = self.style_config.exclude_tags

        # Sources to check
        sources = ['wd14', 'pixai', 'joytag']

        for source in sources:
            source_data = metadata.get(source, {})

            # Handle both dict formats: {"tag": conf} or {"tags": [...], "attributes": {...}}
            if isinstance(source_data, dict):
                if 'tags' in source_data:
                    # New format: {"tags": ["tag1", "tag2"], "attributes": {...}}
                    tag_list = source_data.get('tags', [])
                    for tag in tag_list:
                        tag_clean = self._clean_tag(tag)
                        conf = 0.5  # Default confidence for new format

                        if not tag_clean:
                            continue

                        # Track exclusion
                        if tag_clean.lower() in exclude:
                            analytics.add_tag(tag_clean, conf, source, status="excluded", reason="blacklisted")
                            continue

                        # Track inclusion (keep highest confidence)
                        if tag_clean not in tags:
                            tags[tag_clean] = conf
                            tag_sources[tag_clean] = source
                            analytics.add_tag(tag_clean, conf, source, status="pending")
                else:
                    # Old format: {"tag": confidence, ...}
                    for tag, conf in source_data.items():
                        if not isinstance(conf, (int, float)):
                            continue

                        tag_clean = self._clean_tag(tag)
                        if not tag_clean:
                            continue

                        # Track low confidence
                        if conf < self.config.min_confidence:
                            analytics.add_tag(tag_clean, conf, source, status="skipped", reason=f"low confidence ({conf:.2f})")
                            continue

                        # Track exclusion
                        if tag_clean.lower() in exclude:
                            analytics.add_tag(tag_clean, conf, source, status="excluded", reason="blacklisted")
                            continue

                        # Track inclusion (keep highest confidence)
                        if tag_clean not in tags or conf > tags[tag_clean]:
                            tags[tag_clean] = conf
                            tag_sources[tag_clean] = source
                            analytics.add_tag(tag_clean, conf, source, status="pending")

        return tags, tag_sources

    def extract_tags(self, metadata: Dict) -> Dict[str, float]:
        """Extract and merge tags from all tagging sources (simple version)."""
        analytics = ComposeAnalytics()
        tags, _ = self.extract_tags_with_tracking(metadata, analytics)
        return tags

    def _clean_tag(self, tag: str) -> str:
        """Clean a tag for use."""
        tag = tag.replace('\\(', '(').replace('\\)', ')')
        tag = tag.replace('_', ' ')
        tag = re.sub(r'^\d+', '', tag)  # Remove leading numbers
        tag = re.sub(r'\s+', ' ', tag).strip()
        return tag

    def classify_tags(self, tags: Dict[str, float]) -> Dict[str, List[str]]:
        """Classify tags into categories (no limits)."""
        classified = {cat: [] for cat in self._patterns}
        used_tags = set()

        # Sort tags by confidence
        sorted_tags = sorted(tags.items(), key=lambda x: x[1], reverse=True)

        for tag, conf in sorted_tags:
            tag_lower = tag.lower()

            # Find matching category
            for category, info in self._patterns.items():
                if info['regex'].search(tag_lower):
                    if tag not in used_tags:
                        classified[category].append(tag)
                        used_tags.add(tag)
                    break

        return classified

    def get_florence_caption(self, metadata: Dict) -> str:
        """Extract Florence caption from metadata."""
        # Check various possible locations
        if 'florence_caption' in metadata:
            return metadata['florence_caption']

        if 'florence' in metadata:
            florence = metadata['florence']
            if isinstance(florence, str):
                return florence
            if isinstance(florence, dict):
                return florence.get('caption', florence.get('tags', ['']))[0] if florence.get('tags') else ''

        return ''

    def get_photography_info(self, metadata: Dict) -> Dict[str, float]:
        """Extract photography analysis from metadata."""
        if 'photography' in metadata:
            photo = metadata['photography']
            if isinstance(photo, dict):
                if 'attributes' in photo:
                    return photo['attributes']
                return photo
        return {}

    def get_composition_info(self, metadata: Dict) -> Dict[str, Any]:
        """Extract composition analysis from metadata."""
        if 'composition' in metadata:
            comp = metadata['composition']
            if isinstance(comp, dict):
                if 'attributes' in comp:
                    return comp['attributes']
                return comp
        if 'saliency' in metadata:
            return metadata['saliency']
        return {}

    def get_image_info(self, metadata: Dict) -> Dict[str, int]:
        """Extract image dimensions from metadata."""
        if 'image_info' in metadata:
            return metadata['image_info']
        return {'width': 0, 'height': 0}

    @abstractmethod
    def generate(self, metadata: Dict) -> ComposeResult:
        """Generate prompt from metadata."""
        pass

    def _join_natural(self, items: List[str]) -> str:
        """Join items in natural language style."""
        if not items:
            return ""
        if len(items) == 1:
            return items[0]
        if len(items) == 2:
            return f"{items[0]} and {items[1]}"
        return ', '.join(items[:-1]) + f", and {items[-1]}"
