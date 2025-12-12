"""
Z-Image Prompt Translator

Converts generic prompts to Z-Image optimized format.
Handles:
- Tag soup → Natural language conversion
- Synonym replacement (generic → Z-Image specific)
- Anti-pattern removal
- Missing element injection
- Prompt structure validation

Key Principles:
- Z-Image uses natural language, NOT tag soup
- Optimal prompt length: 80-250 words
- Structure: [Shot] + [Subject] + [Environment] + [Lighting] + [Mood] + [Technical]
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum

from .zimage_vocabulary import (
    ZImageVocabulary,
    VocabCategory,
    get_vocabulary,
    ANTI_PATTERNS,
    SYNONYM_MAP,
    SAFETY_PHRASES,
)


class PromptStyle(Enum):
    """Detected prompt style."""
    NATURAL_LANGUAGE = "natural_language"
    TAG_SOUP = "tag_soup"
    MIXED = "mixed"
    UNKNOWN = "unknown"


@dataclass
class TranslationResult:
    """Result of prompt translation."""
    original: str
    translated: str
    style_detected: PromptStyle
    changes_made: List[str]
    suggestions: List[str]
    word_count: int
    is_optimal_length: bool


@dataclass
class PromptComponents:
    """Extracted components from a prompt."""
    subject: Optional[str] = None
    shot_type: Optional[str] = None
    face_angle: Optional[str] = None
    lighting: Optional[str] = None
    environment: Optional[str] = None
    camera: Optional[str] = None
    lens: Optional[str] = None
    aperture: Optional[str] = None
    film_stock: Optional[str] = None
    color_palette: Optional[str] = None
    mood: Optional[str] = None
    clothing: Optional[str] = None
    safety: Optional[str] = None
    extras: List[str] = None

    def __post_init__(self):
        if self.extras is None:
            self.extras = []


class ZImagePromptTranslator:
    """
    Translates generic prompts to Z-Image optimized format.
    """

    # Patterns for detecting tag soup vs natural language
    TAG_SOUP_INDICATORS = [
        r'^\s*\w+\s*,',  # Starts with word followed by comma
        r',\s*\w+\s*,\s*\w+\s*,',  # Multiple comma-separated single words
        r'\(\w+:\d+\.?\d*\)',  # Weight syntax like (word:1.2)
        r'\[\w+\]',  # Square bracket emphasis
    ]

    NATURAL_LANG_INDICATORS = [
        r'\b(a|an|the|is|are|with|in|on|at|of|for)\b',  # Articles/prepositions
        r'\b(wearing|standing|sitting|looking|holding)\b',  # Verbs
        r'\.\s+[A-Z]',  # Sentences ending with period
    ]

    # Shot type detection patterns
    SHOT_PATTERNS = {
        "extreme close-up": r'\b(extreme\s+)?close[\s-]?up\b|\bmacro\b|\bdetail\s+shot\b',
        "close-up shot": r'\bclose[\s-]?up\b|\bhead\s*shot\b|\btight\s+portrait\b',
        "medium close-up": r'\bmedium\s+close[\s-]?up\b|\bbust\s+shot\b',
        "medium shot": r'\bmedium\s+shot\b|\bmid[\s-]?shot\b|\bwaist[\s-]?up\b',
        "full body shot": r'\bfull[\s-]?body\b|\bfull[\s-]?length\b|\bfull\s+shot\b',
        "wide shot": r'\bwide\s+shot\b|\blong\s+shot\b|\bestablishing\b',
    }

    # Lighting detection patterns
    LIGHTING_PATTERNS = {
        "soft diffused daylight": r'\bsoft\s+(diffused\s+)?light\b|\bnatural\s+light\b|\bdaylight\b',
        "golden hour sunlight": r'\bgolden\s+hour\b|\bsunset\b|\bwarm\s+sunlight\b',
        "studio softbox lighting": r'\bstudio\s+(soft\s*box\s+)?light\b|\bsoftbox\b',
        "cinematic warm key light": r'\bcinematic\s+light\b|\bfilm\s+light\b|\bkey\s+light\b',
        "rim lighting": r'\brim\s+light\b|\bback\s*light\b|\bedge\s+light\b',
        "Rembrandt lighting": r'\brembrandt\b|\btriangle\s+shadow\b',
        "high-contrast noir lighting": r'\bnoir\b|\bhigh[\s-]?contrast\b|\bdramatic\s+shadow\b',
    }

    # Lens detection patterns
    LENS_PATTERNS = {
        "24mm wide angle lens": r'\b24\s*mm\b',
        "35mm lens": r'\b35\s*mm\b',
        "50mm lens": r'\b50\s*mm\b',
        "85mm portrait lens": r'\b85\s*mm\b',
        "105mm macro lens": r'\b105\s*mm\b|\bmacro\s+lens\b',
        "135mm telephoto lens": r'\b135\s*mm\b',
        "200mm telephoto lens": r'\b200\s*mm\b|\btelephoto\b',
    }

    # Aperture detection patterns
    APERTURE_PATTERNS = {
        "f/1.4 wide open aperture, extremely shallow depth of field": r'\bf[/.]?1\.4\b',
        "f/1.8 aperture with shallow depth of field": r'\bf[/.]?1\.8\b',
        "f/2.8 aperture with soft background blur": r'\bf[/.]?2\.8\b',
        "f/5.6 aperture with moderate depth of field": r'\bf[/.]?5\.6\b',
        "f/8 aperture with sharp depth of field": r'\bf[/.]?8\b',
        "shallow depth of field with soft bokeh": r'\bbokeh\b|\bshallow\s+(depth\s+of\s+field|dof)\b|\bblurred?\s+background\b',
    }

    def __init__(self, vocabulary: ZImageVocabulary = None):
        self.vocab = vocabulary or get_vocabulary()

    def detect_style(self, prompt: str) -> PromptStyle:
        """Detect if prompt is tag soup, natural language, or mixed."""
        tag_soup_score = 0
        natural_lang_score = 0

        for pattern in self.TAG_SOUP_INDICATORS:
            if re.search(pattern, prompt, re.IGNORECASE):
                tag_soup_score += 1

        for pattern in self.NATURAL_LANG_INDICATORS:
            if re.search(pattern, prompt, re.IGNORECASE):
                natural_lang_score += 1

        # Count comma-separated elements vs sentences
        comma_count = prompt.count(',')
        period_count = prompt.count('.')
        word_count = len(prompt.split())

        # High comma-to-word ratio suggests tag soup
        if word_count > 0 and comma_count / word_count > 0.15:
            tag_soup_score += 2

        if tag_soup_score > natural_lang_score + 1:
            return PromptStyle.TAG_SOUP
        elif natural_lang_score > tag_soup_score + 1:
            return PromptStyle.NATURAL_LANGUAGE
        elif tag_soup_score > 0 and natural_lang_score > 0:
            return PromptStyle.MIXED
        else:
            return PromptStyle.UNKNOWN

    def extract_components(self, prompt: str) -> PromptComponents:
        """Extract prompt components for analysis."""
        components = PromptComponents()
        prompt_lower = prompt.lower()

        # Detect shot type
        for shot_term, pattern in self.SHOT_PATTERNS.items():
            if re.search(pattern, prompt_lower):
                components.shot_type = shot_term
                break

        # Detect lighting
        for light_term, pattern in self.LIGHTING_PATTERNS.items():
            if re.search(pattern, prompt_lower):
                components.lighting = light_term
                break

        # Detect lens
        for lens_term, pattern in self.LENS_PATTERNS.items():
            if re.search(pattern, prompt_lower):
                components.lens = lens_term
                break

        # Detect aperture/depth of field
        for aperture_term, pattern in self.APERTURE_PATTERNS.items():
            if re.search(pattern, prompt_lower):
                components.aperture = aperture_term
                break

        # Detect face angle
        face_patterns = {
            "front view, facing camera directly": r'\bfront\s+view\b|\bfacing\s+camera\b|\bstraight\s+on\b',
            "three-quarter view, face angled 45 degrees": r'\bthree[\s-]?quarter\b|\b45\s*degree\b',
            "profile view, side of face": r'\bprofile\b|\bside\s+view\b',
            "looking slightly upward": r'\blooking\s+(up|upward)\b',
            "looking slightly downward": r'\blooking\s+down\b',
            "looking over shoulder": r'\bover\s+shoulder\b',
        }
        for angle_term, pattern in face_patterns.items():
            if re.search(pattern, prompt_lower):
                components.face_angle = angle_term
                break

        # Detect mood
        mood_patterns = {
            "serene peaceful atmosphere": r'\bserene\b|\bpeaceful\b|\bcalm\b',
            "dramatic intense atmosphere": r'\bdramatic\b|\bintense\b',
            "cinematic filmic atmosphere": r'\bcinematic\b|\bfilmic\b|\bfilm[\s-]?like\b',
            "ethereal dreamy atmosphere": r'\bethereal\b|\bdreamy\b',
            "nostalgic wistful atmosphere": r'\bnostalgic\b|\bvintage\b',
            "mysterious enigmatic atmosphere": r'\bmysterious\b|\benigmatic\b|\bnoir\b',
        }
        for mood_term, pattern in mood_patterns.items():
            if re.search(pattern, prompt_lower):
                components.mood = mood_term
                break

        # Detect film stock
        film_patterns = {
            "Kodak Portra 400 film aesthetic": r'\bportra\s*400\b',
            "Kodak Portra 800 film grain": r'\bportra\s*800\b',
            "Kodak Tri-X 400 black and white film grain": r'\btri[\s-]?x\b',
            "Cinestill 800T tungsten halation": r'\bcinestill\b',
            "Fuji Pro 400H pastel tones": r'\bfuji\s*(pro\s*)?(400h)?\b',
        }
        for film_term, pattern in film_patterns.items():
            if re.search(pattern, prompt_lower):
                components.film_stock = film_term
                break

        # Detect camera
        camera_patterns = {
            "shot on Canon 5D Mark IV": r'\bcanon\s*(5d|eos)\b',
            "shot on Sony A7 IV": r'\bsony\s*(a7|alpha)\b',
            "shot on Hasselblad medium format": r'\bhasselblad\b',
            "shot on Leica M": r'\bleica\b',
            "smartphone photo": r'\b(iphone|smartphone|mobile)\s*(photo)?\b',
        }
        for camera_term, pattern in camera_patterns.items():
            if re.search(pattern, prompt_lower):
                components.camera = camera_term
                break

        # Try to extract subject (first noun phrase, simplified)
        # Look for patterns like "a woman", "an elderly man", etc.
        subject_match = re.search(
            r'\b(a|an)\s+([\w\s]+?(?:woman|man|person|girl|boy|child|adult|model|portrait))',
            prompt_lower
        )
        if subject_match:
            components.subject = subject_match.group(0)

        return components

    def tag_soup_to_natural(self, prompt: str) -> Tuple[str, List[str]]:
        """
        Convert tag soup to natural language.
        Returns (converted_prompt, list_of_changes).
        """
        changes = []

        # Split by comma
        tags = [t.strip() for t in prompt.split(',') if t.strip()]

        if len(tags) < 3:
            return prompt, []  # Not really tag soup

        # Remove weight syntax (word:1.2)
        cleaned_tags = []
        for tag in tags:
            weight_match = re.match(r'\(?([\w\s]+)(?::\d+\.?\d*)?\)?', tag)
            if weight_match:
                cleaned_tags.append(weight_match.group(1).strip())
            else:
                cleaned_tags.append(tag)

        # Categorize tags
        subject_tags = []
        descriptor_tags = []
        technical_tags = []
        lighting_tags = []
        environment_tags = []

        for tag in cleaned_tags:
            tag_lower = tag.lower()

            # Check if it's a known category
            if any(re.search(p, tag_lower) for p in self.LIGHTING_PATTERNS.values()):
                lighting_tags.append(tag)
            elif any(re.search(p, tag_lower) for p in self.LENS_PATTERNS.values()):
                technical_tags.append(tag)
            elif any(re.search(p, tag_lower) for p in self.APERTURE_PATTERNS.values()):
                technical_tags.append(tag)
            elif any(re.search(p, tag_lower) for p in self.SHOT_PATTERNS.values()):
                descriptor_tags.insert(0, tag)  # Shot type goes first
            elif re.search(r'\b(background|outdoor|indoor|studio|street|room|park)\b', tag_lower):
                environment_tags.append(tag)
            elif re.search(r'\b(woman|man|person|girl|boy|portrait|face)\b', tag_lower):
                subject_tags.append(tag)
            else:
                descriptor_tags.append(tag)

        # Build natural language prompt
        parts = []

        # Start with shot type + subject
        if descriptor_tags and subject_tags:
            shot = descriptor_tags[0] if any(
                re.search(p, descriptor_tags[0].lower()) for p in self.SHOT_PATTERNS.values()
            ) else None
            if shot:
                parts.append(f"{shot} of {' '.join(subject_tags)}")
                descriptor_tags = descriptor_tags[1:]
            else:
                parts.append(f"A {' '.join(subject_tags)}")
        elif subject_tags:
            parts.append(f"A {' '.join(subject_tags)}")

        # Add remaining descriptors
        if descriptor_tags:
            parts.append(', '.join(descriptor_tags))

        # Add environment
        if environment_tags:
            parts.append(f"In {' '.join(environment_tags)}")

        # Add lighting
        if lighting_tags:
            parts.append(f"with {' '.join(lighting_tags)}")

        # Add technical
        if technical_tags:
            parts.append(f"Shot with {', '.join(technical_tags)}")

        # Join into natural sentences
        result = '. '.join(parts) + '.'

        if result != prompt:
            changes.append("Converted tag soup to natural language")

        return result, changes

    def remove_anti_patterns(self, prompt: str) -> Tuple[str, List[str]]:
        """Remove terms Z-Image ignores."""
        changes = []
        result = prompt

        for pattern in ANTI_PATTERNS:
            if pattern.lower() in result.lower():
                # Remove with surrounding punctuation
                result = re.sub(
                    r',?\s*' + re.escape(pattern) + r'\s*,?',
                    ' ',
                    result,
                    flags=re.IGNORECASE
                )
                changes.append(f"Removed '{pattern}' (not needed for Z-Image)")

        # Clean up
        result = re.sub(r',\s*,', ',', result)
        result = re.sub(r'\s+', ' ', result)
        result = result.strip(' ,.')

        return result, changes

    def apply_synonyms(self, prompt: str) -> Tuple[str, List[str]]:
        """Replace generic terms with Z-Image optimized versions."""
        changes = []
        result = prompt

        for generic, optimized in SYNONYM_MAP.items():
            if generic.lower() in result.lower():
                if optimized:
                    result = re.sub(
                        re.escape(generic),
                        optimized,
                        result,
                        flags=re.IGNORECASE
                    )
                    changes.append(f"Replaced '{generic}' with '{optimized}'")
                else:
                    # Remove empty replacements (like "high quality")
                    result = re.sub(
                        r',?\s*' + re.escape(generic) + r'\s*,?',
                        ' ',
                        result,
                        flags=re.IGNORECASE
                    )
                    changes.append(f"Removed '{generic}' (not needed)")

        return result.strip(), changes

    def inject_missing_elements(self, prompt: str, components: PromptComponents) -> Tuple[str, List[str]]:
        """Inject missing essential elements."""
        changes = []
        additions = []

        # Z-Image responds strongly to lighting - add if missing
        if not components.lighting:
            additions.append("soft natural lighting")
            changes.append("Added lighting description (Z-Image responds strongly to lighting)")

        # Add shot type if missing for portraits
        if not components.shot_type and components.subject:
            if 'portrait' in prompt.lower() or 'face' in prompt.lower():
                additions.append("medium close-up")
                changes.append("Added shot type for portrait")

        # Add depth of field if lens specified but no aperture
        if components.lens and not components.aperture:
            additions.append("shallow depth of field")
            changes.append("Added depth of field to complement lens specification")

        if additions:
            prompt = prompt.rstrip('.') + '. ' + ', '.join(additions) + '.'

        return prompt, changes

    def ensure_structure(self, prompt: str, components: PromptComponents) -> str:
        """
        Ensure prompt follows Z-Image optimal structure.
        [Shot] + [Subject] + [Details] + [Environment] + [Lighting] + [Technical]
        """
        # For now, we just clean up and don't restructure completely
        # Full restructuring could lose nuance from the original prompt

        # Ensure proper sentence ending
        prompt = prompt.strip()
        if prompt and not prompt.endswith(('.', '!', '?')):
            prompt += '.'

        return prompt

    def translate(
        self,
        prompt: str,
        add_safety: bool = False,
        safety_level: str = "sfw_basic"
    ) -> TranslationResult:
        """
        Translate a prompt to Z-Image optimized format.

        Args:
            prompt: The original prompt
            add_safety: Whether to add SFW safety phrases
            safety_level: Safety phrase level (sfw_basic, sfw_full, etc.)

        Returns:
            TranslationResult with original, translated, and metadata
        """
        all_changes = []
        suggestions = []

        # Detect style
        style = self.detect_style(prompt)

        # Extract components
        components = self.extract_components(prompt)

        # Start translation
        result = prompt

        # 1. Remove anti-patterns
        result, changes = self.remove_anti_patterns(result)
        all_changes.extend(changes)

        # 2. Convert tag soup if detected
        if style in (PromptStyle.TAG_SOUP, PromptStyle.MIXED):
            result, changes = self.tag_soup_to_natural(result)
            all_changes.extend(changes)

        # 3. Apply synonym replacements
        result, changes = self.apply_synonyms(result)
        all_changes.extend(changes)

        # 4. Re-extract components after changes
        components = self.extract_components(result)

        # 5. Inject missing elements
        result, changes = self.inject_missing_elements(result, components)
        all_changes.extend(changes)

        # 6. Ensure proper structure
        result = self.ensure_structure(result, components)

        # 7. Add safety phrases if requested
        if add_safety:
            safety_phrase = SAFETY_PHRASES.get(safety_level, SAFETY_PHRASES["sfw_basic"])
            result = result.rstrip('.') + '. ' + safety_phrase + '.'
            all_changes.append(f"Added safety phrase: {safety_level}")

        # Generate suggestions
        suggestions = self.vocab.suggest_improvements(result)

        # Calculate metrics
        word_count = len(result.split())
        is_optimal = 80 <= word_count <= 250

        if word_count < 80:
            suggestions.insert(0, f"Prompt is short ({word_count} words). Consider adding more detail for better results.")
        elif word_count > 250:
            suggestions.insert(0, f"Prompt is long ({word_count} words). Consider trimming for efficiency.")

        # Clean up final result
        result = re.sub(r'\s+', ' ', result).strip()
        result = re.sub(r'\.+', '.', result)  # Remove double periods

        return TranslationResult(
            original=prompt,
            translated=result,
            style_detected=style,
            changes_made=all_changes,
            suggestions=suggestions,
            word_count=word_count,
            is_optimal_length=is_optimal,
        )

    def quick_translate(self, prompt: str) -> str:
        """Quick translation without detailed results."""
        return self.translate(prompt).translated


# =============================================================================
# PROMPT BUILDER (Template-based)
# =============================================================================

class ZImagePromptBuilder:
    """
    Build prompts using Z-Image vocabulary and templates.
    """

    def __init__(self, vocabulary: ZImageVocabulary = None):
        self.vocab = vocabulary or get_vocabulary()

    def build_portrait(
        self,
        subject: str,
        shot_type: str = "medium_closeup",
        face_angle: str = "three_quarter",
        lighting: str = "soft_diffused",
        lens: str = "85mm",
        aperture: str = "f1_8",
        environment: str = "studio_gray",
        mood: str = "cinematic",
        clothing: str = None,
        add_safety: bool = True,
    ) -> str:
        """Build a portrait prompt using vocabulary terms."""
        parts = []

        # Shot type
        shot_term = self.vocab.get_term(shot_type)
        if shot_term:
            parts.append(f"{shot_term.term} of {subject}")
        else:
            parts.append(f"Portrait of {subject}")

        # Face angle
        angle_term = self.vocab.get_term(face_angle)
        if angle_term:
            parts.append(angle_term.term)

        # Clothing
        if clothing:
            parts.append(f"wearing {clothing}")

        # Environment
        env_term = self.vocab.get_term(environment)
        if env_term:
            parts.append(env_term.term)

        # Lighting
        light_term = self.vocab.get_term(lighting)
        if light_term:
            parts.append(light_term.term)

        # Technical
        lens_term = self.vocab.get_term(lens)
        aperture_term = self.vocab.get_term(aperture)
        if lens_term and aperture_term:
            parts.append(f"Shot with {lens_term.term}, {aperture_term.term}")
        elif lens_term:
            parts.append(f"Shot with {lens_term.term}")

        # Mood
        mood_term = self.vocab.get_term(mood)
        if mood_term:
            parts.append(mood_term.term)

        # Safety
        if add_safety:
            parts.append(self.vocab.get_safety_phrase("sfw_basic"))

        return '. '.join(parts) + '.'

    def build_product(
        self,
        product: str,
        lighting: str = "studio_softbox",
        environment: str = "studio_white",
        color_palette: str = None,
    ) -> str:
        """Build a product photography prompt."""
        parts = [f"Professional product photography of {product}"]

        # Lighting
        light_term = self.vocab.get_term(lighting)
        if light_term:
            parts.append(light_term.term)

        # Environment
        env_term = self.vocab.get_term(environment)
        if env_term:
            parts.append(env_term.term)

        # Color
        if color_palette:
            color_term = self.vocab.get_term(color_palette)
            if color_term:
                parts.append(color_term.term)

        parts.append("Sharp focus on product details, clean composition")
        parts.append("Commercial photography quality")

        return '. '.join(parts) + '.'

    def enhance_with_photography(
        self,
        base_prompt: str,
        lighting: str = None,
        lens: str = None,
        aperture: str = None,
        film_stock: str = None,
        mood: str = None,
    ) -> str:
        """Enhance an existing prompt with photography vocabulary."""
        additions = []

        if lighting:
            term = self.vocab.get_term(lighting)
            if term:
                additions.append(term.term)

        if lens:
            term = self.vocab.get_term(lens)
            if term:
                additions.append(f"shot with {term.term}")

        if aperture:
            term = self.vocab.get_term(aperture)
            if term:
                additions.append(term.term)

        if film_stock:
            term = self.vocab.get_term(film_stock)
            if term:
                additions.append(term.term)

        if mood:
            term = self.vocab.get_term(mood)
            if term:
                additions.append(term.term)

        if additions:
            return base_prompt.rstrip('.') + '. ' + ', '.join(additions) + '.'
        return base_prompt


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

# Global translator instance
_translator: Optional[ZImagePromptTranslator] = None
_builder: Optional[ZImagePromptBuilder] = None


def get_translator() -> ZImagePromptTranslator:
    """Get the global translator instance."""
    global _translator
    if _translator is None:
        _translator = ZImagePromptTranslator()
    return _translator


def get_builder() -> ZImagePromptBuilder:
    """Get the global builder instance."""
    global _builder
    if _builder is None:
        _builder = ZImagePromptBuilder()
    return _builder


def translate_prompt(prompt: str, add_safety: bool = False) -> str:
    """Quick prompt translation."""
    return get_translator().translate(prompt, add_safety=add_safety).translated


def translate_prompt_full(prompt: str, add_safety: bool = False) -> TranslationResult:
    """Full prompt translation with metadata."""
    return get_translator().translate(prompt, add_safety=add_safety)


def build_portrait_prompt(**kwargs) -> str:
    """Build a portrait prompt."""
    return get_builder().build_portrait(**kwargs)


def enhance_prompt(prompt: str, **kwargs) -> str:
    """Enhance prompt with photography vocabulary."""
    return get_builder().enhance_with_photography(prompt, **kwargs)
