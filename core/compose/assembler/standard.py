"""Standard rule-based prompt assembler.

Assembles classified tokens into natural language prompts using
templates and grammar rules.
"""

from typing import Dict, List, Optional
from collections import defaultdict

from .base import (
    BaseAssembler,
    AssemblerConfig,
    AssembledPrompt,
    CategorySection,
    PromptStyle,
)
from .rules import (
    Gender,
    detect_gender,
    get_pronouns,
    clean_token_for_output,
    join_tokens_natural,
    build_subject_phrase,
    CATEGORY_TEMPLATES,
    SUBJECT_DETAIL_TEMPLATES,
)
from ..classifier.base import ClassifiedImage, TokenClassification
from ..classifier.categories import (
    CanonicalCategory,
    SubjectDetailType,
    CATEGORY_ORDER,
    SUBJECT_DETAIL_ORDER,
)


class StandardAssembler(BaseAssembler):
    """Standard rule-based prompt assembler.

    Converts classified tokens into natural language prompts following
    the canonical category order.
    """

    def assemble(self, classified: ClassifiedImage) -> AssembledPrompt:
        """Assemble classified tokens into a prompt.

        Args:
            classified: ClassifiedImage from the classifier

        Returns:
            AssembledPrompt with the generated prompt
        """
        sections: List[CategorySection] = []

        # Detect gender from subject/detail tokens for pronoun usage
        all_tokens = [c.token.text for c in classified.all_classifications]
        gender = detect_gender(all_tokens)
        subj, poss, obj = get_pronouns(gender)

        # Collect subject details by subcategory
        detail_by_subcat = self._collect_subject_details(classified)

        # Process each category in order
        for category in CATEGORY_ORDER:
            if category == CanonicalCategory.SUBJECT_DETAILS:
                # Handle subject details specially with subcategories
                section = self._build_subject_details_section(
                    classified, detail_by_subcat, gender
                )
            else:
                section = self._build_category_section(classified, category, gender)

            if section and section.tokens:
                sections.append(section)

        # Build final prompt based on style
        if self.config.style == PromptStyle.TAGS:
            prompt = self._build_tags_prompt(sections)
        elif self.config.style == PromptStyle.HYBRID:
            prompt = self._build_hybrid_prompt(sections, gender)
        elif self.config.style == PromptStyle.STRUCTURED:
            prompt = self._build_structured_prompt(sections)
        elif self.config.style == PromptStyle.FULL:
            prompt = self._build_full_prompt(sections, classified.image_info)
        elif self.config.style == PromptStyle.ZIMAGE:
            prompt = self._build_zimage_prompt(sections, classified.image_info, gender)
        else:  # NATURAL
            prompt = self._build_natural_prompt(sections, gender)

        return AssembledPrompt(
            prompt=prompt,
            style=self.config.style,
            sections=sections,
            metadata={
                "gender": gender.value,
                "image_info": classified.image_info,
            }
        )

    def _collect_subject_details(
        self, classified: ClassifiedImage
    ) -> Dict[SubjectDetailType, List[TokenClassification]]:
        """Collect subject detail tokens by subcategory."""
        result = defaultdict(list)

        tokens = classified.get_category(CanonicalCategory.SUBJECT_DETAILS)
        if not tokens:
            return result

        for tc in tokens:
            if tc.token.confidence < self.config.min_confidence:
                continue
            subcat = tc.subcategory or SubjectDetailType.BODY  # Default
            result[subcat].append(tc)

        # Sort each subcategory by confidence and apply limits
        limit = self.config.get_limit_for_category(CanonicalCategory.SUBJECT_DETAILS)
        num_subcats = len(result)

        if num_subcats > 0:
            # Calculate per-subcategory limit that won't exceed total
            # Use ceiling division to distribute evenly
            per_subcat_limit = max(3, limit // num_subcats)

            # Track total tokens to enforce exact limit
            total_tokens = 0

            for subcat in result:
                result[subcat].sort(key=lambda t: t.token.confidence, reverse=True)
                # Calculate how many we can still take
                remaining = limit - total_tokens
                actual_limit = min(per_subcat_limit, remaining, len(result[subcat]))
                result[subcat] = result[subcat][:actual_limit]
                total_tokens += len(result[subcat])

        return result

    def _build_category_section(
        self,
        classified: ClassifiedImage,
        category: CanonicalCategory,
        gender: Gender,
    ) -> Optional[CategorySection]:
        """Build a section for a standard category."""
        tokens = self._get_tokens_for_category(classified, category)
        if not tokens:
            return CategorySection(
                category=category,
                tokens=[],
                text="",
                confidence=0.0
            )

        if self.config.deduplicate:
            tokens = self._deduplicate_tokens(tokens)

        # Clean token texts
        token_texts = [clean_token_for_output(t.token.text) for t in tokens]
        token_texts = [t for t in token_texts if t]  # Remove empty

        if not token_texts:
            return CategorySection(
                category=category,
                tokens=[],
                text="",
                confidence=0.0
            )

        # Build phrase using template
        template = CATEGORY_TEMPLATES.get(category, {})
        join_style = template.get("join", ", ")

        # Determine join style from separator
        if join_style == " and ":
            joined = join_tokens_natural(token_texts, "and")
        elif join_style == " ":
            joined = join_tokens_natural(token_texts, "space")
        else:
            joined = join_tokens_natural(token_texts, "comma")

        # Apply template
        prefix = template.get("prefix", "")
        suffix = template.get("suffix", "")

        if len(token_texts) == 1:
            text = template.get("single", "{token}").format(token=joined)
        else:
            text = template.get("multiple", "{tokens}").format(tokens=joined)

        text = prefix + text + suffix

        # Calculate average confidence
        avg_conf = sum(t.token.confidence for t in tokens) / len(tokens)

        return CategorySection(
            category=category,
            tokens=token_texts,
            text=text.strip(),
            confidence=avg_conf
        )

    def _build_subject_details_section(
        self,
        classified: ClassifiedImage,
        detail_by_subcat: Dict[SubjectDetailType, List[TokenClassification]],
        gender: Gender,
    ) -> Optional[CategorySection]:
        """Build subject details section with subcategory organization."""
        all_tokens = []
        parts = []

        for subcat in SUBJECT_DETAIL_ORDER:
            if subcat not in detail_by_subcat:
                continue

            tokens = detail_by_subcat[subcat]
            if not tokens:
                continue

            # Clean token texts
            token_texts = [clean_token_for_output(t.token.text) for t in tokens]
            token_texts = [t for t in token_texts if t]

            if not token_texts:
                continue

            all_tokens.extend(token_texts)

            # Build phrase for this subcategory
            template = SUBJECT_DETAIL_TEMPLATES.get(subcat, {})

            # Determine join style
            join_sep = template.get("join", ", ")
            if join_sep == " ":
                joined = join_tokens_natural(token_texts, "space")
            elif "and" in join_sep:
                joined = join_tokens_natural(token_texts, "and")
            else:
                joined = join_tokens_natural(token_texts, "comma")

            if len(token_texts) == 1:
                phrase = template.get("single", "{token}").format(token=joined)
            else:
                phrase = template.get("multiple", "{tokens}").format(tokens=joined)

            parts.append(phrase)

        if not all_tokens:
            return CategorySection(
                category=CanonicalCategory.SUBJECT_DETAILS,
                tokens=[],
                text="",
                confidence=0.0
            )

        # Join all parts
        text = ", ".join(parts)

        # Calculate average confidence
        all_tcs = [t for tokens in detail_by_subcat.values() for t in tokens]
        avg_conf = sum(t.token.confidence for t in all_tcs) / len(all_tcs) if all_tcs else 0

        return CategorySection(
            category=CanonicalCategory.SUBJECT_DETAILS,
            tokens=all_tokens,
            text=text,
            confidence=avg_conf
        )

    def _build_natural_prompt(
        self, sections: List[CategorySection], gender: Gender
    ) -> str:
        """Build a natural language prompt from sections."""
        parts = []
        subj, poss, obj = get_pronouns(gender)

        # Quality boosters first (comma-separated)
        quality = next((s for s in sections if s.category == CanonicalCategory.QUALITY_BOOSTERS), None)
        if quality and quality.text:
            parts.append(quality.text)

        # Style/Medium
        style = next((s for s in sections if s.category == CanonicalCategory.STYLE_MEDIUM), None)
        if style and style.text:
            parts.append(style.text)

        # Subject - start of sentence
        subject = next((s for s in sections if s.category == CanonicalCategory.SUBJECT), None)
        subject_text = ""
        if subject and subject.text:
            subject_text = subject.text
            # Capitalize if not already
            if subject_text and subject_text[0].islower():
                subject_text = subject_text[0].upper() + subject_text[1:]

        # Subject details - append to subject
        details = next((s for s in sections if s.category == CanonicalCategory.SUBJECT_DETAILS), None)
        if subject_text:
            if details and details.text:
                parts.append(f"{subject_text}, {details.text}")
            else:
                parts.append(subject_text)
        elif details and details.text:
            parts.append(details.text)

        # Action/Pose
        action = next((s for s in sections if s.category == CanonicalCategory.ACTION_POSE), None)
        if action and action.text:
            parts.append(action.text)

        # Environment
        env = next((s for s in sections if s.category == CanonicalCategory.ENVIRONMENT), None)
        if env and env.text:
            parts.append(env.text)

        # Lighting
        lighting = next((s for s in sections if s.category == CanonicalCategory.LIGHTING), None)
        if lighting and lighting.text:
            parts.append(lighting.text)

        # Composition
        comp = next((s for s in sections if s.category == CanonicalCategory.COMPOSITION), None)
        if comp and comp.text:
            parts.append(comp.text)

        # Technical
        tech = next((s for s in sections if s.category == CanonicalCategory.TECHNICAL), None)
        if tech and tech.text:
            parts.append(tech.text)

        # Join with commas for natural flow
        prompt = ", ".join(p for p in parts if p)

        return prompt

    def _build_tags_prompt(self, sections: List[CategorySection]) -> str:
        """Build a comma-separated tags prompt."""
        all_tokens = []
        for section in sections:
            all_tokens.extend(section.tokens)
        return ", ".join(all_tokens)

    def _build_hybrid_prompt(
        self, sections: List[CategorySection], gender: Gender
    ) -> str:
        """Build a hybrid prompt: sentence intro + tag details."""
        subj, poss, obj = get_pronouns(gender)

        # Build intro sentence from subject/details/action
        intro_parts = []

        subject = next((s for s in sections if s.category == CanonicalCategory.SUBJECT), None)
        if subject and subject.text:
            intro_parts.append(subject.text)

        details = next((s for s in sections if s.category == CanonicalCategory.SUBJECT_DETAILS), None)
        if details and details.text:
            intro_parts.append(details.text)

        action = next((s for s in sections if s.category == CanonicalCategory.ACTION_POSE), None)
        if action and action.text:
            intro_parts.append(action.text)

        intro = ", ".join(intro_parts) if intro_parts else ""

        # Collect remaining tokens as tags
        tag_categories = [
            CanonicalCategory.QUALITY_BOOSTERS,
            CanonicalCategory.STYLE_MEDIUM,
            CanonicalCategory.ENVIRONMENT,
            CanonicalCategory.LIGHTING,
            CanonicalCategory.COMPOSITION,
            CanonicalCategory.TECHNICAL,
        ]
        tag_tokens = []
        for section in sections:
            if section.category in tag_categories:
                tag_tokens.extend(section.tokens)

        tags = ", ".join(tag_tokens) if tag_tokens else ""

        # Combine
        if intro and tags:
            return f"{intro}. {tags}"
        return intro or tags

    def _build_structured_prompt(self, sections: List[CategorySection]) -> str:
        """Build a category-segregated prompt with markdown headers.

        Each category has a markdown header followed by tokens on the next line.
        Example:
            # SUBJECT
            woman, nun

            # DETAILS
            black habit, white wimple, religious attire

            # ENVIRONMENT
            church interior, stained glass window
        """
        # Category display names for cleaner output
        CATEGORY_LABELS = {
            CanonicalCategory.QUALITY_BOOSTERS: "QUALITY",
            CanonicalCategory.STYLE_MEDIUM: "STYLE",
            CanonicalCategory.SUBJECT: "SUBJECT",
            CanonicalCategory.SUBJECT_DETAILS: "DETAILS",
            CanonicalCategory.ACTION_POSE: "POSE",
            CanonicalCategory.ENVIRONMENT: "ENVIRONMENT",
            CanonicalCategory.LIGHTING: "LIGHTING",
            CanonicalCategory.COMPOSITION: "COMPOSITION",
            CanonicalCategory.TECHNICAL: "TECHNICAL",
        }

        blocks = []
        for section in sections:
            if not section.tokens:
                continue

            label = CATEGORY_LABELS.get(section.category, section.category.value.upper())
            tokens_str = ", ".join(section.tokens)
            blocks.append(f"# {label}\n{tokens_str}")

        return "\n\n".join(blocks)

    def _build_full_prompt(self, sections: List[CategorySection], image_info: dict) -> str:
        """Build a full prompt preserving Florence descriptions + structured tags.

        This style preserves maximum information by:
        1. Including the best Florence natural language description
        2. Adding Florence generated tags
        3. Adding structured category tags from classification

        Example output:
            # DESCRIPTION
            A beautiful woman with long, wavy brown hair...

            # FLORENCE TAGS
            1girl, solo, long hair, brown hair, jewelry...

            # QUALITY
            realistic, photograph, sharp...

            # DETAILS
            long hair, brown eyes, necklace...
        """
        blocks = []

        # --- Get best Florence description ---
        # Priority: mixed_caption_plus > mixed_caption > description > caption
        florence_description = None
        for key in ["florence_mixed_caption_plus", "florence_mixed_caption",
                    "florence_description", "florence_caption"]:
            if key in image_info and image_info[key]:
                text = str(image_info[key])
                # Extract just the prose part (before any tags section)
                if "\n\n" in text:
                    text = text.split("\n\n")[0]
                # Skip if too short
                if len(text) > 50:
                    florence_description = text.strip()
                    break

        if florence_description:
            blocks.append(f"# DESCRIPTION\n{florence_description}")

        # --- Get Florence generated tags ---
        florence_tags = image_info.get("florence_generate_tags", "")
        if florence_tags:
            blocks.append(f"# FLORENCE TAGS\n{florence_tags}")

        # --- Add structured category tags ---
        CATEGORY_LABELS = {
            CanonicalCategory.QUALITY_BOOSTERS: "QUALITY",
            CanonicalCategory.STYLE_MEDIUM: "STYLE",
            CanonicalCategory.SUBJECT: "SUBJECT",
            CanonicalCategory.SUBJECT_DETAILS: "DETAILS",
            CanonicalCategory.ACTION_POSE: "POSE",
            CanonicalCategory.ENVIRONMENT: "ENVIRONMENT",
            CanonicalCategory.LIGHTING: "LIGHTING",
            CanonicalCategory.COMPOSITION: "COMPOSITION",
            CanonicalCategory.TECHNICAL: "TECHNICAL",
        }

        for section in sections:
            if not section.tokens:
                continue

            label = CATEGORY_LABELS.get(section.category, section.category.value.upper())
            tokens_str = ", ".join(section.tokens)
            blocks.append(f"# {label}\n{tokens_str}")

        return "\n\n".join(blocks)

    def _build_zimage_prompt(
        self,
        sections: List[CategorySection],
        image_info: dict,
        gender: Gender
    ) -> str:
        """Build Z-Image optimized prompt following the 6-part formula.

        Z-Image Formula: Subject + Scene + Composition + Lighting + Style + Constraints

        Key principles:
        - Natural prose, NOT comma-separated tags
        - Florence descriptions preserved intact
        - Tags converted to descriptive phrases
        - 80-250 words optimal
        - Lighting is critical (always include)

        Output structure:
        1. Florence prose description (preserved intact)
        2. Subject attributes as natural descriptions
        3. Environment/scene description
        4. Lighting description
        5. Style and quality
        6. Technical/composition
        """
        parts = []

        # === PART 1: Florence Description (PRESERVED INTACT) ===
        # Priority: mixed_caption_plus > mixed_caption > description > caption
        florence_prose = None
        for key in ["florence_mixed_caption_plus", "florence_mixed_caption",
                    "florence_description", "florence_caption"]:
            if key in image_info and image_info[key]:
                text = str(image_info[key]).strip()
                # Extract prose part only (before any tag sections)
                if "\n\n" in text:
                    text = text.split("\n\n")[0].strip()
                # Use if it's substantial prose (not just tags)
                if len(text) > 50 and not self._is_tag_list(text):
                    florence_prose = text
                    break

        if florence_prose:
            parts.append(florence_prose)

        # === PART 2: Subject & Appearance ===
        # Convert tags to natural descriptions
        subject_section = next(
            (s for s in sections if s.category == CanonicalCategory.SUBJECT), None
        )
        details_section = next(
            (s for s in sections if s.category == CanonicalCategory.SUBJECT_DETAILS), None
        )

        # Build subject phrase if not already in Florence description
        if not florence_prose:
            subject_phrase = self._tags_to_subject_phrase(
                subject_section, details_section, gender
            )
            if subject_phrase:
                parts.append(subject_phrase)
        elif details_section and details_section.tokens:
            # Add details not covered by Florence
            extra_details = self._tags_to_appearance_phrase(details_section.tokens, gender)
            if extra_details and extra_details.lower() not in florence_prose.lower():
                parts.append(extra_details)

        # === PART 3: Pose/Action ===
        action_section = next(
            (s for s in sections if s.category == CanonicalCategory.ACTION_POSE), None
        )
        if action_section and action_section.tokens:
            pose_phrase = self._tags_to_pose_phrase(action_section.tokens)
            if pose_phrase:
                parts.append(pose_phrase)

        # === PART 4: Environment/Scene ===
        env_section = next(
            (s for s in sections if s.category == CanonicalCategory.ENVIRONMENT), None
        )
        if env_section and env_section.tokens:
            env_phrase = self._tags_to_environment_phrase(env_section.tokens)
            if env_phrase:
                parts.append(env_phrase)

        # === PART 5: Lighting (CRITICAL for Z-Image) ===
        lighting_section = next(
            (s for s in sections if s.category == CanonicalCategory.LIGHTING), None
        )
        if lighting_section and lighting_section.tokens:
            lighting_phrase = self._tags_to_lighting_phrase(lighting_section.tokens)
            if lighting_phrase:
                parts.append(lighting_phrase)
        else:
            # Default lighting if none specified (Z-Image needs lighting)
            parts.append("The lighting is soft and natural")

        # === PART 6: Style & Quality ===
        style_section = next(
            (s for s in sections if s.category == CanonicalCategory.STYLE_MEDIUM), None
        )
        quality_section = next(
            (s for s in sections if s.category == CanonicalCategory.QUALITY_BOOSTERS), None
        )

        style_tokens = (style_section.tokens if style_section else []) + \
                       (quality_section.tokens if quality_section else [])
        if style_tokens:
            style_phrase = self._tags_to_style_phrase(style_tokens)
            if style_phrase:
                parts.append(style_phrase)

        # === PART 7: Composition & Technical ===
        comp_section = next(
            (s for s in sections if s.category == CanonicalCategory.COMPOSITION), None
        )
        tech_section = next(
            (s for s in sections if s.category == CanonicalCategory.TECHNICAL), None
        )

        comp_tokens = (comp_section.tokens if comp_section else []) + \
                      (tech_section.tokens if tech_section else [])
        if comp_tokens:
            comp_phrase = self._tags_to_composition_phrase(comp_tokens)
            if comp_phrase:
                parts.append(comp_phrase)

        # Join with periods for natural prose flow
        prompt = ". ".join(p.rstrip(".") for p in parts if p)
        if prompt and not prompt.endswith("."):
            prompt += "."

        return prompt

    def _is_tag_list(self, text: str) -> bool:
        """Check if text is primarily a comma-separated tag list."""
        if "," not in text:
            return False
        # Count comma-separated items
        items = [i.strip() for i in text.split(",")]
        # If most items are short (< 25 chars), it's likely a tag list
        short_items = sum(1 for i in items if len(i) < 25)
        return short_items / len(items) > 0.7

    def _tags_to_subject_phrase(
        self,
        subject_section: Optional[CategorySection],
        details_section: Optional[CategorySection],
        gender: Gender
    ) -> str:
        """Convert subject and detail tags to a natural subject phrase."""
        subj_pronoun, poss_pronoun, obj_pronoun = get_pronouns(gender)

        subject_words = []
        if subject_section and subject_section.tokens:
            subject_words = subject_section.tokens[:3]  # Main subject terms

        detail_words = []
        if details_section and details_section.tokens:
            detail_words = details_section.tokens[:8]  # Key details

        if not subject_words and not detail_words:
            return ""

        # Build natural phrase
        if subject_words:
            # "A woman" or "A man" or "A person"
            main_subject = subject_words[0]
            if main_subject.lower() in ("woman", "man", "person", "girl", "boy"):
                phrase = f"A {main_subject}"
            else:
                phrase = main_subject.capitalize()

            if detail_words:
                # Group details by type for natural flow
                phrase += f" with {', '.join(detail_words[:4])}"
                if len(detail_words) > 4:
                    phrase += f", {', '.join(detail_words[4:8])}"
        else:
            phrase = ", ".join(detail_words)

        return phrase

    def _tags_to_appearance_phrase(self, tokens: List[str], gender: Gender) -> str:
        """Convert appearance tags to natural description."""
        if not tokens:
            return ""

        subj_pronoun, poss_pronoun, obj_pronoun = get_pronouns(gender)

        # Group tokens by likely type
        hair_tokens = [t for t in tokens if "hair" in t.lower()]
        eye_tokens = [t for t in tokens if "eye" in t.lower()]
        body_tokens = [t for t in tokens if any(
            w in t.lower() for w in ["skin", "body", "tall", "slim", "muscular"]
        )]
        clothing_tokens = [t for t in tokens if any(
            w in t.lower() for w in ["dress", "shirt", "pants", "wearing", "outfit", "suit", "skirt"]
        )]
        other_tokens = [t for t in tokens if t not in hair_tokens + eye_tokens + body_tokens + clothing_tokens]

        parts = []

        if hair_tokens:
            parts.append(f"{poss_pronoun} hair is {', '.join(hair_tokens[:2])}")
        if eye_tokens:
            parts.append(f"{poss_pronoun} eyes are {', '.join(eye_tokens[:2])}")
        if clothing_tokens:
            parts.append(f"wearing {', '.join(clothing_tokens[:3])}")
        if other_tokens[:3]:
            parts.append(", ".join(other_tokens[:3]))

        return ". ".join(parts) if parts else ""

    def _tags_to_pose_phrase(self, tokens: List[str]) -> str:
        """Convert pose/action tags to natural phrase."""
        if not tokens:
            return ""

        # Common pose conversions
        pose_map = {
            "standing": "standing",
            "sitting": "sitting down",
            "walking": "walking",
            "running": "running",
            "looking at viewer": "looking directly at the camera",
            "looking away": "looking away from the camera",
            "hands on hips": "with hands on hips",
            "arms crossed": "with arms crossed",
            "leaning": "leaning",
            "posing": "posing",
        }

        phrases = []
        for token in tokens[:4]:
            token_lower = token.lower()
            for key, phrase in pose_map.items():
                if key in token_lower:
                    phrases.append(phrase)
                    break
            else:
                phrases.append(token)

        if len(phrases) == 1:
            return f"The subject is {phrases[0]}"
        elif len(phrases) > 1:
            return f"The subject is {phrases[0]}, {', '.join(phrases[1:])}"
        return ""

    def _tags_to_environment_phrase(self, tokens: List[str]) -> str:
        """Convert environment tags to natural scene description."""
        if not tokens:
            return ""

        # Classify environment tokens
        indoor_words = ["room", "interior", "indoor", "studio", "office", "home"]
        outdoor_words = ["outdoor", "outside", "street", "nature", "park", "beach", "forest"]
        time_words = ["day", "night", "morning", "evening", "sunset", "dawn", "dusk"]

        indoor_tokens = [t for t in tokens if any(w in t.lower() for w in indoor_words)]
        outdoor_tokens = [t for t in tokens if any(w in t.lower() for w in outdoor_words)]
        time_tokens = [t for t in tokens if any(w in t.lower() for w in time_words)]
        other_tokens = [t for t in tokens if t not in indoor_tokens + outdoor_tokens + time_tokens]

        parts = []

        if indoor_tokens:
            parts.append(f"in a {indoor_tokens[0]}")
        elif outdoor_tokens:
            parts.append(f"in an {outdoor_tokens[0]} setting")
        elif other_tokens:
            parts.append(f"in a {other_tokens[0]} setting")

        if time_tokens:
            parts.append(f"during {time_tokens[0]}")

        # Add additional environment details
        remaining = [t for t in other_tokens if t not in parts][:3]
        if remaining:
            parts.append(f"with {', '.join(remaining)}")

        return "The scene is set " + ", ".join(parts) if parts else ""

    def _tags_to_lighting_phrase(self, tokens: List[str]) -> str:
        """Convert lighting tags to natural description."""
        if not tokens:
            return "The lighting is soft and natural"

        # Lighting descriptors
        light_quality = []
        light_direction = []
        light_color = []

        for token in tokens:
            token_lower = token.lower()
            if any(w in token_lower for w in ["soft", "hard", "diffused", "harsh", "dramatic"]):
                light_quality.append(token)
            elif any(w in token_lower for w in ["side", "back", "front", "rim", "key", "fill"]):
                light_direction.append(token)
            elif any(w in token_lower for w in ["warm", "cool", "golden", "blue", "natural", "artificial"]):
                light_color.append(token)
            else:
                light_quality.append(token)

        parts = ["The lighting is"]

        if light_quality:
            parts.append(light_quality[0])
        if light_color:
            parts.append(f"with {light_color[0]} tones")
        if light_direction:
            parts.append(f"coming from {light_direction[0]}")

        return " ".join(parts) if len(parts) > 1 else "The lighting is natural and balanced"

    def _tags_to_style_phrase(self, tokens: List[str]) -> str:
        """Convert style/quality tags to natural phrase."""
        if not tokens:
            return ""

        # Style categories
        medium = []
        quality = []
        aesthetic = []

        for token in tokens:
            token_lower = token.lower()
            if any(w in token_lower for w in ["photo", "photograph", "painting", "illustration", "render"]):
                medium.append(token)
            elif any(w in token_lower for w in ["8k", "4k", "hd", "detailed", "sharp", "high quality", "masterpiece"]):
                quality.append(token)
            else:
                aesthetic.append(token)

        parts = []

        if medium:
            parts.append(medium[0])
        if aesthetic[:2]:
            parts.append(", ".join(aesthetic[:2]) + " style")
        if quality[:2]:
            parts.append(", ".join(quality[:2]))

        return "Shot as a " + ", ".join(parts) if parts else ""

    def _tags_to_composition_phrase(self, tokens: List[str]) -> str:
        """Convert composition/technical tags to natural phrase."""
        if not tokens:
            return ""

        # Composition elements
        framing = []
        camera = []
        technical = []

        for token in tokens:
            token_lower = token.lower()
            if any(w in token_lower for w in ["close-up", "wide", "medium", "full body", "portrait", "headshot"]):
                framing.append(token)
            elif any(w in token_lower for w in ["lens", "mm", "f/", "aperture", "bokeh", "dof"]):
                camera.append(token)
            else:
                technical.append(token)

        parts = []

        if framing:
            parts.append(f"framed as a {framing[0]}")
        if camera:
            parts.append(f"shot with {camera[0]}")
        if technical[:2]:
            parts.append(", ".join(technical[:2]))

        return "The composition is " + ", ".join(parts) if parts else ""


# Convenience function
def assemble_prompt(
    classified: ClassifiedImage,
    config: Optional[AssemblerConfig] = None
) -> AssembledPrompt:
    """Assemble a prompt from classified tokens.

    Args:
        classified: ClassifiedImage from the classifier
        config: Optional assembler configuration

    Returns:
        AssembledPrompt with the generated prompt
    """
    assembler = StandardAssembler(config)
    return assembler.assemble(classified)
