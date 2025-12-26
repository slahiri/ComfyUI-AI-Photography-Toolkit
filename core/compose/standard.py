"""Standard (NLP) prompt composition with structured output schema."""

import re
from typing import Dict, List, Optional, Tuple

from .base import (
    BaseComposeGenerator, ComposeConfig, ComposeResult, ComposeMode,
    ComposeAnalytics
)


class StandardGenerator(BaseComposeGenerator):
    """
    NLP-based prompt generator with structured output schema.

    Output Schema (canonical format for image generation):
    1. Quality Boosters - Technical quality terms
    2. Subject - Who/what is the main subject
    3. Subject Details - Appearance, clothing, accessories
    4. Action/Pose - What they're doing
    5. Environment/Scene - Setting, location, background
    6. Lighting - Light quality, direction, mood
    7. Style/Medium - Artistic style, medium
    8. Composition - Framing, camera angle
    9. Technical Parameters - Camera/lens settings

    Strategy:
    - Use caption (VLM or Florence) as BASE - always available from Analysis
    - Parse caption to extract content for each section
    - Enhance sections with tags that add missing details
    - Output full descriptive sentences per section
    """

    # Gender detection
    FEMALE_TAGS = {'girl', 'woman', 'female', '1girl', 'lady', 'she'}
    MALE_TAGS = {'boy', 'man', 'male', '1boy', 'he', 'guy'}

    # Quality boosters vocabulary
    QUALITY_TERMS = [
        'masterpiece', 'best quality', 'high resolution', 'highly detailed',
        'professional', 'stunning', 'beautiful', 'elegant', 'exquisite',
        '8k', '4k', 'uhd', 'high definition', 'sharp focus', 'crisp',
    ]

    # Tags to skip as too generic
    SKIP_TAGS = {
        'lips', 'nose', 'face', 'skin', 'eyes', 'solo', '1girl', '1boy',
        'realistic', 'photorealistic', 'photo', 'photograph',
        'bare arms', 'bare shoulders', 'arms',
    }

    def generate(self, metadata: Dict) -> ComposeResult:
        """Generate structured prompt following canonical schema."""
        analytics = ComposeAnalytics()

        # 1. Get base caption (VLM preferred, Florence fallback)
        base_caption = self._get_best_caption(metadata)
        base_caption = self.filter_philosophical(base_caption)
        analytics.florence_used = bool(base_caption)
        if base_caption:
            analytics.florence_sections_used = ["base_caption"]

        # 2. Extract and classify tags
        tags, tag_sources = self.extract_tags_with_tracking(metadata, analytics)
        classified = self.classify_tags(tags)

        for category, category_tags in classified.items():
            if category_tags:
                analytics.categories_used[category] = category_tags

        # 3. Detect gender for pronouns
        gender = self._detect_gender(tags, classified)

        # 4. Extract sections from caption and enhance with tags
        sections = self._extract_sections(base_caption, classified, tags, metadata, gender)

        # 5. Build final prompt in canonical order
        prompt = self._build_canonical_prompt(sections)

        # Finalize analytics
        analytics.finalize()

        return ComposeResult(
            prompt=prompt,
            mode=ComposeMode.STANDARD,
            style=self.style_config.style_name,
            categories_used=list(analytics.categories_used.keys()),
            analytics=analytics,
        )

    def _get_best_caption(self, metadata: Dict) -> str:
        """Get caption based on caption_source setting."""
        source = self.caption_source

        if source == "Tags Only":
            return ''

        if source == "Synthesis (VLM)":
            return metadata.get('vlm_description', '')

        if source == "Analysis (Florence)":
            return self._get_florence_caption(metadata)

        # Auto - VLM > Florence
        vlm_desc = metadata.get('vlm_description', '')
        if vlm_desc and len(vlm_desc) > 100:
            return vlm_desc

        return self._get_florence_caption(metadata)

    def _get_florence_caption(self, metadata: Dict) -> str:
        """Get best Florence caption from metadata."""
        for key in ['florence_description', 'florence_mixed_caption_plus',
                    'florence_mixed_caption', 'florence_caption']:
            caption = metadata.get(key, '')
            if caption and len(caption) > 50:
                if '\n\n' in caption:
                    caption = caption.split('\n\n')[0]
                return caption
        return ''

    def _detect_gender(self, tags: Dict[str, float], classified: Dict[str, List[str]]) -> str:
        """Detect gender from tags."""
        all_text = ' '.join(list(tags.keys()) + classified.get('subject', []))
        all_lower = all_text.lower()

        if any(f in all_lower for f in self.FEMALE_TAGS):
            return 'female'
        if any(m in all_lower for m in self.MALE_TAGS):
            return 'male'
        return 'neutral'

    def _get_pronouns(self, gender: str) -> Tuple[str, str, str]:
        """Get pronouns based on gender."""
        if gender == 'female':
            return ('She', 'Her', 'her')
        elif gender == 'male':
            return ('He', 'His', 'him')
        return ('They', 'Their', 'them')

    def _extract_sections(self, caption: str, classified: Dict[str, List[str]],
                          tags: Dict[str, float], metadata: Dict, gender: str) -> Dict[str, str]:
        """Extract content into canonical schema sections."""
        sections = {
            'quality': '',
            'subject': '',
            'subject_details': '',
            'action': '',
            'environment': '',
            'lighting': '',
            'style': '',
            'composition': '',
            'technical': '',
        }

        caption_lower = caption.lower() if caption else ''
        subj, poss, _ = self._get_pronouns(gender)

        # === QUALITY BOOSTERS ===
        quality_parts = []
        # Check if quality terms are in caption or add defaults
        for term in self.QUALITY_TERMS:
            if term in caption_lower:
                quality_parts.append(term)
        if not quality_parts:
            quality_parts = ['professional photograph', 'high quality', 'detailed']
        sections['quality'] = ', '.join(quality_parts)

        # === SUBJECT ===
        subject_parts = []
        # Extract subject from caption
        subject_match = re.search(
            r'\b((?:a |an )?(?:young |mature |elderly |beautiful |attractive )?'
            r'(?:asian |indian |european |african |latin |japanese |korean |chinese )?'
            r'(?:woman|girl|man|boy|lady|gentleman|person))',
            caption, re.I
        )
        if subject_match:
            subject_parts.append(subject_match.group(1).strip())
        else:
            # Build from tags
            ethnicity = classified.get('ethnicity', [])
            subject = classified.get('subject', [])
            parts = []
            if ethnicity:
                parts.append(ethnicity[0])
            if subject:
                s = subject[0]
                if s.startswith('1'):
                    s = s[1:]
                parts.append(s)
            if parts:
                subj_str = ' '.join(parts)
                article = 'An' if subj_str[0].lower() in 'aeiou' else 'A'
                subject_parts.append(f"{article} {subj_str}")
            else:
                subject_parts.append("A person")

        sections['subject'] = subject_parts[0] if subject_parts else ''

        # === SUBJECT DETAILS ===
        details_parts = []

        # Hair from caption or tags
        hair_match = re.search(r'((?:long|short|medium|curly|straight|wavy)?\s*(?:black|brown|blonde|red|white|gray)\s*hair)', caption, re.I)
        if hair_match:
            details_parts.append(f"{subj} has {hair_match.group(1).strip()}")
        elif classified.get('hair'):
            details_parts.append(f"{subj} has {self._join_natural(classified['hair'])}")

        # Eyes from caption or tags
        eyes_match = re.search(r'((?:brown|blue|green|hazel|dark|light)\s*eyes)', caption, re.I)
        if eyes_match:
            details_parts.append(f"{poss.lower()} eyes are {eyes_match.group(1).strip()}")
        elif classified.get('eyes'):
            details_parts.append(f"{poss.lower()} eyes are {classified['eyes'][0]}")

        # Skin/complexion from caption
        skin_match = re.search(r'((?:fair|pale|tan|olive|dark|brown|warm|smooth)\s*(?:skin|complexion|skin tone))', caption, re.I)
        if skin_match:
            details_parts.append(f"{subj} has {skin_match.group(1).strip()}")

        # Clothing from caption or tags
        clothing_match = re.search(r'(?:wearing|dressed in|in)\s+(?:a\s+)?([^,\.]+(?:dress|saree|sari|outfit|suit|shirt|blouse|gown|robe)[^,\.]*)', caption, re.I)
        if clothing_match:
            details_parts.append(f"{subj} is wearing {clothing_match.group(1).strip()}")
        elif classified.get('clothing'):
            details_parts.append(f"{subj} is wearing {self._join_natural(classified['clothing'])}")

        # Accessories from caption or tags
        acc_match = re.search(r'(?:wearing|with|adorned with)\s+(?:a\s+)?([^,\.]*(?:jewelry|necklace|bracelet|earring|ring|watch)[^,\.]*)', caption, re.I)
        if acc_match:
            details_parts.append(f"{subj} is adorned with {acc_match.group(1).strip()}")
        elif classified.get('accessories'):
            acc_not_in_caption = [a for a in classified['accessories'] if a.lower() not in caption_lower]
            if acc_not_in_caption:
                details_parts.append(f"{subj} is adorned with {self._join_natural(acc_not_in_caption)}")

        sections['subject_details'] = '. '.join(details_parts) + '.' if details_parts else ''

        # === ACTION/POSE ===
        action_parts = []
        action_match = re.search(
            r'(?:is\s+)?((?:standing|sitting|posing|lying|walking|leaning|looking|reclining|kneeling)[^,\.]*)',
            caption, re.I
        )
        if action_match:
            action_parts.append(f"{subj} is {action_match.group(1).strip()}")
        elif classified.get('pose'):
            action_parts.append(f"{subj} is {classified['pose'][0]}")

        # Expression
        expr_match = re.search(r'((?:smiling|serious|pensive|confident|relaxed|sensual|playful)\s*(?:expression)?)', caption, re.I)
        if expr_match:
            action_parts.append(f"{subj} has a {expr_match.group(1).strip()}")
        elif classified.get('expression'):
            action_parts.append(f"{subj} has a {classified['expression'][0]} expression")

        # Gaze
        gaze_match = re.search(r'(looking\s+(?:at the camera|directly|away|to the side|down|up)[^,\.]*)', caption, re.I)
        if gaze_match:
            action_parts.append(gaze_match.group(1).strip())
        elif classified.get('gaze'):
            action_parts.append(classified['gaze'][0])

        sections['action'] = '. '.join(action_parts) + '.' if action_parts else ''

        # === ENVIRONMENT/SCENE ===
        env_parts = []
        # Location from caption
        env_match = re.search(
            r'(?:in|at|against|before)\s+(?:a\s+)?([^,\.]*(?:room|studio|outdoor|indoor|background|setting|location|beach|forest|city|street|garden)[^,\.]*)',
            caption, re.I
        )
        if env_match:
            env_parts.append(f"The scene is set in {env_match.group(1).strip()}")
        elif classified.get('environment'):
            env_parts.append(f"The scene is set {classified['environment'][0]}")

        # Background details
        bg_match = re.search(r'(?:background|backdrop)\s+(?:is\s+)?([^,\.]+)', caption, re.I)
        if bg_match:
            env_parts.append(f"The background features {bg_match.group(1).strip()}")

        sections['environment'] = '. '.join(env_parts) + '.' if env_parts else ''

        # === LIGHTING ===
        light_parts = []
        light_match = re.search(
            r'((?:soft|warm|dim|bright|natural|harsh|dramatic|low[- ]?key|high[- ]?key|ambient|diffused)[^,\.]*light[^,\.]*)',
            caption, re.I
        )
        if light_match:
            light_parts.append(f"The lighting is {light_match.group(1).strip()}")
        elif classified.get('lighting'):
            light_parts.append(f"The lighting features {self._join_natural(classified['lighting'])}")

        # From photography analysis
        photo = self.get_photography_info(metadata)
        if photo:
            if photo.get('low key', 0) > 0.7 or photo.get('low_key', 0) > 0.7:
                if 'low key' not in caption_lower and 'low-key' not in ' '.join(light_parts).lower():
                    light_parts.append("Low-key lighting creates dramatic shadows")
            if photo.get('warm', 0) > 0.7 or photo.get('warm tones', 0) > 0.7:
                if 'warm' not in caption_lower and 'warm' not in ' '.join(light_parts).lower():
                    light_parts.append("Warm tones enhance the atmosphere")

        # Atmosphere/mood
        if classified.get('atmosphere'):
            light_parts.append(f"The mood is {classified['atmosphere'][0]}")

        sections['lighting'] = '. '.join(light_parts) + '.' if light_parts else ''

        # === STYLE/MEDIUM ===
        style_parts = []
        style_match = re.search(
            r'((?:photograph|portrait|cinematic|artistic|editorial|fashion|fine art)[^,\.]*)',
            caption, re.I
        )
        if style_match:
            style_parts.append(style_match.group(1).strip().capitalize())
        elif classified.get('style'):
            style_parts.append(classified['style'][0].capitalize())
        else:
            style_parts.append("Professional photograph")

        sections['style'] = '. '.join(style_parts) + '.' if style_parts else ''

        # === COMPOSITION ===
        comp_parts = []
        # Camera angle/shot type
        shot_match = re.search(
            r'((?:close[- ]?up|medium shot|full body|cowboy shot|portrait|wide shot|headshot)[^,\.]*)',
            caption, re.I
        )
        if shot_match:
            comp_parts.append(shot_match.group(1).strip().capitalize())
        elif classified.get('camera'):
            comp_parts.append(classified['camera'][0].capitalize())

        # From composition analysis
        comp_info = self.get_composition_info(metadata)
        if comp_info:
            if comp_info.get('centered subject', 0) > 0.7 or comp_info.get('centered_subject', 0) > 0.7:
                comp_parts.append("Subject centered in frame")
            if comp_info.get('rule of thirds', 0) > 0.6:
                comp_parts.append("Rule of thirds composition")

        sections['composition'] = '. '.join(comp_parts) + '.' if comp_parts else ''

        # === TECHNICAL PARAMETERS ===
        tech_parts = []
        # Depth of field, bokeh, focus
        if 'bokeh' in caption_lower or classified.get('focus'):
            focus_terms = []
            if 'bokeh' in caption_lower:
                focus_terms.append('creamy bokeh')
            if classified.get('focus'):
                focus_terms.extend([f for f in classified['focus'] if f.lower() not in caption_lower])
            if focus_terms:
                tech_parts.append(f"Shallow depth of field with {self._join_natural(focus_terms)}")

        # Sharp focus
        if 'sharp' in caption_lower or 'crisp' in caption_lower:
            tech_parts.append("Tack-sharp focus on subject")

        sections['technical'] = '. '.join(tech_parts) + '.' if tech_parts else ''

        return sections

    def _build_canonical_prompt(self, sections: Dict[str, str]) -> str:
        """Build final prompt in canonical order with full sentences."""
        parts = []

        # 1. Quality Boosters
        if sections['quality']:
            parts.append(sections['quality'])

        # 2. Subject
        if sections['subject']:
            parts.append(sections['subject'])

        # 3. Subject Details
        if sections['subject_details']:
            parts.append(sections['subject_details'])

        # 4. Action/Pose
        if sections['action']:
            parts.append(sections['action'])

        # 5. Environment/Scene
        if sections['environment']:
            parts.append(sections['environment'])

        # 6. Lighting
        if sections['lighting']:
            parts.append(sections['lighting'])

        # 7. Style/Medium
        if sections['style']:
            parts.append(sections['style'])

        # 8. Composition
        if sections['composition']:
            parts.append(sections['composition'])

        # 9. Technical Parameters
        if sections['technical']:
            parts.append(sections['technical'])

        # Join with spaces (each section already ends with period)
        prompt = ' '.join(parts)

        # Clean up any double periods or spaces
        prompt = re.sub(r'\.+', '.', prompt)
        prompt = re.sub(r'\s+', ' ', prompt)

        return prompt.strip()
