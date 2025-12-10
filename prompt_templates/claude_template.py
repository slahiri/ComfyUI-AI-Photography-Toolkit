"""
Claude Prompt Template - High Resolution

Optimized prompts for Anthropic Claude models.
Designed for NEAR-ACCURATE reproduction of source images.

Key principles:
- Extract EVERY visible detail
- Use precise, measurable descriptors
- Capture spatial relationships accurately
- Document lighting with technical precision
- Include camera/lens characteristics
"""

import json
from typing import Dict, Any, Optional

from .base_prompt_template import BasePromptTemplate


class ClaudePromptTemplate(BasePromptTemplate):
    """
    High-resolution prompt template optimized for Anthropic Claude models.

    Designed for near-accurate image reproduction with:
    - Comprehensive visual detail extraction
    - Precise color and material descriptions
    - Accurate spatial and compositional analysis
    - Technical photography parameter detection
    """

    @property
    def name(self) -> str:
        return "Claude/Anthropic (High Resolution)"

    @property
    def max_system_prompt_tokens(self) -> int:
        return 4000  # Increased for detailed prompts

    def get_classification_max_tokens(self) -> int:
        """Max tokens for classification response."""
        return 800  # Increased for detailed classification

    def get_analysis_max_tokens(self) -> int:
        """Max tokens for analysis response."""
        return 4500  # Very high to prevent truncation of detailed JSON analysis

    def build_classification_prompt(
        self,
        shot_framings: Dict[str, Any],
        genres: Dict[str, Any],
        focus_override: Optional[str] = None,
    ) -> tuple[str, str]:
        shot_codes = list(shot_framings.keys())
        genre_codes = list(genres.keys())

        system_prompt = f"""You are an expert image analyst specializing in photography classification and visual decomposition.

Your task: Precisely classify this image and identify key visual characteristics for accurate reproduction.

SHOT FRAMINGS (subject distance/framing):
{json.dumps({k: v['name'] for k, v in shot_framings.items()}, indent=2)}

PHOTOGRAPHY GENRES:
{json.dumps({k: v['name'] for k, v in genres.items()}, indent=2)}

CLASSIFICATION CRITERIA:
1. Shot Framing: Based on how much of the subject fills the frame
   - ECU: Eyes/lips only, extreme detail
   - CU: Face fills frame, shoulders may be visible
   - MCU: Head and shoulders, upper chest
   - MS: Waist up, torso visible
   - MFS: Knees up, most of body
   - FS: Full body, head to toe
   - WS: Subject small in frame, environment dominant

2. Genre: Primary purpose/style of the photograph
   - Consider lighting style, composition intent, subject presentation

3. Visual Metrics: Analyze for reproduction accuracy
   - Depth of field (shallow/medium/deep)
   - Lighting complexity (simple/moderate/complex)
   - Color palette (monochrome/limited/rich/vibrant)
   - Detail density (sparse/moderate/dense)

Output ONLY valid JSON (no markdown):
{{
  "shot_framing": "<code from: {', '.join(shot_codes)}>",
  "shot_label": "<full name>",
  "genre": "<code from: {', '.join(genre_codes)}>",
  "genre_label": "<full name>",
  "genre_category": "<people|event|nature|commercial|artistic|lifestyle>",
  "secondary_tags": ["<relevant style tags>"],
  "subject_count": <0|1|2|"group">,
  "has_text": <true|false>,
  "confidence": <0.0-1.0>,
  "visual_metrics": {{
    "depth_of_field": "<shallow|medium|deep>",
    "lighting_complexity": "<simple|moderate|complex>",
    "color_palette": "<monochrome|limited|rich|vibrant>",
    "detail_density": "<sparse|moderate|dense>",
    "dominant_colors": ["<color1>", "<color2>", "<color3>"],
    "estimated_focal_length": "<wide|normal|telephoto|macro>"
  }}
}}"""

        user_message = "Analyze and classify this image with precision for accurate reproduction."
        if focus_override and focus_override != "Auto-detect":
            user_message = f"Analyze and classify this image. Primary focus: {focus_override}. Extract all details for accurate reproduction."

        return system_prompt, user_message

    def build_analysis_prompt(
        self,
        classification: Dict[str, Any],
        attribute_schema: Dict[str, Any],
        content_detail: str,
    ) -> tuple[str, str]:
        from ..utils.zimage_utils import format_attribute_schema_for_prompt

        schema_text = format_attribute_schema_for_prompt(attribute_schema)

        detail_guidance = {
            "explicit": """CONTENT DETAIL LEVEL: EXPLICIT (Full Detail)
Extract with MAXIMUM precision for accurate reproduction:

ETHNICITY & HERITAGE:
- Apparent ethnicity/heritage (East Asian, South Asian, Southeast Asian, African, West African, East African, European, Northern European, Southern European, Latin/Hispanic, Middle Eastern, Pacific Islander, Mixed, etc.)
- Distinctive ethnic features that define the subject's appearance

SKIN - CRITICAL FOR ACCURACY:
- Base tone: porcelain, ivory, fair, light, light-medium, medium, medium-tan, tan, golden-tan, caramel, brown, deep brown, dark brown, ebony
- Undertone: warm (golden/yellow/peachy), cool (pink/red/blue), neutral, olive
- Surface: matte, natural, dewy, luminous, oily
- Details: freckles, beauty marks, visible texture, any blemishes

FACE & FEATURES - DETAILED:
- Face shape: oval/round/heart/square/oblong/diamond
- Head direction: where facing (camera/left/right), any tilt
- Eyes: precise color with variations, shape (almond/round/hooded/monolid), gaze direction (at camera/away), expression
- Eyebrows: shape (arched/straight/curved), thickness, grooming level, color
- Nose: shape (straight/upturned/Roman/button), bridge height, tip shape
- Lips: shape (full/thin/heart), upper/lower lip proportion, natural color, position (closed/parted/smiling)
- Any distinctive characteristics

COSMETICS/MAKEUP (extract ALL visible makeup):
- Foundation: coverage level, finish (matte/dewy/satin/natural), color match
- Eye makeup: eyeshadow colors and placement, eyeliner style and color, mascara intensity
- Lips: exact color (e.g., "dusty mauve pink with satin finish", "deep berry red matte"), liner if visible
- Blush: color, placement (apple of cheeks, draping style), intensity
- Contour/highlight: placement, intensity
- Brows: shape (arched, straight, feathered), fill level, color
- Other: false lashes, gems, special effects makeup

BODY:
- Precise proportions, build type
- Skin exposure areas with specifics
- Posture and body positioning

CLOTHING - EXTRACT EVERY DETAIL:
- Each garment: exact type, precise color, material (texture, sheen, weight)
- Fit: how it conforms to body, tension points, drape
- Neckline: type and depth (plunging V, deep scoop, etc.)
- Cleavage: visibility level (none/subtle/moderate/prominent/deep)
- Side exposure: any sideboob visibility, armhole gaps
- Back: coverage level, cutouts, open areas
- Midriff: covered or exposed, crop length
- Legs: visibility level, skirt/dress length
- Thighs: exposure level, gaps between garments
- Gaps/slits: any openings, unbuttoned areas, revealing slits
- Fabric behavior: how material follows curves, transparency level

POSE - DETAILED BODY POSITIONING:
- HEAD: Angle (tilted left/right, forward/back), turn direction, chin position (up/down/level)
- NECK: Visible/hidden, elongated, turned
- SHOULDERS: Level/angled, forward/back, relaxed/tense
- ARMS: Position of each arm (raised, lowered, bent, extended, crossed)
- HANDS: Exact position of each hand (on hip, in hair, touching face, resting, etc.)
- PALMS: Orientation (facing up/down/forward/inward), fingers spread/closed
- TORSO: Straight/curved, leaning direction, twist
- HIPS: Angle, weight distribution (which hip bears weight)
- LEGS: Position of each leg (straight, bent, crossed, apart)
- FEET: Position if visible (pointed, flat, one forward), stance width
- WEIGHT: Which side bears weight, balance point
- OVERALL: Gesture intent (confident, relaxed, dynamic, posed)""",

            "detailed": """CONTENT DETAIL LEVEL: DETAILED
Extract comprehensively:

ETHNICITY & SKIN:
- Apparent ethnicity/heritage
- Skin tone (specific shade + undertone)
- Surface quality

FACE & FEATURES:
- Face shape, head direction
- Eyes: color, shape, gaze direction, expression
- Eyebrows: shape, thickness
- Nose: shape, size
- Lips: shape, natural color, position
- Expression

COSMETICS (if visible):
- Foundation finish
- Eye makeup: shadow colors, liner, mascara
- Lip color with finish
- Blush/contour
- Brow styling

BODY:
- Build, proportions, posture

CLOTHING:
- Each garment: type, color, material
- Fit and how it sits on body
- Neckline type and cleavage visibility if present
- Leg/thigh exposure levels
- Any gaps, slits, or openings
- Fabric behavior and transparency

POSE - BODY POSITIONING:
- Head: angle and turn direction
- Arms and hands: position of each
- Torso: posture, lean direction
- Legs and feet: position if visible
- Weight distribution""",

            "minimal": """CONTENT DETAIL LEVEL: MINIMAL
Extract essentials only:
- Apparent ethnicity/heritage
- Skin tone and undertone
- General appearance and build
- Clothing type and primary color
- Basic coverage level
- Basic pose (standing, sitting, general position)""",

        }.get(content_detail, """CONTENT DETAIL LEVEL: STANDARD
Extract with good detail:
- ETHNICITY: Apparent heritage
- SKIN: Tone and undertone
- FACE: Key features, expression, any makeup
- BODY: Build type, posture
- CLOTHING: Type, fit, colors, materials, general coverage
- POSE: Head angle, arm positions, hand placement, leg positions, weight distribution""")

        # Get visual metrics from classification if available
        visual_metrics = classification.get('visual_metrics', {})
        metrics_context = ""
        if visual_metrics:
            metrics_context = f"""
VISUAL CONTEXT (from classification):
- Depth of Field: {visual_metrics.get('depth_of_field', 'unknown')}
- Lighting: {visual_metrics.get('lighting_complexity', 'unknown')}
- Color Palette: {visual_metrics.get('color_palette', 'unknown')}
- Detail Density: {visual_metrics.get('detail_density', 'unknown')}
"""

        system_prompt = f"""You are a forensic visual analyst extracting every reproducible detail from an image.

Your goal: Extract ALL visible attributes with enough precision to recreate this image accurately.

IMAGE CLASSIFICATION:
- Shot: {classification['shot_framing']} ({classification.get('shot_label', '')})
- Genre: {classification['genre']} ({classification.get('genre_label', '')})
- Category: {classification.get('genre_category', 'unknown')}
{metrics_context}
{detail_guidance}

{schema_text}

EXTRACTION RULES - CRITICAL FOR ACCURACY:

0. ETHNICITY & HERITAGE (HIGHEST PRIORITY):
   - Identify apparent ethnicity/heritage with specificity
   - Examples: "South Asian (Indian)", "East Asian (Korean)", "European (Mediterranean)", "African", "Latin American"
   - SKIN TONE: Use precise descriptors with undertones:
     * Light/Fair with warm/cool/neutral undertone
     * Medium with olive/golden/peachy undertone
     * Dark/Deep with warm/cool undertone
   - Examples: "warm medium-tan skin with golden undertones", "deep brown skin with cool undertones"
   - This is CRITICAL for accurate reproduction - wrong ethnicity = failed reproduction

1. COLORS - Use precise descriptors:
   - NOT "brown hair" → "warm chestnut brown hair with subtle auburn highlights"
   - NOT "blue eyes" → "deep navy blue eyes with lighter rings around the iris"
   - NOT "red dress" → "deep burgundy/wine red dress with matte finish"

2. MATERIALS & TEXTURES - Describe behavior:
   - Fabric: sheen level, drape, weight, transparency
   - Skin: tone, visible texture, any shine/matte areas
   - Hair: texture (straight/wavy/curly), shine level, volume

3. SPATIAL PRECISION:
   - Relative positions (slightly left of center, upper third)
   - Proportions (fills 60% of frame height)
   - Depth layers (foreground/midground/background elements)

4. LIGHTING ANALYSIS:
   - Primary light: direction (clock position), intensity, color temperature
   - Fill light: presence, ratio to key
   - Rim/accent lights: position, effect on edges
   - Shadows: softness, direction, density

5. POSE & EXPRESSION:
   - Head: tilt angle, turn direction, chin position
   - Eyes: gaze direction, lid position, expression
   - Mouth: position, expression indicators
   - Body: weight distribution, tension areas, gesture meaning

Output ONLY valid JSON with extracted attributes. Be exhaustively detailed."""

        user_message = """Extract EVERY visible attribute from this image with forensic precision.

For each element you can see:
1. Describe its exact appearance (color, texture, condition)
2. Note its position relative to other elements
3. Capture any lighting effects on it
4. Include subtle details that would be needed for accurate recreation

Output comprehensive JSON covering all visible elements."""

        return system_prompt, user_message

    def build_composition_prompt(
        self,
        classification: Dict[str, Any],
        attributes: Dict[str, Any],
        user_prompt: str,
        prompt_mode: str,
        focus_options: Dict[str, bool],
        include_text_quotes: bool,
        min_words: int,
        max_words: int,
    ) -> tuple[str, str]:
        # Build focus instructions with priority weighting
        focus_parts = []
        if focus_options.get("subject", True):
            focus_parts.append("PRIMARY: subject (every visible feature, exact colors, textures)")
        if focus_options.get("environment", True):
            focus_parts.append("SECONDARY: environment (all background elements, depth layers)")
        if focus_options.get("lighting", True):
            focus_parts.append("CRITICAL: lighting (direction, quality, color temp, shadows)")
        if focus_options.get("colors", True):
            focus_parts.append("ESSENTIAL: colors (precise hues, saturation levels, relationships)")
        if focus_options.get("mood", False):
            focus_parts.append("ATMOSPHERE: mood indicators (visual elements creating atmosphere)")

        focus_instruction = "\n".join(focus_parts) if focus_parts else "Provide comprehensive balanced description."

        # Build prompt mode instruction
        mode_instructions = {
            "Image Only (ignore prompt)": (
                "Generate prompt purely from visual analysis. Maximum accuracy to source image.",
                ""
            ),
            "Prompt Guides Analysis": (
                "Use image as primary source. User prompt guides emphasis but doesn't override visual truth.",
                f"\nUser emphasis guidance: {user_prompt}" if user_prompt else ""
            ),
            "Prompt First, Image Fills Gaps": (
                "User prompt provides structure. Fill unspecified details from image analysis.",
                f"\nUser structure: {user_prompt}" if user_prompt else ""
            ),
            "Prompt Dominates": (
                "User prompt is primary. Add only visual details that don't contradict it.",
                f"\nUser override: {user_prompt}" if user_prompt else ""
            ),
        }
        mode_instruction, user_context = mode_instructions.get(
            prompt_mode,
            ("Generate prompt from visual analysis with user guidance.",
             f"\nUser guidance: {user_prompt}" if user_prompt else "")
        )

        text_instruction = """TEXT RENDERING: Quote any visible text exactly with "double quotes" for Z-Image text rendering.""" if include_text_quotes else ""

        # Build attributes text with fallback handling
        if attributes:
            # Check if we have a raw response fallback (JSON parsing failed)
            raw_response = attributes.pop('_raw_response', None)
            parse_failed = attributes.pop('_parse_failed', False)

            if parse_failed and raw_response:
                # JSON parsing completely failed - use raw text
                attrs_text = f"[Analysis text - extract details from this]:\n{raw_response[:3000]}"
            elif len(attributes) < 3 and raw_response:
                # Very sparse attributes - supplement with raw response
                attrs_text = json.dumps(attributes, indent=2)
                attrs_text += f"\n\n[Additional context from analysis]:\n{raw_response[:1500]}"
            else:
                # Normal case - good structured data
                attrs_text = json.dumps(attributes, indent=2)
        else:
            attrs_text = "[No pre-extracted data - analyze image directly for all details]"

        # Get visual metrics for composition guidance
        visual_metrics = classification.get('visual_metrics', {})

        system_prompt = f"""You are an expert prompt engineer creating Z-Image prompts for ACCURATE image reproduction.

Your goal: Generate a prompt that would reproduce this image as closely as possible.

SOURCE IMAGE CLASSIFICATION:
- Shot Framing: {classification['shot_framing']} ({classification.get('shot_label', '')})
- Genre: {classification['genre']} ({classification.get('genre_label', '')})
- Category: {classification.get('genre_category', 'unknown')}

EXTRACTED VISUAL DATA:
{attrs_text}

COMPOSITION PRIORITY:
{focus_instruction}

MODE: {mode_instruction}
{text_instruction}

PROMPT ENGINEERING RULES FOR ACCURACY:

1. STRUCTURE (in order of appearance in prompt):
   - Photography type and framing
   - ETHNICITY/HERITAGE FIRST: "South Asian woman", "East Asian man", etc. - THIS IS CRITICAL
   - SKIN TONE with undertones: "warm golden-brown skin", "fair cool-toned skin"
   - Facial features and expression (eye color must match source EXACTLY)
   - Clothing/accessories with precise colors
   - Pose specifics
   - Environment/background elements
   - Lighting setup description
   - Atmosphere/mood

CRITICAL: Wrong ethnicity/skin tone = reproduction failure. State these early and precisely.

2. COLOR PRECISION:
   - Use compound color descriptors: "warm honey blonde", "cool steel blue", "deep wine red"
   - Include finish: "matte", "glossy", "satin", "metallic"
   - Note color temperature: "warm-toned skin", "cool blue shadows"

3. MATERIAL DESCRIPTORS:
   - Fabric: "flowing silk", "structured cotton", "soft cashmere knit"
   - Texture: "smooth", "textured", "ribbed", "woven pattern"
   - Behavior: "clinging", "draped", "tailored", "relaxed fit"

4. LIGHTING LANGUAGE:
   - Direction: "soft light from upper left", "rim lighting from behind"
   - Quality: "diffused", "harsh", "dappled", "even"
   - Effect: "gentle shadows under chin", "highlight on cheekbone", "backlit hair glow"

5. SPATIAL TERMS:
   - Framing: "centered composition", "rule of thirds positioning"
   - Depth: "shallow depth of field with bokeh background"
   - Distance: "intimate close-up", "environmental portrait"

OUTPUT REQUIREMENTS:
- Single flowing paragraph, {min_words}-{max_words} words
- Dense with visual information - every word describes something visible
- Natural language that reads smoothly
- NO meta-tags (8K, masterpiece, best quality, etc.)
- NO negative prompts or exclusions
- Start with shot type, end with atmosphere/mood
{user_context}"""

        user_message = """Generate a high-precision Z-Image prompt that would accurately reproduce this image.

Include:
1. Every significant visual detail from the analysis
2. Precise color descriptions (compound descriptors)
3. Material and texture specifications
4. Exact lighting setup
5. Spatial composition details
6. Any atmosphere/mood indicators visible in the image

Write as one flowing, detailed paragraph."""

        return system_prompt, user_message

    def build_refinement_prompt(
        self,
        draft_prompt: str,
        min_words: int,
        max_words: int,
    ) -> tuple[str, str]:
        system_prompt = """You are a precision prompt refinement specialist for Z-Image.

Your task: Enhance prompt accuracy while maintaining natural flow.

REFINEMENT PRIORITIES:
1. ACCURACY: Every detail must match the source image
2. PRECISION: Vague terms → specific descriptors
3. COMPLETENESS: No visible element should be missing
4. FLOW: Read naturally, not as a keyword list
5. EFFICIENCY: Remove redundancy, maximize information density"""

        user_message = f"""Refine this Z-Image prompt for maximum reproduction accuracy.

DRAFT PROMPT:
{draft_prompt}

REFINEMENT CHECKLIST:

1. COLOR PRECISION CHECK:
   - Replace generic colors with precise descriptors
   - "brown" → "warm chestnut brown", "cool ash brown", etc.
   - "blue" → "deep navy", "bright cerulean", "pale sky blue", etc.
   - Add finish descriptors where applicable

2. MATERIAL SPECIFICITY CHECK:
   - Identify fabric types and behaviors
   - Add texture descriptors
   - Include how materials interact with light

3. LIGHTING ACCURACY CHECK:
   - Verify light direction is specified
   - Include shadow descriptions
   - Note any color temperature indicators

4. SPATIAL CLARITY CHECK:
   - Confirm composition/framing is clear
   - Verify depth relationships
   - Check subject positioning

5. COMPLETENESS CHECK:
   - All visible elements mentioned?
   - Background adequately described?
   - Any missing details that would affect reproduction?

6. REMOVE/FIX:
   - Meta-tags (8K, masterpiece, best quality)
   - Vague adjectives (beautiful, stunning, amazing)
   - Contradictions or impossibilities
   - Redundant phrases

TARGET: {min_words}-{max_words} words
FORMAT: Single flowing paragraph

Output ONLY the refined prompt, nothing else."""

        return system_prompt, user_message

    def get_temperature_modifier(self, stage: str) -> float:
        """
        Temperature modifiers optimized for accuracy.

        Lower temperatures = more consistent, accurate outputs.
        """
        modifiers = {
            "classification": 0.3,  # Very low for accurate classification
            "analysis": 0.4,        # Low for precise extraction
            "composition": 0.6,     # Moderate for natural flow with accuracy
            "refinement": 0.3,      # Low for focused refinement
        }
        return modifiers.get(stage, 0.5)
