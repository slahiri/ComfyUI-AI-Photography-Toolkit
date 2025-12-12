"""
SID_ZImageOptimizer Node

Optimizes prompts specifically for Z-Image/Z-Image-Turbo model.
Uses the Z-Image vocabulary system to:
- Convert tag soup to natural language
- Replace generic terms with Z-Image optimized vocabulary
- Remove anti-patterns (quality boosters Z-Image ignores)
- Inject missing essential elements (lighting, composition)
- Validate prompt structure

No LLM required - pure Python vocabulary-based optimization.
"""

from comfy_api.latest import io as comfy_io

from .zimage_vocabulary import (
    ZImageVocabulary,
    VocabCategory,
    get_vocabulary,
    ANTI_PATTERNS,
)
from .zimage_prompt_translator import (
    ZImagePromptTranslator,
    ZImagePromptBuilder,
    PromptStyle,
    get_translator,
    get_builder,
)


# =============================================================================
# Z-Image Prompt Optimizer Node
# =============================================================================

@comfy_io.Title("SID Z-Image Prompt Optimizer")
@comfy_io.Description(
    "Optimizes prompts for Z-Image/Z-Image-Turbo model. "
    "Converts tag soup to natural language, applies Z-Image vocabulary, "
    "removes anti-patterns, and injects missing elements. No LLM required."
)
@comfy_io.Category("SID/Photography")
class SID_ZImageOptimizer(comfy_io.ComfyNode):
    """
    Optimizes prompts for Z-Image model using vocabulary-based translation.
    """

    @classmethod
    def define_schema(cls):
        return comfy_io.Schema(
            node_id="SID_ZImageOptimizer",
            display_name="SID Z-Image Prompt Optimizer",
            description="Optimize prompts for Z-Image model",
            category="SID/Photography",
            inputs=[
                comfy_io.String.Input(
                    "prompt",
                    display_name="Prompt",
                    multiline=True,
                    default="",
                    tooltip="The prompt to optimize for Z-Image"
                ),
                comfy_io.Combo.Input(
                    "optimization_level",
                    display_name="Optimization Level",
                    options=["Light", "Standard", "Aggressive"],
                    default="Standard",
                    tooltip="Light: Synonym replacement only. Standard: Full optimization. Aggressive: Restructure prompt."
                ),
                comfy_io.Boolean.Input(
                    "add_safety",
                    display_name="Add Safety Phrases",
                    default=False,
                    tooltip="Add SFW safety phrases to the prompt"
                ),
                comfy_io.Combo.Input(
                    "safety_level",
                    display_name="Safety Level",
                    options=["Basic", "Full", "Modest Clothing", "Anatomy Fix"],
                    default="Basic",
                    tooltip="Level of safety phrases to add"
                ),
                comfy_io.Boolean.Input(
                    "inject_lighting",
                    display_name="Inject Lighting",
                    default=True,
                    tooltip="Add lighting description if missing (Z-Image responds strongly to lighting)"
                ),
                comfy_io.Boolean.Input(
                    "inject_composition",
                    display_name="Inject Composition",
                    default=True,
                    tooltip="Add shot type if missing for portraits"
                ),
                comfy_io.Boolean.Input(
                    "show_changes",
                    display_name="Show Changes",
                    default=True,
                    tooltip="Print changes made to console"
                ),
            ],
            outputs=[
                comfy_io.String.Output(
                    "optimized_prompt",
                    display_name="Optimized Prompt",
                    tooltip="The Z-Image optimized prompt"
                ),
                comfy_io.String.Output(
                    "original_prompt",
                    display_name="Original Prompt",
                    tooltip="The original input prompt"
                ),
                comfy_io.String.Output(
                    "changes_report",
                    display_name="Changes Report",
                    tooltip="Summary of changes made"
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        prompt: str,
        optimization_level: str,
        add_safety: bool,
        safety_level: str,
        inject_lighting: bool,
        inject_composition: bool,
        show_changes: bool,
    ) -> comfy_io.NodeOutput:
        """Execute the Z-Image optimization."""

        if not prompt or not prompt.strip():
            return comfy_io.NodeOutput("", "", "No prompt provided")

        prompt = prompt.strip()
        translator = get_translator()

        # Map safety level
        safety_map = {
            "Basic": "sfw_basic",
            "Full": "sfw_full",
            "Modest Clothing": "modest_clothing",
            "Anatomy Fix": "anatomy_fix",
        }
        safety_key = safety_map.get(safety_level, "sfw_basic")

        # Translate prompt
        result = translator.translate(
            prompt,
            add_safety=add_safety,
            safety_level=safety_key,
        )

        # Build changes report
        report_lines = [
            f"Style Detected: {result.style_detected.value}",
            f"Word Count: {result.word_count} ({'optimal' if result.is_optimal_length else 'suboptimal'})",
            "",
            "Changes Made:",
        ]
        if result.changes_made:
            for change in result.changes_made:
                report_lines.append(f"  - {change}")
        else:
            report_lines.append("  - No changes needed")

        if result.suggestions:
            report_lines.append("")
            report_lines.append("Suggestions:")
            for suggestion in result.suggestions[:5]:  # Limit to 5
                report_lines.append(f"  - {suggestion}")

        report = "\n".join(report_lines)

        # Print to console if requested
        if show_changes:
            print("\n" + "=" * 60)
            print("[Z-Image Optimizer] Optimization Report")
            print("=" * 60)
            print(f"Original ({len(prompt.split())} words):")
            print(f"  {prompt[:200]}{'...' if len(prompt) > 200 else ''}")
            print(f"\nOptimized ({result.word_count} words):")
            print(f"  {result.translated[:200]}{'...' if len(result.translated) > 200 else ''}")
            print(f"\n{report}")
            print("=" * 60 + "\n")

        return comfy_io.NodeOutput(
            result.translated,
            prompt,
            report,
        )


# =============================================================================
# Z-Image Portrait Builder Node
# =============================================================================

@comfy_io.Title("SID Z-Image Portrait Builder")
@comfy_io.Description(
    "Build portrait prompts using Z-Image optimized vocabulary. "
    "Select from pre-defined lighting, lens, and composition options."
)
@comfy_io.Category("SID/Photography")
class SID_ZImagePortraitBuilder(comfy_io.ComfyNode):
    """
    Build portrait prompts using Z-Image vocabulary templates.
    """

    # Get vocabulary for options
    _vocab = get_vocabulary()

    # Build option lists from vocabulary
    SHOT_OPTIONS = ["auto"] + list({
        "extreme_closeup", "closeup", "medium_closeup",
        "medium_shot", "medium_full", "full_shot", "wide_shot"
    })

    FACE_ANGLE_OPTIONS = ["auto"] + list({
        "front_view", "three_quarter", "profile",
        "looking_up", "looking_down", "over_shoulder"
    })

    LIGHTING_OPTIONS = ["auto"] + list({
        "soft_diffused", "golden_hour", "blue_hour", "overcast",
        "studio_softbox", "rembrandt", "butterfly", "split", "rim",
        "cinematic_warm", "noir", "neon", "volumetric", "hazy"
    })

    LENS_OPTIONS = ["auto"] + list({
        "24mm", "35mm", "50mm", "85mm", "105mm", "135mm", "200mm"
    })

    APERTURE_OPTIONS = ["auto"] + list({
        "f1_4", "f1_8", "f2_8", "f5_6", "f8", "f11_sharp"
    })

    ENVIRONMENT_OPTIONS = ["auto"] + list({
        "studio_white", "studio_gray", "studio_black",
        "blurred_urban", "nature_outdoor", "office_modern",
        "home_cozy", "cafe", "beach", "street"
    })

    MOOD_OPTIONS = ["auto"] + list({
        "serene", "dramatic", "intimate", "ethereal",
        "gritty", "nostalgic", "mysterious", "joyful",
        "melancholic", "cinematic"
    })

    FILM_STOCK_OPTIONS = ["none"] + list({
        "portra_400", "portra_800", "ektar", "tri_x",
        "hp5", "fuji_400h", "cinestill_800t", "velvia"
    })

    @classmethod
    def define_schema(cls):
        return comfy_io.Schema(
            node_id="SID_ZImagePortraitBuilder",
            display_name="SID Z-Image Portrait Builder",
            description="Build portrait prompts with Z-Image vocabulary",
            category="SID/Photography",
            inputs=[
                comfy_io.String.Input(
                    "subject",
                    display_name="Subject",
                    multiline=False,
                    default="a young woman with auburn hair",
                    tooltip="Description of the subject (e.g., 'a young woman with auburn hair')"
                ),
                comfy_io.String.Input(
                    "clothing",
                    display_name="Clothing",
                    multiline=False,
                    default="",
                    tooltip="Optional clothing description (e.g., 'dark blazer over white shirt')"
                ),
                comfy_io.String.Input(
                    "additional_details",
                    display_name="Additional Details",
                    multiline=True,
                    default="",
                    tooltip="Any additional details to include"
                ),
                comfy_io.Combo.Input(
                    "shot_type",
                    display_name="Shot Type",
                    options=cls.SHOT_OPTIONS,
                    default="medium_closeup",
                    tooltip="Type of shot/framing"
                ),
                comfy_io.Combo.Input(
                    "face_angle",
                    display_name="Face Angle",
                    options=cls.FACE_ANGLE_OPTIONS,
                    default="three_quarter",
                    tooltip="Face angle/direction"
                ),
                comfy_io.Combo.Input(
                    "lighting",
                    display_name="Lighting",
                    options=cls.LIGHTING_OPTIONS,
                    default="soft_diffused",
                    tooltip="Lighting style"
                ),
                comfy_io.Combo.Input(
                    "lens",
                    display_name="Lens",
                    options=cls.LENS_OPTIONS,
                    default="85mm",
                    tooltip="Lens focal length"
                ),
                comfy_io.Combo.Input(
                    "aperture",
                    display_name="Aperture",
                    options=cls.APERTURE_OPTIONS,
                    default="f1_8",
                    tooltip="Aperture/depth of field"
                ),
                comfy_io.Combo.Input(
                    "environment",
                    display_name="Environment",
                    options=cls.ENVIRONMENT_OPTIONS,
                    default="studio_gray",
                    tooltip="Background/environment"
                ),
                comfy_io.Combo.Input(
                    "mood",
                    display_name="Mood",
                    options=cls.MOOD_OPTIONS,
                    default="cinematic",
                    tooltip="Atmosphere/mood"
                ),
                comfy_io.Combo.Input(
                    "film_stock",
                    display_name="Film Stock",
                    options=cls.FILM_STOCK_OPTIONS,
                    default="none",
                    tooltip="Optional film stock aesthetic"
                ),
                comfy_io.Boolean.Input(
                    "add_safety",
                    display_name="Add Safety Phrases",
                    default=True,
                    tooltip="Add SFW safety phrases"
                ),
            ],
            outputs=[
                comfy_io.String.Output(
                    "prompt",
                    display_name="Prompt",
                    tooltip="The built Z-Image portrait prompt"
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        subject: str,
        clothing: str,
        additional_details: str,
        shot_type: str,
        face_angle: str,
        lighting: str,
        lens: str,
        aperture: str,
        environment: str,
        mood: str,
        film_stock: str,
        add_safety: bool,
    ) -> comfy_io.NodeOutput:
        """Build the portrait prompt."""

        vocab = get_vocabulary()
        parts = []

        # Shot type + subject
        if shot_type != "auto":
            shot_term = vocab.get_term(shot_type)
            if shot_term:
                parts.append(f"{shot_term.term} of {subject}")
            else:
                parts.append(f"Portrait of {subject}")
        else:
            parts.append(f"Portrait of {subject}")

        # Face angle
        if face_angle != "auto":
            angle_term = vocab.get_term(face_angle)
            if angle_term:
                parts.append(angle_term.term)

        # Clothing
        if clothing and clothing.strip():
            parts.append(f"wearing {clothing.strip()}")

        # Additional details
        if additional_details and additional_details.strip():
            parts.append(additional_details.strip())

        # Environment
        if environment != "auto":
            env_term = vocab.get_term(environment)
            if env_term:
                parts.append(env_term.term)

        # Lighting
        if lighting != "auto":
            light_term = vocab.get_term(lighting)
            if light_term:
                parts.append(light_term.term)

        # Technical (lens + aperture)
        tech_parts = []
        if lens != "auto":
            lens_term = vocab.get_term(lens)
            if lens_term:
                tech_parts.append(lens_term.term)
        if aperture != "auto":
            aperture_term = vocab.get_term(aperture)
            if aperture_term:
                tech_parts.append(aperture_term.term)
        if tech_parts:
            parts.append(f"Shot with {', '.join(tech_parts)}")

        # Film stock
        if film_stock != "none":
            film_term = vocab.get_term(film_stock)
            if film_term:
                parts.append(film_term.term)

        # Mood
        if mood != "auto":
            mood_term = vocab.get_term(mood)
            if mood_term:
                parts.append(mood_term.term)

        # Safety
        if add_safety:
            parts.append(vocab.get_safety_phrase("sfw_basic"))

        # Build final prompt
        prompt = '. '.join(parts) + '.'

        print(f"[Z-Image Portrait Builder] Built prompt ({len(prompt.split())} words):")
        print(f"  {prompt[:150]}{'...' if len(prompt) > 150 else ''}")

        return comfy_io.NodeOutput(prompt)


# =============================================================================
# Z-Image Vocabulary Lookup Node
# =============================================================================

@comfy_io.Title("SID Z-Image Vocabulary Lookup")
@comfy_io.Description(
    "Look up Z-Image vocabulary terms by category. "
    "Useful for discovering optimal terms for lighting, composition, etc."
)
@comfy_io.Category("SID/Photography")
class SID_ZImageVocabLookup(comfy_io.ComfyNode):
    """
    Look up vocabulary terms by category.
    """

    CATEGORY_OPTIONS = [
        "composition", "shot_type", "face_angle", "lighting",
        "camera", "lens", "film_stock", "color_palette",
        "mood", "material", "environment", "clothing"
    ]

    @classmethod
    def define_schema(cls):
        return comfy_io.Schema(
            node_id="SID_ZImageVocabLookup",
            display_name="SID Z-Image Vocabulary Lookup",
            description="Look up Z-Image vocabulary by category",
            category="SID/Photography",
            inputs=[
                comfy_io.Combo.Input(
                    "category",
                    display_name="Category",
                    options=cls.CATEGORY_OPTIONS,
                    default="lighting",
                    tooltip="Vocabulary category to look up"
                ),
                comfy_io.Integer.Input(
                    "top_n",
                    display_name="Top N Terms",
                    default=10,
                    min=1,
                    max=50,
                    tooltip="Number of top terms to return"
                ),
            ],
            outputs=[
                comfy_io.String.Output(
                    "terms",
                    display_name="Terms",
                    tooltip="Top vocabulary terms for this category"
                ),
                comfy_io.String.Output(
                    "report",
                    display_name="Full Report",
                    tooltip="Detailed report with effectiveness scores"
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        category: str,
        top_n: int,
    ) -> comfy_io.NodeOutput:
        """Look up vocabulary terms."""

        vocab = get_vocabulary()

        # Map string to enum
        category_map = {
            "composition": VocabCategory.COMPOSITION,
            "shot_type": VocabCategory.SHOT_TYPE,
            "face_angle": VocabCategory.FACE_ANGLE,
            "lighting": VocabCategory.LIGHTING,
            "camera": VocabCategory.CAMERA,
            "lens": VocabCategory.LENS,
            "film_stock": VocabCategory.FILM_STOCK,
            "color_palette": VocabCategory.COLOR_PALETTE,
            "mood": VocabCategory.MOOD,
            "material": VocabCategory.MATERIAL,
            "environment": VocabCategory.ENVIRONMENT,
            "clothing": VocabCategory.CLOTHING,
        }

        cat_enum = category_map.get(category)
        if not cat_enum:
            return comfy_io.NodeOutput("Unknown category", "Unknown category")

        # Get top terms
        terms = vocab.get_best_terms(cat_enum, top_n)

        # Build outputs
        term_list = [t.term for t in terms]
        terms_str = "\n".join(term_list)

        report_lines = [f"Z-Image Vocabulary: {category.upper()}", "=" * 40, ""]
        for i, term in enumerate(terms, 1):
            report_lines.append(f"{i}. {term.term}")
            report_lines.append(f"   Effectiveness: {term.effectiveness:.0%}")
            report_lines.append(f"   Use case: {term.use_case}")
            if term.alternatives:
                report_lines.append(f"   Alternatives: {', '.join(term.alternatives[:3])}")
            report_lines.append("")

        report = "\n".join(report_lines)

        print(f"\n[Z-Image Vocabulary] Top {len(terms)} terms for {category}:")
        for term in terms[:5]:
            print(f"  - {term.term} ({term.effectiveness:.0%})")

        return comfy_io.NodeOutput(terms_str, report)


# =============================================================================
# NODE MAPPINGS
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "SID_ZImageOptimizer": SID_ZImageOptimizer,
    "SID_ZImagePortraitBuilder": SID_ZImagePortraitBuilder,
    "SID_ZImageVocabLookup": SID_ZImageVocabLookup,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SID_ZImageOptimizer": "SID Z-Image Prompt Optimizer",
    "SID_ZImagePortraitBuilder": "SID Z-Image Portrait Builder",
    "SID_ZImageVocabLookup": "SID Z-Image Vocabulary Lookup",
}
