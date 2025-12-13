# Photography Expert Node - Implementation Plan

**Status:** TODO
**Priority:** High
**Target Version:** 4.3.0
**Created:** 2024-12-13

## Overview

Create an agentic node that acts as a **professional photography expert**, analyzing images and providing detailed critique based on industry-standard frameworks used in photography competitions and professional reviews.

---

## Features

### Core Capabilities

1. **12 Elements Analysis** - Based on PPA (Professional Photographers of America) Merit Image standards
2. **Technical Evaluation** - Exposure, focus, sharpness, noise, color accuracy
3. **Composition Analysis** - Rule of thirds, leading lines, balance, geometry
4. **Style Matching** - Match to 50+ master photographers' styles
5. **Genre Classification** - Portrait, Landscape, Street, Fashion, etc.
6. **Actionable Feedback** - Prioritized improvement suggestions
7. **Competition Readiness** - Score and merit assessment

---

## Analysis Modes

| Mode | Passes | Est. Time | Use Case |
|------|--------|-----------|----------|
| **Quick** | 1 | ~2-5s | Fast feedback, basic issues |
| **Standard** | 2 | ~10-15s | Balanced critique |
| **Detailed** | 4 | ~30-45s | Competition-level analysis |
| **Extreme** | 6 | ~60-90s | Master photographer review |

### Quick Mode (1 Pass)
- Overall impression
- Top 3 issues identified
- Single score with brief explanation

### Standard Mode (2 Passes)
1. **Technical Analysis** - Exposure, focus, sharpness, noise
2. **Artistic Analysis** - Style, impact, storytelling, composition

### Detailed Mode (4 Passes)
1. **Technical Merit** - Exposure, focus, sharpness, noise, color accuracy
2. **Composition & Lighting** - Rules, balance, light quality, direction
3. **Artistic & Style** - Creativity, emotion, narrative, originality
4. **Style Matching + Recommendations** - Master photographer comparison, improvements

### Extreme Mode (6 Passes)
1. **Technical Deep-Dive** - Forensic-level technical analysis
2. **Composition Geometry** - Golden ratio, fibonacci, dynamic symmetry
3. **Lighting & Color Science** - Color theory, temperature, grading analysis
4. **Emotional Impact & Storytelling** - Narrative, mood, viewer connection
5. **Master Photographer Style Matching** - Detailed comparison to masters
6. **Synthesis + Improvement Roadmap** - Comprehensive action plan

---

## The 12 Elements Framework

Based on [Professional Photographers of America](https://www.ppa.com/events/photo-competitions/the-12-elements-of-a-merit-image) competition standards:

| # | Element | Description |
|---|---------|-------------|
| 1 | **Impact** | Emotional response evoked on first viewing |
| 2 | **Technical Excellence** | Focus, exposure, sharpness, noise levels |
| 3 | **Creativity** | Unique vision, originality, fresh perspective |
| 4 | **Style** | Consistent artistic approach and visual identity |
| 5 | **Composition** | Arrangement, balance, rule of thirds, leading lines |
| 6 | **Print Presentation** | Final output quality and presentation |
| 7 | **Color Balance** | Harmony, temperature, grading, palette |
| 8 | **Center of Interest** | Clear focal point that draws the eye |
| 9 | **Lighting** | Quality, direction, mood, contrast |
| 10 | **Subject Matter** | Relevance, interest, appropriateness |
| 11 | **Technique** | Mastery of photographic craft |
| 12 | **Story Telling** | Narrative, meaning, emotional connection |

### Scoring Scale (PPA Standard)

| Score Range | Rating | Description |
|-------------|--------|-------------|
| 95-100 | Exceptional | Museum/gallery quality |
| 90-94 | Superior | Award-winning caliber |
| 85-89 | Excellent | Professional excellence |
| 80-84 | Merit | Deserving of recognition |
| 75-79 | Above Average | Good with minor issues |
| 70-74 | Average | Competent but unremarkable |
| 65-69 | Below Average | Significant issues |
| <65 | Needs Work | Major improvements needed |

---

## Master Photographer Style Database

### Landscape Photography

| Photographer | Era | Key Characteristics |
|--------------|-----|---------------------|
| **Ansel Adams** | 1930-1980 | Zone system, B&W, dramatic tonal range, American West |
| **Galen Rowell** | 1970-2002 | Adventure, golden light, alpenglow, dynamic compositions |
| **Michael Kenna** | 1980-present | Minimalist, long exposure, ethereal, zen-like |
| **Peter Lik** | 1990-present | Panoramic, saturated colors, dramatic landscapes |
| **Art Wolfe** | 1980-present | Patterns in nature, wildlife, cultural landscapes |

### Portrait & Fashion Photography

| Photographer | Era | Key Characteristics |
|--------------|-----|---------------------|
| **Annie Leibovitz** | 1970-present | Dramatic staging, celebrity, theatrical lighting |
| **Peter Lindbergh** | 1980-2019 | B&W, raw beauty, minimal retouching, emotional depth |
| **Richard Avedon** | 1950-2004 | White backgrounds, movement, emotional intensity |
| **Irving Penn** | 1940-2009 | Minimalist, stark elegance, masterful lighting |
| **Helmut Newton** | 1960-2004 | Provocative, noir, powerful women, high contrast |
| **Mario Testino** | 1980-present | Glamorous, vibrant, celebrity fashion |
| **Patrick Demarchelier** | 1970-present | Clean, elegant, natural beauty |
| **Steven Meisel** | 1980-present | Conceptual, transformative, editorial |
| **Herb Ritts** | 1980-2002 | Sculptural bodies, B&W, classical beauty |
| **Tim Walker** | 2000-present | Fantastical, surreal, whimsical sets |

### Street Photography

| Photographer | Era | Key Characteristics |
|--------------|-----|---------------------|
| **Henri Cartier-Bresson** | 1930-2004 | Decisive moment, geometric precision, humanist |
| **Vivian Maier** | 1950-2009 | Intimate street scenes, self-portraits, mid-century America |
| **Daido Moriyama** | 1960-present | High contrast, grain, blur, urban chaos |
| **Garry Winogrand** | 1960-1984 | American life, tilted frames, social observation |
| **Robert Frank** | 1950-2019 | Raw, unfiltered, The Americans, social critique |
| **Elliott Erwitt** | 1950-present | Witty, humorous, dogs, everyday moments |
| **Joel Meyerowitz** | 1960-present | Color street, Cape Light, contemplative |
| **Alex Webb** | 1970-present | Complex layering, saturated color, Latin America |
| **Martin Parr** | 1980-present | Satirical, British life, consumer culture, flash |
| **Bruce Gilden** | 1980-present | In-your-face, flash street, confrontational |

### Documentary & Photojournalism

| Photographer | Era | Key Characteristics |
|--------------|-----|---------------------|
| **Steve McCurry** | 1980-present | Vibrant color, human faces, cultural stories |
| **Sebastião Salgado** | 1970-present | Epic B&W, humanitarian, environmental |
| **James Nachtwey** | 1980-present | War, conflict, human suffering, witness |
| **Dorothea Lange** | 1930-1965 | Depression era, migrant workers, social documentary |
| **W. Eugene Smith** | 1940-1978 | Photo essays, Minamata, Country Doctor |
| **Don McCullin** | 1960-present | War, poverty, stark B&W, humanitarian |

### Fine Art Photography

| Photographer | Era | Key Characteristics |
|--------------|-----|---------------------|
| **Cindy Sherman** | 1980-present | Conceptual self-portraits, identity, film stills |
| **Gregory Crewdson** | 1990-present | Cinematic tableaux, suburban uncanny, elaborate sets |
| **Andreas Gursky** | 1990-present | Large format, globalization, digitally enhanced |
| **Jeff Wall** | 1980-present | Lightbox transparencies, constructed reality |
| **Edward Weston** | 1920-1958 | Form, texture, peppers, shells, nudes |
| **Man Ray** | 1920-1976 | Surrealist, rayographs, experimental |

### Nature & Wildlife

| Photographer | Era | Key Characteristics |
|--------------|-----|---------------------|
| **Frans Lanting** | 1980-present | Intimate wildlife, environmental storytelling |
| **Nick Brandt** | 2000-present | African wildlife, B&W, environmental crisis |
| **Art Wolfe** | 1980-present | Patterns, camouflage, cultural-wildlife blend |
| **Paul Nicklen** | 2000-present | Polar regions, underwater, climate documentation |
| **Tim Flach** | 2000-present | Studio animal portraits, endangered species |

### Architecture & Urban

| Photographer | Era | Key Characteristics |
|--------------|-----|---------------------|
| **Julius Shulman** | 1940-2009 | Mid-century modern, California architecture |
| **Iwan Baan** | 2000-present | Contemporary architecture in context |
| **Ezra Stoller** | 1940-2004 | Modernist buildings, precise compositions |
| **Candida Höfer** | 1980-present | Empty interiors, symmetry, cultural spaces |

---

## Output Structure

```json
{
  "analysis_mode": "Detailed",
  "processing_time_seconds": 32.5,

  "executive_summary": {
    "overall_score": 82,
    "merit_level": "Deserving of a Point",
    "one_liner": "Strong portrait with excellent lighting but minor composition issues",
    "strengths": ["Dramatic Rembrandt lighting", "Sharp focus on eyes", "Emotional connection"],
    "weaknesses": ["Slight highlight clipping", "Distracting background element", "Could benefit from tighter crop"]
  },

  "12_elements": {
    "impact": {
      "score": 8,
      "rating": "Excellent",
      "feedback": "Strong emotional connection established through subject's gaze"
    },
    "technical_excellence": {
      "score": 7,
      "rating": "Good",
      "feedback": "Sharp focus on eyes, but slight highlight clipping in background"
    },
    "creativity": {
      "score": 8,
      "rating": "Excellent",
      "feedback": "Fresh perspective with unconventional framing"
    },
    "style": {
      "score": 8,
      "rating": "Excellent",
      "feedback": "Consistent moody aesthetic with intentional color grading"
    },
    "composition": {
      "score": 7,
      "rating": "Good",
      "feedback": "Rule of thirds applied, but left edge slightly cluttered"
    },
    "print_presentation": {
      "score": 7,
      "rating": "Good",
      "feedback": "Good resolution, minor sharpening artifacts visible"
    },
    "color_balance": {
      "score": 8,
      "rating": "Excellent",
      "feedback": "Warm tones complement skin, cohesive palette"
    },
    "center_of_interest": {
      "score": 9,
      "rating": "Superior",
      "feedback": "Eyes immediately draw viewer, no competing elements"
    },
    "lighting": {
      "score": 9,
      "rating": "Superior",
      "feedback": "Classic Rembrandt triangle, beautiful rim light separation"
    },
    "subject_matter": {
      "score": 7,
      "rating": "Good",
      "feedback": "Compelling subject, could benefit from more context"
    },
    "technique": {
      "score": 8,
      "rating": "Excellent",
      "feedback": "Masterful shallow DOF, appropriate for portrait"
    },
    "story_telling": {
      "score": 7,
      "rating": "Good",
      "feedback": "Hints at narrative through expression, room for more depth"
    }
  },

  "technical_analysis": {
    "exposure": {
      "rating": "good",
      "ev_estimate": "+0.3",
      "histogram_shape": "right_weighted",
      "issues": ["Minor highlight clipping in specular reflections"],
      "recommendations": ["Reduce exposure by 0.3-0.5 stops", "Use highlight recovery in post"]
    },
    "focus": {
      "rating": "excellent",
      "focus_point": "eyes",
      "depth_of_field": "shallow",
      "aperture_estimate": "f/2.0-2.8",
      "sharpness_score": 9,
      "notes": "Tack sharp on nearest eye, beautiful focus falloff"
    },
    "noise": {
      "rating": "acceptable",
      "iso_estimate": "800-1600",
      "noise_type": "luminance",
      "visibility": "visible in shadows",
      "recommendations": ["Apply selective noise reduction to shadow areas"]
    },
    "white_balance": {
      "rating": "good",
      "temperature_estimate": "5200K",
      "tint": "slight_magenta",
      "skin_tone_accuracy": "natural"
    },
    "sharpness": {
      "rating": "excellent",
      "in_focus_areas": ["eyes", "face"],
      "blur_quality": "creamy_bokeh",
      "artifacts": "none"
    }
  },

  "composition_analysis": {
    "overall_rating": "good",
    "rules_applied": {
      "rule_of_thirds": true,
      "golden_ratio": false,
      "leading_lines": true,
      "symmetry": false,
      "framing": false,
      "negative_space": "effective"
    },
    "balance": "asymmetric_dynamic",
    "visual_weight_distribution": "right_heavy",
    "eye_flow": ["eyes", "lips", "hands", "background"],
    "cropping_suggestions": [
      "Consider 4:5 crop to eliminate left edge distraction",
      "Headroom is appropriate for the mood"
    ],
    "geometric_elements": {
      "triangles": ["face_triangle_formed_by_lighting"],
      "diagonals": ["shoulder_line_creates_dynamic_angle"],
      "curves": ["arm_creates_leading_curve"]
    }
  },

  "lighting_analysis": {
    "type": "studio_strobe",
    "pattern": "rembrandt",
    "quality": "soft_with_contrast",
    "direction": "45_degrees_camera_left",
    "ratio": "3:1",
    "key_light": "large_softbox_overhead",
    "fill": "minimal_reflector",
    "rim_light": "present_camera_right",
    "catchlights": "visible_natural_shape",
    "mood": "dramatic_intimate",
    "color_temperature": "neutral_to_warm"
  },

  "style_matching": {
    "primary_match": {
      "photographer": "Peter Lindbergh",
      "confidence": 0.78,
      "matching_elements": ["emotional_depth", "natural_beauty", "dramatic_lighting", "intimate_mood"]
    },
    "secondary_matches": [
      {"photographer": "Annie Leibovitz", "confidence": 0.65, "reason": "theatrical_lighting"},
      {"photographer": "Richard Avedon", "confidence": 0.58, "reason": "emotional_intensity"}
    ],
    "genre": "Portrait/Fashion",
    "sub_genre": "Editorial Portrait",
    "era_influence": "Contemporary",
    "movement": "Modern Naturalism"
  },

  "genre_classification": {
    "primary": "Portrait",
    "secondary": "Fashion",
    "sub_type": "Editorial",
    "setting": "Studio",
    "mood": "Dramatic/Intimate"
  },

  "competition_assessment": {
    "competition_ready": true,
    "recommended_categories": ["Portrait", "Fashion", "Creative"],
    "strengths_for_competition": [
      "Strong lighting technique",
      "Clear center of interest",
      "Professional execution"
    ],
    "areas_to_address": [
      "Minor technical issues (highlight clipping)",
      "Composition refinement needed"
    ],
    "predicted_score_range": "78-84",
    "verdict": "Would likely earn a merit point with minor adjustments"
  },

  "improvements": [
    {
      "priority": "high",
      "category": "technical",
      "issue": "Highlight clipping in specular areas",
      "suggestion": "Reduce exposure by 0.5 stops or use ND gel on rim light",
      "impact": "Would improve technical score by 0.5-1 point"
    },
    {
      "priority": "medium",
      "category": "composition",
      "issue": "Slight distraction on left edge",
      "suggestion": "Crop to 4:5 ratio or clone out distracting element",
      "impact": "Would improve composition score by 0.5 points"
    },
    {
      "priority": "low",
      "category": "post_processing",
      "issue": "Minor luminance noise in shadows",
      "suggestion": "Apply selective noise reduction to shadow areas only",
      "impact": "Minor refinement for print presentation"
    }
  ],

  "learning_resources": [
    {
      "topic": "Rembrandt Lighting Mastery",
      "reason": "To refine your already strong lighting technique"
    },
    {
      "topic": "Highlight Management in Portraits",
      "reason": "To address recurring clipping issues"
    }
  ]
}
```

---

## Node Schema

```python
class SID_ZImagePhotographyExpert(comfy_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> comfy_io.Schema:
        return comfy_io.Schema(
            node_id="SID_ZImagePhotographyExpert",
            display_name="SID Z-Image Photography Expert",
            category="SID Photography Toolkit/Analysis",
            description="Professional photography critique and analysis",
            inputs=[
                # LLM Model (required for analysis)
                LLM_MODEL_Type.Input("llm_model"),

                # Image to analyze
                comfy_io.Image.Input("image"),

                # Analysis depth
                comfy_io.Combo.Input(
                    "analysis_mode",
                    options=["Quick", "Standard", "Detailed", "Extreme"],
                    default="Standard"
                ),

                # Focus areas
                comfy_io.Combo.Input(
                    "focus",
                    options=["All", "Technical", "Composition", "Lighting", "Style", "Story"],
                    default="All"
                ),

                # Genre hint (helps with context)
                comfy_io.Combo.Input(
                    "genre_hint",
                    options=["Auto-Detect", "Portrait", "Landscape", "Street", "Fashion",
                             "Wildlife", "Architecture", "Product", "Event", "Fine Art"],
                    default="Auto-Detect"
                ),

                # Output format
                comfy_io.Combo.Input(
                    "output_format",
                    options=["Summary", "Detailed", "JSON", "Competition Report"],
                    default="Detailed"
                ),

                # Seed for reproducibility
                comfy_io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff)
            ],
            outputs=[
                comfy_io.String.Output("critique", display_name="Critique"),
                comfy_io.String.Output("score_summary", display_name="Score Summary"),
                comfy_io.String.Output("style_match", display_name="Style Match"),
                comfy_io.String.Output("improvements", display_name="Improvements"),
                comfy_io.String.Output("json_output", display_name="JSON Output")
            ]
        )
```

---

## Implementation Phases

### Phase 1: Core Framework
- [ ] Create base node structure
- [ ] Implement 12 Elements analysis prompts
- [ ] Build scoring system
- [ ] Quick mode implementation

### Phase 2: Style Database
- [ ] Create photographer style database (50+ masters)
- [ ] Implement style matching algorithm
- [ ] Genre classification system
- [ ] Era and movement detection

### Phase 3: Multi-Pass Analysis
- [ ] Standard mode (2 passes)
- [ ] Detailed mode (4 passes)
- [ ] Extreme mode (6 passes)
- [ ] Pass synthesis and conflict resolution

### Phase 4: Output & Polish
- [ ] Multiple output formats
- [ ] Competition report generator
- [ ] Learning resource suggestions
- [ ] Caching for repeated analysis

---

## Technical Considerations

### Prompt Engineering
- Each pass should be focused and specific
- Use structured output (JSON) for parsing
- Include examples in few-shot prompts
- Balance between speed and depth

### Performance
- Quick mode: Single comprehensive pass
- Cache style database in memory
- Progressive detail loading
- Cancel support for long analyses

### Accuracy
- Cross-reference between passes
- Confidence scores for style matching
- Acknowledge uncertainty
- Avoid over-criticism

---

## References

- [PPA 12 Elements of Merit Image](https://www.ppa.com/events/photo-competitions/the-12-elements-of-a-merit-image)
- [PPAM Scoring Criteria](https://www.ppam.com/12-elements-and-scoring)
- [Visual Wilderness Critique Framework](https://visualwilderness.com/q-and-a/a-simple-framework-to-critique-photos-like-a-pro)
- [Famous Photographers Analysis](https://www.photoworkout.com/most-famous-photographers/)
- [Adobe Photography Styles Guide](https://www.adobe.com/creativecloud/photography/type.html)
- [Getty Visual Analysis Methods](https://www.getty.edu/education/for_teachers/curricula/exploring_photographs/background1.html)
