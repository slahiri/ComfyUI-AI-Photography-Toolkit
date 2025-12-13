# Vision Prompting Best Practices Guide

**Purpose:** Reference guide for implementing vision-based image analysis in the Photography Expert Node
**Created:** 2024-12-13
**Sources:** Research from GitHub repositories, Microsoft Azure, NVIDIA, Anthropic, OpenAI, and academic papers

---

## Table of Contents

1. [Core Principles](#core-principles)
2. [Prompt Structure Patterns](#prompt-structure-patterns)
3. [Chain-of-Thought for Vision](#chain-of-thought-for-vision)
4. [Structured Output (JSON)](#structured-output-json)
5. [Multi-Pass Agentic Workflows](#multi-pass-agentic-workflows)
6. [Provider-Specific Techniques](#provider-specific-techniques)
7. [Photography Analysis Prompts](#photography-analysis-prompts)
8. [Common Pitfalls & Solutions](#common-pitfalls--solutions)
9. [Prompt Templates Library](#prompt-templates-library)

---

## Core Principles

### 1. Contextual Specificity

Generic prompts produce generic results. Always provide context about the analysis purpose.

**Bad:**
```
Describe this image.
```

**Good:**
```
You are a professional photography judge evaluating this image for a portrait competition.
Analyze the lighting technique, focusing on how it shapes the subject's face and creates mood.
```

### 2. Task-Oriented Framing

Frame requests around specific tasks with clear objectives.

**Pattern:**
```
You are a [ROLE] performing [TASK] for [PURPOSE].
Focus on [SPECIFIC ASPECTS].
Output in [FORMAT].
```

**Example:**
```
You are an expert photography critic evaluating this portrait for a professional competition.
Focus on: lighting quality, composition balance, emotional impact, and technical execution.
Output your analysis as a structured JSON object.
```

### 3. Image-First Placement

Place images before text instructions for optimal understanding.

**Recommended order:**
1. Image(s)
2. System context
3. Task instructions
4. Output format specification

### 4. Specificity Over Generality

Be explicit about what to analyze and how to analyze it.

**Vague:** "Is the lighting good?"
**Specific:** "Evaluate the lighting pattern: Is it Rembrandt, butterfly, split, or loop lighting? Describe the light-to-shadow ratio and how the catchlights appear in the subject's eyes."

---

## Prompt Structure Patterns

### Pattern 1: Role-Task-Format (RTF)

```
[ROLE DEFINITION]
You are an expert [domain expert] with [X] years of experience in [specialty].

[TASK SPECIFICATION]
Analyze this [image type] and evaluate:
1. [Aspect 1]
2. [Aspect 2]
3. [Aspect 3]

[FORMAT REQUIREMENTS]
Provide your analysis as [format] with the following structure:
{structure example}

[CONSTRAINTS]
- [Constraint 1]
- [Constraint 2]
```

### Pattern 2: Context-Examples-Task (CET)

```
[CONTEXT]
You are evaluating photographs for [purpose]. The scoring scale is [X-Y].

[EXAMPLES]
Example 1: [Image] Score: 85 - "Excellent use of natural light..."
Example 2: [Image] Score: 72 - "Good composition but exposure issues..."

[TASK]
Now evaluate this image using the same criteria and scoring approach.
```

### Pattern 3: Decomposition Pattern

```
[OVERVIEW]
Analyze this photograph in three phases:

[PHASE 1: TECHNICAL]
First, evaluate only the technical aspects:
- Exposure accuracy
- Focus sharpness
- Noise levels
- Color accuracy

[PHASE 2: ARTISTIC]
Next, evaluate the artistic elements:
- Composition
- Lighting mood
- Color harmony
- Visual storytelling

[PHASE 3: SYNTHESIS]
Finally, synthesize your findings into an overall assessment.
```

---

## Chain-of-Thought for Vision

### Why CoT Matters for Image Analysis

Chain-of-thought prompting guides the model through step-by-step reasoning, reducing hallucinations and improving accuracy for complex visual analysis tasks.

### Zero-Shot CoT

Add "Let's analyze this step by step" to trigger reasoning:

```
Evaluate this portrait photograph for a professional competition.

Let's analyze this step by step:
1. First, I'll examine the technical execution (exposure, focus, noise)
2. Then, I'll evaluate the lighting setup and quality
3. Next, I'll assess composition and visual balance
4. Finally, I'll consider the emotional impact and storytelling

Provide your step-by-step analysis:
```

### Structured CoT Template

```
Analyze this photograph using the following reasoning process:

STEP 1 - FIRST IMPRESSION (5 seconds)
What is your immediate emotional response? What draws your eye first?

STEP 2 - TECHNICAL SCAN
Evaluate: exposure, focus, sharpness, noise, color accuracy.
For each, rate 1-10 and explain why.

STEP 3 - COMPOSITION ANALYSIS
Identify: rule of thirds, leading lines, symmetry, balance, negative space.
How do these elements guide the viewer's eye?

STEP 4 - LIGHTING EVALUATION
Determine: light source, direction, quality (hard/soft), color temperature.
How does lighting contribute to mood?

STEP 5 - STORY & EMOTION
What story does this image tell? What emotion does it evoke?
How effectively does it communicate its message?

STEP 6 - SYNTHESIS
Combine your observations into a final score (0-100) with justification.
```

### Visual Chain-of-Thought (VCoT)

For complex scenes, have the model describe what it sees before analyzing:

```
Before evaluating this photograph, first describe in detail:
1. What objects and subjects are present
2. Their spatial relationships
3. The lighting conditions you observe
4. Any notable colors or tones

Then, using this description as reference, evaluate the photograph's quality.
```

---

## Structured Output (JSON)

### Why JSON for Photography Analysis

- Consistent parsing for downstream processing
- Type-safe scores and ratings
- Easy integration with databases and UIs
- Enables comparison across analyses

### Enforcing JSON Output

**Method 1: Explicit Instruction**
```
CRITICAL: Respond ONLY with a valid JSON object. No additional text before or after.
```

**Method 2: Schema Definition**
```
Output your analysis as JSON matching this exact schema:
{
  "overall_score": <integer 0-100>,
  "technical": {
    "exposure": {"score": <1-10>, "notes": "<string>"},
    "focus": {"score": <1-10>, "notes": "<string>"}
  },
  "artistic": {
    "composition": {"score": <1-10>, "notes": "<string>"}
  }
}
```

**Method 3: Function Calling (Recommended)**
Use the model's function calling API to guarantee JSON structure with all required keys.

### Photography Analysis JSON Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["overall_score", "merit_level", "summary", "elements", "improvements"],
  "properties": {
    "overall_score": {
      "type": "integer",
      "minimum": 0,
      "maximum": 100,
      "description": "Overall quality score on PPA scale"
    },
    "merit_level": {
      "type": "string",
      "enum": ["Exceptional", "Superior", "Excellent", "Merit", "Above Average", "Average", "Below Average", "Needs Work"],
      "description": "PPA merit classification"
    },
    "summary": {
      "type": "string",
      "maxLength": 500,
      "description": "One-paragraph executive summary"
    },
    "elements": {
      "type": "object",
      "description": "12 Elements of Merit Image scores",
      "properties": {
        "impact": {"$ref": "#/definitions/element_score"},
        "technical_excellence": {"$ref": "#/definitions/element_score"},
        "creativity": {"$ref": "#/definitions/element_score"},
        "style": {"$ref": "#/definitions/element_score"},
        "composition": {"$ref": "#/definitions/element_score"},
        "presentation": {"$ref": "#/definitions/element_score"},
        "color_balance": {"$ref": "#/definitions/element_score"},
        "center_of_interest": {"$ref": "#/definitions/element_score"},
        "lighting": {"$ref": "#/definitions/element_score"},
        "subject_matter": {"$ref": "#/definitions/element_score"},
        "technique": {"$ref": "#/definitions/element_score"},
        "story_telling": {"$ref": "#/definitions/element_score"}
      }
    },
    "style_match": {
      "type": "object",
      "properties": {
        "photographer": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "matching_elements": {"type": "array", "items": {"type": "string"}}
      }
    },
    "improvements": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "priority": {"type": "string", "enum": ["high", "medium", "low"]},
          "category": {"type": "string"},
          "issue": {"type": "string"},
          "suggestion": {"type": "string"}
        }
      }
    }
  },
  "definitions": {
    "element_score": {
      "type": "object",
      "required": ["score", "feedback"],
      "properties": {
        "score": {"type": "integer", "minimum": 1, "maximum": 10},
        "feedback": {"type": "string", "maxLength": 200}
      }
    }
  }
}
```

---

## Multi-Pass Agentic Workflows

### Why Multi-Pass?

Single-pass analysis often misses nuances. Multi-pass approaches:
- Allow specialized focus per pass
- Enable cross-referencing between analyses
- Reduce cognitive load per prompt
- Improve accuracy through decomposition

### Pass Design Principles

1. **Single Responsibility**: Each pass focuses on one aspect
2. **Progressive Detail**: Start broad, then narrow focus
3. **Cross-Referencing**: Later passes can reference earlier findings
4. **Synthesis Pass**: Final pass integrates all findings

### 4-Pass Photography Analysis Workflow

```
PASS 1: TECHNICAL MERIT
────────────────────────
Focus ONLY on technical aspects:
- Exposure: Is it correct? Any clipping?
- Focus: Where is the focal point? Is it appropriate?
- Sharpness: Is the in-focus area crisp?
- Noise: What's the noise level? Is it intrusive?
- Color: Is white balance accurate? Any color casts?

Output: JSON with technical scores and observations.

PASS 2: COMPOSITION & LIGHTING
──────────────────────────────
Focus ONLY on composition and lighting:
- Composition rules applied (thirds, golden ratio, etc.)
- Balance and visual weight distribution
- Leading lines and eye flow
- Lighting pattern (Rembrandt, butterfly, etc.)
- Light quality (hard/soft) and direction
- Light-to-shadow ratio

Output: JSON with composition and lighting analysis.

PASS 3: ARTISTIC & EMOTIONAL
────────────────────────────
Focus ONLY on artistic and emotional impact:
- First impression and emotional response
- Creativity and originality
- Style consistency
- Storytelling effectiveness
- Mood and atmosphere
- Connection with viewer

Output: JSON with artistic scores and observations.

PASS 4: SYNTHESIS & RECOMMENDATIONS
───────────────────────────────────
Given the previous analyses:
[Insert Pass 1-3 results]

Synthesize into:
1. Overall score (0-100)
2. Merit classification
3. Style matching to known photographers
4. Top 3 strengths
5. Top 3 areas for improvement
6. Competition readiness assessment

Output: Final comprehensive JSON report.
```

### Conflict Resolution

When passes produce conflicting assessments:

```
SYNTHESIS INSTRUCTIONS:
If earlier passes conflict, apply these rules:
1. Technical issues override artistic scores (a blurry masterpiece is still blurry)
2. Impact trumps technical perfection (a technically perfect but boring image scores lower)
3. When in doubt, favor the more conservative (lower) assessment
4. Note conflicts explicitly in the final report
```

---

## Provider-Specific Techniques

### Anthropic Claude

**Strengths:**
- Excellent at nuanced analysis
- Good at following complex instructions
- Strong reasoning capabilities

**Best Practices:**
- Use XML tags for structure: `<context>`, `<task>`, `<format>`
- Place images before text
- Enable extended thinking for complex analysis
- Use the "crop tool" technique for detailed region analysis

**Template:**
```xml
<context>
You are a professional photography judge with 20 years of experience.
</context>

<image>
[IMAGE]
</image>

<task>
Evaluate this photograph for a portrait competition.
Focus on: lighting, composition, emotional impact, technical execution.
</task>

<format>
Provide your analysis as JSON with scores 1-10 for each element.
</format>
```

### OpenAI GPT-4 Vision

**Strengths:**
- Good at structured output
- Strong visual understanding
- Effective function calling

**Best Practices:**
- Use system messages for role definition
- Define output format explicitly
- Use visual pointing (markup on images) for specific regions
- Combine with function calling for guaranteed JSON

**Template:**
```
System: You are an expert photography critic. Always respond in valid JSON format.

User: [IMAGE]
Evaluate this photograph's technical quality and artistic merit.
Return JSON with this structure:
{
  "technical_score": 1-10,
  "artistic_score": 1-10,
  "overall": 1-10,
  "strengths": ["..."],
  "improvements": ["..."]
}
```

### Local Models (Florence, Moondream, Qwen-VL)

**Strengths:**
- Fast inference
- Privacy-preserving
- No API costs

**Limitations:**
- Less nuanced analysis
- Smaller context windows
- May struggle with complex instructions

**Best Practices:**
- Use extremely concise prompts (10-20 words)
- Focus on single aspects per call
- Avoid complex JSON; use simple formats
- Multiple simple calls > one complex call

**Template (Local):**
```
Describe this photo's lighting. Be specific about direction and quality.
```

```
Rate the composition 1-10. Explain briefly.
```

---

## Photography Analysis Prompts

### Quick Analysis (Single Pass)

```
You are a photography expert. Analyze this image quickly.

Provide:
1. Overall impression (1 sentence)
2. Score (0-100)
3. Top strength
4. Main issue to fix

Format as JSON:
{"impression": "", "score": 0, "strength": "", "issue": ""}
```

### Technical Analysis

```
Evaluate ONLY the technical aspects of this photograph:

EXPOSURE:
- Is the histogram balanced or clipped?
- Are shadows/highlights properly exposed?
- Estimate the EV correction needed (if any)

FOCUS:
- Where is the focal plane?
- Is the intended subject sharp?
- Evaluate depth of field appropriateness

NOISE:
- Estimate ISO range used
- Is noise intrusive or acceptable?
- Does noise affect image quality?

SHARPNESS:
- Rate overall sharpness in focus areas
- Any motion blur or camera shake?
- Lens quality indicators

COLOR:
- White balance accuracy
- Color cast presence
- Saturation levels

Output as JSON with scores 1-10 for each category.
```

### Composition Analysis

```
Analyze the composition of this photograph:

RULES & TECHNIQUES:
- Rule of thirds: Is it applied? How?
- Golden ratio/spiral: Evidence of use?
- Leading lines: What leads the eye?
- Symmetry/asymmetry: Which and why?
- Framing: Natural frames present?
- Negative space: Effective use?

BALANCE:
- Visual weight distribution
- Tension and harmony
- Dynamic vs static feel

EYE FLOW:
- Where does the eye land first?
- Path through the image
- Exit points (good or bad?)

CROPPING:
- Is current crop optimal?
- Would alternative aspect ratios improve it?

Output JSON with analysis and improvement suggestions.
```

### Lighting Analysis

```
Perform a detailed lighting analysis:

SOURCE IDENTIFICATION:
- Natural or artificial?
- Single or multiple sources?
- Key, fill, rim, background lights?

PATTERN:
- Rembrandt, butterfly, split, loop, broad, short?
- Light-to-shadow ratio estimate
- Transition quality (hard/soft edge)

QUALITY:
- Hard or soft light?
- Size of light source relative to subject
- Modifier used (umbrella, softbox, natural)?

DIRECTION:
- Angle from camera axis
- Height relative to subject
- Front/side/back position

MOOD:
- How does lighting contribute to mood?
- Emotional impact of light choices
- Consistency with subject/story

COLOR:
- Color temperature
- Gels or mixed sources?
- Color harmony with scene

Output JSON with lighting breakdown and score.
```

### Style Matching Analysis

```
Compare this photograph's style to known master photographers.

Consider these characteristics:
- Lighting approach
- Composition tendencies
- Color/tonal treatment
- Subject interaction
- Emotional quality
- Technical choices

REFERENCE PHOTOGRAPHERS:
Landscape: Ansel Adams, Michael Kenna, Galen Rowell
Portrait: Peter Lindbergh, Annie Leibovitz, Richard Avedon, Irving Penn
Street: Henri Cartier-Bresson, Daido Moriyama, Vivian Maier
Fashion: Helmut Newton, Mario Testino, Steven Meisel
Documentary: Steve McCurry, Sebastião Salgado

Identify:
1. Primary style match (photographer + confidence 0-1)
2. Secondary matches (up to 2)
3. Specific matching elements
4. Genre classification
5. Era/movement influence

Output as JSON.
```

---

## Common Pitfalls & Solutions

### Pitfall 1: Overly Generic Analysis

**Problem:** Model gives vague, non-specific feedback
**Solution:** Add explicit requirements for specificity

```
BAD: "The lighting is good."
GOOD: "The Rembrandt lighting creates a triangle of light on the left cheek,
with a 3:1 ratio to the shadow side. The catchlight at 10 o'clock suggests
a large softbox positioned 45 degrees camera left."
```

**Fix prompt:**
```
Provide SPECIFIC observations, not generic assessments.
Instead of "good lighting," describe the exact pattern, ratio, and effect.
```

### Pitfall 2: Hallucinated Details

**Problem:** Model invents details not visible in image
**Solution:** Ground analysis in visible evidence

```
Before making any claim, first identify the visual evidence.
Only state what you can directly observe.
If uncertain, say "appears to be" or "suggests" rather than stating definitively.
```

### Pitfall 3: Inconsistent Scoring

**Problem:** Scores don't align with written feedback
**Solution:** Require explicit score justification

```
For each score, you MUST provide:
1. The score (1-10)
2. Why it's not one point higher
3. Why it's not one point lower
This ensures scores align with actual observations.
```

### Pitfall 4: Overly Positive Bias

**Problem:** Model hesitates to give low scores or negative feedback
**Solution:** Explicitly request critical analysis

```
You are a strict judge known for high standards.
Your role is to identify issues, not to praise.
A score of 7 is "good" - only exceptional work scores 9+.
Finding no issues is suspicious; look harder.
```

### Pitfall 5: Technical vs Artistic Confusion

**Problem:** Technical excellence conflated with artistic merit
**Solution:** Explicitly separate the two

```
Evaluate technical and artistic merit SEPARATELY.

A technically perfect but artistically uninspired image might score:
- Technical: 9/10
- Artistic: 5/10
- Overall: 7/10

An artistically brilliant but technically flawed image might score:
- Technical: 5/10
- Artistic: 9/10
- Overall: 7/10 (or lower if flaws are severe)
```

---

## Prompt Templates Library

### Template 1: Competition Judge

```
ROLE: You are a certified PPA (Professional Photographers of America) judge.

CONTEXT: You are evaluating this image for the International Photographic Competition.
Images are scored 0-100 against the 12 Elements of a Merit Image.

SCORING GUIDE:
95-100: Exceptional (top 1%)
90-94: Superior (top 5%)
85-89: Excellent (top 15%)
80-84: Merit worthy
75-79: Above average
70-74: Average
<70: Below exhibition standards

TASK: Evaluate this photograph using the 12 Elements framework.

OUTPUT FORMAT: JSON
{
  "overall_score": <int>,
  "merit_level": "<string>",
  "elements": {
    "impact": {"score": <int 1-10>, "notes": "<string>"},
    "technical_excellence": {"score": <int 1-10>, "notes": "<string>"},
    "creativity": {"score": <int 1-10>, "notes": "<string>"},
    "style": {"score": <int 1-10>, "notes": "<string>"},
    "composition": {"score": <int 1-10>, "notes": "<string>"},
    "presentation": {"score": <int 1-10>, "notes": "<string>"},
    "color_balance": {"score": <int 1-10>, "notes": "<string>"},
    "center_of_interest": {"score": <int 1-10>, "notes": "<string>"},
    "lighting": {"score": <int 1-10>, "notes": "<string>"},
    "subject_matter": {"score": <int 1-10>, "notes": "<string>"},
    "technique": {"score": <int 1-10>, "notes": "<string>"},
    "story_telling": {"score": <int 1-10>, "notes": "<string>"}
  },
  "verdict": "<string: would this earn a merit point?>"
}
```

### Template 2: Technical Reviewer

```
ROLE: You are a camera technician and post-processing expert.

TASK: Analyze ONLY the technical execution of this photograph.
Ignore artistic merit entirely - focus on technical quality.

EVALUATE:
1. EXPOSURE
   - Histogram analysis (clipping?)
   - Shadow/highlight detail retention
   - Dynamic range utilization

2. FOCUS
   - Focal plane placement
   - Depth of field appropriateness
   - Any focus errors?

3. SHARPNESS
   - In-focus area crispness
   - Motion blur presence
   - Camera shake indicators

4. NOISE
   - ISO estimate
   - Noise type (luminance/chroma)
   - Impact on image quality

5. COLOR
   - White balance accuracy
   - Color cast presence
   - Saturation levels

OUTPUT: JSON with score 1-10 for each category and specific observations.
```

### Template 3: Style Matcher

```
ROLE: You are a photography historian and style analyst.

TASK: Analyze this photograph's visual style and match it to known photographers.

ANALYZE:
1. Lighting approach (dramatic, natural, studio, available)
2. Tonal treatment (high contrast, low contrast, B&W, color palette)
3. Composition tendencies (minimal, complex, geometric, organic)
4. Subject relationship (intimate, distant, candid, posed)
5. Technical choices (DOF, motion, grain)
6. Emotional quality (mood, atmosphere, feeling)

COMPARE TO DATABASE:
[List of 50+ photographers with their signature characteristics]

OUTPUT:
{
  "primary_match": {
    "photographer": "<name>",
    "confidence": <0.0-1.0>,
    "matching_elements": ["<element1>", "<element2>"]
  },
  "secondary_matches": [
    {"photographer": "<name>", "confidence": <0.0-1.0>}
  ],
  "genre": "<primary genre>",
  "era_influence": "<decade/movement>",
  "unique_elements": ["<what makes this distinct>"]
}
```

### Template 4: Improvement Coach

```
ROLE: You are a photography mentor focused on helping photographers improve.

TASK: Identify specific, actionable improvements for this photograph.

APPROACH:
1. Identify what works well (briefly)
2. Focus on what could be improved
3. Prioritize by impact (high/medium/low)
4. Provide specific, actionable advice

FOR EACH ISSUE:
- What's the problem?
- Why does it matter?
- How to fix it (in-camera or post-processing)?
- Example of the fix

OUTPUT:
{
  "quick_wins": [
    {"issue": "", "fix": "", "impact": "high/medium/low"}
  ],
  "technique_improvements": [
    {"skill": "", "current_level": "", "how_to_improve": ""}
  ],
  "gear_suggestions": [
    {"limitation": "", "solution": ""}
  ],
  "learning_resources": [
    {"topic": "", "why": ""}
  ]
}
```

---

## References

- [Microsoft Azure: Image Prompt Engineering](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/concepts/gpt-4-v-prompt-engineering)
- [NVIDIA: Vision Language Model Prompt Engineering Guide](https://developer.nvidia.com/blog/vision-language-model-prompt-engineering-guide-for-image-and-video-understanding/)
- [Anthropic Claude Vision Documentation](https://docs.anthropic.com/claude/docs/vision)
- [Roboflow: Prompting Tips for Vision LLMs](https://blog.roboflow.com/prompting-tips-for-large-language-models-with-vision/)
- [Prompt Engineering Guide: Chain of Thought](https://www.promptingguide.ai/techniques/cot)
- [Awesome Prompting Papers in Computer Vision](https://github.com/ttengwang/Awesome_Prompting_Papers_in_Computer_Vision)
- [IBM: Chain of Thought Prompting](https://www.ibm.com/think/topics/chain-of-thoughts)
- [Visual Chain-of-Thought Prompting (AAAI 2024)](https://zfchenunique.github.io/files/aaai24_vcot_arxiv.pdf)
