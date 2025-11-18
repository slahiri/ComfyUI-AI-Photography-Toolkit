"""
SID_AIPromptGenerator Node
Takes an image input and uses Anthropic's Claude API to generate detailed positive and negative prompts.
"""

import base64
import io
import numpy as np
from PIL import Image
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io as comfy_io


class SID_AIPromptGenerator(comfy_io.ComfyNode):
    """
    AI-powered prompt generator that analyzes an image using Anthropic's Claude API
    and generates comprehensive ComfyUI prompts for image generation workflows.

    Features multiple photography styles (Artistic, Technical, Minimal, Ultra-Detailed),
    model-specific optimization (FLUX, SDXL, SD 1.5, Universal), color styles,
    famous photographer aesthetics, and lighting conditions.

    Inputs:
    - image: The input image to analyze
    - api_key: Anthropic API key
    - model: Claude model to use (Sonnet 4.5, Haiku 4.5, Opus 4.1, etc.)
    - user_prompt: Additional context or specific instructions
    - photography_style: Style of prompt (Artistic, Technical, Minimal, Ultra-Detailed)
    - target_model: Target image generation model (FLUX, SDXL, SD 1.5, Universal)
    - color_style: Color treatment (None, B&W, Color, Sepia, etc.)
    - photographer_style: Emulate famous photographer (None, Helmut Newton, Peter Lindbergh, etc.)
    - lighting_condition: Specific lighting setup (None, Studio, Outdoor, Techniques, Special)
    - seed: Seed for randomization (use different seeds for variation)
    - temperature: Creativity level (0.0-1.0)
    - max_tokens: Maximum response length

    Outputs:
    - positive_prompt: Comprehensive positive prompt optimized for selected style, model, and photographer
    - negative_prompt: Categorized negative prompt with quality/lighting/composition/subject/style/technical exclusions

    Version: 1.3.3
    """

    @classmethod
    def define_schema(cls) -> comfy_io.Schema:
        """
        Define the node schema with all inputs and outputs.
        """
        return comfy_io.Schema(
            node_id="SID_AIPromptGenerator",
            display_name="SID AI Prompt Generator",
            category="SID Photography Toolkit/Prompt",
            inputs=[
                comfy_io.Image.Input("image", tooltip="Input image to analyze"),
                comfy_io.String.Input(
                    "api_key",
                    default="",
                    multiline=False,
                    tooltip="Anthropic API key (get from https://console.anthropic.com/)"
                ),
                comfy_io.Combo.Input(
                    "model",
                    options=[
                        "claude-sonnet-4-5-20250929",
                        "claude-haiku-4-5-20251001",
                        "claude-opus-4-1-20250805",
                        "claude-3-5-haiku-20241022",
                        "claude-3-haiku-20240307"
                    ],
                    default="claude-sonnet-4-5-20250929",
                    tooltip="Claude model to use for analysis"
                ),
                comfy_io.String.Input(
                    "user_prompt",
                    default="Analyze this image and create detailed prompts for recreating it.",
                    multiline=True,
                    tooltip="Additional context or specific instructions"
                ),
                comfy_io.Combo.Input(
                    "photography_style",
                    options=["Artistic", "Technical", "Minimal", "Ultra-Detailed"],
                    default="Technical",
                    tooltip="Photography prompt style: Artistic (creative focus), Technical (camera specs), Minimal (concise), Ultra-Detailed (comprehensive)"
                ),
                comfy_io.Combo.Input(
                    "target_model",
                    options=["FLUX", "SDXL", "SD 1.5", "Universal"],
                    default="FLUX",
                    tooltip="Target image generation model (affects prompt format and negative prompt style)"
                ),
                comfy_io.Combo.Input(
                    "color_style",
                    options=["None", "Black and White", "Color", "Monochrome", "Sepia", "Cross-Processed", "High Contrast B&W", "Low Key", "High Key"],
                    default="None",
                    tooltip="Photography color treatment (None = based on image analysis)"
                ),
                comfy_io.Combo.Input(
                    "photographer_style",
                    options=[
                        "None",
                        "Helmut Newton", "Peter Lindbergh", "Annie Leibovitz", "Richard Avedon",
                        "Irving Penn", "Herb Ritts", "Mario Testino", "Steven Meisel",
                        "Patrick Demarchelier", "Paolo Roversi", "Tim Walker", "Ellen von Unwerth",
                        "David LaChapelle", "Steven Klein", "Mert and Marcus", "Juergen Teller",
                        "Bruce Weber", "Terry Richardson", "Rankin", "Albert Watson"
                    ],
                    default="None",
                    tooltip="Style after a famous photographer (None = no specific photographer style)"
                ),
                comfy_io.Combo.Input(
                    "lighting_condition",
                    options=[
                        "None",
                        # Studio Lighting
                        "Natural Window Light", "Studio Strobes", "Softbox Lighting", "Beauty Dish",
                        "Ring Light", "Umbrella Lighting", "Grid Spot", "Reflector Fill",
                        # Studio Techniques
                        "Rembrandt Lighting", "Split Lighting", "Butterfly Lighting", "Loop Lighting",
                        "Broad Lighting", "Short Lighting", "High Key Studio", "Low Key Studio",
                        "Clamshell Lighting", "Edge/Rim Lighting",
                        # Outdoor/Natural
                        "Golden Hour", "Blue Hour", "Harsh Midday Sun", "Overcast Diffused",
                        "Open Shade", "Backlit", "Dusk/Twilight", "Sunrise", "Sunset",
                        # Special/Creative
                        "Chiaroscuro", "Dramatic Side Light", "Silhouette", "Candlelight",
                        "Neon/Colorful", "Practical Lights", "Mixed Lighting", "Night Photography"
                    ],
                    default="None",
                    tooltip="Lighting setup/condition (None = based on image analysis)"
                ),
                comfy_io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    tooltip="Seed for randomization (use different seeds for variation)"
                ),
                comfy_io.Float.Input(
                    "temperature",
                    default=0.7,
                    min=0.0,
                    max=1.0,
                    step=0.1,
                    round=0.1,
                    display_mode=comfy_io.NumberDisplay.slider,
                    tooltip="Controls randomness (0=focused, 1=creative)"
                ),
                comfy_io.Int.Input(
                    "max_tokens",
                    default=1024,
                    min=256,
                    max=4096,
                    step=128,
                    display_mode=comfy_io.NumberDisplay.number,
                    tooltip="Maximum tokens in response"
                ),
            ],
            outputs=[
                comfy_io.String.Output("positive_prompt"),
                comfy_io.String.Output("negative_prompt"),
            ],
        )

    @classmethod
    def execute(
        cls,
        image,
        api_key,
        model,
        user_prompt,
        photography_style,
        target_model,
        color_style,
        photographer_style,
        lighting_condition,
        seed,
        temperature,
        max_tokens
    ) -> comfy_io.NodeOutput:
        """
        Execute the node - analyze image and generate prompts using Claude API.
        """

        # Handle None values for backward compatibility with old workflows
        if seed is None:
            seed = 0
        if lighting_condition is None:
            lighting_condition = "None"
        if color_style is None:
            color_style = "None"
        if photographer_style is None:
            photographer_style = "None"

        # Validate API key
        if not api_key or api_key.strip() == "":
            error_msg = "ERROR: Anthropic API key is required. Get one at https://console.anthropic.com/"
            return comfy_io.NodeOutput(error_msg, error_msg)

        try:
            # Import anthropic library
            try:
                import anthropic
            except ImportError:
                error_msg = "ERROR: anthropic library not installed. Run: pip install anthropic"
                return comfy_io.NodeOutput(error_msg, error_msg)

            # Convert image tensor to base64
            base64_image = cls._image_to_base64(image)

            # Build the system prompt based on photography style, target model, color style, photographer, and lighting
            system_prompt = cls._build_system_prompt(photography_style, target_model, color_style, photographer_style, lighting_condition, seed)

            # Initialize Anthropic client
            client = anthropic.Anthropic(api_key=api_key.strip())

            # Create the message
            message = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": base64_image,
                                },
                            },
                            {
                                "type": "text",
                                "text": user_prompt
                            }
                        ],
                    }
                ],
            )

            # Extract the response
            response_text = message.content[0].text

            # Parse the response to separate positive and negative prompts
            positive_prompt, negative_prompt = cls._parse_response(response_text)

            # Log the results
            print(f"\n{'='*60}")
            print(f"SID AI Prompt Generator - Results")
            print(f"{'='*60}")
            print(f"Model: {model}")
            print(f"Photography Style: {photography_style}")
            print(f"Target Model: {target_model}")
            print(f"Color Style: {color_style}")
            print(f"Photographer Style: {photographer_style}")
            print(f"Lighting Condition: {lighting_condition}")
            print(f"Seed: {seed}")
            print(f"\nPOSITIVE PROMPT:")
            print(f"{positive_prompt}")
            print(f"\nNEGATIVE PROMPT:")
            print(f"{negative_prompt}")
            print(f"{'='*60}\n")

            return comfy_io.NodeOutput(positive_prompt, negative_prompt)

        except anthropic.APIError as e:
            error_msg = f"Anthropic API Error: {str(e)}"
            print(f"ERROR: {error_msg}")
            return comfy_io.NodeOutput(error_msg, error_msg)
        except Exception as e:
            error_msg = f"Error generating prompts: {str(e)}"
            print(f"ERROR: {error_msg}")
            return comfy_io.NodeOutput(error_msg, error_msg)

    @staticmethod
    def _image_to_base64(image_tensor) -> str:
        """
        Convert ComfyUI image tensor to base64 string.
        Image tensor is in shape (B, H, W, C) with values in range [0, 1]
        """
        # Take the first image if batch
        if len(image_tensor.shape) == 4:
            image_np = image_tensor[0].cpu().numpy()
        else:
            image_np = image_tensor.cpu().numpy()

        # Convert from [0, 1] float to [0, 255] uint8
        image_np = (image_np * 255).astype(np.uint8)

        # Convert to PIL Image
        pil_image = Image.fromarray(image_np)

        # Convert to base64
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')

        return img_base64

    @staticmethod
    def _build_system_prompt(photography_style: str, target_model: str, color_style: str, photographer_style: str, lighting_condition: str, seed: int) -> str:
        """
        Build the system prompt based on the requested photography style, target model, color style, photographer, and lighting condition.
        """
        base_prompt = """You are an expert professional photographer and prompt engineer specializing in creating comprehensive ComfyUI/Stable Diffusion/FLUX prompts for AI image generation.

Your task is to:
1. Analyze the provided image in extreme detail
2. Generate a POSITIVE PROMPT optimized for the specified photography style and target model
3. Generate a comprehensive NEGATIVE PROMPT with categorized exclusions

Format your response EXACTLY as follows:
POSITIVE:
[Your detailed positive prompt here]

NEGATIVE:
[Your categorized negative prompt here]"""

        # Add color style instruction if specified
        color_style_instruction = ""
        if color_style != "None":
            color_style_instruction = f"""

COLOR STYLE REQUIREMENT:
The prompt MUST incorporate "{color_style}" photography style.
- If "Black and White" or "Monochrome": Specify B&W, monochrome, black and white photography
- If "Color": Emphasize vibrant colors, color photography, RGB
- If "Sepia": Include sepia tone, vintage sepia photography
- If "Cross-Processed": Mention cross-processing, experimental color treatment
- If "High Contrast B&W": Emphasize high contrast black and white, dramatic tones
- If "Low Key": Focus on low key lighting, dark moody tones, minimal highlights
- If "High Key": Focus on high key lighting, bright airy feel, minimal shadows

IMPORTANT: Make sure this color treatment is prominent in the positive prompt."""

        # Add photographer style instruction if specified
        photographer_style_instruction = ""
        if photographer_style != "None":
            photographer_descriptions = {
                "Helmut Newton": "Helmut Newton style: bold, provocative, high contrast black and white, strong shadows, powerful poses, fashion noir, dramatic lighting, Berlin school aesthetic",
                "Peter Lindbergh": "Peter Lindbergh style: raw, intimate black and white, natural beauty, minimal retouching, cinematic quality, film noir influenced, emotional depth, authentic portraiture",
                "Annie Leibovitz": "Annie Leibovitz style: dramatic environmental portraits, rich color, elaborate setups, storytelling, celebrity portraiture, bold compositions, Vanity Fair aesthetic",
                "Richard Avedon": "Richard Avedon style: stark white backgrounds, intense close-ups, psychological depth, minimalist approach, high fashion, raw emotion, museum-quality prints",
                "Irving Penn": "Irving Penn style: elegant simplicity, neutral backgrounds, perfect lighting, modernist approach, still life mastery, platinum prints, timeless sophistication",
                "Herb Ritts": "Herb Ritts style: classical Greek influence, sculptural bodies, high contrast, outdoor locations, natural light, minimalist black and white, athletic elegance",
                "Mario Testino": "Mario Testino style: vibrant colors, glamorous, sensual, warm tones, joyful energy, Vogue aesthetic, Peruvian influence, sun-drenched locations",
                "Steven Meisel": "Steven Meisel style: editorial perfection, trend-setting, high fashion, versatile aesthetics, avant-garde, cinematic references, Vogue Italia influence",
                "Patrick Demarchelier": "Patrick Demarchelier style: soft elegant lighting, natural beauty, romantic feel, classic portraiture, Harper's Bazaar aesthetic, timeless grace",
                "Paolo Roversi": "Paolo Roversi style: dreamlike, soft focus, ethereal quality, Polaroid large format, romantic, painterly, muted colors, nostalgic beauty",
                "Tim Walker": "Tim Walker style: fantastical, surreal, whimsical, elaborate sets, fairy tale aesthetic, British Vogue, theatrical, magical realism",
                "Ellen von Unwerth": "Ellen von Unwerth style: playful, feminine, vintage-inspired, bold colors, Paris nightlife, provocative yet fun, candid energy, film aesthetic",
                "David LaChapelle": "David LaChapelle style: hyper-saturated colors, pop surrealism, baroque maximalism, celebrity culture commentary, elaborate staging, kitsch aesthetic",
                "Steven Klein": "Steven Klein style: dark, edgy, provocative, fashion as art, surreal elements, controversial themes, high contrast, avant-garde fashion",
                "Mert and Marcus": "Mert and Marcus style: high glamour, digital perfection, saturated colors, retouched to perfection, fashion maximalism, bold makeup, Vogue covers",
                "Juergen Teller": "Juergen Teller style: raw, unpolished, snapshot aesthetic, flash photography, anti-fashion fashion, controversial, authentic moments, artistic irreverence",
                "Bruce Weber": "Bruce Weber style: American classics, nostalgic, romantic, athletic bodies, natural settings, Abercrombie & Fitch aesthetic, golden era Hollywood",
                "Terry Richardson": "Terry Richardson style: snapshot flash aesthetic, raw energy, white backdrop, amateur look, controversial, direct flash, candid provocative",
                "Rankin": "Rankin style: bold, graphic, high impact, British aesthetic, celebrity portraiture, fashion editorial, vibrant colors, strong personality",
                "Albert Watson": "Albert Watson style: dramatic lighting, technical mastery, diverse subjects, intense contrast, fashion and celebrity, powerful black and white, iconic imagery"
            }

            photographer_desc = photographer_descriptions.get(photographer_style, f"{photographer_style} style photography")
            photographer_style_instruction = f"""

PHOTOGRAPHER STYLE REQUIREMENT:
Style the prompt to emulate: {photographer_desc}

IMPORTANT: Incorporate the photographer's signature style, techniques, and aesthetic throughout the positive prompt. This should influence:
- Lighting approach
- Composition style
- Post-processing aesthetic
- Subject treatment
- Overall mood and feel
- Technical approach"""

        # Add lighting condition instruction if specified
        lighting_instruction = ""
        if lighting_condition != "None":
            lighting_descriptions = {
                # Studio Lighting
                "Natural Window Light": "soft directional natural window light, diffused daylight, organic shadows, window as main light source, authentic natural feel",
                "Studio Strobes": "professional studio flash/strobe lighting, controlled power output, precise lighting control, sharp shadows, frozen motion",
                "Softbox Lighting": "large softbox modifier, soft diffused light, even illumination, gentle shadows, professional studio setup, flattering skin tones",
                "Beauty Dish": "beauty dish modifier, focused yet soft light, circular catchlights, glamour/beauty lighting, moderate contrast",
                "Ring Light": "ring light/ring flash, shadowless frontal lighting, distinctive circular catchlights, even illumination, fashion/beauty aesthetic",
                "Umbrella Lighting": "umbrella modifier, broad soft light source, budget-friendly studio lighting, wide coverage, soft shadows",
                "Grid Spot": "grid spot/honeycomb grid modifier, focused directional light, controlled spill, dramatic accent lighting, hair/rim light",
                "Reflector Fill": "reflector fill light, bounced natural light, subtle fill, reduced shadows, cost-effective lighting solution",

                # Studio Lighting Techniques
                "Rembrandt Lighting": "Rembrandt lighting technique, triangular light patch on cheek, 45-degree key light, dramatic portrait lighting, classic technique",
                "Split Lighting": "split lighting, half face illuminated half in shadow, 90-degree side lighting, dramatic contrasty look, moody atmosphere",
                "Butterfly Lighting": "butterfly lighting (Paramount lighting), key light above and in front, butterfly shadow under nose, glamour lighting, beauty aesthetic",
                "Loop Lighting": "loop lighting, small nose shadow, key light 30-45 degrees, natural flattering light, versatile portrait technique",
                "Broad Lighting": "broad lighting, illuminates side of face toward camera, wider face appearance, side lighting technique",
                "Short Lighting": "short lighting, illuminates side of face away from camera, slimming effect, dimensional modeling",
                "High Key Studio": "high key lighting setup, bright even illumination, minimal shadows, light airy feel, white/light backgrounds, optimistic mood",
                "Low Key Studio": "low key lighting setup, dramatic shadows, minimal fill light, dark moody atmosphere, black backgrounds, cinematic feel",
                "Clamshell Lighting": "clamshell lighting (butterfly + reflector), beauty lighting setup, soft wraparound light, glamour photography",
                "Edge/Rim Lighting": "edge/rim lighting, backlight creating luminous outline, subject separation from background, dramatic halo effect",

                # Outdoor/Natural Lighting
                "Golden Hour": "golden hour lighting, warm soft sunlight, low sun angle, magical quality, long shadows, amber/golden tones, 1 hour after sunrise or before sunset",
                "Blue Hour": "blue hour lighting, deep blue twilight, soft even light, no direct sun, magical atmosphere, civil twilight period",
                "Harsh Midday Sun": "harsh midday sun, direct overhead sunlight, hard shadows, high contrast, bright highlights, challenging lighting",
                "Overcast Diffused": "overcast sky diffused lighting, soft even illumination, natural softbox effect, no harsh shadows, muted colors, gentle flattering light",
                "Open Shade": "open shade lighting, subject in shadow with open sky, soft directional light, even skin tones, outdoor portrait technique",
                "Backlit": "backlit/backlighting, subject with light source behind, rim light effect, lens flare potential, silhouette or halo lighting",
                "Dusk/Twilight": "dusk/twilight lighting, transitional light quality, soft ambient glow, blue hour approaching, romantic atmosphere",
                "Sunrise": "sunrise lighting, warm directional light, fresh morning quality, soft shadows, optimistic mood, low angle sun",
                "Sunset": "sunset lighting, warm golden/orange tones, dramatic sky colors, low angle directional light, romantic mood, long shadows",

                # Special/Creative Lighting
                "Chiaroscuro": "chiaroscuro lighting technique, dramatic contrast between light and dark, Renaissance painting influence, sculptural lighting, artistic shadows",
                "Dramatic Side Light": "dramatic side lighting, strong directional light from 90 degrees, high contrast, depth and dimension, moody atmospheric",
                "Silhouette": "silhouette lighting, backlit subject, solid dark shape against bright background, no subject detail visible, graphic effect",
                "Candlelight": "candlelight/warm practical light, warm amber glow, soft flickering quality, intimate atmosphere, low light romantic mood",
                "Neon/Colorful": "neon/colorful lighting, vibrant colored lights, urban night aesthetic, cyan/magenta/RGB, creative color grading, cyberpunk mood",
                "Practical Lights": "practical lights visible in scene, lamps/fixtures as light sources, natural motivation, environmental lighting, realistic setup",
                "Mixed Lighting": "mixed lighting sources, combination of natural and artificial, multiple color temperatures, complex realistic lighting",
                "Night Photography": "night photography lighting, artificial lights in darkness, long exposure potential, high ISO, city lights or moon/stars"
            }

            lighting_desc = lighting_descriptions.get(lighting_condition, lighting_condition)
            lighting_instruction = f"""

LIGHTING CONDITION REQUIREMENT:
The prompt MUST incorporate this specific lighting: {lighting_condition}

Lighting characteristics: {lighting_desc}

IMPORTANT: Make the lighting setup/condition a prominent element of the positive prompt. This should specify:
- Type of light source
- Quality of light (hard/soft)
- Direction and position
- Color temperature if relevant
- Shadow characteristics
- Overall mood created by the lighting"""

        base_prompt += color_style_instruction + photographer_style_instruction + lighting_instruction

        photography_style_instructions = {
            "Artistic": """
PHOTOGRAPHY STYLE: ARTISTIC FOCUS

Create a descriptive, mood-focused prompt that emphasizes:
- Subject description with emotional/artistic context
- Pose, expression, and body language details
- Clothing/styling/accessories description
- Setting and environment (indoor/outdoor, specific location)
- Lighting quality and atmosphere (dramatic, soft, natural, etc.)
- Artistic style references (film noir, editorial, fine art, etc.)
- Mood and emotional tone
- Color palette or B&W specification
- Photography aesthetic (cinematic, elegant, intimate, etc.)

Structure: Subject + Pose/Expression + Clothing + Setting + Lighting + Artistic Style + Mood + Quality
Keep it flowing and descriptive, like describing a scene to an artist.
""",

            "Technical": """
PHOTOGRAPHY STYLE: TECHNICAL WITH CAMERA & STUDIO DETAILS

Create a comprehensive technical prompt including:

CAMERA EQUIPMENT:
- Specific camera body (e.g., Hasselblad H6D-100c, Canon EOS R5, Sony A7R V)
- Specific lens (e.g., HC 100mm f/2.2, 85mm f/1.4, 50mm f/1.8)
- Camera settings: ISO (e.g., ISO 100), Aperture (e.g., f/2.8), Shutter speed (e.g., 1/125)
- Medium format/full frame/crop sensor specification

STUDIO SETUP (if applicable):
- Backdrop type and color (seamless, textured, size like 12ft)
- Studio dimensions and floor type
- Shooting environment details

LIGHTING SETUP:
- Main/key light: type (Profoto, Godox), modifier (softbox size), position, power
- Fill light: type, position, power ratio
- Rim/hair light: position, modifier (snoot, grid), purpose
- Background light (if used)
- Light ratios (e.g., 1:3 key to fill)
- Color temperature (e.g., 5500K daylight)

SUBJECT & COMPOSITION:
- Detailed subject description
- Pose and positioning
- Clothing/styling specifics
- Composition and framing
- Depth of field notes

POST-PROCESSING:
- Software used (Capture One, Lightroom, Photoshop)
- B&W conversion method (Silver Efex Pro, etc.)
- Specific adjustments (contrast, grain, dodging/burning)
- Film simulation if applicable

AESTHETIC:
- Photography style (fashion, editorial, fine art)
- Photographer references (Helmut Newton, Peter Lindbergh, etc.)
- Technical approach (chiaroscuro, high key, low key)

Structure: Technical specs + Studio setup + Lighting details + Subject + Post-processing + Aesthetic
""",

            "Minimal": """
PHOTOGRAPHY STYLE: MINIMAL/OPTIMIZED (BEST FOR FLUX)

Create a concise, efficient prompt with essential elements only:
- Core subject description (brief)
- Key pose/expression
- Essential clothing/styling
- Setting (brief - studio, outdoor, location)
- Main lighting characteristic
- Style/aesthetic (1-2 words)
- Quality markers (2-3 terms)

Structure: Subject, pose, clothing, setting, lighting, style, quality
Target: 30-50 words maximum
Example format: "Medium format portrait, woman in black dress, elegant pose, studio lighting, dark backdrop, fashion editorial, high contrast"
""",

            "Ultra-Detailed": """
PHOTOGRAPHY STYLE: ULTRA-DETAILED COMPREHENSIVE REALISM

Create an exhaustively detailed prompt covering EVERY aspect:

CAMERA & TECHNICAL SPECIFICATIONS:
- Exact camera model with sensor details (e.g., Hasselblad H6D-100c medium format, 100MP sensor)
- Exact lens model with specifications (e.g., Hasselblad HC 100mm f/2.2 lens)
- Precise camera settings: ISO value, f-stop, shutter speed, color depth, RAW format
- Tethered workflow if applicable
- Processing software with version

STUDIO ENVIRONMENT:
- Exact backdrop specifications (type, color, dimensions like 15x20ft seamless)
- Studio size and layout
- Floor type and color
- Climate/environment details
- Shooting space dimensions

LIGHTING SETUP (EXTREMELY DETAILED):
- Main key light: exact model (Profoto D2 1000Ws), modifier type and size (3x4ft softbox), exact position (45-degrees camera left, 6ft from subject), power output
- Fill light: exact model, modifier, position, power ratio to key (e.g., 1:3)
- Rim/hair light: exact model, modifier (snoot, grid, etc.), position (clock position like 7 o'clock), purpose (separation, highlight)
- Background light (if any): specifications
- Light meter reading
- Exact color temperature (e.g., 5500K daylight balanced)

SUBJECT DESCRIPTION (COMPREHENSIVE):
- Physical characteristics (physique, build, specific features)
- Exact pose description (every limb, hand position, head tilt angle)
- Facial expression (eyes, mouth, emotional state)
- Hair (length, style, flow, movement)
- Clothing (exact items, materials, textures, colors, fit, details like straps, seams)
- Accessories (if any)
- Skin tone and texture

COMPOSITION & FRAMING:
- Aspect ratio (2:3, 16:9, etc.)
- Framing (full body, 3/4, torso, etc.)
- Subject position in frame (centered, rule of thirds, etc.)
- Negative space usage
- Foreground/background elements

POST-PROCESSING WORKFLOW:
- Capture/processing software (Phase One Capture One Pro, Adobe Lightroom Classic, etc.)
- Black and white conversion method (Silver Efex Pro with specific preset)
- Tonal curve adjustments (high contrast, specific curve shape)
- Dodging and burning specifics
- Film grain simulation (type like Tri-X 400, grain size)
- Sharpening method (unsharp mask with radius/amount)
- Skin retouching approach (natural texture preserved, smoothing level)

AESTHETIC & STYLE:
- Specific photography style (fine art, fashion editorial, commercial, etc.)
- Lighting technique name (chiaroscuro, Rembrandt, split lighting, etc.)
- Photographer/artist references (2-3 names)
- Film era or style references
- Mood descriptors (intimate, dramatic, serene, powerful, etc.)
- Print/output quality (museum quality, gallery print, etc.)

QUALITY & TECHNICAL EXCELLENCE:
- Resolution details
- Tonal range (pure black to white, mid-tone handling)
- Sharpness and focus areas
- Depth of field specifics
- Color/monochrome treatment details
- Final output specifications

Structure: Camera specs → Studio environment → Complete lighting setup → Detailed subject → Composition → Post-processing workflow → Aesthetic → Quality
Target: 200-400 words for maximum detail and realism
"""
        }

        # Model-specific positive prompt guidance
        positive_prompt_guidance = {
            "FLUX": """
POSITIVE PROMPT GUIDELINES FOR FLUX:
- FLUX excels with natural language descriptions - write as you would describe the image to a person
- Use clear, specific descriptive phrases rather than keyword lists
- Structure: Subject + Action/Pose + Setting + Lighting + Style + Quality
- Lighting is crucial: specify direction, quality (soft/hard), color temperature (warm/cool/neutral)
- Be specific about materials and textures (e.g., "brushed metal", "rough canvas", "smooth silk")
- Style descriptors work well: "cinematic", "photorealistic", "dramatic", "ethereal", "vibrant"
- Camera/lens terms: "wide angle", "telephoto compression", "shallow depth of field", "bokeh"
- Quality boosters: "high detail", "sharp focus", "professional photography", "8k uhd", "masterpiece"
- Avoid over-stuffing keywords - FLUX understands context and relationships
- Example structure: "A portrait of [subject] [doing what] in [location], [lighting description], [style], [quality terms]"
""",

            "SDXL": """
POSITIVE PROMPT GUIDELINES FOR SDXL (Stable Diffusion XL):
- SDXL works well with both natural language and weighted keywords
- Effective structure: Main subject + Details + Environment + Lighting + Style + Quality
- Use commas to separate distinct concepts and elements
- Emphasis syntax available: (keyword:1.2) increases weight, (keyword:0.8) decreases
- Art style references work excellently: mention specific artists, art movements, or periods
- Photography terms: specify camera type (DSLR, mirrorless), lens (50mm, 85mm), settings (f/1.8, ISO 100)
- Lighting descriptors: "golden hour", "studio lighting", "rim light", "backlighting", "diffused light"
- Quality terms: "highly detailed", "sharp focus", "professional", "award-winning", "trending on artstation"
- Medium specification: "oil painting", "digital art", "photograph", "3D render", "watercolor"
- Include aspect ratio preferences if important to composition
- SDXL understands context well - you can be descriptive without excessive keywords
""",

            "SD 1.5": """
POSITIVE PROMPT GUIDELINES FOR SD 1.5 (Stable Diffusion 1.5):
- SD 1.5 works best with structured, keyword-based prompts
- Use clear keyword separation with commas
- Front-load important elements - put the most important concepts first
- Heavy use of quality boosters: "highly detailed", "intricate", "sharp focus", "professional", "8k", "hdr"
- Emphasis syntax: (keyword:1.3) or ((keyword)) for importance, [keyword] to reduce weight
- Artist references are powerful: mention 2-3 artists whose styles match your vision
- Specific technical terms: "unreal engine", "octane render", "physically based rendering"
- Lighting keywords: "dramatic lighting", "volumetric lighting", "studio lighting", "cinematic lighting"
- Style modifiers: "hyperrealistic", "photorealistic", "concept art", "fantasy art", "sci-fi"
- Common effective additions: "trending on artstation", "winner of award", "featured on behance"
- Include medium: "digital painting", "oil on canvas", "photograph", "3d render"
- SD 1.5 benefits from more keywords - be comprehensive but organized
""",

            "Universal": """
POSITIVE PROMPT GUIDELINES (UNIVERSAL):
- Create prompts that work across multiple model types
- Clear structure: Subject + Context + Style + Quality
- Be descriptive but not overly verbose
- Include: main subject, action/pose, environment, lighting, artistic style, quality terms
- Use natural descriptive language mixed with effective keywords
- Specify medium (photo, painting, digital art, etc.)
- Include lighting direction and quality
- Add 2-4 quality enhancers (detailed, sharp, professional, high quality)
- Mention camera/artistic technique if relevant
- Balance between natural language (FLUX-friendly) and keywords (SD-friendly)
"""
        }

        # Model-specific negative prompt guidance - CATEGORIZED AND COMPREHENSIVE
        negative_prompt_guidance = {
            "FLUX": """
NEGATIVE PROMPT GUIDELINES FOR FLUX (CATEGORIZED):

Create a comprehensive negative prompt organized by categories. FLUX works best with specific, targeted exclusions.

QUALITY ISSUES:
low quality, low resolution, blurry, out of focus, soft focus, motion blur, compression artifacts, jpeg artifacts, pixelated, poor quality, amateur photography, smartphone photo

LIGHTING ISSUES:
flat lighting, overhead lighting, harsh direct flash, on-camera flash, overexposed, underexposed, blown highlights, crushed blacks, low contrast, washed out, muddy midtones, mixed lighting, fluorescent lighting

COMPOSITION ISSUES:
cluttered background, busy background, cropped face, cropped head, tilted horizon, off-center (unless intended), too much negative space, cramped composition, poor framing

SUBJECT ISSUES (if applicable):
stiff pose, awkward pose, unnatural pose, looking at camera (unless intended), smiling (unless intended), heavy makeup, shiny skin, oily skin, sweaty, over-smoothed skin, plastic skin, airbrushed

ANATOMICAL ISSUES (if people):
bad anatomy, extra limbs, missing limbs, deformed hands, bad hands, fused fingers, extra fingers, missing fingers, malformed hands, poorly drawn face, distorted proportions, unrealistic proportions, long neck, asymmetrical face (unless natural)

STYLE ISSUES:
cartoon, anime, illustration, 3D render, CGI, painting, drawing, oil painting (if photo), sketch, watercolor (if photo)

TECHNICAL ISSUES:
text, watermark, logo, signature, timestamp, frame, border, lens flare, vignetting (unless intended), chromatic aberration, barrel distortion, fisheye effect, wide angle distortion

UNWANTED ELEMENTS:
Based on image analysis, exclude specific unwanted elements like: wrong setting, incorrect props, mismatched clothing style, wrong time period, anachronistic elements, visible equipment, backdrop wrinkles

Format: Organize negative terms with commas, keep concise but comprehensive
""",

            "SDXL": """
NEGATIVE PROMPT GUIDELINES FOR SDXL (CATEGORIZED):

SDXL requires moderate negative prompts - more focused than Universal but less than SD 1.5.

QUALITY ISSUES:
low quality, worst quality, low resolution, blurry, out of focus, soft focus, compression artifacts, jpeg artifacts, pixelated, poor quality, grainy (unless film grain intended), bad detail, blurry-image, bad contrast

LIGHTING ISSUES:
flat lighting, harsh flash, on-camera flash, overexposed, underexposed, poor lighting, mixed color temperature, overhead fluorescent lighting, low contrast lighting, washed out

COMPOSITION ISSUES:
cluttered, busy background, cropped (unless intended), poor composition, off-center (unless rule of thirds), tilted, poorly framed

SUBJECT ISSUES:
poorly rendered face, poorly drawn hands, poorly drawn feet, heavy makeup (unless intended), shiny oily skin, over-retouched, plastic skin, unnatural skin texture

ANATOMICAL ISSUES:
bad anatomy, bad proportions, deformed, duplicate, extra arms, extra legs, extra limbs, missing limbs, cloned face, malformed, disfigured, mutated

STYLE ISSUES:
cartoon, anime, 3D render, CGI, illustration, painting (if photo intended), drawing, sketch

TECHNICAL ISSUES:
watermark, signature, username, text, logo, error, cropped edges, frame border, timestamp

UNWANTED ELEMENTS:
Specific exclusions based on image content - wrong setting, incorrect objects, mismatched style elements

Format: Comma-separated, focused on main issues, avoid over-stuffing
""",

            "SD 1.5": """
NEGATIVE PROMPT GUIDELINES FOR SD 1.5 (COMPREHENSIVE - CATEGORIZED):

SD 1.5 REQUIRES extensive negative prompts. Include ALL relevant categories.

QUALITY ISSUES:
low quality, worst quality, normal quality, low resolution, low res, blurry, bad art, blurred, out of focus, soft focus, motion blur, compression artifacts, jpeg artifacts, noise, pixelated, poor quality, amateur, bad quality, grainy (unless film grain), dull, washed out, undersaturated (unless B&W)

LIGHTING ISSUES:
flat lighting, overhead lighting, harsh flash, direct flash, on-camera flash, red eye, overexposed, underexposed, blown out, dark, too bright, poor lighting, bad lighting, muddy midtones, low contrast, high contrast (unless intended), mixed lighting, fluorescent, tungsten mismatch

COMPOSITION ISSUES:
cluttered, busy, cropped, poorly cropped, bad composition, bad framing, tilted horizon, dutch angle (unless intended), too much negative space, cramped, off-center (unless thirds rule), poorly framed, cut off, out of frame, bad angle

SUBJECT & POSE ISSUES:
ugly, poorly drawn, stiff posture, stiff pose, awkward pose, awkward hands, unnatural pose, bad expression, blank stare, dead eyes, looking away (unless intended), looking at camera (unless intended), smiling (unless intended), teeth showing (unless intended)

ANATOMICAL ISSUES (COMPREHENSIVE):
bad anatomy, poorly drawn face, poorly drawn hands, poorly drawn feet, bad hands, bad feet, deformed, disfigured, extra limbs, missing limbs, extra arms, missing arms, extra legs, missing legs, extra fingers, missing fingers, fused fingers, too many fingers, long neck, short neck, thick neck, malformed, mutated, mutation, body horror, duplicate, cloned face, asymmetrical eyes, asymmetrical face (unless natural), distorted proportions, wrong proportions, elongated, compressed, stretched

SKIN & TEXTURE ISSUES:
shiny skin, oily skin, greasy, sweaty, blemished (unless intended), acne (unless intended), over-smoothed, plastic skin, waxy skin, airbrushed, fake skin, unnatural skin, mannequin, doll-like

STYLE ISSUES (if realistic photo intended):
cartoon, anime, manga, illustration, digital art, painting, drawing, sketch, oil painting, watercolor, 3D render, CGI, computer generated, render, clay, sculpture, comic book, graphic novel, stylized

TECHNICAL ISSUES:
watermark, text, signature, username, logo, copyright, watermark text, artist name, photographer credit, timestamp, date, frame, border, vignetting (unless intended), lens flare (unless intended), chromatic aberration, purple fringing, barrel distortion, fisheye, wide angle distortion, perspective distortion

ARTIFACTS & ERRORS:
tiling, repetition, tessellation, mirrored, duplicated elements, copy-paste artifacts, seams, stitching errors, blending errors, morphing, melting, dissolving, fragmented, glitchy, corrupted, artifacting

UNWANTED ELEMENTS:
Multiple people (unless intended), crowd (unless intended), other faces, hands in frame (unless intended), props (unless intended), furniture (except intended), visible equipment, studio equipment, light stands, backdrop wrinkles, backdrop seams, cables, reflections (unless intended), shadows (unless intended)

PROCESSING ISSUES:
oversharpened, over-processed, over-saturated (unless intended), HDR effect (unless intended), artificial, fake, filters, Instagram filter, beauty filter, Snapchat filter, face app, artificial bokeh, fake depth of field

Format: Comprehensive comma-separated list, front-load most important exclusions, longer is better for SD 1.5
""",

            "Universal": """
NEGATIVE PROMPT GUIDELINES (UNIVERSAL - CATEGORIZED):

Create a balanced negative prompt with essential categories that work across all models.

QUALITY ISSUES:
low quality, low resolution, blurry, out of focus, pixelated, compression artifacts, poor quality, amateur

LIGHTING ISSUES:
flat lighting, harsh flash, overexposed, underexposed, poor lighting

COMPOSITION ISSUES:
cluttered background, busy, poorly framed, cropped (unless intended)

SUBJECT ISSUES:
stiff pose, awkward pose, poorly drawn

ANATOMICAL ISSUES (if applicable):
bad anatomy, extra limbs, deformed hands, poorly drawn face, bad proportions, distorted proportions

STYLE ISSUES:
cartoon, anime, 3D render, CGI, illustration (if photo intended)

TECHNICAL ISSUES:
watermark, signature, text, logo, lens flare, distortion

UNWANTED ELEMENTS:
Context-specific exclusions based on image analysis

Format: Balanced, moderate length, works across FLUX/SDXL/SD 1.5
"""
        }

        # Combine all parts
        full_prompt = base_prompt + "\n\n" + photography_style_instructions.get(photography_style, photography_style_instructions["Technical"])
        full_prompt += "\n\n" + positive_prompt_guidance.get(target_model, positive_prompt_guidance["Universal"])
        full_prompt += "\n\n" + negative_prompt_guidance.get(target_model, negative_prompt_guidance["Universal"])

        return full_prompt

    @staticmethod
    def _parse_response(response_text: str) -> tuple[str, str]:
        """
        Parse the Claude response to extract positive and negative prompts.
        """
        positive_prompt = ""
        negative_prompt = ""

        # Split by sections
        parts = response_text.split("NEGATIVE:")

        if len(parts) >= 2:
            # Extract positive part (remove "POSITIVE:" label)
            positive_part = parts[0].replace("POSITIVE:", "").strip()
            positive_prompt = positive_part

            # Extract negative part
            negative_prompt = parts[1].strip()
        else:
            # Fallback: if format not followed, put everything in positive
            positive_prompt = response_text.strip()
            negative_prompt = "low quality, blurry, distorted, deformed, bad anatomy, poorly drawn"

        return positive_prompt, negative_prompt
