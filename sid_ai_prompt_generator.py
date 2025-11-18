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
    Intelligent AI prompting agent that analyzes images using Anthropic's Claude API
    and generates model-optimized prompts with automatic model research and sampler recommendations.

    INTELLIGENT AGENT FEATURES:
    - Automatic model research and type detection (FLUX, SDXL, SD 1.5)
    - Model-specific prompt optimization based on research
    - Token limit management per model type
    - Sampler configuration recommendations (markdown format)
    - Priority-based prompt sequencing (style → subject → technical → quality)

    CREATIVE FEATURES:
    - Multiple photography styles (Artistic, Technical, Minimal, Ultra-Detailed)
    - Color style options (B&W, Color, Sepia, High Contrast, etc.)
    - 20 famous photographer aesthetics
    - 31 professional lighting conditions (Studio, Outdoor, Techniques, Special)

    Inputs:
    - image: The input image to analyze
    - api_key: Anthropic API key
    - model: Claude model to use (Sonnet 4.5, Haiku 4.5, Opus 4.1, etc.)
    - user_prompt: Additional context or specific instructions
    - photography_style: Style of prompt (Artistic, Technical, Minimal, Ultra-Detailed)
    - target_model: IMAGE GENERATION model name (e.g., FLUX.1-dev, SDXL 1.0, juggernaut_reborn)
    - color_style: Color treatment (None, B&W, Color, Sepia, etc.)
    - photographer_style: Emulate famous photographer (None, Helmut Newton, Peter Lindbergh, etc.)
    - lighting_condition: Specific lighting setup (None, Studio, Outdoor, Techniques, Special)
    - seed: Seed for randomization (use different seeds for variation)
    - temperature: Creativity level (0.0-1.0)
    - max_tokens: Maximum response length

    Outputs:
    - positive: Comma-separated prompt optimized for target model with priority sequencing
    - negative: Model-appropriate negative terms (concise/moderate/comprehensive)
    - sampler_config: Markdown-formatted optimal sampler settings for the target model

    Version: 3.0.1 - Fixed color style enforcement and enhanced model detection
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
                comfy_io.String.Input(
                    "target_model",
                    default="FLUX.1-dev",
                    multiline=False,
                    force_input=False,
                    tooltip="Target image generation model name (e.g., FLUX.1-dev, SDXL 1.0, SD 1.5, juggernaut_reborn). Can be typed or connected from model loader. Agent will research and optimize prompts for this specific model."
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
                    control_after_generate=True,
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
                comfy_io.String.Output("positive"),
                comfy_io.String.Output("negative"),
                comfy_io.String.Output("sampler_config"),
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

        # Research the target model to get detailed information
        model_info = cls._research_model(target_model)

        # Validate API key
        if not api_key or api_key.strip() == "":
            error_msg = "ERROR: Anthropic API key is required. Get one at https://console.anthropic.com/"
            return comfy_io.NodeOutput(error_msg, error_msg, error_msg)

        try:
            # Import anthropic library
            try:
                import anthropic
            except ImportError:
                error_msg = "ERROR: anthropic library not installed. Run: pip install anthropic"
                return comfy_io.NodeOutput(error_msg, error_msg, error_msg)

            # Convert image tensor to base64
            base64_image = cls._image_to_base64(image)

            # Build the system prompt based on photography style, model_info, color style, photographer, and lighting
            system_prompt = cls._build_system_prompt(photography_style, model_info, color_style, photographer_style, lighting_condition, seed)

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

            # Generate sampler configuration markdown
            sampler_config = cls._generate_sampler_config(model_info)

            # Log the results
            print(f"\n{'='*60}")
            print(f"SID AI Prompt Generator - Results")
            print(f"{'='*60}")
            print(f"Model: {model}")
            print(f"Photography Style: {photography_style}")
            print(f"Target Model: {model_info['name']} (Type: {model_info['type']})")
            print(f"Color Style: {color_style}")
            print(f"Photographer Style: {photographer_style}")
            print(f"Lighting Condition: {lighting_condition}")
            print(f"Seed: {seed}")
            print(f"\nMODEL RESEARCH:")
            print(f"  Base Model: {model_info['base_model']}")
            print(f"  Token Limit: {model_info['token_limit']}")
            print(f"  Negative Strategy: {model_info['negative_strategy']}")
            if model_info['characteristics']:
                print(f"  Characteristics:")
                for char in model_info['characteristics']:
                    print(f"    - {char}")
            print(f"\nPOSITIVE PROMPT:")
            print(f"{positive_prompt}")
            print(f"\nNEGATIVE PROMPT:")
            print(f"{negative_prompt}")
            print(f"\nSAMPLER CONFIG:")
            print(f"{sampler_config}")
            print(f"{'='*60}\n")

            return comfy_io.NodeOutput(positive_prompt, negative_prompt, sampler_config)

        except anthropic.APIError as e:
            error_msg = f"Anthropic API Error: {str(e)}"
            print(f"ERROR: {error_msg}")
            return comfy_io.NodeOutput(error_msg, error_msg, error_msg)
        except Exception as e:
            error_msg = f"Error generating prompts: {str(e)}"
            print(f"ERROR: {error_msg}")
            return comfy_io.NodeOutput(error_msg, error_msg, error_msg)

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
    def _research_model(model_name: str) -> dict:
        """
        Research the target model using pattern matching and inference.

        NOTE: Currently uses intelligent pattern matching. Real-time WebSearch/MCP integration
        requires access to external APIs which are not available in node execution context.

        Future enhancement: Can be extended with CivitAI/HuggingFace API calls if API keys provided.

        Returns comprehensive model information for prompt tailoring.
        """
        print(f"\n{'='*60}")
        print(f"RESEARCHING MODEL: {model_name}")
        print(f"{'='*60}")

        model_info = {
            "name": model_name,
            "type": "Unknown",
            "base_model": "Unknown",
            "characteristics": [],
            "optimal_settings": {},
            "prompt_style": "comma-separated",
            "token_limit": 120,
            "negative_strategy": "moderate",
            "research_source": "pattern_matching"
        }

        try:
            print(f"[Pattern Matching] Analyzing model name: {model_name}")

            # Infer model type from name
            model_name_lower = model_name.lower()

            # Detect FLUX models
            if "flux" in model_name_lower:
                model_info["type"] = "FLUX"
                model_info["base_model"] = "FLUX"
                model_info["token_limit"] = 256
                model_info["negative_strategy"] = "concise"
                model_info["characteristics"] = [
                    "natural language understanding",
                    "excellent with detailed descriptions",
                    "strong lighting comprehension",
                    "material/texture aware"
                ]
                model_info["optimal_settings"] = {
                    "sampler": "Euler a",
                    "steps": "20-30",
                    "cfg_scale": "3.5-7.0",
                    "guidance": "3.5 recommended"
                }
                print(f"[Detected] FLUX model - optimizing for natural language prompts")

            # Detect SDXL models (expanded pattern list)
            elif any(x in model_name_lower for x in [
                "sdxl", "xl", "juggernaut", "proteus", "copax", "pony", "animagine",
                "realvis xl", "realism engine", "zavychroma", "duchaiten", "dreamshaper xl",
                "crystal clear", "hellonijicute", "counterfeit xl"
            ]):
                model_info["type"] = "SDXL"
                model_info["base_model"] = "SDXL"
                model_info["token_limit"] = 75
                model_info["negative_strategy"] = "moderate"
                model_info["characteristics"] = [
                    "keyword and natural language hybrid",
                    "strong art style references",
                    "emphasis syntax support (term:1.2)",
                    "excellent composition"
                ]
                model_info["optimal_settings"] = {
                    "sampler": "DPM++ 2M Karras or Euler a",
                    "steps": "20-40",
                    "cfg_scale": "7.0-9.0",
                    "clip_skip": "1-2"
                }
                print(f"[Detected] SDXL model - optimizing for balanced prompts")

            # Detect SD 1.5 models (expanded pattern list)
            elif any(x in model_name_lower for x in [
                "sd15", "sd 1.5", "sd1.5", "v1-5", "v1.5", "1.5", "realistic vision", "dreamshaper",
                "deliberate", "epicrealism", "cyberrealistic", "majicmix", "chilloutmix",
                "revanimated", "absolutereality", "analog madness", "perfect world"
            ]):
                model_info["type"] = "SD 1.5"
                model_info["base_model"] = "SD 1.5"
                model_info["token_limit"] = 75
                model_info["negative_strategy"] = "comprehensive"
                model_info["characteristics"] = [
                    "keyword-based prompting",
                    "front-loading important terms",
                    "heavy quality boosters needed",
                    "artist name references powerful"
                ]
                model_info["optimal_settings"] = {
                    "sampler": "DPM++ 2M Karras or Euler a",
                    "steps": "25-50",
                    "cfg_scale": "7.0-12.0",
                    "clip_skip": "1-2"
                }
                print(f"[Detected] SD 1.5 model - optimizing for keyword-heavy prompts")

            # Default/Unknown
            else:
                model_info["type"] = "Universal"
                model_info["optimal_settings"] = {
                    "sampler": "DPM++ 2M Karras",
                    "steps": "20-30",
                    "cfg_scale": "7.0-9.0"
                }
                print(f"[Unknown] Using universal optimization")

            print(f"[Research Complete] Type: {model_info['type']}, Token Limit: {model_info['token_limit']}")

        except Exception as e:
            print(f"[Research Error] {str(e)} - using defaults")

        print(f"{'='*60}\n")
        return model_info

    @staticmethod
    def _build_system_prompt(photography_style: str, model_info: dict, color_style: str, photographer_style: str, lighting_condition: str, seed: int) -> str:
        """
        Build the system prompt for comma-separated output with token limits based on target model.
        Token limits: FLUX (200-256 optimal, max 512), SDXL (27-75), SD 1.5 (75), Universal (100-120)
        """

        # Extract model type and get token limits
        model_type = model_info.get("type", "Universal")
        model_name = model_info.get("name", "Unknown")
        characteristics = model_info.get("characteristics", [])

        # Model-specific token limits
        token_limits = {
            "FLUX": {"optimal": "200-256 tokens", "max": "512 tokens", "negative": "concise (30-50 tokens)"},
            "SDXL": {"optimal": "27-75 tokens", "max": "75 tokens", "negative": "moderate (40-60 tokens)"},
            "SD 1.5": {"optimal": "50-75 tokens", "max": "75 tokens", "negative": "comprehensive (75-100 tokens)"},
            "Universal": {"optimal": "100-120 tokens", "max": "120 tokens", "negative": "balanced (50-70 tokens)"}
        }

        limits = token_limits.get(model_type, token_limits["Universal"])

        # Photographer style descriptions (kept as-is per requirements)
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

        # Lighting descriptions (kept as-is per requirements)
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
            # Studio Techniques
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

        # Build base prompt
        # Build model characteristics string
        char_str = "\n- ".join(characteristics) if characteristics else "General purpose"

        base_prompt = f"""You are an expert prompt engineer for AI image generation. Your task is to analyze the provided image and generate COMMA-SEPARATED prompts optimized for {model_name} ({model_type}).

MODEL INFORMATION:
- Model Name: {model_name}
- Base Type: {model_type}
- Characteristics:
- {char_str}

CRITICAL OUTPUT FORMAT:
- Generate ONE LINE of comma-separated terms for positive prompt
- Generate ONE LINE of comma-separated terms for negative prompt
- NO markdown, NO bullet points, NO structured lists
- Format EXACTLY as:

POSITIVE:
term1, term2, term3, term4, etc

NEGATIVE:
term1, term2, term3, term4, etc

TOKEN LIMITS FOR {model_name}:
- Positive prompt: {limits["optimal"]} (maximum {limits["max"]})
- Negative prompt: {limits["negative"]}

PROMPT SEQUENCING PRIORITY (CRITICAL):
Priority 1: User instructions + Style requirements (color, photographer, lighting)
Priority 2: Subject description (pose, expression, clothing)
Priority 3: Technical details (camera, settings, composition)
Priority 4: Quality terms (professional, high detail, sharp focus)

Place the MOST IMPORTANT terms FIRST in the sequence."""

        # Add color style requirement if specified
        if color_style != "None":
            color_terms_map = {
                "Black and White": "black and white photography, monochrome, grayscale, B&W, no color",
                "Color": "vibrant color photography, full color, saturated colors, colorful, RGB color",
                "Monochrome": "monochrome photography, single color tone, monotone",
                "Sepia": "sepia tone, vintage sepia photography, warm brown tones, sepia tinted",
                "Cross-Processed": "cross-processed, experimental color grading, color shifted",
                "High Contrast B&W": "high contrast black and white, dramatic monochrome, stark B&W",
                "Low Key": "low key lighting, dark moody tones, predominantly dark, minimal highlights",
                "High Key": "high key lighting, bright airy, predominantly bright, minimal shadows"
            }
            color_terms = color_terms_map.get(color_style, color_style)
            base_prompt += f"""

⚠️ CRITICAL COLOR REQUIREMENT (HIGHEST PRIORITY):
The image MUST be {color_style.upper()}.

MANDATORY: Start your positive prompt with these EXACT comma-separated terms:
{color_terms}

ENFORCE: Do NOT include contradicting color terms. If "{color_style}" is specified:
- For B&W/Monochrome: DO NOT include "color", "colorful", "vibrant colors", "RGB"
- For Color: DO NOT include "black and white", "monochrome", "B&W", "grayscale"

This is NON-NEGOTIABLE. The color style MUST be the FIRST terms in the prompt."""

        # Add photographer style requirement if specified
        if photographer_style != "None":
            photographer_desc = photographer_descriptions.get(photographer_style, f"{photographer_style} style photography")
            base_prompt += f"""

PHOTOGRAPHER STYLE REQUIREMENT (Priority 1):
Style: {photographer_desc}
Incorporate the photographer's signature aesthetic as PRIMARY terms early in the prompt."""

        # Add lighting requirement if specified
        if lighting_condition != "None":
            lighting_desc = lighting_descriptions.get(lighting_condition, lighting_condition)
            base_prompt += f"""

LIGHTING REQUIREMENT (Priority 1):
Lighting: {lighting_desc}
Include lighting characteristics as PRIMARY terms early in the prompt."""

        # Photography style-specific instructions
        photography_style_instructions = {
            "Artistic": f"""

PHOTOGRAPHY STYLE: ARTISTIC FOCUS
Generate comma-separated terms emphasizing:
- Mood and atmosphere terms
- Artistic style references
- Emotional tone descriptors
- Scene and environment
- Photography aesthetic terms

Target length: {limits["optimal"]}
Example sequence: style terms, mood, subject, setting, lighting, artistic references, quality""",

            "Technical": f"""

PHOTOGRAPHY STYLE: TECHNICAL DETAILS
Generate comma-separated terms including:
- Camera specifications (Hasselblad H6D-100c, Canon EOS R5, etc)
- Lens details (85mm f/1.4, 50mm f/1.2, etc)
- Camera settings (ISO 100, f/2.8, 1/125)
- Lighting setup terms
- Studio equipment mentions
- Post-processing terms

Target length: {limits["optimal"]}
Example sequence: camera model, lens, settings, subject, lighting setup, technical terms, quality""",

            "Minimal": f"""

PHOTOGRAPHY STYLE: MINIMAL/CONCISE
Generate essential comma-separated terms only:
- Core subject descriptor
- Key pose or action
- Primary setting
- Main lighting characteristic
- Style aesthetic
- 2-3 quality terms

Target length: {limits["optimal"]}
Example sequence: subject, pose, setting, lighting, style, quality""",

            "Ultra-Detailed": f"""

PHOTOGRAPHY STYLE: ULTRA-DETAILED COMPREHENSIVE
Generate extensive comma-separated terms covering:
- Exact camera model and specifications
- Precise lens details
- Complete lighting setup description
- Detailed subject characteristics
- Composition specifics
- Post-processing workflow terms
- Technical excellence terms

Target length: {limits["optimal"]} (can approach {limits["max"]} for this style)
Example sequence: camera specs, lens, lighting details, subject description, composition, post-processing, quality"""
        }

        base_prompt += photography_style_instructions.get(photography_style, photography_style_instructions["Technical"])

        # Model-specific guidance
        model_specific_guidance = {
            "FLUX": f"""

FLUX OPTIMIZATION:
- Use natural descriptive terms in logical sequence
- Lighting terms are crucial - be specific
- Material and texture descriptors work well
- Style terms: cinematic, photorealistic, dramatic, ethereal
- Camera terms: wide angle, shallow depth of field, bokeh
- Quality terms: high detail, sharp focus, professional photography, 8k uhd, masterpiece
- Keep to {limits["optimal"]} for optimal results""",

            "SDXL": f"""

SDXL OPTIMIZATION:
- Comma-separated weighted terms work best
- Can use emphasis syntax: (term:1.2) for importance
- Art style and artist references effective
- Photography terms: DSLR, 85mm lens, f/1.8, ISO 100
- Lighting: golden hour, studio lighting, rim light, backlighting
- Quality terms: highly detailed, sharp focus, professional, award-winning
- Medium specification: photograph, digital art, oil painting
- Keep to {limits["optimal"]} tokens""",

            "SD 1.5": f"""

SD 1.5 OPTIMIZATION:
- Structured keyword-based comma-separated terms
- Front-load most important concepts first
- Heavy quality boosters: highly detailed, intricate, sharp focus, 8k, hdr
- Emphasis syntax: (term:1.3) or ((term)) for importance
- Artist references powerful: mention 2-3 artists
- Technical terms: unreal engine, octane render, physically based rendering
- Style modifiers: hyperrealistic, photorealistic, concept art
- Common additions: trending on artstation, award winner
- Maximum {limits["max"]} tokens""",

            "Universal": f"""

UNIVERSAL OPTIMIZATION:
- Balanced comma-separated terms for cross-model compatibility
- Clear sequence: style, subject, context, lighting, quality
- Natural descriptive language mixed with keywords
- Specify medium: photo, painting, digital art
- Include lighting direction and quality
- Add 3-5 quality enhancers
- Camera/artistic technique terms
- Keep to {limits["optimal"]} tokens"""
        }

        base_prompt += model_specific_guidance.get(model_type, model_specific_guidance["Universal"])

        # Negative prompt guidance
        negative_prompt_guidance = {
            "FLUX": f"""

NEGATIVE PROMPT FOR FLUX ({limits["negative"]}):
Generate concise comma-separated exclusion terms:
- Quality issues: low quality, blurry, out of focus, pixelated, compression artifacts
- Lighting issues: flat lighting, harsh flash, overexposed, underexposed
- Composition issues: cluttered background, cropped face, poor framing
- Anatomical issues (if people): bad anatomy, extra limbs, deformed hands, distorted proportions
- Style issues: cartoon, anime, 3D render (if photo intended)
- Technical issues: watermark, text, signature, lens flare
Keep concise, comma-separated, one line.""",

            "SDXL": f"""

NEGATIVE PROMPT FOR SDXL ({limits["negative"]}):
Generate moderate comma-separated exclusion terms:
- Quality: low quality, worst quality, blurry, pixelated, compression artifacts, poor quality
- Lighting: flat lighting, harsh flash, overexposed, underexposed, poor lighting, washed out
- Composition: cluttered, busy background, cropped, poorly framed
- Subject: poorly rendered face, poorly drawn hands, plastic skin, over-retouched
- Anatomy: bad anatomy, bad proportions, deformed, extra limbs, malformed
- Style: cartoon, anime, 3D render, CGI, illustration (if photo)
- Technical: watermark, signature, text, logo, timestamp
Moderate length, comma-separated, one line.""",

            "SD 1.5": f"""

NEGATIVE PROMPT FOR SD 1.5 ({limits["negative"]}):
Generate comprehensive comma-separated exclusion terms:
- Quality: low quality, worst quality, blurry, bad art, out of focus, pixelated, poor quality, amateur, grainy, dull, washed out
- Lighting: flat lighting, harsh flash, overexposed, underexposed, poor lighting, low contrast, muddy midtones
- Composition: cluttered, busy, poorly cropped, bad composition, tilted, poorly framed, cut off, out of frame
- Subject/Pose: ugly, poorly drawn, stiff pose, awkward pose, bad expression, blank stare
- Anatomy: bad anatomy, poorly drawn face, poorly drawn hands, deformed, extra limbs, missing limbs, fused fingers, long neck, distorted proportions
- Skin: shiny skin, oily skin, plastic skin, waxy skin, airbrushed, fake skin, mannequin
- Style: cartoon, anime, illustration, painting, drawing, 3D render, CGI (if photo)
- Technical: watermark, text, signature, logo, frame, border, chromatic aberration, distortion
- Artifacts: tiling, repetition, duplicated elements, glitchy, corrupted
- Processing: oversharpened, over-processed, artificial, filters, fake depth of field
Comprehensive length up to {limits["negative"]}, comma-separated, one line.""",

            "Universal": f"""

NEGATIVE PROMPT UNIVERSAL ({limits["negative"]}):
Generate balanced comma-separated exclusion terms:
- Quality: low quality, blurry, out of focus, pixelated, poor quality, amateur
- Lighting: flat lighting, harsh flash, overexposed, underexposed
- Composition: cluttered background, busy, poorly framed
- Subject: stiff pose, awkward pose, poorly drawn
- Anatomy: bad anatomy, extra limbs, deformed hands, bad proportions
- Style: cartoon, anime, 3D render, CGI (if photo intended)
- Technical: watermark, signature, text, logo, distortion
Balanced length, comma-separated, one line."""
        }

        base_prompt += negative_prompt_guidance.get(model_type, negative_prompt_guidance["Universal"])

        base_prompt += """

FINAL REMINDER:
- Output ONLY comma-separated terms on ONE LINE each for positive and negative
- NO markdown formatting, NO lists, NO categories in the output
- Respect token limits strictly
- Sequence by priority: style requirements first, then subject, then technical, then quality
- Format as:

POSITIVE:
comma, separated, terms, here

NEGATIVE:
comma, separated, terms, here"""

        return base_prompt

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

    @staticmethod
    def _generate_sampler_config(model_info: dict) -> str:
        """
        Generate a markdown-formatted sampler configuration guide based on model research.
        Returns a pretty markdown table with optimal settings.
        """
        model_name = model_info.get("name", "Unknown")
        model_type = model_info.get("type", "Universal")
        optimal_settings = model_info.get("optimal_settings", {})

        # Build markdown table
        markdown = f"# Optimal Sampler Settings for {model_name}\n\n"
        markdown += f"**Model Type:** {model_type}\n\n"

        if optimal_settings:
            markdown += "| Setting | Recommended Value |\n"
            markdown += "|---------|------------------|\n"

            # Add each setting to the table
            for key, value in optimal_settings.items():
                # Convert snake_case to Title Case for display
                setting_name = key.replace("_", " ").title()
                markdown += f"| {setting_name} | {value} |\n"
        else:
            markdown += "*No specific settings available. Use default sampler configuration.*\n"

        # Add additional guidance based on model type
        markdown += f"\n## Guidance\n\n"

        if model_type == "FLUX":
            markdown += "- FLUX models work best with lower CFG values (3.5-7.0)\n"
            markdown += "- Use fewer steps (20-30) for optimal quality/speed balance\n"
            markdown += "- Euler a sampler provides excellent results\n"
        elif model_type == "SDXL":
            markdown += "- SDXL benefits from DPM++ samplers with Karras noise\n"
            markdown += "- Moderate CFG (7.0-9.0) provides best results\n"
            markdown += "- 20-40 steps recommended depending on detail needs\n"
        elif model_type == "SD 1.5":
            markdown += "- SD 1.5 may need higher CFG (7.0-12.0) for prompt adherence\n"
            markdown += "- 25-50 steps recommended for quality\n"
            markdown += "- DPM++ 2M Karras or Euler a samplers work well\n"
        else:
            markdown += "- Start with standard DPM++ 2M Karras sampler\n"
            markdown += "- Use CFG 7.0-9.0 as baseline\n"
            markdown += "- Adjust steps based on quality needs (20-30 typical)\n"

        return markdown
