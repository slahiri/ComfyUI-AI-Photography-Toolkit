# ComfyUI-AI-Photography-Toolkit

A collection of AI-powered photography and image generation tools for ComfyUI. All nodes are prefixed with `SID_` for easy identification.

## Features

### Current Nodes

#### SID_AIPromptGenerator
AI-powered prompt generator that analyzes images using Anthropic's Claude API (with vision capabilities) and generates detailed positive and negative prompts for image generation workflows.

**Features:**
- Analyzes input images using Claude's vision capabilities
- Generates detailed positive prompts describing the image
- Creates negative prompts to avoid unwanted elements
- Multiple detail levels (Basic, Detailed, Very Detailed)
- Supports all Claude 3.x models including Sonnet, Opus, and Haiku
- Adjustable creativity via temperature control
- Customizable maximum token length

### Future Nodes (Planned)
- SID_ImageAnalyzer - Detailed image analysis and metadata extraction
- SID_StyleTransfer - AI-powered style transfer using LLM guidance
- SID_OpenAIPromptGenerator - OpenAI GPT-4 Vision based prompt generation
- And more photography-focused AI tools!

## Installation

### Method 1: Manual Installation (Recommended)

1. Navigate to your ComfyUI custom_nodes directory:
   ```bash
   cd ComfyUI/custom_nodes
   ```

2. Clone this repository:
   ```bash
   git clone https://github.com/[YOUR_USERNAME]/ComfyUI-AI-Photography-Toolkit.git
   ```

3. Install dependencies:
   ```bash
   cd ComfyUI-AI-Photography-Toolkit
   pip install -r requirements.txt
   ```

4. Restart ComfyUI

### Method 2: Via ComfyUI Manager

1. Open ComfyUI Manager
2. Search for "ComfyUI-AI-Photography-Toolkit"
3. Click Install
4. Restart ComfyUI

## Usage

### SID_AIPromptGenerator

1. **Get an Anthropic API Key**
   - Visit https://console.anthropic.com/
   - Sign up or log in
   - Create an API key from the dashboard
   - Copy your API key

2. **Add the Node to Your Workflow**
   - In ComfyUI, add the "SID AI Prompt Generator" node
   - Connect an image to the `image` input

3. **Configure the Node**
   - **api_key**: Paste your Anthropic API key
   - **model**: Choose your preferred Claude model
     - `claude-sonnet-4-5-20250929` (Recommended - Latest and most capable Sonnet)
     - `claude-haiku-4-5-20251001` (Latest Haiku - Fast and economical)
     - `claude-opus-4-1-20250805` (Latest Opus - Most capable)
     - `claude-3-5-haiku-20241022` (Claude 3.5 Haiku)
     - `claude-3-haiku-20240307` (Claude 3 Haiku)
   - **user_prompt**: Add specific instructions or context
   - **photography_style**: Choose the prompt format style
     - `Artistic`: Creative, mood-focused descriptions
     - `Technical`: Camera specs, lighting setup, studio details (Recommended)
     - `Minimal`: Concise, optimized for quick generation
     - `Ultra-Detailed`: Exhaustive specs - camera, lighting, post-processing (200-400 words)
   - **target_model**: Select your target image generation model
     - `FLUX`: Optimized for FLUX models (concise, specific negatives)
     - `SDXL`: Optimized for Stable Diffusion XL (moderate negatives)
     - `SD 1.5`: Optimized for SD 1.5 (comprehensive negatives)
     - `Universal`: Works across all models
   - **color_style**: ⭐ NEW! Choose color treatment
     - `None`: Based on image analysis (default)
     - `Black and White`: B&W photography
     - `Color`: Vibrant color photography
     - `Monochrome`: Single color tone
     - `Sepia`: Vintage sepia tone
     - `Cross-Processed`: Experimental color treatment
     - `High Contrast B&W`: Dramatic black and white
     - `Low Key`: Dark, moody tones
     - `High Key`: Bright, airy feel
   - **photographer_style**: ⭐ NEW! Emulate famous photographer
     - `None`: No specific photographer style (default)
     - 20 Famous photographers including:
       - Helmut Newton (bold, provocative, high contrast B&W)
       - Peter Lindbergh (raw, intimate, cinematic B&W)
       - Annie Leibovitz (dramatic environmental portraits)
       - Richard Avedon (stark white backgrounds, psychological depth)
       - And 16 more legendary photographers!
   - **lighting_condition**: ⭐ NEW! Specific lighting setup
     - `None`: Based on image analysis (default)
     - **Studio Lighting (8 options)**: Natural Window Light, Studio Strobes, Softbox Lighting, Beauty Dish, Ring Light, Umbrella Lighting, Grid Spot, Reflector Fill
     - **Studio Techniques (10 options)**: Rembrandt, Split, Butterfly, Loop, Broad, Short, High Key Studio, Low Key Studio, Clamshell, Edge/Rim Lighting
     - **Outdoor/Natural (9 options)**: Golden Hour, Blue Hour, Harsh Midday Sun, Overcast Diffused, Open Shade, Backlit, Dusk/Twilight, Sunrise, Sunset
     - **Special/Creative (8 options)**: Chiaroscuro, Dramatic Side Light, Silhouette, Candlelight, Neon/Colorful, Practical Lights, Mixed Lighting, Night Photography
   - **seed**: ⭐ NEW! Randomization seed (change for variations)
   - **temperature**: Control creativity (0.0 = focused, 1.0 = creative)
   - **max_tokens**: Maximum response length (1024 recommended)

4. **Run the Workflow**
   - The node will output two text strings:
     - `positive_prompt`: Use this as your main prompt
     - `negative_prompt`: Use this to exclude unwanted elements

5. **Use the Generated Prompts**
   - Connect `positive_prompt` to your text encoder or prompt node
   - Connect `negative_prompt` to your negative prompt input

### Example Workflow

```
Image Input → SID_AIPromptGenerator → Positive Prompt → CLIP Text Encode
                      ↓
                Negative Prompt → CLIP Text Encode (Negative)
```

### Understanding Photography Styles

The `photography_style` parameter controls the level and type of detail in generated prompts:

#### Artistic (Creative Focus)
- **Best for**: Creative, mood-driven image generation
- **Includes**: Subject, pose, expression, clothing, setting, lighting quality, artistic style, mood
- **Structure**: Flowing, descriptive language like describing a scene to an artist
- **Length**: Moderate (100-150 words)
- **Example elements**: "dramatic portrait", "serene expression", "soft golden hour lighting", "film noir aesthetic"

#### Technical (Camera & Studio Details) - RECOMMENDED
- **Best for**: Professional photography recreation with realistic technical specs
- **Includes**:
  - Camera equipment (Hasselblad H6D, Canon R5, Sony A7R V)
  - Lens specs (85mm f/1.4, 100mm f/2.2)
  - Camera settings (ISO, aperture, shutter speed)
  - Studio setup (backdrop, dimensions)
  - Lighting setup (Profoto, Godox, positions, modifiers, ratios)
  - Post-processing (Capture One, Silver Efex Pro, etc.)
  - Photographer references (Helmut Newton, Peter Lindbergh)
- **Structure**: Technical specs → Studio setup → Lighting → Subject → Post-processing → Aesthetic
- **Length**: Comprehensive (150-250 words)

#### Minimal (Optimized for FLUX)
- **Best for**: Quick generation, FLUX models, testing
- **Includes**: Core subject, pose, clothing, setting, main lighting, style, quality
- **Structure**: Concise keywords
- **Length**: Short (30-50 words)
- **Example**: "Medium format portrait, woman in black dress, elegant pose, studio lighting, dark backdrop, fashion editorial, high contrast"

#### Ultra-Detailed (Maximum Realism) - MOST COMPREHENSIVE
- **Best for**: Photorealistic recreation with exhaustive detail
- **Includes EVERYTHING**:
  - Exact camera model with sensor details
  - Exact lens with specifications
  - Precise camera settings and workflow
  - Complete studio environment specifications
  - Extremely detailed lighting setup (specific models, positions, power ratios, color temperature)
  - Comprehensive subject description (every detail of pose, clothing, expression)
  - Detailed composition and framing
  - Complete post-processing workflow
  - Aesthetic references and style
  - Quality specifications
- **Structure**: All technical aspects in extreme detail
- **Length**: Very comprehensive (200-400 words)
- **Use when**: Maximum realism and photographic accuracy required

### Understanding Target Models

The `target_model` parameter optimizes prompts for specific image generation models:

#### FLUX Models
- **Best for**: Latest FLUX models (Dev, Schnell, Pro)
- **Positive prompts**: Natural language descriptions, contextual and flowing
- **Negative prompts**: Concise and specific
- **Example positive**: "A portrait of an elderly woman with silver hair, wearing a red scarf, standing in a sunlit garden, soft golden hour lighting, photorealistic style, high detail, sharp focus"
- **When to use**: Default choice for modern FLUX workflows

#### SDXL (Stable Diffusion XL)
- **Best for**: SDXL 1.0 and derived models
- **Positive prompts**: Mix of natural language and weighted keywords
- **Negative prompts**: Moderate length, focused on quality
- **Example positive**: "portrait of elderly woman, silver hair, red scarf, sunlit garden, (golden hour lighting:1.2), photorealistic, highly detailed, sharp focus, professional photography"
- **When to use**: When working with SDXL or Pony Diffusion models

#### SD 1.5 (Stable Diffusion 1.5)
- **Best for**: SD 1.5 and older Stable Diffusion models
- **Positive prompts**: Keyword-heavy, front-loaded with important elements
- **Negative prompts**: Comprehensive and extensive
- **Example positive**: "elderly woman portrait, silver hair, red scarf, sunlit garden, golden hour, photorealistic, highly detailed, intricate, sharp focus, professional, 8k, trending on artstation"
- **When to use**: Legacy SD 1.5 workflows or models based on SD 1.5

#### Universal
- **Best for**: When you're unsure or using multiple models
- **Positive prompts**: Balanced format that works across models
- **Negative prompts**: Moderate, broadly compatible
- **When to use**: Testing, multi-model workflows, or general purpose

## Configuration

### API Key Security

**Important:** Never commit your API key to version control!

Options for storing your API key:
1. Enter it directly in the node (least secure)
2. Use environment variables (recommended for local use)
3. Use a secrets management system (recommended for production)

### Cost Considerations

Using the Anthropic API incurs costs based on:
- Model selected (Haiku is most economical)
- Token usage (both input and output)
- Number of requests

Tips to reduce costs:
- Use `claude-haiku-4-5-20251001` for most tasks (fastest and most economical)
- Use "Basic" or "Detailed" level instead of "Very Detailed"
- Reduce `max_tokens` if you don't need very long prompts
- Cache API responses if analyzing the same image multiple times

## Troubleshooting

### "ERROR: anthropic library not installed"
```bash
cd ComfyUI/custom_nodes/ComfyUI-AI-Photography-Toolkit
pip install -r requirements.txt
```

### "Anthropic API Error: authentication_error"
- Check that your API key is correct
- Verify your API key has not expired
- Ensure you have credits in your Anthropic account

### "Node not appearing in ComfyUI"
- Restart ComfyUI completely
- Check the console for any error messages
- Verify the installation path is correct

### Empty or Invalid Prompts
- Ensure the image input is valid
- Try increasing `max_tokens`
- Check the console output for any error messages

## Development

### Adding New Nodes

1. Create a new Python file in the package directory (e.g., `sid_new_node.py`)
2. Follow the ComfyUI node structure using `comfy_api.latest`
3. Import and register the node in `__init__.py`
4. Update this README with documentation

### Code Standards

- All nodes must be prefixed with `SID_`
- Follow ComfyUI's modern API (`comfy_api.latest`)
- Include comprehensive docstrings
- Add error handling for API calls
- Log important information to console

## Requirements

- ComfyUI (latest version recommended)
- Python 3.10+
- Anthropic API key (for SID_AIPromptGenerator)

## Dependencies

- `anthropic>=0.39.0` - Anthropic Claude API client
- `pillow>=10.0.0` - Image processing
- `numpy>=1.24.0` - Array operations

## License

MIT License - See LICENSE file for details

## Support

- GitHub Issues: [Report bugs or request features](https://github.com/[YOUR_USERNAME]/ComfyUI-AI-Photography-Toolkit/issues)
- Discord: [ComfyUI Community](https://discord.gg/comfyui)

## Changelog

### Version 1.3.2 (2025-01-18)
- **BUGFIX**: Fixed seed parameter displaying "nan"
  - Changed seed max value from 0xffffffffffffffff to 2147483647 (standard 32-bit int max)
  - Seed now displays correctly in ComfyUI interface

### Version 1.3.1 (2025-01-18)
- **NEW FEATURE**: Lighting Condition dropdown
  - 31 lighting options across 4 categories
  - **Studio Lighting (8)**: Natural Window Light, Studio Strobes, Softbox, Beauty Dish, Ring Light, Umbrella, Grid Spot, Reflector Fill
  - **Studio Techniques (10)**: Rembrandt, Split, Butterfly, Loop, Broad, Short, High/Low Key Studio, Clamshell, Edge/Rim
  - **Outdoor/Natural (9)**: Golden Hour, Blue Hour, Harsh Midday, Overcast, Open Shade, Backlit, Dusk/Twilight, Sunrise, Sunset
  - **Special/Creative (8)**: Chiaroscuro, Dramatic Side Light, Silhouette, Candlelight, Neon/Colorful, Practical, Mixed, Night
  - Each lighting type includes detailed characteristics (light source, quality, direction, color temperature, shadows, mood)
  - Automatically incorporates lighting setup into positive prompts
  - "None" option to skip specific lighting instructions
- **IMPROVED**: System prompts now incorporate lighting conditions with comprehensive technical details
- Enhanced logging to show selected lighting condition

### Version 1.3.0 (2025-01-18)
- **NEW FEATURE**: Color Style dropdown
  - 9 color treatment options: None, B&W, Color, Monochrome, Sepia, Cross-Processed, High Contrast B&W, Low Key, High Key
  - Automatically incorporates color treatment into prompts
- **NEW FEATURE**: Photographer Style dropdown
  - 20 famous photographers to emulate their signature style
  - Includes: Helmut Newton, Peter Lindbergh, Annie Leibovitz, Richard Avedon, Irving Penn, Herb Ritts, Mario Testino, Steven Meisel, Patrick Demarchelier, Paolo Roversi, Tim Walker, Ellen von Unwerth, David LaChapelle, Steven Klein, Mert and Marcus, Juergen Teller, Bruce Weber, Terry Richardson, Rankin, Albert Watson
  - Each photographer's style includes specific lighting, composition, and post-processing characteristics
  - "None" option to skip photographer styling
- **NEW FEATURE**: Seed parameter for randomization
  - Use different seeds to get variations on the same image
  - Standard ComfyUI seed system (0 to max int64)
- **IMPROVED**: System prompts now incorporate color and photographer styles throughout
- Enhanced logging to show color style, photographer style, and seed
- All new features have "None" options to maintain backward compatibility

### Version 1.2.0 (2025-01-18)
- **MAJOR UPDATE**: Complete redesign based on professional photography workflow
- **NEW**: Photography Style system replacing detail levels
  - `Artistic`: Creative, mood-focused prompts
  - `Technical`: Professional camera/studio/lighting specifications (Recommended)
  - `Minimal`: Concise, optimized prompts
  - `Ultra-Detailed`: Exhaustive 200-400 word comprehensive specs
- **IMPROVED**: Categorized negative prompts for all models
  - Quality Issues category
  - Lighting Issues category
  - Composition Issues category
  - Subject Issues category
  - Anatomical Issues category
  - Style Issues category
  - Technical Issues category
  - Unwanted Elements category
- **ENHANCED**: Model-specific prompt formatting
  - FLUX: Natural language with specific structure
  - SDXL: Balanced keywords with weighted syntax
  - SD 1.5: Comprehensive keyword-heavy format
  - Universal: Cross-compatible balanced format
- **IMPROVED**: Negative prompts now include 300+ organized exclusion terms
- **ENHANCED**: Ultra-Detailed style includes exact equipment models, lighting ratios, post-processing workflow
- Follows professional ComfyUI prompt engineering standards
- Documentation completely updated with examples

### Version 1.1.1 (2025-01-18)
- **IMPROVEMENT**: Added model-specific positive prompt guidelines
  - FLUX: Natural language, contextual descriptions with proper structure
  - SDXL: Balanced natural language and weighted keywords
  - SD 1.5: Structured keyword-based prompts with emphasis syntax
  - Universal: Cross-compatible prompt format
- Updated author attribution to Siddhartha Lahiri
- Enhanced Claude with comprehensive positive prompt standards
- Better prompt generation following each model's optimal format

### Version 1.1.0 (2025-01-18)
- **NEW FEATURE**: Added `target_model` parameter for model-specific prompt optimization
- **MAJOR IMPROVEMENT**: Model-specific negative prompt generation
  - FLUX: Concise, targeted negative prompts optimized for FLUX workflow
  - SDXL: Moderate negative prompts focusing on quality issues
  - SD 1.5: Comprehensive negative prompts for best SD 1.5 results
  - Universal: Balanced prompts that work across all models
- Enhanced system prompts with detailed negative prompt guidelines
- Improved negative prompt quality based on community best practices
- Better logging to show target model in console output

### Version 1.0.2 (2025-01-18)
- **CRITICAL FIX**: Updated to current Claude model IDs
- Removed deprecated Claude 3.5 Sonnet models
- Added Claude 4.5 Sonnet (default), Claude 4.5 Haiku, and Claude 4.1 Opus
- Updated documentation with correct model names

### Version 1.0.1 (2025-01-18)
- Added automatic dependency installation
- Fixed anthropic library auto-install on ComfyUI startup
- Improved error messages and installation feedback

### Version 1.0.0 (2025-01-18)
- Initial release
- Added SID_AIPromptGenerator node
- Support for all Claude 3.x models
- Multiple detail levels
- Temperature and token control

## Credits

Created by Siddhartha Lahiri

Special thanks to:
- ComfyUI team for the amazing framework
- Anthropic for the Claude API
- The ComfyUI community

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

Please ensure:
- Code follows existing patterns
- Nodes are prefixed with `SID_`
- Documentation is updated
- Changes are tested

## Roadmap

- [ ] Add OpenAI GPT-4 Vision support
- [ ] Implement batch processing
- [ ] Add prompt caching for repeated analyses
- [ ] Create preset templates for common photography styles
- [ ] Add support for other LLM providers (Gemini, etc.)
- [ ] Build style analyzer node
- [ ] Implement color palette extraction
- [ ] Add composition analysis tools

---

**Enjoy creating better prompts with AI! 📸✨**
