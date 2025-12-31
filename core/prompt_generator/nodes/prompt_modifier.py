"""
SID_PromptModifier - Semantically modify prompts using LLM.

Takes an input prompt and modification instructions for each semantic section,
then uses an LLM to intelligently modify the prompt while preserving structure.

Sections (Z-Image structure):
1. Subject - age, ethnicity, build, features
2. Clothing - materials, colors, fit, accessories
3. Pose - expression, body position, gesture
4. Environment - background, setting, location
5. Lighting - quality, direction, color temperature
6. Camera - angle, framing, depth of field
"""

import base64
import hashlib
import io
import json
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image
import comfy.model_management


def check_interrupt():
    """Check if execution was interrupted and raise exception if so."""
    comfy.model_management.throw_exception_if_processing_interrupted()


class SID_PromptModifier:
    """
    Semantically modify prompts using LLM with section-based instructions.
    """

    CATEGORY = "SID Photography Toolkit"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt", "caption")
    FUNCTION = "modify"

    # Class-level cache for prompt/caption results
    _cache: Dict[str, Tuple[str, str]] = {}

    # Section definitions for Z-Image prompt structure
    SECTIONS = {
        "subject": {
            "name": "Subject",
            "description": "Main subject attributes (age, ethnicity, build, features, hair, face)",
            "keywords": ["woman", "man", "person", "model", "face", "hair", "eyes", "skin", "features"],
        },
        "clothing": {
            "name": "Clothing & Accessories",
            "description": "Clothing items, materials, colors, fit, jewelry, accessories",
            "keywords": ["wearing", "dress", "shirt", "pants", "shoes", "jewelry", "accessories", "fabric"],
        },
        "pose": {
            "name": "Pose & Expression",
            "description": "Body position, stance, gesture, facial expression, mood",
            "keywords": ["standing", "sitting", "pose", "expression", "looking", "hands", "arms", "gesture"],
        },
        "environment": {
            "name": "Environment & Background",
            "description": "Setting, location, background elements, props, atmosphere",
            "keywords": ["background", "setting", "location", "room", "outdoor", "indoor", "scene"],
        },
        "lighting": {
            "name": "Lighting",
            "description": "Light quality, direction, color temperature, shadows, highlights",
            "keywords": ["lighting", "light", "shadows", "highlights", "golden hour", "soft", "dramatic"],
        },
        "camera": {
            "name": "Camera & Framing",
            "description": "Camera angle, shot type, framing, depth of field, focus",
            "keywords": ["camera", "angle", "shot", "close-up", "portrait", "wide", "depth of field", "bokeh"],
        },
    }

    # Photography Templates - Pre-defined photography styles/setups
    PHOTOGRAPHY_TEMPLATES = {
        "None": "",
        "Studio Portrait": "Professional studio portrait with seamless background, controlled lighting setup with key light, fill light, and hair light, clean and polished look",
        "Fashion Editorial": "High-fashion editorial style, dramatic poses, bold styling, magazine-quality composition, striking visual impact",
        "Street Photography": "Candid street photography aesthetic, urban environment, natural moments, documentary style, authentic atmosphere",
        "Beauty/Cosmetic": "Close-up beauty photography, flawless skin, detailed makeup visibility, soft diffused lighting, commercial beauty aesthetic",
        "Lifestyle": "Natural lifestyle photography, relaxed and authentic moments, warm inviting atmosphere, relatable everyday settings",
        "Glamour": "Glamorous photography style, luxurious setting, elegant styling, soft flattering light, aspirational mood",
        "Boudoir": "Intimate boudoir photography, soft romantic lighting, elegant poses, private luxurious setting, tasteful sensuality",
        "Fine Art": "Fine art photography aesthetic, artistic composition, conceptual elements, gallery-worthy presentation",
        "Commercial": "Clean commercial photography, product-focused when applicable, professional lighting, advertising quality",
        "Vintage/Retro": "Vintage photography aesthetic, period-appropriate styling, nostalgic color grading, classic film look",
        "Cinematic": "Cinematic photography style, movie-like framing, dramatic lighting, widescreen composition, narrative mood",
        "Minimalist": "Minimalist photography, clean simple backgrounds, negative space, focus on essential elements only",
        "Environmental Portrait": "Environmental portrait showing subject in their natural setting, context-rich background, storytelling composition",
        "High Key": "High key photography, bright white background, minimal shadows, clean ethereal look, overexposed aesthetic",
        "Low Key": "Low key photography, dark moody background, dramatic shadows, rim lighting, mysterious atmosphere",
    }

    # Photography Effects - Visual effects to apply
    PHOTOGRAPHY_EFFECTS = {
        "None": "",
        "Film Grain": "Add subtle film grain texture, analog film aesthetic, organic noise pattern",
        "Soft Focus": "Dreamy soft focus effect, gentle blur, romantic ethereal quality",
        "High Contrast": "High contrast look, deep blacks and bright highlights, bold tonal separation",
        "Desaturated": "Muted desaturated colors, subtle pastel tones, understated color palette",
        "Rich Colors": "Vibrant rich saturated colors, punchy color palette, vivid tones",
        "Golden Hour": "Warm golden hour color grading, orange and amber tones, sunset warmth",
        "Blue Hour": "Cool blue hour tones, twilight color palette, serene blue cast",
        "Cross Process": "Cross-processed film look, shifted colors, experimental color cast",
        "Faded Film": "Faded vintage film effect, lifted blacks, reduced contrast, nostalgic fade",
        "HDR": "HDR-style processing, enhanced details in shadows and highlights, surreal clarity",
        "Matte Finish": "Matte film finish, lifted shadows, reduced contrast, modern film look",
        "Teal and Orange": "Teal and orange color grading, cinematic color contrast, Hollywood look",
        "Black and White": "Classic black and white, rich grayscale tones, timeless monochrome",
        "Sepia": "Warm sepia toning, antique photograph aesthetic, brown monochrome",
        "Split Toning": "Split toning effect, different colors in highlights and shadows, artistic color separation",
        "Lens Flare": "Natural lens flare, sun rays, organic light artifacts, atmospheric glow",
        "Bokeh": "Pronounced bokeh effect, creamy out-of-focus areas, shallow depth of field aesthetic",
        "Vignette": "Subtle vignette, darkened edges, focus drawn to center, classic finish",
    }

    # Famous Photographer Styles
    PHOTOGRAPHER_STYLES = {
        "None": "",
        "Annie Leibovitz": "Annie Leibovitz style - dramatic theatrical portraits, conceptual storytelling, bold artistic vision, celebrity portrait mastery, elaborate production value",
        "Peter Lindbergh": "Peter Lindbergh style - raw natural beauty, black and white mastery, minimal retouching, emotional authenticity, supermodel portraits, timeless elegance",
        "Helmut Newton": "Helmut Newton style - provocative glamour, strong powerful women, high contrast black and white, bold sexuality, cinematic noir aesthetic",
        "Richard Avedon": "Richard Avedon style - stark white backgrounds, dynamic movement, psychological intensity, fashion innovation, raw emotional portraits",
        "Irving Penn": "Irving Penn style - elegant simplicity, meticulous composition, neutral backgrounds, still life precision, timeless sophistication",
        "Mario Testino": "Mario Testino style - vibrant glamorous energy, sun-kissed skin, joyful spontaneity, luxury fashion, warm golden tones",
        "Steven Meisel": "Steven Meisel style - chameleon versatility, trendsetting fashion, narrative editorial stories, transformative character studies",
        "Patrick Demarchelier": "Patrick Demarchelier style - classic French elegance, natural beauty, soft romantic lighting, effortless sophistication",
        "David LaChapelle": "David LaChapelle style - hyper-saturated colors, surreal fantasy worlds, pop culture commentary, extravagant maximalism",
        "Tim Walker": "Tim Walker style - whimsical fairytale aesthetic, elaborate fantasy sets, dreamlike storytelling, magical romanticism",
        "Ellen von Unwerth": "Ellen von Unwerth style - playful feminine energy, retro pin-up glamour, flirtatious spontaneity, empowered sensuality",
        "Herb Ritts": "Herb Ritts style - sculptural body photography, graphic black and white, classical beauty, athletic forms, sun-drenched California aesthetic",
        "Guy Bourdin": "Guy Bourdin style - surrealist fashion, bold graphic colors, mysterious narratives, provocative compositions, artistic tension",
        "Paolo Roversi": "Paolo Roversi style - ethereal Polaroid aesthetic, soft diffused light, intimate portraits, romantic timelessness, painterly quality",
        "Juergen Teller": "Juergen Teller style - raw unpolished aesthetic, flash photography, anti-glamour authenticity, candid imperfection",
        "Terry Richardson": "Terry Richardson style - harsh flash photography, snapshot aesthetic, provocative directness, high contrast pop",
        "Nick Knight": "Nick Knight style - experimental digital innovation, avant-garde fashion, futuristic aesthetic, boundary-pushing imagery",
        "Rankin": "Rankin style - bold pop portraits, vibrant colors, direct eye contact, confident subjects, commercial edge",
        "Bruce Weber": "Bruce Weber style - all-American athleticism, outdoor naturalism, youthful energy, golden hour warmth, Americana nostalgia",
        "Nadav Kander": "Nadav Kander style - contemplative portraits, muted color palette, psychological depth, understated power, fine art approach",
    }

    # Creative Effects - Concise technical terms (3-5 max per effect)
    CREATIVE_EFFECTS = {
        "None": "",
        # Motion & Blur
        "Ghost Motion Blur": "long exposure motion trail, translucent motion echo, 1/4 second shutter",
        "Soft Focus": "soft focus lens, f/2.8 diffusion, slight gaussian blur",
        "Tilt Shift": "tilt shift lens, selective focus plane, miniature effect",
        "Bokeh Background": "f/1.4 aperture, circular bokeh, blurred background lights",
        "Light Trails": "long exposure light trails, 30 second shutter, light streaks",
        "Zoom Blur": "zoom burst effect, radial motion blur, center focus",
        "Stroboscopic": "stroboscopic flash, multiple exposure freeze, motion sequence",
        # Color & Tone
        "Chromatic Aberration": "RGB color split, chromatic aberration, color fringing edges",
        "Duotone": "duotone color grade, two-color palette, high contrast",
        "Prism Split": "prism light refraction, rainbow spectrum split, light dispersion",
        "Infrared": "infrared photography, false color, 720nm filter look",
        "Thermal": "thermal imaging colors, heat map gradient, FLIR style",
        "Cross Process": "cross processed film, shifted colors, green shadows magenta highlights",
        "Solarization": "solarization effect, partial tone inversion, Sabattier effect",
        "Posterize": "posterized colors, limited color palette, flat color zones",
        "Teal Orange": "teal and orange color grade, complementary split toning",
        # Lighting
        "Neon Glow": "neon lighting, electric glow, saturated color cast",
        "Rim Light Silhouette": "backlit rim lighting, silhouette, edge glow",
        "Split Lighting": "split lighting, half-face shadow, dramatic side light",
        "Backlit Haze": "backlit with haze, lens flare, volumetric light rays",
        "Anamorphic Flare": "anamorphic lens flare, horizontal blue streak, oval bokeh",
        "Soft Glow": "soft diffused glow, bloom lighting, gentle luminosity",
        "UV Blacklight": "UV blacklight, fluorescent glow, neon reactive",
        "Bioluminescent": "bioluminescent glow, inner light emission, organic luminescence",
        # Texture & Film
        "Film Grain": "heavy film grain, ISO 3200 noise, analog texture",
        "Halftone": "halftone dot pattern, CMYK print texture, Ben-Day dots",
        "VHS": "VHS scan lines, tape distortion, color bleeding",
        "Aged Film": "scratched film, dust particles, vintage film damage",
        "Static Noise": "TV static noise, signal interference texture",
        # Digital & Glitch
        "Glitch": "digital glitch, data corruption, pixel artifacts",
        "Pixel Sort": "pixel sorting effect, vertical data streaks",
        "Holographic": "holographic iridescent, rainbow shimmer, pearlescent",
        "Wireframe": "wireframe mesh overlay, 3D polygon lines, digital grid",
        "Anaglyph 3D": "anaglyph 3D, red-cyan offset, stereoscopic",
        "CRT Scanlines": "CRT scanlines, retro monitor lines, phosphor glow",
        # Artistic
        "Oil Paint": "oil painting texture, visible brushstrokes, impasto",
        "Watercolor": "watercolor effect, soft bleeding edges, color wash",
        "Charcoal": "charcoal sketch, rough strokes, monochrome graphite",
        "Stained Glass": "stained glass segments, colored glass panels, lead lines",
        "Kaleidoscope": "kaleidoscope symmetry, radial mirror pattern",
        "Pop Art": "pop art style, bold flat colors, screen print look",
        # Surreal
        "Double Exposure": "double exposure blend, overlaid images, composite",
        "Disintegration": "particle disintegration, scatter dissolve effect",
        "Levitation": "levitation, floating suspended, no ground contact",
        "Mirror Reflection": "mirror reflection, perfect water symmetry",
        "Portal": "dimensional portal effect, circular gateway, reality warp",
        "Liquid Melt": "melting effect, dripping liquid distortion",
        # Environmental
        "Underwater": "underwater photography, caustic light patterns, aquatic",
        "Fog": "dense fog, atmospheric haze, low visibility",
        "Dust Particles": "visible dust motes, particles in light beam",
        "Rain": "falling rain, wet surfaces, water droplets visible",
        "Frost": "frost covered, ice crystals, frozen texture",
        "Fire Glow": "fire light, orange ember glow, heat distortion",
    }

    # Cinematic Styles - Concise movie/director visual references
    CINEMATIC_STYLES = {
        "None": "",
        # Tarantino
        "Kill Bill Yellow": "high saturation yellow, bold primary colors, black stripe accent",
        "Kill Bill Anime": "anime cell shading, stylized action, manga aesthetic",
        "Pulp Fiction Glow": "warm golden glow, divine light source, amber tones",
        "Grindhouse": "film scratches, missing frames, 70s exploitation print damage",
        # Sci-Fi
        "Tron Neon Grid": "neon blue circuit lines, black background, digital grid glow",
        "Matrix Green": "green tinted, digital rain overlay, dark cyber aesthetic",
        "Matrix Bullet Time": "frozen mid-action, 360 degree angle, time slice effect",
        "Star Wars Hologram": "blue holographic projection, flickering scanlines, translucent",
        "Hyperspace": "stretched star trails, motion blur tunnel, light speed streaks",
        "Lightsaber Glow": "colored light source glow, plasma ambient light, blade reflection",
        "Blade Runner Neon Rain": "neon reflections, rain, dark cyberpunk, wet streets",
        "Blade Runner 2049 Orange": "orange atmospheric haze, dust particles, apocalyptic sky",
        # Zack Snyder
        "300 Bronze": "desaturated bronze tones, crushed blacks, high contrast sepia",
        "300 Speed Ramp": "motion blur trails, slow motion freeze, dynamic action",
        "Sin City": "high contrast black and white, single selective color accent",
        "Snyder Desaturated": "desaturated color grade, dark tones, dramatic shadows",
        # Wes Anderson
        "Anderson Symmetry": "perfect center composition, pastel colors, symmetrical framing",
        "Grand Budapest Pink": "pastel pink palette, whimsical staging, vintage texture",
        "Moonrise Kingdom": "yellow-khaki tones, 60s vintage color grade, warm nostalgic",
        "Anderson Diorama": "cross-section framing, dollhouse miniature staging, flat lighting",
        # Christopher Nolan
        "Inception": "impossible geometry, folding architecture, surreal perspective",
        "Interstellar": "cosmic scale, IMAX wide angle, space vastness, black hole lighting",
        "Oppenheimer": "IMAX black and white, stark contrast, dramatic monochrome",
        # Denis Villeneuve
        "Dune Desert": "vast orange desert, massive scale, epic wide shot, dust haze",
        "Dune Spice": "blue-within-blue eyes, teal eye color, prescient gaze",
        "Arrival": "grey fog atmosphere, muted colors, overcast diffused light",
        # Other Directors
        "Kubrick Symmetry": "one-point perspective, centered frame, symmetrical composition",
        "Kubrick Stare": "head tilted down, eyes looking up, intense direct gaze",
        "Spielberg Light": "strong backlight from above, silhouette rim, wonder lighting",
        "Fincher Dark": "green-yellow tint, underexposed shadows, dark moody",
        "Midsommar Daylight": "bright overexposed daylight, floral setting, sunny horror",
        "Michael Bay": "orange explosion light, low angle hero shot, action dynamic",
    }

    # Anime Styles - Concise anime visual references
    ANIME_STYLES = {
        "None": "",
        # Dragon Ball
        "Super Saiyan Aura": "golden energy aura, spiked blonde hair, DBZ style",
        "Kamehameha Beam": "blue energy beam, ki blast, charging pose",
        "Ultra Instinct": "silver white aura, glowing silver hair, divine energy",
        "DBZ Power Up": "energy aura, cracking ground, floating debris",
        "Anime Speed Lines": "radial speed lines, motion blur, impact frame",
        # Naruto
        "Chakra Aura": "blue energy aura, ninja style, spiritual energy",
        "Sharingan Eye": "red iris with tomoe pattern, glowing red eyes",
        "Rasengan": "blue spinning energy sphere, hand-held orb",
        "Nine Tails Cloak": "orange energy cloak, fox-like aura, fiery outline",
        "Susanoo": "giant spectral armor, ethereal purple warrior projection",
        # Demon Slayer
        "Water Breathing": "flowing water effects, blue wave patterns, sword trails",
        "Flame Breathing": "fire sword trails, ember particles, orange flames",
        "Thunder Breathing": "yellow lightning effects, electric aura, speed blur",
        "Ufotable Sky": "gradient sunset sky, detailed clouds, anime background",
        # Attack on Titan
        "Titan Steam": "heavy steam emission, hot mist, giant form",
        "ODM Gear": "cable wire trails, aerial acrobatic pose",
        # Studio Ghibli
        "Ghibli Clouds": "fluffy detailed clouds, soft lighting, Miyazaki sky",
        "Ghibli Forest": "magical woodland, soft dappled light, nature spirits",
        "Ghibli Steampunk": "whimsical machinery, brass gears, fantasy mechanical",
        # Cyberpunk Anime
        "Akira": "red color accent, Neo-Tokyo neon, cyberpunk anime",
        "Ghost in Shell": "cybernetic aesthetic, digital overlay, cyborg chrome",
        "Cowboy Bebop": "noir lighting, jazz mood, retro-future anime",
        "Evangelion": "mecha aesthetic, existential mood, NERV style",
        # Modern Anime
        "Jujutsu Kaisen": "dark cursed energy, black-purple aura, domain effect",
        "One Punch Impact": "impact shockwave, devastating force, speed lines",
        "Makoto Shinkai": "beautiful light rays, detailed clouds, lens flare bokeh",
    }

    # Gaming & TV Styles - Concise visual references
    GAMING_STYLES = {
        "None": "",
        # Retro Gaming
        "8-Bit Pixel": "8-bit pixel art style, limited color palette, blocky pixels",
        "16-Bit Sprite": "16-bit sprite art, SNES graphics style, detailed pixels",
        "Arcade CRT": "CRT scanlines, curved screen glow, phosphor colors",
        # Modern Gaming
        "GTA Art Style": "stylized character portrait, bold outlines, loading screen art",
        "Cyberpunk 2077": "neon city lights, chrome implants, cyberpunk aesthetic",
        "Dark Souls": "dark fantasy, bonfire lighting, gothic atmosphere",
        "Elden Ring": "golden grace light, dark fantasy, ethereal glow",
        "Borderlands Cell Shade": "heavy cell shading, thick black outlines, comic style",
        "Bioshock Art Deco": "art deco style, 1920s futurism, golden ornate",
        # Nintendo
        "Zelda Fairy": "magical fairy glow, sparkle trail, fantasy light",
        "Pokemon Evolution": "transformation glow, energy burst, type-colored light",
        # TV Series
        "Stranger Things": "80s aesthetic, Christmas lights, dark supernatural",
        "Squid Game": "pink and green contrast, geometric shapes, stark lighting",
        "Euphoria": "glitter makeup, neon party lighting, dramatic color",
        "Wednesday Gothic": "gothic desaturated, dark academia, Tim Burton style",
        "Last of Us": "post-apocalyptic overgrowth, nature reclaiming, decay",
        "Breaking Bad": "yellow desert tint, New Mexico lighting, stark contrast",
        "True Detective": "southern gothic, decay aesthetic, rust tones",
        "Twin Peaks": "surreal red room, zigzag pattern, Lynch aesthetic",
        # Comic & Superhero
        "Spider-Verse": "halftone dots, comic print texture, chromatic offset",
        "Kirby Cosmic": "cosmic energy dots, Kirby crackle pattern, space power",
        "Batman Noir": "high contrast shadows, noir lighting, dark silhouette",
    }

    # Poses - Concise pose descriptions
    POSES = {
        "None": "",
        # Standing
        "Standing Straight": "standing upright, feet together, arms at sides",
        "Standing Relaxed": "casual stance, weight on one leg",
        "Standing Confident": "shoulders back, chin up, powerful stance",
        "Hip Pop": "hip popped to side, weight shifted, S-curve",
        "Contrapposto": "contrapposto, weight on one leg, twisted torso",
        "Hands on Hips": "hands on hips, elbows out",
        "Arms Crossed": "arms crossed over chest",
        "Arms Behind Back": "hands clasped behind back",
        "Hand on Face": "hand touching face or chin",
        "Hands in Pockets": "hands in pockets, casual stance",
        # Walking & Movement
        "Walking Forward": "walking towards camera, mid-stride",
        "Looking Back": "walking away, looking over shoulder",
        "Runway Walk": "runway stride, one foot in front",
        "Twirling": "twirling, hair or dress in motion",
        "Jumping": "mid-jump, feet off ground",
        # Sitting
        "Sitting Straight": "sitting upright, good posture",
        "Sitting Relaxed": "sitting casual, relaxed posture",
        "Cross-Legged Floor": "sitting cross-legged on floor",
        "Legs Crossed Chair": "seated, legs crossed elegantly",
        "Sitting Reclined": "reclined, leaning back",
        # Leaning
        "Leaning Wall": "leaning against wall",
        "Leaning Forward": "leaning forward, engaged",
        # Lying
        "Lying Back": "lying on back, supine",
        "Lying Side": "lying on side, propped on elbow",
        "Lying Stomach": "lying prone, chin on hands",
        # Fashion
        "Editorial Pose": "high fashion angular pose",
        "S-Curve": "S-curve body line, feminine",
        "Fashion Crouch": "crouching, low angle fashion",
        "Fashion Kneel": "kneeling elegantly",
        # Portrait
        "Three-Quarter": "body angled 45 degrees to camera",
        "Profile": "side profile view",
        "Over Shoulder": "looking over shoulder",
        "Chin Down Eyes Up": "chin down, eyes looking up",
        "Head Tilt": "gentle head tilt",
        # Yoga
        "Tree Pose": "one foot on inner thigh, arms raised",
        "Warrior Pose": "lunge stance, arms extended",
        "Downward Dog": "inverted V position",
        "Cobra": "chest lifted, back arch",
        "Lotus": "seated cross-legged meditation",
        "Dancer Pose": "standing backbend, one leg raised",
        # Influencer
        "Selfie Angle": "phone held up, selfie angle",
        "Peace Sign": "peace sign hand gesture",
        "Hair Flip": "tossing hair, movement",
        "Candid Laugh": "natural laughing, candid",
        # Action
        "Running": "mid-stride running",
        "Stretching": "athletic stretch pose",
        "Reaching Up": "arms reaching upward",
    }

    # Facial Expressions - Concise emotion descriptors
    FACIAL_EXPRESSIONS = {
        "None": "",
        # Happy
        "Smile": "genuine smile, eyes crinkled",
        "Soft Smile": "soft subtle smile, lips slightly upturned",
        "Grin": "wide grin, teeth showing",
        "Smirk": "slight smirk, one corner raised",
        "Laughing": "laughing, mouth open, joyful",
        "Playful": "playful expression, mischievous glint",
        # Serious
        "Neutral": "neutral expression, relaxed face",
        "Serious": "serious expression, no smile",
        "Stoic": "stoic, emotionless calm",
        "Determined": "determined expression, resolute",
        "Focused": "focused, concentrated gaze",
        "Contemplative": "contemplative, deep in thought",
        "Pensive": "pensive, thoughtful",
        # Confident
        "Confident": "confident expression, self-assured",
        "Fierce": "fierce, intense powerful gaze",
        "Defiant": "defiant expression, challenging",
        "Smoldering": "smoldering, intense sultry gaze",
        "Intense": "intense, piercing stare",
        # Sad
        "Sad": "sad expression, downturned mouth",
        "Melancholic": "melancholic, wistful",
        "Tearful": "tearful, eyes glistening",
        "Crying": "crying, tears streaming",
        # Angry
        "Angry": "angry, furrowed brow",
        "Annoyed": "annoyed, irritated",
        "Scowling": "scowling, deep frown",
        "Glaring": "glaring, hostile stare",
        # Surprised
        "Surprised": "surprised, wide eyes, raised eyebrows",
        "Shocked": "shocked, jaw dropped",
        "Amazed": "amazed, wonder expression",
        # Fear
        "Afraid": "afraid, fearful eyes",
        "Anxious": "anxious, worried expression",
        "Nervous": "nervous, apprehensive",
        # Romantic
        "Seductive": "seductive, alluring gaze",
        "Flirty": "flirty, playful coy",
        "Romantic": "romantic, tender loving",
        "Longing": "longing, yearning gaze",
        "Dreamy": "dreamy, dazed romantic",
        # Other
        "Confused": "confused, furrowed brow",
        "Skeptical": "skeptical, raised eyebrow",
        "Curious": "curious, interested",
        "Shy": "shy, avoiding eye contact",
        "Peaceful": "peaceful, serene calm",
        "Mysterious": "mysterious, enigmatic",
        "Blank": "blank expression, vacant",
    }

    # Hair Styles - Concise hair descriptors
    HAIR_STYLES = {
        "None": "",
        # Length & Cut
        "Pixie Cut": "pixie cut, very short cropped",
        "Bob": "bob cut, chin-length",
        "Lob": "long bob, shoulder-length",
        "Long Hair": "long hair, past shoulders",
        "Very Long": "very long hair, waist-length",
        "Buzz Cut": "buzz cut, very short",
        "Undercut": "undercut, shaved sides, longer top",
        # Texture
        "Straight Sleek": "straight sleek hair, smooth",
        "Beach Waves": "beach waves, tousled texture",
        "Soft Waves": "soft waves, gentle undulation",
        "Hollywood Waves": "Hollywood waves, glamour curls",
        "Tight Curls": "tight curls, springy coils",
        "Loose Curls": "loose curls, bouncy spirals",
        "Afro": "afro, natural voluminous curls",
        "Natural Curls": "natural textured curls",
        # Updos
        "High Bun": "high bun, top of head",
        "Low Bun": "low bun, nape of neck",
        "Messy Bun": "messy bun, casual undone",
        "Sleek Bun": "sleek polished bun",
        "Chignon": "chignon, elegant twisted bun",
        "French Twist": "French twist updo",
        # Ponytails
        "High Ponytail": "high ponytail",
        "Low Ponytail": "low ponytail, sleek",
        "Side Ponytail": "side ponytail, asymmetric",
        # Braids
        "French Braid": "French braid",
        "Dutch Braid": "Dutch braid, raised pattern",
        "Fishtail Braid": "fishtail braid",
        "Box Braids": "box braids, sectioned",
        "Cornrows": "cornrows, braided along scalp",
        "Double Braids": "double braids, pigtails",
        "Locs": "locs, dreadlocks",
        # Bangs
        "Blunt Bangs": "blunt straight bangs",
        "Side Bangs": "side swept bangs",
        "Curtain Bangs": "curtain bangs, parted fringe",
        "Wispy Bangs": "wispy thin bangs",
        "Long Bangs": "long bangs, eye-covering",
        # Special
        "Shag": "shag haircut, layered choppy",
        "Wolf Cut": "wolf cut, shaggy layers",
        "Mohawk": "mohawk, shaved sides center strip",
        # Vintage
        "Victory Rolls": "victory rolls, 1940s style",
        "Finger Waves": "finger waves, 1920s style",
        "Bouffant": "bouffant, teased volume",
    }

    # Clothing Styles - Concise clothing descriptors
    CLOTHING_STYLES = {
        "None": "",
        # Dresses
        "Evening Gown": "evening gown, floor-length formal",
        "Cocktail Dress": "cocktail dress, knee-length",
        "Little Black Dress": "little black dress, LBD",
        "Maxi Dress": "maxi dress, floor-length flowing",
        "Mini Dress": "mini dress, short hemline",
        "Bodycon": "bodycon dress, form-fitting",
        "Wrap Dress": "wrap dress, crossover front",
        "Slip Dress": "slip dress, satin straps",
        "Sundress": "sundress, light summer",
        "Ball Gown": "ball gown, full skirt",
        # Tops
        "Blouse": "elegant blouse, dressy",
        "Button-Up": "button-up shirt, collared",
        "T-Shirt": "casual t-shirt",
        "Tank Top": "tank top, sleeveless",
        "Crop Top": "crop top, midriff-baring",
        "Turtleneck": "turtleneck, high neck",
        "Off-Shoulder": "off-shoulder top",
        "Halter Top": "halter top, neck-tied",
        "Corset Top": "corset top, structured",
        # Bottoms
        "Skinny Jeans": "skinny jeans, fitted denim",
        "Wide-Leg Pants": "wide-leg pants, flowing",
        "High-Waisted": "high-waisted pants",
        "Pencil Skirt": "pencil skirt, fitted",
        "Mini Skirt": "mini skirt, short",
        "Maxi Skirt": "maxi skirt, floor-length",
        "Pleated Skirt": "pleated skirt",
        "Leather Pants": "leather pants, edgy",
        "Shorts": "shorts, casual",
        # Suits & Formal
        "Pantsuit": "tailored pantsuit",
        "Blazer": "structured blazer",
        "Power Suit": "power suit, bold shoulders",
        # Outerwear
        "Leather Jacket": "leather jacket, biker",
        "Denim Jacket": "denim jacket",
        "Trench Coat": "trench coat, belted",
        "Cardigan": "cardigan sweater",
        "Fur Coat": "fur coat, luxurious",
        "Bomber Jacket": "bomber jacket, sporty",
        # Swimwear
        "Bikini": "bikini, two-piece",
        "One-Piece": "one-piece swimsuit",
        "High-Waisted Bikini": "high-waisted bikini, retro",
        # Lingerie
        "Lace Lingerie": "lace lingerie, delicate",
        "Silk Robe": "silk robe, loungewear",
        "Bralette": "bralette, soft",
        # Athleisure
        "Yoga Outfit": "yoga outfit, athletic",
        "Sports Bra": "sports bra, athletic",
        "Leggings": "leggings, fitted stretch",
        "Hoodie": "hoodie, casual",
        # Cultural
        "Kimono": "Japanese kimono, wrapped",
        "Sari": "Indian sari, draped",
        "Cheongsam": "Chinese cheongsam, fitted",
        # Vintage
        "1920s Flapper": "1920s flapper, beaded fringe",
        "1950s Pin-Up": "1950s style, full skirt",
        "1970s Bohemian": "1970s bohemian, flowing",
        # Bridal
        "Wedding Dress": "wedding dress, white gown",
    }

    # Eye Styles - Concise eye descriptors
    EYE_STYLES = {
        "None": "",
        # Colors
        "Blue Eyes": "blue eyes, clear iris",
        "Green Eyes": "green eyes, emerald iris",
        "Brown Eyes": "brown eyes, warm",
        "Hazel Eyes": "hazel eyes, mixed tones",
        "Amber Eyes": "amber eyes, golden",
        "Gray Eyes": "gray eyes, cool silver",
        "Heterochromia": "heterochromia, different colored eyes",
        # Shapes
        "Almond Eyes": "almond shaped eyes",
        "Round Eyes": "round eyes, large",
        "Hooded Eyes": "hooded eyes, heavy lid",
        "Monolid": "monolid eyes, no crease",
        "Upturned": "upturned eyes, cat-like",
        "Doe Eyes": "doe eyes, large innocent",
        # Makeup
        "Natural Eye": "natural eye makeup, subtle",
        "Smoky Eye": "smoky eye, blended dark shadow",
        "Cat Eye": "cat eye liner, winged",
        "Glitter Eye": "glitter eye makeup, sparkle",
        "No Makeup": "no eye makeup, bare",
        "Bold Shadow": "bold colorful eyeshadow",
        # Liner
        "Winged Liner": "winged eyeliner",
        "Thick Liner": "thick eyeliner, bold",
        "No Liner": "no eyeliner, natural",
        # Lashes
        "Natural Lashes": "natural eyelashes",
        "Long Lashes": "long eyelashes, dramatic",
        "False Lashes": "false lashes, voluminous",
        # Brows
        "Natural Brows": "natural eyebrows",
        "Thick Brows": "thick bold eyebrows",
        "Arched Brows": "arched eyebrows",
        "Straight Brows": "straight eyebrows",
        # Gaze
        "Direct Gaze": "direct eye contact",
        "Side Glance": "side glance, looking away",
        "Looking Down": "eyes looking downward",
        "Intense Stare": "intense stare, piercing",
    }

    # Fashion Styles - Concise style descriptors
    FASHION_STYLES = {
        "None": "",
        # Classic
        "Classic Elegant": "classic elegant, timeless refined",
        "Preppy": "preppy, collegiate polished",
        "Business": "business professional, formal corporate",
        "Minimalist": "minimalist, clean simple lines",
        "Old Money": "old money aesthetic, quiet luxury",
        # Trendy
        "Streetwear": "streetwear, urban casual",
        "Athleisure": "athleisure, sporty casual",
        "Y2K": "Y2K style, early 2000s",
        "Dark Academia": "dark academia, scholarly vintage",
        "Cottagecore": "cottagecore, rural romantic",
        # Edgy
        "Grunge": "grunge, distressed 90s",
        "Punk": "punk style, rebellious",
        "Goth": "gothic, all black dark",
        "Rock Chic": "rock chic, leather edgy",
        # Bohemian
        "Bohemian": "bohemian, flowing eclectic",
        "Boho Chic": "boho chic, refined bohemian",
        "Hippie": "hippie style, 70s free spirit",
        # Glamorous
        "Hollywood Glamour": "Hollywood glamour, red carpet",
        "Haute Couture": "haute couture, high fashion",
        "Evening Glamour": "evening glamour, formal elegant",
        # Romantic
        "Romantic": "romantic fashion, soft feminine",
        "Victorian": "Victorian, historical romantic",
        "Fairycore": "fairycore, whimsical magical",
        # Sporty
        "Sporty": "sporty, athletic casual",
        "Surf Style": "surf style, beach casual",
        # Cultural
        "Japanese Street": "Japanese street, Harajuku",
        "Korean Fashion": "Korean fashion, K-style",
        "Parisian Chic": "Parisian chic, French elegant",
        # Vintage
        "1950s Style": "1950s style, full skirts",
        "1970s Style": "1970s style, bohemian disco",
        "1980s Style": "1980s style, bold power shoulders",
    }

    # Ethnicity
    ETHNICITY = {
        "None": "",
        # European
        "Northern European": "Northern European features, Scandinavian, fair complexion",
        "Scandinavian": "Scandinavian features, Nordic appearance, light coloring",
        "Swedish": "Swedish features, Nordic appearance, typically fair",
        "Norwegian": "Norwegian features, Nordic, often fair complexion",
        "Danish": "Danish features, Scandinavian appearance",
        "Finnish": "Finnish features, Nordic-Baltic appearance",
        "Icelandic": "Icelandic features, Nordic appearance",
        "British": "British features, Anglo-Saxon appearance",
        "Irish": "Irish features, Celtic appearance, often fair with freckles",
        "Scottish": "Scottish features, Celtic appearance",
        "Welsh": "Welsh features, Celtic British appearance",
        "German": "German features, Central European appearance",
        "Dutch": "Dutch features, Northwestern European appearance",
        "French": "French features, Western European appearance",
        "Belgian": "Belgian features, Northwestern European appearance",
        "Swiss": "Swiss features, Central European appearance",
        "Austrian": "Austrian features, Central European appearance",
        "Italian": "Italian features, Mediterranean appearance, Southern European",
        "Spanish": "Spanish features, Mediterranean Iberian appearance",
        "Portuguese": "Portuguese features, Iberian Mediterranean appearance",
        "Greek": "Greek features, Mediterranean appearance",
        "Polish": "Polish features, Eastern European Slavic appearance",
        "Russian": "Russian features, Eastern European Slavic appearance",
        "Ukrainian": "Ukrainian features, Eastern European Slavic appearance",
        "Czech": "Czech features, Central European Slavic appearance",
        "Romanian": "Romanian features, Southeastern European appearance",
        "Hungarian": "Hungarian features, Central European appearance",
        "Serbian": "Serbian features, Balkan Slavic appearance",
        "Croatian": "Croatian features, Balkan Slavic appearance",
        "Bulgarian": "Bulgarian features, Balkan Slavic appearance",
        # Asian
        "East Asian": "East Asian features, monolid eyes, straight black hair common",
        "Chinese": "Chinese features, East Asian appearance",
        "Japanese": "Japanese features, East Asian appearance",
        "Korean": "Korean features, East Asian appearance",
        "Taiwanese": "Taiwanese features, East Asian appearance",
        "Vietnamese": "Vietnamese features, Southeast Asian appearance",
        "Thai": "Thai features, Southeast Asian appearance",
        "Filipino": "Filipino features, Southeast Asian appearance, Austronesian",
        "Indonesian": "Indonesian features, Southeast Asian appearance",
        "Malaysian": "Malaysian features, Southeast Asian appearance",
        "Singaporean": "Singaporean features, Southeast Asian mixed appearance",
        "Cambodian": "Cambodian features, Southeast Asian appearance",
        "Laotian": "Laotian features, Southeast Asian appearance",
        "Burmese": "Burmese features, Southeast Asian appearance",
        "Indian": "Indian features, South Asian appearance, diverse",
        "Pakistani": "Pakistani features, South Asian appearance",
        "Bangladeshi": "Bangladeshi features, South Asian appearance",
        "Sri Lankan": "Sri Lankan features, South Asian appearance",
        "Nepali": "Nepali features, South Asian Himalayan appearance",
        "Afghan": "Afghan features, Central-South Asian appearance",
        "Central Asian": "Central Asian features, mixed heritage appearance",
        "Kazakh": "Kazakh features, Central Asian Turkic appearance",
        "Uzbek": "Uzbek features, Central Asian Turkic appearance",
        "Mongolian": "Mongolian features, Central-East Asian appearance",
        # Middle Eastern & North African
        "Middle Eastern": "Middle Eastern features, MENA appearance",
        "Arab": "Arab features, Middle Eastern appearance",
        "Lebanese": "Lebanese features, Levantine Middle Eastern appearance",
        "Syrian": "Syrian features, Levantine appearance",
        "Palestinian": "Palestinian features, Levantine appearance",
        "Jordanian": "Jordanian features, Levantine appearance",
        "Iraqi": "Iraqi features, Mesopotamian Middle Eastern appearance",
        "Iranian": "Iranian features, Persian appearance",
        "Persian": "Persian features, Iranian appearance, diverse",
        "Turkish": "Turkish features, Anatolian appearance",
        "Egyptian": "Egyptian features, North African appearance",
        "Moroccan": "Moroccan features, North African Maghrebi appearance",
        "Algerian": "Algerian features, North African Maghrebi appearance",
        "Tunisian": "Tunisian features, North African appearance",
        "Libyan": "Libyan features, North African appearance",
        "Israeli": "Israeli features, diverse Middle Eastern appearance",
        "Saudi": "Saudi features, Arabian Peninsula appearance",
        "Emirati": "Emirati features, Gulf Arab appearance",
        "Yemeni": "Yemeni features, Arabian Peninsula appearance",
        # African
        "African": "African features, Sub-Saharan appearance, diverse",
        "West African": "West African features, diverse appearances",
        "Nigerian": "Nigerian features, West African appearance",
        "Ghanaian": "Ghanaian features, West African appearance",
        "Senegalese": "Senegalese features, West African appearance",
        "Ivorian": "Ivorian features, West African appearance",
        "Cameroonian": "Cameroonian features, Central-West African appearance",
        "East African": "East African features, Horn of Africa appearance",
        "Ethiopian": "Ethiopian features, East African appearance",
        "Eritrean": "Eritrean features, East African Horn appearance",
        "Somali": "Somali features, East African Horn appearance",
        "Kenyan": "Kenyan features, East African appearance",
        "Tanzanian": "Tanzanian features, East African appearance",
        "Ugandan": "Ugandan features, East African appearance",
        "Rwandan": "Rwandan features, Central-East African appearance",
        "South African": "South African features, diverse appearances",
        "Zimbabwean": "Zimbabwean features, Southern African appearance",
        "Congolese": "Congolese features, Central African appearance",
        "Sudanese": "Sudanese features, Northeast African appearance",
        # Americas
        "Native American": "Native American features, Indigenous American appearance",
        "Indigenous American": "Indigenous American features, First Nations appearance",
        "Inuit": "Inuit features, Arctic Indigenous appearance",
        "Latinx": "Latinx features, Latin American diverse appearance",
        "Mexican": "Mexican features, Latin American Mestizo appearance",
        "Puerto Rican": "Puerto Rican features, Caribbean Latin appearance",
        "Cuban": "Cuban features, Caribbean Latin appearance",
        "Dominican": "Dominican features, Caribbean Latin appearance",
        "Colombian": "Colombian features, South American appearance",
        "Venezuelan": "Venezuelan features, South American appearance",
        "Brazilian": "Brazilian features, diverse South American appearance",
        "Argentinian": "Argentinian features, South American appearance",
        "Chilean": "Chilean features, South American appearance",
        "Peruvian": "Peruvian features, Andean South American appearance",
        "Ecuadorian": "Ecuadorian features, Andean appearance",
        "Bolivian": "Bolivian features, Andean Indigenous appearance",
        # Pacific & Oceania
        "Pacific Islander": "Pacific Islander features, Polynesian Melanesian appearance",
        "Hawaiian": "Hawaiian features, Polynesian appearance",
        "Polynesian": "Polynesian features, Pacific Islander appearance",
        "Samoan": "Samoan features, Polynesian appearance",
        "Tongan": "Tongan features, Polynesian appearance",
        "Fijian": "Fijian features, Melanesian-Polynesian appearance",
        "Maori": "Maori features, New Zealand Indigenous Polynesian appearance",
        "Aboriginal Australian": "Aboriginal Australian features, Indigenous Australian appearance",
        "Torres Strait Islander": "Torres Strait Islander features, Melanesian appearance",
        "Melanesian": "Melanesian features, Pacific Islander appearance",
        "Micronesian": "Micronesian features, Pacific Islander appearance",
        # Mixed Heritage
        "Mixed Race": "mixed race features, multiracial appearance, blended heritage",
        "Biracial": "biracial features, dual heritage appearance",
        "Multiracial": "multiracial features, multiple heritage appearance",
        "Eurasian": "Eurasian features, European-Asian mixed appearance",
        "Afro-Asian": "Afro-Asian features, African-Asian mixed appearance",
        "Afro-Latinx": "Afro-Latinx features, African-Latin American appearance",
        "Mixed European-African": "mixed European-African features, blended appearance",
        "Mixed Asian-Latinx": "mixed Asian-Latinx features, blended appearance",
    }

    # Skin Styles - Tones and Textures
    SKIN_STYLES = {
        "None": "",
        # Skin Tones - Fair
        "Porcelain": "porcelain skin, very fair, pale luminous complexion",
        "Ivory": "ivory skin tone, fair with warm undertone, creamy",
        "Fair": "fair skin, light complexion, delicate",
        "Light": "light skin tone, pale complexion",
        "Pale": "pale skin, very light complexion, ethereal",
        "Alabaster": "alabaster skin, smooth pale white, classic",
        "Cream": "cream skin tone, fair with warmth, soft",
        "Peaches and Cream": "peaches and cream complexion, fair with pink undertones",
        "Rose": "rose-tinted fair skin, pink undertones, delicate",
        # Skin Tones - Light Medium
        "Light Medium": "light medium skin tone, transitional complexion",
        "Beige": "beige skin tone, neutral light-medium, versatile",
        "Sand": "sand skin tone, warm light-medium, golden",
        "Nude": "nude skin tone, neutral medium, natural",
        "Buff": "buff skin tone, warm light-medium complexion",
        "Natural": "natural skin tone, healthy medium complexion",
        # Skin Tones - Medium
        "Medium": "medium skin tone, balanced complexion",
        "Tan": "tan skin, sun-kissed medium, warm",
        "Golden": "golden skin tone, warm radiant medium, glowing",
        "Olive": "olive skin tone, Mediterranean, greenish undertone",
        "Honey": "honey skin tone, warm golden medium, luminous",
        "Caramel": "caramel skin tone, warm medium-tan, rich",
        "Bronze": "bronze skin tone, deep tan, metallic warmth",
        "Tawny": "tawny skin tone, warm brown-tan, earthy",
        "Warm Beige": "warm beige skin, golden undertones",
        "Cool Beige": "cool beige skin, pink undertones",
        # Skin Tones - Medium Deep
        "Medium Deep": "medium deep skin tone, rich complexion",
        "Cinnamon": "cinnamon skin tone, warm reddish-brown, spicy",
        "Chestnut": "chestnut skin tone, warm brown, rich",
        "Copper": "copper skin tone, reddish-brown, metallic warmth",
        "Amber": "amber skin tone, golden-brown, warm radiant",
        "Toffee": "toffee skin tone, rich medium-brown, warm",
        "Sienna": "sienna skin tone, earthy reddish-brown",
        # Skin Tones - Deep
        "Deep": "deep skin tone, rich dark complexion",
        "Brown": "brown skin tone, rich warm brown complexion",
        "Dark Brown": "dark brown skin, deep warm brown",
        "Chocolate": "chocolate skin tone, rich deep brown, luxurious",
        "Mahogany": "mahogany skin tone, reddish deep brown, rich",
        "Espresso": "espresso skin tone, very deep brown, intense",
        "Mocha": "mocha skin tone, cool deep brown, rich",
        "Cocoa": "cocoa skin tone, deep warm brown, velvety",
        "Coffee": "coffee skin tone, deep rich brown",
        "Ebony": "ebony skin tone, very deep black-brown, striking",
        "Onyx": "onyx skin tone, darkest deep brown-black, dramatic",
        "Deep Ebony": "deep ebony skin, richest dark complexion",
        # Undertones
        "Warm Undertone": "warm undertone skin, golden yellow peachy",
        "Cool Undertone": "cool undertone skin, pink red blue",
        "Neutral Undertone": "neutral undertone skin, balanced mix",
        "Golden Undertone": "golden undertone, yellow-gold warmth",
        "Peachy Undertone": "peachy undertone, warm pink-orange",
        "Pink Undertone": "pink undertone, cool rosy tones",
        "Red Undertone": "red undertone, warm reddish tones",
        "Blue Undertone": "blue undertone, cool blue-purple tones",
        "Yellow Undertone": "yellow undertone, warm sallow tones",
        "Olive Undertone": "olive undertone, green-gray neutral",
        # Skin Textures
        "Smooth Skin": "smooth skin texture, even flawless surface",
        "Poreless": "poreless skin, refined smooth texture, airbrushed",
        "Dewy Skin": "dewy skin, hydrated glowing, fresh moisturized",
        "Matte Skin": "matte skin finish, shine-free, velvety",
        "Glowing Skin": "glowing radiant skin, luminous healthy, lit from within",
        "Glass Skin": "glass skin, Korean beauty, translucent dewy flawless",
        "Satin Skin": "satin skin finish, soft subtle sheen, refined",
        "Velvet Skin": "velvet skin texture, soft matte luxurious",
        "Natural Texture": "natural skin texture, realistic pores, authentic",
        "Soft Focus Skin": "soft focus skin, gently blurred, dreamy",
        # Skin Characteristics
        "Freckled": "freckled skin, natural freckles, sun-kissed spots",
        "Heavy Freckles": "heavily freckled skin, abundant natural freckles",
        "Light Freckles": "light freckles, subtle sparse freckling",
        "Moles": "skin with beauty marks, natural moles",
        "Beauty Mark": "prominent beauty mark, Cindy Crawford style mole",
        "Clear Skin": "clear skin, blemish-free, healthy complexion",
        "Youthful Skin": "youthful skin, plump firm, young complexion",
        "Mature Skin": "mature skin, graceful aging, wisdom lines",
        "Sun-Kissed": "sun-kissed skin, natural tan, healthy glow",
        "Bronzed": "bronzed skin, deep tan, summer glow",
        "Pale Luminous": "pale luminous skin, ethereal glow, vampire-like",
        # Skin Conditions (for representation)
        "Vitiligo": "vitiligo skin, depigmented patches, unique pattern",
        "Rosacea": "rosacea skin, rosy cheeks, flushed appearance",
        "Albinism": "albinism, very pale skin, light features",
        # Skin Finish
        "Highlighted Skin": "highlighted skin, strategic glow points, sculpted",
        "Contoured": "contoured skin, sculpted shadows highlights, defined",
        "Natural Finish": "natural skin finish, no heavy makeup, authentic",
        "Full Glam": "full glam skin, flawless makeup, perfected",
        "Bare Skin": "bare skin, no makeup, natural beauty",
        "Minimal Makeup": "minimal makeup skin, enhanced natural, fresh",
    }

    # Conflicting preset combinations to warn about
    PRESET_CONFLICTS = {
        # Creative effects conflicts
        ("Soft Focus", "High Contrast"): "Soft focus typically reduces contrast",
        ("Soft Focus", "HDR"): "Soft focus conflicts with HDR sharpness",
        ("Black and White", "Rich Colors"): "Black and white has no color",
        ("Black and White", "Teal Orange"): "Black and white has no color",
        ("Black and White", "Cross Process"): "Black and white has no color",
        ("Black and White", "Golden Hour"): "Black and white has no color tint",
        ("Golden Hour", "Blue Hour"): "Conflicting warm vs cool color temperatures",
        ("High Contrast", "Matte Finish"): "Matte finish reduces contrast",
        ("High Contrast", "Faded Film"): "Faded film reduces contrast",
        # Lighting conflicts
        ("High Key", "Low Key"): "Opposite lighting approaches",
        ("Split Lighting", "High Key"): "Split lighting creates shadows incompatible with high key",
        # Skin/makeup conflicts
        ("Full Glam", "Bare Skin"): "Contradictory makeup levels",
        ("Full Glam", "Minimal Makeup"): "Contradictory makeup levels",
        ("Bare Skin", "Glitter Eye"): "Bare skin implies no makeup",
        ("Bare Skin", "Smoky Eye"): "Bare skin implies no makeup",
        # Expression conflicts
        ("Smile", "Angry"): "Contradictory expressions",
        ("Smile", "Sad"): "Contradictory expressions",
        ("Laughing", "Crying"): "Contradictory expressions",
        ("Neutral", "Intense"): "Conflicting emotional intensity",
    }

    @classmethod
    def _check_preset_conflicts(cls, presets: dict) -> list:
        """Check for conflicting preset combinations and return warnings."""
        warnings = []
        # Get all selected preset names
        selected = []
        for key, value in presets.items():
            if value:
                # Find the preset name by matching the value
                for dict_name in ['CREATIVE_EFFECTS', 'PHOTOGRAPHY_EFFECTS', 'PHOTOGRAPHY_TEMPLATES',
                                  'CINEMATIC_STYLES', 'ANIME_STYLES', 'GAMING_STYLES',
                                  'FACIAL_EXPRESSIONS', 'SKIN_STYLES', 'EYE_STYLES']:
                    preset_dict = getattr(cls, dict_name, {})
                    for name, desc in preset_dict.items():
                        if desc == value:
                            selected.append(name)
                            break

        # Check for conflicts
        for (preset1, preset2), reason in cls.PRESET_CONFLICTS.items():
            if preset1 in selected and preset2 in selected:
                warnings.append(f"⚠️ Conflict: '{preset1}' + '{preset2}' - {reason}")

        return warnings

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        """Define ComfyUI input types."""
        return {
            "required": {
                "prompt": ("STRING", {"forceInput": True, "tooltip": "Input prompt - can be basic text (for Expand mode) or detailed prompt (for Modify modes)"}),
                "llm_model": ("LLM_MODEL",),
            },
            "optional": {
                "processing_mode": (["Expand", "Single Pass", "Section by Section", "6-Section Generate"], {"default": "Single Pass", "tooltip": "Expand: generate from basic text. Single Pass/Section: modify existing prompt. 6-Section Generate: create each section independently (6 LLM passes)"}),
                # Photography presets
                "photography_template": (list(cls.PHOTOGRAPHY_TEMPLATES.keys()), {"default": "None", "tooltip": "Apply a photography style template"}),
                "photography_effect": (list(cls.PHOTOGRAPHY_EFFECTS.keys()), {"default": "None", "tooltip": "Apply a visual effect"}),
                "photographer_style": (list(cls.PHOTOGRAPHER_STYLES.keys()), {"default": "None", "tooltip": "Emulate a famous photographer's style"}),
                # Creative effects
                "creative_effect": (list(cls.CREATIVE_EFFECTS.keys()), {"default": "None", "tooltip": "Apply creative visual effects (motion, color, texture, surreal)"}),
                "cinematic_style": (list(cls.CINEMATIC_STYLES.keys()), {"default": "None", "tooltip": "Apply movie/director aesthetic (Tarantino, Nolan, Snyder, etc.)"}),
                "anime_style": (list(cls.ANIME_STYLES.keys()), {"default": "None", "tooltip": "Apply anime/manga style (DBZ, Naruto, Ghibli, etc.)"}),
                "gaming_style": (list(cls.GAMING_STYLES.keys()), {"default": "None", "tooltip": "Apply gaming/TV series aesthetic (Cyberpunk, Souls, Spider-Verse, etc.)"}),
                # Subject presets (dropdowns above subject_instruction)
                "ethnicity_preset": (list(cls.ETHNICITY.keys()), {"default": "None", "tooltip": "Apply ethnicity/cultural features"}),
                "skin_preset": (list(cls.SKIN_STYLES.keys()), {"default": "None", "tooltip": "Apply skin tone and texture"}),
                "hair_preset": (list(cls.HAIR_STYLES.keys()), {"default": "None", "tooltip": "Apply hair style"}),
                "eye_preset": (list(cls.EYE_STYLES.keys()), {"default": "None", "tooltip": "Apply eye style/makeup"}),
                "expression_preset": (list(cls.FACIAL_EXPRESSIONS.keys()), {"default": "None", "tooltip": "Apply facial expression"}),
                "subject_instruction": ("STRING", {"multiline": True, "default": "", "tooltip": "How to modify subject (e.g., 'Make younger, add freckles')"}),
                # Clothing presets (dropdowns above clothing_instruction)
                "fashion_preset": (list(cls.FASHION_STYLES.keys()), {"default": "None", "tooltip": "Apply fashion style aesthetic"}),
                "clothing_preset": (list(cls.CLOTHING_STYLES.keys()), {"default": "None", "tooltip": "Apply specific clothing style"}),
                "clothing_instruction": ("STRING", {"multiline": True, "default": "", "tooltip": "How to modify clothing (e.g., 'Change to elegant red dress')"}),
                # Pose preset (dropdown above pose_instruction)
                "pose_preset": (list(cls.POSES.keys()), {"default": "None", "tooltip": "Apply pose from presets"}),
                "pose_instruction": ("STRING", {"multiline": True, "default": "", "tooltip": "How to modify pose (e.g., 'More confident stance, hands on hips')"}),
                "environment_instruction": ("STRING", {"multiline": True, "default": "", "tooltip": "How to modify environment (e.g., 'Move to beach at sunset')"}),
                "lighting_instruction": ("STRING", {"multiline": True, "default": "", "tooltip": "How to modify lighting (e.g., 'Golden hour, warm tones')"}),
                "camera_instruction": ("STRING", {"multiline": True, "default": "", "tooltip": "How to modify camera (e.g., 'Close-up portrait, shallow DOF')"}),
                "generate_caption": ("BOOLEAN", {"default": False, "tooltip": "Generate Instagram caption from modified prompt"}),
                "release_vram": ("BOOLEAN", {"default": True, "tooltip": "Release VRAM after execution (recommended)"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Random seed for reproducibility (change to force re-generation)"}),
            },
        }

    def modify(
        self,
        prompt: str,
        llm_model: Any,
        processing_mode: str = "Single Pass",
        photography_template: str = "None",
        photography_effect: str = "None",
        photographer_style: str = "None",
        creative_effect: str = "None",
        cinematic_style: str = "None",
        anime_style: str = "None",
        gaming_style: str = "None",
        # Subject presets
        ethnicity_preset: str = "None",
        skin_preset: str = "None",
        hair_preset: str = "None",
        eye_preset: str = "None",
        expression_preset: str = "None",
        subject_instruction: str = "",
        # Clothing presets
        fashion_preset: str = "None",
        clothing_preset: str = "None",
        clothing_instruction: str = "",
        # Pose preset
        pose_preset: str = "None",
        pose_instruction: str = "",
        environment_instruction: str = "",
        lighting_instruction: str = "",
        camera_instruction: str = "",
        generate_caption: bool = False,
        release_vram: bool = True,
        seed: int = 0,
    ) -> Tuple[str, str]:
        """
        Modify prompt based on section instructions and photography presets.

        Args:
            prompt: Input prompt to modify
            llm_model: LLM configuration (temperature and other params from LLM node)
            processing_mode: "Single Pass" or "Section by Section"
            photography_template: Photography style template to apply
            photography_effect: Visual effect to apply
            photographer_style: Famous photographer style to emulate
            creative_effect: Creative visual effect to apply
            cinematic_style: Movie/director aesthetic to apply
            anime_style: Anime/manga style to apply
            gaming_style: Gaming/TV series aesthetic to apply
            ethnicity_preset: Ethnicity/cultural features preset
            skin_preset: Skin tone and texture preset
            hair_preset: Hair style preset
            eye_preset: Eye style/makeup preset
            expression_preset: Facial expression preset
            subject_instruction: Modification for subject section
            fashion_preset: Fashion style aesthetic preset
            clothing_preset: Specific clothing style preset
            clothing_instruction: Modification for clothing section
            pose_preset: Pose preset
            pose_instruction: Modification for pose section
            environment_instruction: Modification for environment section
            lighting_instruction: Modification for lighting section
            camera_instruction: Modification for camera section
            generate_caption: Whether to generate Instagram caption
            release_vram: Release VRAM after execution
            seed: Random seed for reproducibility (change to force re-generation)

        Returns:
            Tuple of (modified_prompt, caption)
        """
        start_time = time.time()

        # Build cache key from all inputs
        cache_key = self._build_cache_key(
            prompt=prompt,
            seed=seed,
            processing_mode=processing_mode,
            photography_template=photography_template,
            photography_effect=photography_effect,
            photographer_style=photographer_style,
            creative_effect=creative_effect,
            cinematic_style=cinematic_style,
            anime_style=anime_style,
            gaming_style=gaming_style,
            ethnicity_preset=ethnicity_preset,
            skin_preset=skin_preset,
            hair_preset=hair_preset,
            eye_preset=eye_preset,
            expression_preset=expression_preset,
            subject_instruction=subject_instruction,
            fashion_preset=fashion_preset,
            clothing_preset=clothing_preset,
            clothing_instruction=clothing_instruction,
            pose_preset=pose_preset,
            pose_instruction=pose_instruction,
            environment_instruction=environment_instruction,
            lighting_instruction=lighting_instruction,
            camera_instruction=camera_instruction,
            generate_caption=generate_caption,
            model=llm_model.model if llm_model else "",
        )

        # Check cache
        if cache_key in SID_PromptModifier._cache:
            cached_prompt, cached_caption = SID_PromptModifier._cache[cache_key]
            print(f"[SID_PromptModifier] Cache hit - returning cached result (seed: {seed})")
            return (cached_prompt, cached_caption)

        # Check for interrupt before starting
        check_interrupt()

        # Get temperature from LLM model config
        temperature = getattr(llm_model, 'temperature', 0.7)

        # Collect section instructions
        instructions = {
            "subject": subject_instruction.strip(),
            "clothing": clothing_instruction.strip(),
            "pose": pose_instruction.strip(),
            "environment": environment_instruction.strip(),
            "lighting": lighting_instruction.strip(),
            "camera": camera_instruction.strip(),
        }

        # Filter out empty instructions
        active_instructions = {k: v for k, v in instructions.items() if v}

        # Collect photography presets
        presets = {}
        if photography_template != "None":
            presets["template"] = self.PHOTOGRAPHY_TEMPLATES.get(photography_template, "")
        if photography_effect != "None":
            presets["effect"] = self.PHOTOGRAPHY_EFFECTS.get(photography_effect, "")
        if photographer_style != "None":
            presets["photographer"] = self.PHOTOGRAPHER_STYLES.get(photographer_style, "")
        # Creative effects
        if creative_effect != "None":
            presets["creative"] = self.CREATIVE_EFFECTS.get(creative_effect, "")
        if cinematic_style != "None":
            presets["cinematic"] = self.CINEMATIC_STYLES.get(cinematic_style, "")
        if anime_style != "None":
            presets["anime"] = self.ANIME_STYLES.get(anime_style, "")
        if gaming_style != "None":
            presets["gaming"] = self.GAMING_STYLES.get(gaming_style, "")
        # Subject presets
        if ethnicity_preset != "None":
            presets["ethnicity"] = self.ETHNICITY.get(ethnicity_preset, "")
        if skin_preset != "None":
            presets["skin"] = self.SKIN_STYLES.get(skin_preset, "")
        if hair_preset != "None":
            presets["hair"] = self.HAIR_STYLES.get(hair_preset, "")
        if eye_preset != "None":
            presets["eye"] = self.EYE_STYLES.get(eye_preset, "")
        if expression_preset != "None":
            presets["expression"] = self.FACIAL_EXPRESSIONS.get(expression_preset, "")
        # Clothing presets
        if fashion_preset != "None":
            presets["fashion"] = self.FASHION_STYLES.get(fashion_preset, "")
        if clothing_preset != "None":
            presets["clothing"] = self.CLOTHING_STYLES.get(clothing_preset, "")
        # Pose preset
        if pose_preset != "None":
            presets["pose"] = self.POSES.get(pose_preset, "")

        # If no instructions or presets, return original prompt
        if not active_instructions and not presets:
            print("[SID_PromptModifier] No modifications requested, returning original prompt")
            caption = ""
            if generate_caption:
                caption = self._generate_caption(llm_model, prompt, temperature)
            return (prompt, caption)

        # Log what we're applying
        if presets:
            preset_names = []
            if photography_template != "None":
                preset_names.append(f"Template: {photography_template}")
            if photography_effect != "None":
                preset_names.append(f"Effect: {photography_effect}")
            if photographer_style != "None":
                preset_names.append(f"Style: {photographer_style}")
            if creative_effect != "None":
                preset_names.append(f"Creative: {creative_effect}")
            if cinematic_style != "None":
                preset_names.append(f"Cinematic: {cinematic_style}")
            if anime_style != "None":
                preset_names.append(f"Anime: {anime_style}")
            if gaming_style != "None":
                preset_names.append(f"Gaming: {gaming_style}")
            if ethnicity_preset != "None":
                preset_names.append(f"Ethnicity: {ethnicity_preset}")
            if skin_preset != "None":
                preset_names.append(f"Skin: {skin_preset}")
            if hair_preset != "None":
                preset_names.append(f"Hair: {hair_preset}")
            if eye_preset != "None":
                preset_names.append(f"Eye: {eye_preset}")
            if expression_preset != "None":
                preset_names.append(f"Expression: {expression_preset}")
            if fashion_preset != "None":
                preset_names.append(f"Fashion: {fashion_preset}")
            if clothing_preset != "None":
                preset_names.append(f"Clothing: {clothing_preset}")
            if pose_preset != "None":
                preset_names.append(f"Pose: {pose_preset}")
            print(f"[SID_PromptModifier] Applying: {', '.join(preset_names)}")

            # Check for conflicting presets
            conflicts = self._check_preset_conflicts(presets)
            for warning in conflicts:
                print(f"[SID_PromptModifier] {warning}")

        # Process based on mode
        if processing_mode == "Expand":
            # Expand basic text into detailed prompt, then apply modifications
            print(f"[SID_PromptModifier] Expanding basic text into detailed prompt...")
            expanded_prompt = self._expand_prompt(prompt, llm_model, temperature, presets)
            # Apply any section modifications to the expanded prompt
            if active_instructions:
                modified_prompt = self._single_pass_modify(expanded_prompt, llm_model, active_instructions, temperature, {})
            else:
                modified_prompt = expanded_prompt
        elif processing_mode == "Single Pass":
            modified_prompt = self._single_pass_modify(prompt, llm_model, active_instructions, temperature, presets)
        elif processing_mode == "6-Section Generate":
            # Generate each section independently with 6 focused LLM passes
            print(f"[SID_PromptModifier] Generating 6 sections independently...")
            modified_prompt = self._six_section_generate(prompt, llm_model, temperature, presets, active_instructions)
        else:
            modified_prompt = self._section_by_section_modify(prompt, llm_model, active_instructions, temperature, presets)

        # Check for interrupt after modification
        check_interrupt()

        # Generate caption if requested
        caption = ""
        if generate_caption:
            check_interrupt()
            caption = self._generate_caption(llm_model, modified_prompt, temperature)

        # Print analysis report
        self._print_analysis_report(
            original_prompt=prompt,
            modified_prompt=modified_prompt,
            instructions=active_instructions,
            presets=presets,
        )

        # Release VRAM if requested
        if release_vram:
            self._release_vram()

        elapsed = int((time.time() - start_time) * 1000)
        print(f"[SID_PromptModifier] Completed in {elapsed}ms (mode: {processing_mode}, temp: {temperature})")

        # Store in cache
        SID_PromptModifier._cache[cache_key] = (modified_prompt, caption)
        print(f"[SID_PromptModifier] Result cached (seed: {seed})")

        return (modified_prompt, caption)

    def _build_cache_key(self, **kwargs) -> str:
        """Build a cache key from all input parameters."""
        # Create a stable string representation of all inputs
        key_parts = []
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}={v}")
        key_string = "|".join(key_parts)
        # Hash it for a compact key
        return hashlib.md5(key_string.encode()).hexdigest()

    def _single_pass_modify(
        self,
        prompt: str,
        llm_model: Any,
        instructions: Dict[str, str],
        temperature: float,
        presets: Dict[str, str] = None,
    ) -> str:
        """Modify prompt in a single LLM call."""
        presets = presets or {}

        # Build instruction list for sections
        instruction_text = ""
        for section_key, instruction in instructions.items():
            section_name = self.SECTIONS[section_key]["name"]
            instruction_text += f"- **{section_name}**: {instruction}\n"

        # Build preset instructions
        preset_text = ""
        if presets.get("template"):
            preset_text += f"- **Photography Template**: {presets['template']}\n"
        if presets.get("effect"):
            preset_text += f"- **Visual Effect**: {presets['effect']}\n"
        if presets.get("photographer"):
            preset_text += f"- **Photographer Style**: {presets['photographer']}\n"
        if presets.get("creative"):
            preset_text += f"- **Creative Effect**: {presets['creative']}\n"
        if presets.get("cinematic"):
            preset_text += f"- **Cinematic Style**: {presets['cinematic']}\n"
        if presets.get("anime"):
            preset_text += f"- **Anime Style**: {presets['anime']}\n"
        if presets.get("gaming"):
            preset_text += f"- **Gaming/TV Style**: {presets['gaming']}\n"
        # Character presets
        if presets.get("ethnicity"):
            preset_text += f"- **Ethnicity**: {presets['ethnicity']}\n"
        if presets.get("skin"):
            preset_text += f"- **Skin**: {presets['skin']}\n"
        if presets.get("hair"):
            preset_text += f"- **Hair Style**: {presets['hair']}\n"
        if presets.get("eye"):
            preset_text += f"- **Eye Style**: {presets['eye']}\n"
        if presets.get("expression"):
            preset_text += f"- **Expression**: {presets['expression']}\n"
        if presets.get("fashion"):
            preset_text += f"- **Fashion Style**: {presets['fashion']}\n"
        if presets.get("clothing"):
            preset_text += f"- **Clothing**: {presets['clothing']}\n"
        if presets.get("pose"):
            preset_text += f"- **Pose**: {presets['pose']}\n"

        system_prompt = """You are an expert prompt engineer for AI image generation (Stable Diffusion, Flux, Z-Image).

Your task is to modify an existing image prompt based on specific instructions and style presets.

CAMERA THINKING - Write like setting camera parameters:
- Use technical, concrete terms (not poetic/literary language)
- "f/1.8, shallow depth of field, subject sharp, background blur" NOT "dreamy ethereal softness"
- "golden hour, warm 3200K, low angle sun" NOT "bathed in the warm embrace of sunset"

RULES:
1. PRESERVE the overall structure, sections, and formatting of the original prompt exactly
2. KEEP all original details unless explicitly changed by a preset
3. Apply modifications by ADDING or REPLACING relevant phrases only
4. Keep unmentioned sections COMPLETELY UNCHANGED - copy them verbatim
5. Limit each section to 3-5 key visual concepts - avoid overloading
6. NO contradictory terms (soft focus + sharp focus, warm + cool lighting)
7. Output ONLY the modified prompt - no explanations or commentary
8. The output length should be similar to or longer than the input - NEVER shorten or condense"""

        # Build user prompt with all modifications
        user_prompt = f"""ORIGINAL PROMPT:
{prompt}

"""
        if preset_text:
            user_prompt += f"""STYLE PRESETS TO APPLY:
{preset_text}
"""
        if instruction_text:
            user_prompt += f"""SECTION MODIFICATIONS:
{instruction_text}
"""
        user_prompt += "Modify the prompt according to the instructions above. Integrate all style presets and modifications naturally. Output only the modified prompt."

        try:
            client = self._get_client(llm_model)
            response = self._call_llm(client, llm_model, system_prompt, user_prompt, temperature, prompt)
            return self._clean_response(response)
        except Exception as e:
            print(f"[SID_PromptModifier] Error in single pass: {e}")
            return prompt

    def _section_by_section_modify(
        self,
        prompt: str,
        llm_model: Any,
        instructions: Dict[str, str],
        temperature: float,
        presets: Dict[str, str] = None,
    ) -> str:
        """Modify prompt section by section with separate LLM calls."""
        presets = presets or {}

        modified_prompt = prompt

        # First, apply presets if any (as a single pass for style integration)
        if presets:
            preset_text = ""
            if presets.get("template"):
                preset_text += f"- **Photography Template**: {presets['template']}\n"
            if presets.get("effect"):
                preset_text += f"- **Visual Effect**: {presets['effect']}\n"
            if presets.get("photographer"):
                preset_text += f"- **Photographer Style**: {presets['photographer']}\n"
            if presets.get("creative"):
                preset_text += f"- **Creative Effect**: {presets['creative']}\n"
            if presets.get("cinematic"):
                preset_text += f"- **Cinematic Style**: {presets['cinematic']}\n"
            if presets.get("anime"):
                preset_text += f"- **Anime Style**: {presets['anime']}\n"
            if presets.get("gaming"):
                preset_text += f"- **Gaming/TV Style**: {presets['gaming']}\n"
            # Character presets
            if presets.get("ethnicity"):
                preset_text += f"- **Ethnicity**: {presets['ethnicity']}\n"
            if presets.get("skin"):
                preset_text += f"- **Skin**: {presets['skin']}\n"
            if presets.get("hair"):
                preset_text += f"- **Hair Style**: {presets['hair']}\n"
            if presets.get("eye"):
                preset_text += f"- **Eye Style**: {presets['eye']}\n"
            if presets.get("expression"):
                preset_text += f"- **Expression**: {presets['expression']}\n"
            if presets.get("fashion"):
                preset_text += f"- **Fashion Style**: {presets['fashion']}\n"
            if presets.get("clothing"):
                preset_text += f"- **Clothing**: {presets['clothing']}\n"
            if presets.get("pose"):
                preset_text += f"- **Pose**: {presets['pose']}\n"

            preset_system = """You are an expert prompt engineer for AI image generation.

Your task is to integrate photography style presets into an existing prompt.

CAMERA THINKING - Write like setting camera parameters:
- Use technical, concrete terms (not poetic/literary language)
- "f/1.8, shallow depth of field" NOT "dreamy ethereal softness"
- "golden hour, warm 3200K" NOT "bathed in warm embrace of sunset"

RULES:
1. PRESERVE the overall structure and formatting of the original prompt exactly
2. KEEP all original details unless explicitly changed by a preset
3. Integrate presets by ADDING phrases to appropriate sections, not by rewriting
4. Limit each section to 3-5 key visual concepts - avoid overloading
5. NO contradictory terms (soft focus + sharp focus, warm + cool lighting)
6. Output ONLY the modified prompt - no explanations
7. The output length should be similar to or longer than the input - NEVER shorten"""

            preset_user = f"""CURRENT PROMPT:
{modified_prompt}

STYLE PRESETS TO APPLY:
{preset_text}

Integrate these style presets into the prompt. Output only the modified prompt."""

            try:
                client = self._get_client(llm_model)
                response = self._call_llm(client, llm_model, preset_system, preset_user, temperature, modified_prompt)
                modified_prompt = self._clean_response(response)
                print(f"[SID_PromptModifier] Applied style presets")
            except Exception as e:
                print(f"[SID_PromptModifier] Error applying presets: {e}")

        # Then apply section-by-section modifications
        section_system = """You are an expert prompt engineer for AI image generation.

Your task is to modify ONLY ONE specific section of an image prompt based on the instruction.

CAMERA THINKING - Write like setting camera parameters:
- Use technical, concrete terms (not poetic/literary language)
- "f/1.8, shallow depth of field" NOT "dreamy softness"
- "golden hour, warm 3200K" NOT "warm embrace of sunset"

RULES:
1. ONLY modify the specified section - add or replace relevant phrases
2. Keep ALL other parts of the prompt EXACTLY the same - copy them verbatim
3. PRESERVE the overall structure and formatting of the original prompt
4. Limit modified section to 3-5 key visual concepts - avoid overloading
5. NO contradictory terms within the same section
6. Output ONLY the complete modified prompt - no explanations
7. The output length should be similar to or longer than the input - NEVER shorten"""

        for section_key, instruction in instructions.items():
            section_info = self.SECTIONS[section_key]
            section_name = section_info["name"]
            section_desc = section_info["description"]

            user_prompt = f"""CURRENT PROMPT:
{modified_prompt}

SECTION TO MODIFY: {section_name}
SECTION DESCRIPTION: {section_desc}
INSTRUCTION: {instruction}

Modify ONLY the {section_name} section according to the instruction. Keep everything else unchanged. Output only the complete modified prompt."""

            try:
                client = self._get_client(llm_model)
                response = self._call_llm(client, llm_model, section_system, user_prompt, temperature, modified_prompt)
                modified_prompt = self._clean_response(response)
                print(f"[SID_PromptModifier] Modified section: {section_name}")
            except Exception as e:
                print(f"[SID_PromptModifier] Error modifying {section_name}: {e}")
                # Continue with current prompt

        return modified_prompt

    def _six_section_generate(
        self,
        basic_text: str,
        llm_model: Any,
        temperature: float,
        presets: Dict[str, str] = None,
        instructions: Dict[str, str] = None,
    ) -> str:
        """Generate each of 6 sections independently with focused LLM passes."""
        presets = presets or {}
        instructions = instructions or {}

        # Build preset context for each section
        preset_context = self._build_preset_context(presets)

        # Define the 6 sections with focused prompts
        sections = [
            {
                "name": "Subject",
                "prompt": """Analyze the subject description and generate ONLY the SUBJECT section.

Describe with technical precision (2-3 sentences max):
- Gender, apparent age range, ethnicity/racial features
- Face shape, skin tone, hair color/length/texture/style
- Distinctive facial features (cheekbones, lips, jaw, etc.)

Use concrete visual terms only. No poetic language.""",
                "presets": ["ethnicity", "skin", "hair"],
            },
            {
                "name": "Clothing",
                "prompt": """Analyze the description and generate ONLY the CLOTHING section.

Describe with technical precision (2-3 sentences max):
- Each garment: type, color, material/fabric, fit
- Style: casual, formal, streetwear, elegant, etc.
- Accessories: jewelry, bags, glasses, hats, shoes

Use concrete visual terms only. No poetic language.""",
                "presets": ["clothing", "fashion"],
            },
            {
                "name": "Pose & Expression",
                "prompt": """Analyze the description and generate ONLY the POSE & EXPRESSION section.

Describe with technical precision (2-3 sentences max):
POSE: body orientation, posture, arm/hand position, weight distribution
EXPRESSION: facial expression, eye direction, mouth, mood conveyed

Use concrete visual terms only. No poetic language.""",
                "presets": ["pose", "expression"],
            },
            {
                "name": "Scene",
                "prompt": """Analyze the description and generate ONLY the SCENE section.

Describe with technical precision (2-3 sentences max):
- Location type: studio, outdoor, indoor, urban, nature
- Background elements and props
- Atmosphere and environmental details

Use concrete visual terms only. No poetic language.""",
                "presets": [],
            },
            {
                "name": "Lighting",
                "prompt": """Analyze the description and generate ONLY the LIGHTING section.

Describe with technical precision (2-3 sentences max):
- Light quality: soft/diffused vs hard/direct
- Direction: front, side, back, overhead
- Color temperature: warm (3200K) vs cool (5600K+)
- Shadow characteristics and highlights

Use camera terminology (color temps, ratios). No poetic language.""",
                "presets": ["creative", "effect"],
            },
            {
                "name": "Camera",
                "prompt": """Analyze the description and generate ONLY the CAMERA section.

Describe with technical precision (2-3 sentences max):
- Shot type: CU, MCU, MS, MFS, FS, LS
- Camera angle: eye-level, high, low
- Depth of field: shallow (f/1.4-2.8), medium (f/4-5.6), deep (f/8+)
- Framing and composition

Use camera terminology (f-stops, focal lengths). No poetic language.""",
                "presets": ["template"],
            },
        ]

        # System prompt for focused section generation
        system_prompt = """You are an expert prompt engineer for AI image generation.

Your task is to generate ONLY ONE specific section of an image prompt.

RULES:
1. Generate ONLY the section requested - nothing else
2. Use technical, concrete visual terms only
3. NO poetic, philosophical, or emotional language
4. Keep output to 2-3 focused sentences (40-60 words max)
5. Apply any provided presets naturally
6. Output ONLY the section content - no labels or headers"""

        # Determine max tokens: 512 for local LLM, user-specified for API
        is_local = llm_model.provider.lower() == "local"
        section_max_tokens = 512 if is_local else llm_model.max_tokens

        section_outputs = []

        for section in sections:
            # Build section-specific user prompt
            user_prompt = f"""CONCEPT TO DESCRIBE:
{basic_text}

"""
            # Add relevant presets for this section
            section_presets = ""
            for preset_key in section["presets"]:
                if presets.get(preset_key):
                    section_presets += f"- {preset_key}: {presets[preset_key]}\n"

            if section_presets:
                user_prompt += f"""STYLE REQUIREMENTS:
{section_presets}
"""
            # Add any manual instructions for related sections
            section_instructions = ""
            for inst_key, inst_value in instructions.items():
                if inst_key.lower() in section["name"].lower() or section["name"].lower() in inst_key.lower():
                    section_instructions += f"- {inst_value}\n"

            if section_instructions:
                user_prompt += f"""ADDITIONAL INSTRUCTIONS:
{section_instructions}
"""

            user_prompt += section["prompt"]

            try:
                client = self._get_client(llm_model)
                response = self._call_llm_short(client, llm_model, system_prompt, user_prompt, temperature, max_tokens=section_max_tokens)
                cleaned = self._clean_response(response)
                section_outputs.append(cleaned)
                print(f"[SID_PromptModifier] Generated section: {section['name']} ({len(cleaned.split())} words)")
            except Exception as e:
                print(f"[SID_PromptModifier] Error generating {section['name']}: {e}")
                section_outputs.append("")

        # Assemble final prompt from all sections
        final_prompt = " ".join([s for s in section_outputs if s])
        word_count = len(final_prompt.split())
        print(f"[SID_PromptModifier] 6-Section generation complete: {word_count} total words")

        return final_prompt

    def _build_preset_context(self, presets: Dict[str, str]) -> str:
        """Build preset context string for prompts."""
        context = ""
        preset_names = {
            "template": "Photography Template",
            "effect": "Visual Effect",
            "photographer": "Photographer Style",
            "creative": "Creative Effect",
            "cinematic": "Cinematic Style",
            "anime": "Anime Style",
            "gaming": "Gaming/TV Style",
            "ethnicity": "Ethnicity",
            "skin": "Skin",
            "hair": "Hair Style",
            "eye": "Eye Style",
            "expression": "Expression",
            "fashion": "Fashion Style",
            "clothing": "Clothing",
            "pose": "Pose",
        }
        for key, name in preset_names.items():
            if presets.get(key):
                context += f"- **{name}**: {presets[key]}\n"
        return context

    def _call_llm_short(
        self,
        client: Any,
        llm_model: Any,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int = 200,
    ) -> str:
        """Make LLM call with short token limit for focused output."""
        if hasattr(client, 'messages'):
            # Anthropic API
            response = client.messages.create(
                model=llm_model.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}]
            )
            return response.content[0].text
        else:
            # OpenAI-compatible API
            response = client.chat.completions.create(
                model=llm_model.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            if response and response.choices and response.choices[0].message:
                return response.choices[0].message.content
            return ""

    def _expand_prompt(
        self,
        basic_text: str,
        llm_model: Any,
        temperature: float,
        presets: Dict[str, str] = None,
    ) -> str:
        """Expand basic text into a detailed image generation prompt."""
        presets = presets or {}

        # Build preset instructions
        preset_text = ""
        if presets.get("template"):
            preset_text += f"- **Photography Template**: {presets['template']}\n"
        if presets.get("effect"):
            preset_text += f"- **Visual Effect**: {presets['effect']}\n"
        if presets.get("photographer"):
            preset_text += f"- **Photographer Style**: {presets['photographer']}\n"
        if presets.get("creative"):
            preset_text += f"- **Creative Effect**: {presets['creative']}\n"
        if presets.get("cinematic"):
            preset_text += f"- **Cinematic Style**: {presets['cinematic']}\n"
        if presets.get("anime"):
            preset_text += f"- **Anime Style**: {presets['anime']}\n"
        if presets.get("gaming"):
            preset_text += f"- **Gaming/TV Style**: {presets['gaming']}\n"
        # Character presets
        if presets.get("ethnicity"):
            preset_text += f"- **Ethnicity**: {presets['ethnicity']}\n"
        if presets.get("skin"):
            preset_text += f"- **Skin**: {presets['skin']}\n"
        if presets.get("hair"):
            preset_text += f"- **Hair Style**: {presets['hair']}\n"
        if presets.get("eye"):
            preset_text += f"- **Eye Style**: {presets['eye']}\n"
        if presets.get("expression"):
            preset_text += f"- **Expression**: {presets['expression']}\n"
        if presets.get("fashion"):
            preset_text += f"- **Fashion Style**: {presets['fashion']}\n"
        if presets.get("clothing"):
            preset_text += f"- **Clothing**: {presets['clothing']}\n"
        if presets.get("pose"):
            preset_text += f"- **Pose**: {presets['pose']}\n"

        system_prompt = """You are an expert prompt engineer for AI image generation (Stable Diffusion, Flux, Z-Image).

Your task is to expand a basic concept into a detailed image generation prompt.

CAMERA THINKING - Write like setting camera parameters, not prose:
- Technical terms: "f/1.8, shallow DOF, subject sharp, background blur"
- Lighting specs: "golden hour, warm 3200K, low angle sun, soft shadows"
- NOT poetic: Avoid "dreamy ethereal", "bathed in warmth", "whispers of light"

6-PART STRUCTURE (3-5 concepts per section max):

1. **Subject**: age, build, ethnicity, facial features, hair, skin tone
2. **Clothing**: garments, fabrics, colors, fit, accessories
3. **Pose & Expression**: body position, gesture, facial expression, gaze
4. **Scene**: setting, background, props, atmosphere
5. **Lighting**: quality, direction, color temp, shadows
6. **Camera**: shot type, angle, framing, depth of field, lens

RULES:
1. Be specific and visual - describe what can be SEEN
2. Use concrete descriptors: "high cheekbones, full lips" NOT "beautiful"
3. Include colors, textures, materials
4. 3-5 key concepts per section - avoid overloading
5. NO contradictory terms (soft focus + sharp, warm + cool)
6. Write as connected prose (not bullet points)
7. Output ONLY the prompt - no headers, no explanations
8. Aim for 150-300 words of focused, technical description"""

        user_prompt = f"""BASIC CONCEPT:
{basic_text}
"""
        if preset_text:
            user_prompt += f"""
STYLE REQUIREMENTS TO INCORPORATE:
{preset_text}
"""
        user_prompt += """
Expand this concept into a detailed, comprehensive image generation prompt. Include all visual details for subject, clothing, pose, environment, lighting, and camera. Output only the expanded prompt."""

        try:
            client = self._get_client(llm_model)
            response = self._call_llm(client, llm_model, system_prompt, user_prompt, temperature, basic_text)
            expanded = self._clean_response(response)
            word_count = len(expanded.split())
            print(f"[SID_PromptModifier] Expanded to {word_count} words")
            return expanded
        except Exception as e:
            print(f"[SID_PromptModifier] Error expanding prompt: {e}")
            return basic_text

    def _generate_caption(self, llm_model: Any, prompt: str, temperature: float) -> str:
        """Generate Instagram caption from prompt."""

        system_prompt = """You are a social media expert creating Instagram captions for photographers.

Generate 3 different caption styles. Each style should include:
- A caption (1-2 engaging sentences)
- A description (2-3 sentences)
- 10-15 relevant hashtags

FORMAT YOUR RESPONSE EXACTLY LIKE THIS:

## Poetic
**Caption:** [Evocative, emotional, artistic caption]
**Description:** [Lyrical description with metaphors and feeling]
**Hashtags:** #hashtag1 #hashtag2 #hashtag3...

## Technical
**Caption:** [Professional, precise, photography-focused caption]
**Description:** [Details about technique, composition, style]
**Hashtags:** #hashtag1 #hashtag2 #hashtag3...

## Personal
**Caption:** [Casual, relatable, storytelling caption]
**Description:** [Behind-the-scenes, personal connection, authentic voice]
**Hashtags:** #hashtag1 #hashtag2 #hashtag3...

Keep each section concise."""

        user_prompt = f"""Create 3 Instagram caption styles based on this image description:

{prompt}

Generate Poetic, Technical, and Personal caption styles following the exact format specified."""

        try:
            client = self._get_client(llm_model)
            response = self._call_llm(client, llm_model, system_prompt, user_prompt, temperature, prompt)
            return response
        except Exception as e:
            print(f"[SID_PromptModifier] Caption generation error: {e}")
            return f"[Caption generation error: {str(e)}]"

    def _get_client(self, llm_model: Any) -> Any:
        """Get LLM client based on provider."""
        provider = llm_model.provider.lower()

        if provider == "anthropic":
            import anthropic
            import httpx
            timeout = httpx.Timeout(timeout=120.0, connect=30.0)
            return anthropic.Anthropic(api_key=llm_model.api_key, timeout=timeout)

        elif provider == "local_text":
            # Local text-only models (Qwen text)
            from ....llm_providers.sid_llm_local_text import LocalTextModelClient
            extra = llm_model.extra_params or {}
            return LocalTextModelClient(
                model_name=llm_model.model,
                quantization=extra.get("quantization", "4-bit"),
                device=extra.get("device", "auto"),
                keep_model_loaded=extra.get("keep_model_loaded", True),
                repo_id=extra.get("repo_id", ""),
                hf_token=extra.get("hf_token"),
                repetition_penalty=extra.get("repetition_penalty", 1.3),
                top_p=extra.get("top_p", 0.9),
            )

        elif provider == "local":
            # Local vision models - fallback to OpenAI-compatible
            from openai import OpenAI
            return OpenAI(
                api_key="not-needed",
                base_url=llm_model.api_url if llm_model.api_url else "http://localhost:11434/v1",
            )

        else:
            # OpenAI-compatible API
            from openai import OpenAI
            return OpenAI(
                api_key=llm_model.api_key or "not-needed",
                base_url=llm_model.api_url if llm_model.api_url else None,
            )

    def _calculate_max_tokens(self, input_prompt: str, llm_model: Any) -> int:
        """Calculate max tokens based on input size + buffer for modifications."""
        # Estimate tokens from input (roughly 1 token per 4 characters)
        input_chars = len(input_prompt)
        estimated_input_tokens = input_chars // 4

        # Output should be at least as long as input, plus 100% buffer for modifications
        # This ensures we don't truncate prompts during modification (presets add significant content)
        buffer_multiplier = 2.0
        calculated_tokens = int(estimated_input_tokens * buffer_multiplier)

        # Set minimum (for short prompts) and maximum bounds
        min_tokens = 2000  # Generous minimum for detailed prompt modification
        max_tokens = getattr(llm_model, 'max_tokens', 8000)

        # Clamp to bounds
        result = max(min_tokens, min(calculated_tokens, max_tokens))
        print(f"[SID_PromptModifier] Token calculation: input ~{estimated_input_tokens} tokens, allocated {result} max tokens")
        return result

    def _call_llm(
        self,
        client: Any,
        llm_model: Any,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        input_prompt: str = "",
    ) -> str:
        """Make LLM call (text-only, no image)."""
        # Calculate max tokens based on input size + 30%
        max_tokens = self._calculate_max_tokens(input_prompt or user_prompt, llm_model)

        if hasattr(client, 'messages'):
            # Anthropic API
            response = client.messages.create(
                model=llm_model.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=[{
                    "role": "user",
                    "content": user_prompt
                }]
            )
            return response.content[0].text
        else:
            # OpenAI-compatible API
            response = client.chat.completions.create(
                model=llm_model.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )

            if response and response.choices and response.choices[0].message:
                return response.choices[0].message.content
            return ""

    def _clean_response(self, text: str) -> str:
        """Clean LLM response text."""
        import re

        if not text:
            return ""

        # Remove thinking blocks (<think>...</think>) - Qwen3 internal reasoning
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        if '<think>' in text:
            think_pos = text.find('<think>')
            if think_pos > 0:
                text = text[:think_pos].rstrip()
                print(f"[SID_PromptModifier] Removed incomplete <think> block")
            else:
                text = ""
                print(f"[SID_PromptModifier] Removed <think> block at start")

        if not text:
            return ""

        # Remove common preambles
        preambles = [
            r'^(?:Here is|Here\'s|This is|The modified prompt|Modified prompt)[:\s]*',
            r'^(?:Sure|Okay|Of course)[,.\s]*(?:here is|here\'s)?[:\s]*',
        ]
        for pattern in preambles:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)

        # Remove markdown code blocks if present
        text = re.sub(r'^```[\w]*\n?', '', text)
        text = re.sub(r'\n?```$', '', text)

        # Remove quotes if the entire response is quoted
        text = text.strip()
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]

        # Clean whitespace
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    def _print_analysis_report(
        self,
        original_prompt: str,
        modified_prompt: str,
        instructions: Dict[str, str],
        presets: Dict[str, str],
    ) -> None:
        """Print a console report analyzing the prompt modification."""
        print("\n" + "=" * 70)
        print("PROMPT MODIFICATION ANALYSIS REPORT")
        print("=" * 70)

        # Word count comparison
        orig_words = len(original_prompt.split())
        mod_words = len(modified_prompt.split())
        word_diff = mod_words - orig_words
        word_pct = ((mod_words / orig_words) - 1) * 100 if orig_words > 0 else 0

        print(f"\n📊 LENGTH ANALYSIS:")
        print(f"   Original: {orig_words} words")
        print(f"   Modified: {mod_words} words")
        print(f"   Change:   {word_diff:+d} words ({word_pct:+.1f}%)")

        # Find added and removed words
        orig_word_set = set(original_prompt.lower().split())
        mod_word_set = set(modified_prompt.lower().split())

        added_words = mod_word_set - orig_word_set
        removed_words = orig_word_set - mod_word_set

        print(f"\n📝 SEMANTIC CHANGES:")
        if added_words:
            # Show up to 15 most significant added words (filter common words)
            common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'her', 'his', 'its', 'their', 'this', 'that', 'these', 'those', 'as', 'if', 'then', 'so', 'than', 'too', 'very', 'just', 'also', 'now', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'not', 'only', 'own', 'same', 'she', 'he', 'it', 'we', 'they', 'you', 'who', 'which', 'what'}
            significant_added = [w for w in added_words if w not in common_words and len(w) > 2][:15]
            if significant_added:
                print(f"   ✅ Added:   {', '.join(sorted(significant_added))}")
        if removed_words:
            common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'her', 'his', 'its', 'their', 'this', 'that', 'these', 'those', 'as', 'if', 'then', 'so', 'than', 'too', 'very', 'just', 'also', 'now', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'not', 'only', 'own', 'same', 'she', 'he', 'it', 'we', 'they', 'you', 'who', 'which', 'what'}
            significant_removed = [w for w in removed_words if w not in common_words and len(w) > 2][:15]
            if significant_removed:
                print(f"   ❌ Removed: {', '.join(sorted(significant_removed))}")

        # Check if requested changes were applied
        print(f"\n🎯 MODIFICATION VERIFICATION:")

        # Check presets
        if presets:
            for preset_type, preset_value in presets.items():
                # Extract key terms from preset
                preset_keywords = [w.lower() for w in preset_value.split() if len(w) > 4][:5]
                found = sum(1 for kw in preset_keywords if kw in modified_prompt.lower())
                match_pct = (found / len(preset_keywords)) * 100 if preset_keywords else 0
                status = "✅" if match_pct >= 40 else "⚠️" if match_pct >= 20 else "❌"
                print(f"   {status} {preset_type.title()}: {match_pct:.0f}% keywords detected")

        # Check section instructions
        if instructions:
            for section, instruction in instructions.items():
                # Extract key terms from instruction
                instruction_keywords = [w.lower() for w in instruction.split() if len(w) > 3][:5]
                found = sum(1 for kw in instruction_keywords if kw in modified_prompt.lower())
                match_pct = (found / len(instruction_keywords)) * 100 if instruction_keywords else 0
                status = "✅" if match_pct >= 40 else "⚠️" if match_pct >= 20 else "❌"
                section_name = self.SECTIONS[section]["name"]
                print(f"   {status} {section_name}: {match_pct:.0f}% instruction keywords found")

        print("=" * 70 + "\n")

    def _release_vram(self):
        """Release VRAM by clearing GPU memory and running garbage collection."""
        import gc

        gc.collect()

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                print("[SID_PromptModifier] VRAM released")
        except ImportError:
            pass


# Node registration
NODE_CLASS_MAPPINGS = {
    "SID_PromptModifier": SID_PromptModifier,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SID_PromptModifier": "SID Prompt Modifier",
}
