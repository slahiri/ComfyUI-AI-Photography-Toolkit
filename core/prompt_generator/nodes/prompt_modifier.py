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

    # Creative Effects - Motion, Blur, Color, Lighting, Texture, Digital, Artistic, Surreal
    CREATIVE_EFFECTS = {
        "None": "",
        # Motion & Blur
        "Ghost Motion Blur": "ghost motion blur, long exposure portrait, motion trail effect, ethereal double, teal and orange color grading, translucent echo, cinematic movement",
        "Dreamy Soft Focus": "dreamy soft focus, gaussian blur, romantic haze, light bloom, soft diffusion, pastel tones, vaseline lens effect, ethereal portrait",
        "Tilt Shift Miniature": "tilt shift photography, miniature effect, selective focus, shallow depth of field, diorama style, toy-like, Lensbaby effect",
        "Bokeh Lights": "bokeh lights, circular blur, out of focus highlights, fairy lights background, dreamy bokeh, light orbs, f/1.4 aperture",
        "Light Trail Long Exposure": "long exposure light trails, light painting, streaking lights, night photography, motion blur lights, car light trails",
        "Radial Zoom Blur": "radial zoom blur, tunnel vision effect, zoom burst, speed blur, vortex motion, dynamic energy",
        "Freeze Frame Burst": "freeze frame burst, stroboscopic effect, motion sequence, multiple exposure movement, action freeze",
        # Color & Tone
        "Chromatic Aberration": "chromatic aberration, RGB split, color fringing, lens distortion, color channel separation, prismatic edge effect, glitch photography",
        "Duotone": "duotone effect, two-color palette, gradient map, Spotify aesthetic, bold color contrast, graphic poster style",
        "Prism Rainbow Split": "prism effect, rainbow light dispersion, spectral split, Pink Floyd prism, light refraction, ROYGBIV spectrum",
        "Infrared Photography": "infrared photography, false color IR, pink foliage, surreal landscape, 720nm filter, Aerochrome film, magenta trees",
        "Thermal Vision": "thermal imaging, heat map vision, infrared thermal, FLIR camera, temperature gradient, heat signature, predator vision",
        "Cross Process": "cross processed film, E6 in C41, color shift, high contrast, saturated colors, green shadows, magenta highlights",
        "Solarization": "solarization effect, Sabattier effect, tone reversal, Man Ray style, partially inverted, psychedelic tones",
        "Negative Inversion": "negative film effect, inverted colors, color negative, complementary inversion, reversed tones",
        "Posterize": "posterized effect, limited color palette, color banding, pop art poster, flat color zones, Andy Warhol style",
        "Split Complementary Light": "split complementary lighting, orange teal, magenta green, complementary gel lighting, color contrast",
        # Lighting
        "Neon Glow": "neon glow, fluorescent lighting, electric colors, light bloom, cyberpunk neon, glowing edges, synthwave glow",
        "Silhouette Rim Light": "silhouette, rim lighting, backlit portrait, golden hour backlight, edge glow, contre-jour, halo effect",
        "Split Lighting": "split lighting, half-face shadow, chiaroscuro, dramatic side light, Rembrandt lighting, film noir lighting",
        "Backlit Haze": "backlit haze, atmospheric fog, lens flare, golden haze, dusty atmosphere, god rays, volumetric light",
        "Anamorphic Lens Flare": "anamorphic lens flare, horizontal streak, JJ Abrams flare, cinema lens, blue streak flare, oval bokeh",
        "Ethereal Glow": "ethereal glow, angelic radiance, supernatural light, divine illumination, heavenly aura, soft luminescence",
        "UV Blacklight": "UV blacklight, fluorescent glow, ultraviolet lighting, rave aesthetic, neon paint glow, phosphorescent",
        "Bioluminescent": "bioluminescent glow, organic inner light, glowing organism, avatar style, living light, natural fluorescence",
        # Texture & Film
        "Heavy Film Grain": "heavy film grain, analog texture, ISO 3200, grainy photograph, 35mm film grain, noisy film aesthetic",
        "Halftone Dots": "halftone dots, Ben-Day dots, comic book print, newspaper print effect, CMYK halftone, Lichtenstein style",
        "VHS Retro": "VHS effect, scan lines, tape distortion, retro video, color bleeding, tracking errors, 80s camcorder",
        "Scratched Film": "scratched film, dust particles, aged film stock, vintage scratches, film damage, old movie reel",
        "Static Noise": "TV static noise, signal interference, white noise texture, analog static, dead channel, broadcast noise",
        # Digital & Glitch
        "Data Glitch": "glitch art, data corruption, pixel sorting, digital artifacts, databending, corrupted image, broken data",
        "Pixel Sort": "pixel sorting, algorithmic art, data manipulation, sorted pixels, Kim Asendorf style, digital distortion",
        "Holographic": "holographic effect, iridescent, rainbow shimmer, spectrum shift, pearlescent, oil slick colors, CD reflection",
        "Wireframe Mesh": "wireframe mesh, 3D polygons, digital blueprint, CAD rendering, Tron aesthetic, grid overlay, vector lines",
        "Anaglyph 3D": "anaglyph 3D, red-cyan stereoscopic, retro 3D glasses, depth offset, chromatic 3D, stereoscopy",
        "CRT Scanlines": "CRT scanlines, retro monitor, cathode ray tube, arcade screen, old TV effect, phosphor glow",
        # Artistic & Painterly
        "Oil Painting": "oil painting effect, impasto texture, visible brushstrokes, painterly style, canvas texture, classical painting",
        "Watercolor": "watercolor effect, soft bleeding edges, color wash, wet-on-wet technique, translucent layers, aquarelle style",
        "Charcoal Sketch": "charcoal sketch, rough strokes, smudged shadows, graphite drawing, pencil texture, monochrome sketch",
        "Stained Glass": "stained glass effect, geometric segments, lead lines, colored glass, cathedral window, Tiffany style",
        "Kaleidoscope": "kaleidoscope effect, radial symmetry, mirrored pattern, mandala style, rotational symmetry, prismatic reflection",
        "Pop Art": "pop art style, bold flat colors, high contrast, Andy Warhol style, screen print, graphic art, comic colors",
        # Surreal & Conceptual
        "Double Exposure": "double exposure, multiple exposure, blended images, portrait nature blend, silhouette fill, composite photography",
        "Disintegration": "disintegration effect, particle scatter, dissolving, dust particles, Thanos snap, breaking apart",
        "Levitation": "levitation photography, floating, suspended in air, gravity defying, weightless, magical floating",
        "Mirror Reflection": "mirror reflection, water reflection, perfect symmetry, reflected image, still water mirror",
        "Portal Dimensional": "portal effect, dimensional gateway, reality warp, interdimensional, wormhole, cosmic doorway",
        "Liquid Melt": "melting effect, liquid drip, wax melt, surreal melting, Dali style, flowing forms",
        # Environmental
        "Underwater Caustics": "underwater photography, submerged, caustic light patterns, aquatic atmosphere, deep sea blue",
        "Fog Diffusion": "fog diffusion, heavy mist, atmospheric haze, foggy isolation, mysterious fog, dense atmosphere",
        "Dust Motes": "dust motes, floating particles, light beam dust, atmospheric particles, magical dust, sunbeam particles",
        "Rain Atmosphere": "rain photography, falling rain, wet reflections, rainy atmosphere, water droplets, storm mood",
        "Frozen Ice": "frozen effect, ice crystals, frost covered, glacial blue, crystalline texture, cryogenic, winter frozen",
        "Fire Ember": "fire effect, burning embers, flame glow, heat distortion, molten, volcanic glow, inferno light",
    }

    # Cinematic Styles - Movie and Director aesthetics
    CINEMATIC_STYLES = {
        "None": "",
        # Tarantino
        "Kill Bill Yellow": "Kill Bill style, high saturation yellow, black stripe, revenge aesthetic, Tarantino visual style, bold primary colors",
        "Kill Bill Anime": "Kill Bill anime sequence, stylized blood splatter, Japanese animation style, manga action, O-Ren Ishii style",
        "Pulp Fiction Glow": "Pulp Fiction style, mysterious golden glow, divine briefcase light, warm heavenly illumination, Tarantino aesthetic",
        "Grindhouse Damaged": "grindhouse film damage, exploitation cinema, scratched print, missing frames, 70s sleaze aesthetic, drive-in movie",
        # Sci-Fi
        "Tron Grid": "Tron Legacy style, glowing neon circuit lines, digital grid, neon blue orange, light cycle trails, digital frontier",
        "Tron Derezz": "Tron derezzing effect, digital disintegration, pixel scatter, fragmenting cubes, digital death, data dissolution",
        "Matrix Green Code": "Matrix code rain, green digital cascade, falling characters, simulation code, digital rain, green tint",
        "Matrix Bullet Time": "bullet time effect, frozen action, time slice, rotating freeze frame, Neo dodge, 360 degree freeze",
        "Star Wars Hologram": "Star Wars hologram, blue holographic projection, R2D2 message, flickering transmission, sci-fi hologram",
        "Hyperspace Jump": "hyperspace tunnel, star streak, light speed jump, warp drive effect, stretched starfield, FTL travel",
        "Lightsaber Glow": "lightsaber glow, plasma blade light, colored energy sword, ambient blade glow, Jedi weapon, Force energy",
        "Blade Runner Rain": "Blade Runner style, neon rain reflections, dystopian downpour, replicant noir, cyberpunk rain, Ridley Scott",
        "Blade Runner Orange": "Blade Runner 2049, apocalyptic orange sky, nuclear sunset, wasteland haze, Las Vegas dust, Villeneuve style",
        # Zack Snyder
        "300 Bronze": "300 movie style, desaturated bronze, Spartan aesthetic, Frank Miller, crushed blacks, sepia combat, epic battle",
        "300 Speed Ramp": "300 speed ramping, variable speed combat, slow motion action, Snyder speed ramp, dramatic slow-mo",
        "Sin City Noir": "Sin City style, high contrast black and white, selective color, Frank Miller noir, neo-noir comic, graphic novel",
        "Snyder Desaturated": "Zack Snyder style, desaturated dramatic, dark superhero aesthetic, DCEU look, grim tones, epic scale",
        # Wes Anderson
        "Anderson Symmetry": "Wes Anderson symmetry, centered composition, pastel colors, symmetrical framing, quirky aesthetic, storybook",
        "Grand Budapest Pink": "Grand Budapest Hotel style, signature pink, whimsical miniature, Mendl's aesthetic, Anderson pastel",
        "Moonrise Kingdom": "Moonrise Kingdom style, yellow khaki tones, 60s nostalgia, scouting aesthetic, vintage Americana",
        "Anderson Diorama": "Wes Anderson diorama, cross-section view, dollhouse staging, theatrical set design, miniature world",
        # Christopher Nolan
        "Inception Paradox": "Inception style, impossible architecture, folding reality, Penrose stairs, dream architecture, gravity paradox",
        "Interstellar Cosmic": "Interstellar style, wormhole visualization, tesseract dimensions, Gargantua black hole, cosmic scale",
        "Oppenheimer Stark": "Oppenheimer style, IMAX black and white, stark contrast, dramatic monochrome, Trinity test, historical epic",
        # Denis Villeneuve
        "Dune Desert Scale": "Dune Arrakis style, vast orange desert, massive scale, sandworm territory, epic desert, Villeneuve scale",
        "Dune Spice Vision": "Dune spice eyes, blue within blue, prescient vision, Kwisatz Haderach, spice trance, prophetic",
        "Arrival Atmospheric": "Arrival style, grey fog atmosphere, mysterious alien contact, heptapod aesthetic, linguistic sci-fi",
        # Other Directors
        "Kubrick Symmetry": "Stanley Kubrick symmetry, one-point perspective, centered framing, unsettling symmetry, precise composition",
        "Kubrick Stare": "Kubrick stare, tilted head eyes up, menacing gaze, psychotic look, A Clockwork Orange, intense stare",
        "Spielberg Wonder": "Spielberg wonder, awe-struck face, light from above, amazement expression, E.T. lighting, magical wonder",
        "Fincher Dark": "David Fincher style, sickly green-yellow tint, oppressive shadow, Se7en aesthetic, Fight Club darkness",
        "Ari Aster Daylight": "Midsommar style, bright daylight horror, floral nightmare, Ari Aster, Swedish horror, sunny dread",
        "Jordan Peele Duality": "Get Out style, Jordan Peele aesthetic, sunken place, social horror, duality theme, psychological horror",
        "Michael Bay Explosion": "Michael Bay style, orange explosions, dramatic low angle, action explosion, blockbuster pyrotechnics",
        "Edgar Wright Kinetic": "Edgar Wright style, quick cuts, whip pan transitions, comedic timing, Scott Pilgrim energy, visual comedy",
    }

    # Anime Styles - Japanese animation aesthetics
    ANIME_STYLES = {
        "None": "",
        # Dragon Ball
        "Super Saiyan Aura": "Super Saiyan aura, golden ki flame, power up glow, DBZ energy, spiked golden hair, blazing aura",
        "Kamehameha Beam": "Kamehameha wave, blue ki blast, energy beam attack, charging power, DBZ blast, energy wave",
        "Ultra Instinct": "Ultra Instinct form, silver white aura, divine ki, godly power, autonomous movement, silver hair glow",
        "DBZ Power Up": "DBZ power up, ground cracking, floating debris, energy explosion, power level rising, screaming power",
        "DBZ Speed Lines": "anime speed lines, DBZ action lines, impact frame, motion blur lines, manga action, dynamic movement",
        # Naruto
        "Chakra Aura": "chakra aura, blue spiritual energy, ninja power, Naruto chakra, energy emanation, shinobi power",
        "Sharingan Eye": "Sharingan eye, red iris tomoe pattern, Uchiha dojutsu, spinning eye pattern, kekkei genkai",
        "Rasengan": "Rasengan sphere, blue spinning energy ball, palm technique, wind chakra, spiral sphere, jutsu effect",
        "Nine Tails Cloak": "Nine-Tails chakra cloak, orange demon energy, Kurama mode, jinchuriki power, fox aura, bijuu chakra",
        "Susanoo": "Susanoo projection, ethereal giant warrior, chakra avatar, spectral armor, purple energy giant, Uchiha ultimate",
        # Demon Slayer
        "Water Breathing": "Water Breathing technique, Tanjiro style, flowing water slash, blue water effects, wave pattern sword",
        "Flame Breathing": "Flame Breathing technique, Rengoku fire, ember trails, blazing sword, fire kata, flame effects",
        "Thunder Breathing": "Thunder Breathing, Zenitsu lightning, yellow electric, thunderclap flash, lightning speed, electric aura",
        "Demon Slayer Sky": "Demon Slayer sky, Ufotable quality, gorgeous gradient sunset, anime background, beautiful clouds",
        # Attack on Titan
        "Titan Steam": "Titan transformation steam, steaming giant, AOT steam effect, regenerating titan, hot mist, colossal emergence",
        "ODM Gear Trails": "ODM gear wire trails, vertical maneuvering, Scout Regiment, acrobatic movement, grappling lines",
        "Founding Titan Paths": "Paths dimension, coordinate realm, sand world, Founding Titan realm, memory dimension, Ymir paths",
        # Studio Ghibli
        "Ghibli Clouds": "Studio Ghibli clouds, fluffy cumulus, Miyazaki sky, wonder atmosphere, beautiful cloud formations",
        "Spirited Away Magic": "Spirited Away style, spirit world, bathhouse magic, Ghibli supernatural, Japanese spirits, No-Face",
        "Totoro Forest": "Totoro forest style, magical woodland, Ghibli nature, forest spirits, enchanted woods, gentle magic",
        "Howl's Castle": "Howl's Moving Castle style, Ghibli steampunk, magical machinery, fantasy contraption, whimsical mechanical",
        "Mononoke Forest": "Princess Mononoke style, ancient forest spirits, nature god, kodama spirits, Ghibli environmental",
        # Cyberpunk Anime
        "Akira Red": "Akira style, iconic red, Neo-Tokyo, Kaneda bike, red cape, cyberpunk motorcycle, Otomo aesthetic",
        "Akira Psychic": "Akira psychic power, Tetsuo awakening, telekinetic destruction, psychic energy, mental explosion",
        "Ghost in Shell": "Ghost in the Shell style, cybernetic body, digital consciousness, cyborg aesthetic, Kusanagi style",
        "Cowboy Bebop": "Cowboy Bebop style, space jazz noir, bounty hunter aesthetic, Spike Spiegel, retro future, jazz western",
        "Evangelion": "Neon Genesis Evangelion style, mecha psychological, Eva unit, angel battle, existential anime, NERV",
        # Modern Anime
        "Jujutsu Kaisen Curse": "Jujutsu Kaisen style, cursed energy, domain expansion, dark sorcery, Sukuna, curse technique",
        "My Hero Academia": "My Hero Academia style, quirk activation, hero costume, Plus Ultra, superpower manifestation",
        "One Punch Impact": "One Punch Man style, serious punch impact, devastating blow, impact frame, Saitama power",
        "Mob Psycho Aura": "Mob Psycho 100 style, psychic percentage, esper aura, emotional explosion, ???% power",
        "Chainsaw Devil": "Chainsaw Man style, devil power, brutal action, Fujimoto aesthetic, dark energy, hybrid form",
        "Makoto Shinkai": "Makoto Shinkai style, Your Name aesthetic, beautiful light rays, detailed clouds, romantic atmosphere",
    }

    # Gaming & TV Styles
    GAMING_STYLES = {
        "None": "",
        # Retro Gaming
        "8-Bit Pixel": "8-bit pixel art, NES style graphics, retro pixels, limited color palette, chiptune era, classic gaming",
        "16-Bit Sprite": "16-bit sprite art, SNES Genesis style, detailed pixels, retro gaming graphics, classic console",
        "Arcade CRT": "arcade CRT monitor, scanlines, curved screen glow, retro arcade, phosphor display, cabinet screen",
        # Modern Gaming
        "GTA Loading Screen": "GTA art style, loading screen portrait, Rockstar aesthetic, crime game art, stylized character",
        "Cyberpunk 2077": "Cyberpunk 2077 style, Night City neon, chrome implants, V aesthetic, corpo punk, futuristic dystopia",
        "Dark Souls Bonfire": "Dark Souls style, bonfire rest, dark fantasy, Souls-like atmosphere, ember glow, gothic dark",
        "Elden Ring Golden": "Elden Ring style, golden grace, Erdtree light, FromSoftware aesthetic, dark fantasy gold",
        "Borderlands Cell": "Borderlands style, heavy cell shading, thick outlines, comic book game, stylized shooter",
        "Bioshock Art Deco": "Bioshock Rapture style, art deco dystopia, underwater city, 1920s futurism, retro-futuristic",
        # Nintendo
        "Zelda Fairy": "Zelda fairy glow, Navi light, magical companion, Hyrule magic, fairy trail, Nintendo fantasy",
        "Pokemon Evolution": "Pokemon evolution glow, transformation light, type energy, leveling up effect, creature evolution",
        # TV Series
        "Stranger Things 80s": "Stranger Things style, Christmas lights, Upside Down, 80s horror nostalgia, Hawkins aesthetic",
        "Squid Game": "Squid Game style, pink guard green player, geometric shapes, deadly game, Korean thriller, contrast",
        "Euphoria Glitter": "Euphoria style, glitter makeup, neon party, Gen Z aesthetic, dramatic teen, sparkle tears",
        "Wednesday Gothic": "Wednesday Addams style, gothic desaturated, Nevermore aesthetic, Tim Burton gothic, dark academia",
        "Last of Us Overgrown": "Last of Us style, post-apocalyptic overgrowth, cordyceps aesthetic, nature reclaiming, fungal horror",
        "Breaking Bad Yellow": "Breaking Bad style, New Mexico yellow tint, desert aesthetic, Albuquerque, meth blue accent",
        "True Detective Decay": "True Detective style, southern gothic, Louisiana decay, rust aesthetic, Carcosa, yellow king",
        "Twin Peaks Surreal": "Twin Peaks style, red room aesthetic, Black Lodge, David Lynch surreal, zigzag floor, surrealist TV",
        # Comic & Superhero
        "Spider-Verse Halftone": "Spider-Verse style, halftone dots, comic print texture, frame stutter, Miles Morales, animated comic",
        "Spider-Verse Glitch": "Spider-Verse glitch, dimension break, multiverse tear, chromatic comic, reality fracture",
        "Marvel Kirby Cosmic": "Jack Kirby cosmic style, Kirby crackle, cosmic energy dots, Marvel cosmic, divine power",
        "Batman Noir": "Batman noir style, Gotham shadows, dark knight aesthetic, gothic superhero, DC noir, cape shadow",
        "Watchmen Gritty": "Watchmen style, gritty 80s superhero, Dr Manhattan blue glow, deconstructed hero, Alan Moore",
    }

    # Poses - Model, Studio, Fashion, Yoga, Influencer poses
    POSES = {
        "None": "",
        # Standing Poses - Basic
        "Standing Straight": "standing straight, upright posture, feet together, arms at sides, formal stance",
        "Standing Relaxed": "standing relaxed, casual stance, weight on one leg, natural posture",
        "Standing Confident": "standing confident, shoulders back, chin up, powerful stance, assertive posture",
        "Standing Elegant": "standing elegant, graceful posture, elongated neck, refined stance",
        "Standing Hip Pop": "standing with hip popped to side, weight shifted, casual fashion pose",
        "Standing Crossed Legs": "standing with legs crossed at ankle, elegant stance, fashion pose",
        "Standing Wide Stance": "standing with wide stance, feet apart, powerful grounded pose",
        "Standing Contrapposto": "contrapposto stance, classical pose, weight on one leg, twisted torso",
        # Standing Poses - Arms
        "Hands on Hips": "hands on hips, confident pose, elbows out, assertive stance",
        "Arms Crossed": "arms crossed over chest, confident, thoughtful pose",
        "Hand on Hip": "one hand on hip, other arm relaxed, casual confident pose",
        "Arms Behind Back": "arms behind back, hands clasped, formal professional pose",
        "Arms Raised": "arms raised above head, celebratory pose, joyful expression",
        "One Arm Up": "one arm raised, touching hair or head, casual elegant pose",
        "Arms Outstretched": "arms outstretched to sides, open welcoming pose, freedom",
        "Hand on Face": "hand gently touching face, thoughtful contemplative pose",
        "Hand Under Chin": "hand under chin, thinking pose, intellectual expression",
        "Hands in Pockets": "hands in pockets, casual relaxed stance, effortless cool",
        "Hand on Chest": "hand placed on chest, sincere emotional pose",
        "Hands Clasped Front": "hands clasped in front, modest professional pose",
        # Walking & Movement
        "Walking Towards Camera": "walking towards camera, mid-stride, confident approach",
        "Walking Away": "walking away from camera, looking back over shoulder",
        "Walking Side Profile": "walking in profile view, elegant stride, fashion walk",
        "Runway Walk": "runway model walk, elongated stride, fierce expression, catwalk pose",
        "Casual Stroll": "casual strolling pose, relaxed walking, natural movement",
        "Power Walk": "power walking pose, determined stride, businesslike",
        "Dancing Movement": "mid-dance movement, dynamic pose, flowing motion",
        "Twirling": "twirling pose, dress or hair in motion, spinning elegantly",
        "Jumping": "jumping pose, feet off ground, joyful leap, dynamic",
        "Mid-Step": "caught mid-step, one foot forward, transitional pose",
        # Sitting Poses
        "Sitting Straight": "sitting straight, good posture, formal seated pose",
        "Sitting Relaxed": "sitting relaxed, casual posture, comfortable pose",
        "Sitting Cross-Legged": "sitting cross-legged, meditation style, floor seated",
        "Sitting Legs Crossed": "sitting with legs crossed, elegant seated pose",
        "Sitting Sideways": "sitting sideways on chair, profile seated pose",
        "Sitting on Edge": "sitting on edge of seat, leaning forward, engaged pose",
        "Sitting Reclined": "sitting reclined, leaning back, relaxed lounging",
        "Sitting on Floor": "sitting on floor, casual grounded pose",
        "Sitting Knees Up": "sitting with knees pulled up, casual intimate pose",
        "Sitting Legs Extended": "sitting with legs extended forward, relaxed pose",
        "Perched Sitting": "perched on edge, one leg dangling, casual seated",
        # Leaning Poses
        "Leaning on Wall": "leaning against wall, casual cool pose, relaxed",
        "Leaning Forward": "leaning forward, engaged interested pose",
        "Leaning Back": "leaning back, relaxed confident pose",
        "Leaning on Elbow": "leaning on elbow, thoughtful casual pose",
        "Leaning Sideways": "leaning to one side, casual asymmetric pose",
        "Leaning on Object": "leaning on furniture or prop, supported pose",
        # Lying Poses
        "Lying on Back": "lying on back, supine pose, looking up",
        "Lying on Side": "lying on side, propped on elbow, relaxed pose",
        "Lying on Stomach": "lying on stomach, prone pose, chin on hands",
        "Lying Curled": "lying in curled fetal position, intimate vulnerable pose",
        "Lying Stretched": "lying fully stretched out, relaxed extended pose",
        # Fashion & Editorial Poses
        "High Fashion Stance": "high fashion editorial stance, dramatic angles, editorial pose",
        "Editorial Lean": "editorial leaning pose, angular body, fashion photography",
        "Fashion Forward": "fashion forward pose, one leg in front, dynamic stance",
        "Model S-Curve": "S-curve pose, curved body line, feminine silhouette",
        "Editorial Crouch": "crouching editorial pose, low angle, fashion forward",
        "Fashion Kneel": "kneeling fashion pose, elegant floor pose",
        "Editorial Twist": "twisted torso editorial pose, dynamic angles",
        "High Fashion Recline": "high fashion reclining pose, dramatic lounging",
        "Vogue Pose": "classic Vogue magazine pose, elegant dramatic",
        "Fashion Jump": "fashion jumping pose, dynamic mid-air, editorial",
        # Portrait Poses
        "Classic Portrait": "classic portrait pose, three-quarter turn, traditional",
        "Headshot Pose": "professional headshot pose, shoulders angled, face to camera",
        "Profile Portrait": "profile portrait pose, side view of face",
        "Three-Quarter View": "three-quarter view, body angled 45 degrees to camera",
        "Looking Over Shoulder": "looking over shoulder, mysterious glance back",
        "Chin Down": "chin tilted down, eyes up, intense gaze",
        "Chin Up": "chin tilted up, confident regal pose",
        "Head Tilt": "gentle head tilt, approachable friendly pose",
        "Looking Away": "looking away from camera, candid thoughtful",
        "Looking Down": "looking downward, contemplative introspective pose",
        # Yoga Poses
        "Mountain Pose": "Tadasana mountain pose, standing tall, grounded",
        "Tree Pose": "Vrikshasana tree pose, one foot on inner thigh, arms up",
        "Warrior I": "Virabhadrasana I warrior one pose, lunge with arms raised",
        "Warrior II": "Virabhadrasana II warrior two pose, arms extended sides",
        "Warrior III": "Virabhadrasana III warrior three, balancing on one leg, body horizontal",
        "Downward Dog": "Adho Mukha Svanasana downward facing dog, inverted V shape",
        "Cobra Pose": "Bhujangasana cobra pose, chest lifted, back bend",
        "Child's Pose": "Balasana child's pose, kneeling forward fold, restful",
        "Lotus Pose": "Padmasana lotus pose, seated meditation, crossed legs",
        "Dancer Pose": "Natarajasana dancer pose, standing backbend, graceful balance",
        "Crow Pose": "Bakasana crow pose, arm balance, advanced yoga",
        "Headstand": "Sirsasana headstand, inverted pose, balanced on head",
        "Shoulder Stand": "Sarvangasana shoulder stand, inverted on shoulders",
        "Bridge Pose": "Setu Bandhasana bridge pose, back arch, hips lifted",
        "Pigeon Pose": "Eka Pada Rajakapotasana pigeon pose, hip opener",
        "Seated Forward Fold": "Paschimottanasana seated forward fold, stretching",
        "Triangle Pose": "Trikonasana triangle pose, side stretch, extended",
        "Half Moon Pose": "Ardha Chandrasana half moon pose, balancing side stretch",
        "Camel Pose": "Ustrasana camel pose, kneeling backbend",
        "Bow Pose": "Dhanurasana bow pose, lying backbend, holding feet",
        # Influencer & Social Media Poses
        "Selfie Pose": "selfie pose, phone held up, casual self-portrait angle",
        "Mirror Selfie": "mirror selfie pose, phone visible, reflection shot",
        "OOTD Pose": "outfit of the day pose, full body fashion display",
        "Peace Sign": "peace sign pose, fingers in V, playful casual",
        "Blowing Kiss": "blowing kiss pose, playful flirty expression",
        "Hair Flip": "hair flip pose, tossing hair, dynamic movement",
        "Looking Up at Camera": "looking up at camera, doe eyes, flattering angle",
        "Candid Laugh": "candid laughing pose, genuine joy, unstaged feel",
        "Coffee Cup Pose": "holding coffee cup, lifestyle aesthetic, cozy",
        "Phone in Hand": "casually holding phone, modern lifestyle pose",
        "Beach Pose": "beach vacation pose, relaxed summer aesthetic",
        "Travel Pose": "travel influencer pose, scenic background, wanderlust",
        "Fitness Pose": "fitness influencer pose, athletic strong stance",
        "Food Pose": "holding or presenting food, foodie aesthetic",
        "Reading Pose": "reading book pose, intellectual cozy aesthetic",
        # Couple & Group Poses
        "Embrace": "embracing pose, arms wrapped around, intimate",
        "Hand Holding": "holding hands, connected romantic pose",
        "Back to Back": "standing back to back, connected but independent",
        "Forehead Touch": "foreheads touching, intimate romantic moment",
        "Piggyback": "piggyback pose, playful fun couple pose",
        "Dip Pose": "dance dip pose, romantic dramatic lean",
        # Action & Dynamic Poses
        "Running": "running pose, mid-stride, athletic motion",
        "Stretching": "stretching pose, athletic warm-up, flexible",
        "Reaching Up": "reaching upward, stretching towards sky",
        "Kicking": "kicking pose, leg extended, dynamic action",
        "Punching": "punching pose, fist forward, powerful action",
        "Spinning": "spinning motion pose, rotational movement blur",
    }

    # Facial Expressions
    FACIAL_EXPRESSIONS = {
        "None": "",
        # Happy & Positive
        "Genuine Smile": "genuine warm smile, Duchenne smile, eyes crinkled, authentic joy",
        "Soft Smile": "soft gentle smile, subtle upturn of lips, serene",
        "Beaming": "beaming bright smile, radiant joy, teeth showing",
        "Grinning": "wide grin, playful happy expression, ear to ear smile",
        "Smirk": "slight smirk, one corner of mouth raised, knowing expression",
        "Laughing": "laughing expression, mouth open, genuine mirth, joyful",
        "Giggling": "giggling expression, suppressed laughter, playful",
        "Chuckling": "chuckling expression, amused, warm humor",
        "Ecstatic": "ecstatic expression, overwhelming joy, elated",
        "Blissful": "blissful expression, peaceful happiness, content",
        "Delighted": "delighted expression, pleasantly surprised, happy",
        "Cheerful": "cheerful expression, bright optimistic, upbeat",
        "Playful": "playful expression, mischievous glint, fun-loving",
        "Amused": "amused expression, entertained, light humor",
        "Satisfied": "satisfied expression, content accomplishment, pleased",
        # Serious & Neutral
        "Neutral": "neutral expression, relaxed face, no particular emotion",
        "Serious": "serious expression, focused intense, no smile",
        "Stoic": "stoic expression, emotionless calm, controlled",
        "Stern": "stern expression, firm disapproving, strict",
        "Determined": "determined expression, resolute focused, strong will",
        "Focused": "focused expression, concentrated attention, engaged",
        "Contemplative": "contemplative expression, deep in thought, reflective",
        "Pensive": "pensive expression, thoughtfully sad, melancholic thinking",
        "Thoughtful": "thoughtful expression, considering, intellectual",
        "Brooding": "brooding expression, dark contemplation, moody",
        "Composed": "composed expression, calm collected, poised",
        "Dignified": "dignified expression, noble bearing, proud",
        # Confident & Powerful
        "Confident": "confident expression, self-assured, bold",
        "Fierce": "fierce expression, intense powerful, commanding",
        "Defiant": "defiant expression, rebellious challenging, bold",
        "Proud": "proud expression, accomplished dignified, head high",
        "Smoldering": "smoldering expression, intense sultry, magnetic gaze",
        "Intense": "intense expression, piercing focused, powerful",
        "Commanding": "commanding expression, authoritative, leader presence",
        "Assertive": "assertive expression, confident direct, self-assured",
        "Bold": "bold expression, daring fearless, striking",
        "Powerful": "powerful expression, strong dominant, commanding",
        # Sad & Melancholic
        "Sad": "sad expression, downturned mouth, sorrowful eyes",
        "Melancholic": "melancholic expression, wistful sadness, bittersweet",
        "Tearful": "tearful expression, eyes glistening, about to cry",
        "Crying": "crying expression, tears streaming, emotional",
        "Grieving": "grieving expression, deep sorrow, loss",
        "Disappointed": "disappointed expression, let down, deflated",
        "Heartbroken": "heartbroken expression, devastated, emotional pain",
        "Wistful": "wistful expression, longing nostalgia, yearning",
        "Forlorn": "forlorn expression, hopeless lonely, abandoned",
        "Dejected": "dejected expression, disheartened, low spirits",
        # Angry & Frustrated
        "Angry": "angry expression, furrowed brow, tight lips, rage",
        "Furious": "furious expression, intense anger, livid",
        "Annoyed": "annoyed expression, irritated, mild frustration",
        "Frustrated": "frustrated expression, exasperated, fed up",
        "Scowling": "scowling expression, deep frown, disapproval",
        "Glaring": "glaring expression, intense stare, hostile",
        "Enraged": "enraged expression, uncontrolled anger, fury",
        "Resentful": "resentful expression, bitter grudge, simmering anger",
        "Irritated": "irritated expression, bothered, impatient",
        "Displeased": "displeased expression, unhappy, dissatisfied",
        # Surprised & Shocked
        "Surprised": "surprised expression, raised eyebrows, wide eyes, open mouth",
        "Shocked": "shocked expression, stunned, jaw dropped",
        "Astonished": "astonished expression, amazed wonder, disbelief",
        "Startled": "startled expression, caught off guard, alert",
        "Amazed": "amazed expression, wonder awe, impressed",
        "Bewildered": "bewildered expression, confused surprised, lost",
        "Stunned": "stunned expression, frozen surprise, speechless",
        "Awestruck": "awestruck expression, overwhelming wonder, amazement",
        "Flabbergasted": "flabbergasted expression, completely shocked, dumbfounded",
        # Fear & Anxiety
        "Afraid": "afraid expression, fearful eyes, scared",
        "Terrified": "terrified expression, extreme fear, horror",
        "Anxious": "anxious expression, worried nervous, uneasy",
        "Nervous": "nervous expression, apprehensive, jittery",
        "Worried": "worried expression, concerned frown, troubled",
        "Fearful": "fearful expression, frightened, scared eyes",
        "Panicked": "panicked expression, terror, flight response",
        "Apprehensive": "apprehensive expression, cautious worried, uncertain",
        "Uneasy": "uneasy expression, uncomfortable, on edge",
        "Horrified": "horrified expression, extreme shock and fear, disturbed",
        # Romantic & Sensual
        "Seductive": "seductive expression, alluring sultry, bedroom eyes",
        "Flirty": "flirty expression, playful romantic, coy smile",
        "Romantic": "romantic expression, loving tender, affectionate",
        "Longing": "longing expression, yearning desire, passionate want",
        "Loving": "loving expression, warm affection, tender care",
        "Adoring": "adoring expression, deep love, devoted gaze",
        "Passionate": "passionate expression, intense desire, fervent",
        "Sultry": "sultry expression, sensual smoldering, alluring",
        "Coy": "coy expression, shy flirtatious, demure",
        "Dreamy": "dreamy expression, romantic daze, starry-eyed",
        # Confused & Uncertain
        "Confused": "confused expression, furrowed brow, questioning",
        "Puzzled": "puzzled expression, trying to understand, perplexed",
        "Skeptical": "skeptical expression, doubtful, raised eyebrow",
        "Doubtful": "doubtful expression, uncertain, questioning",
        "Perplexed": "perplexed expression, deeply confused, bewildered",
        "Questioning": "questioning expression, curious inquiry, wondering",
        "Uncertain": "uncertain expression, unsure, hesitant",
        "Suspicious": "suspicious expression, distrustful, wary",
        # Disgust & Disapproval
        "Disgusted": "disgusted expression, nose wrinkled, repulsed",
        "Disapproving": "disapproving expression, judgmental frown",
        "Contemptuous": "contemptuous expression, disdain, superior",
        "Sneering": "sneering expression, mocking disdain, cruel",
        "Revolted": "revolted expression, extreme disgust, nauseated",
        "Unimpressed": "unimpressed expression, bored disapproval, jaded",
        # Other Expressions
        "Bored": "bored expression, disinterested, unenthused",
        "Tired": "tired expression, fatigued, weary eyes",
        "Sleepy": "sleepy expression, drooping eyelids, drowsy",
        "Exhausted": "exhausted expression, deeply fatigued, drained",
        "Curious": "curious expression, interested, inquisitive",
        "Innocent": "innocent expression, pure naive, childlike",
        "Mischievous": "mischievous expression, playfully naughty, impish",
        "Shy": "shy expression, bashful, avoiding eye contact",
        "Embarrassed": "embarrassed expression, blushing, flustered",
        "Hopeful": "hopeful expression, optimistic anticipation, bright eyes",
        "Relieved": "relieved expression, tension released, grateful",
        "Grateful": "grateful expression, thankful, appreciative",
        "Peaceful": "peaceful expression, serene calm, tranquil",
        "Ethereal": "ethereal expression, otherworldly, angelic",
        "Mysterious": "mysterious expression, enigmatic, secretive",
        "Vacant": "vacant expression, empty distant, thousand-yard stare",
        "Blank": "blank expression, emotionless, void",
    }

    # Hair Styles
    HAIR_STYLES = {
        "None": "",
        # Length
        "Pixie Cut": "pixie cut, very short cropped hair, elfin style",
        "Bob Cut": "bob cut, chin-length hair, blunt ends",
        "Lob": "lob haircut, long bob, shoulder-length",
        "Medium Length": "medium length hair, past shoulders",
        "Long Hair": "long flowing hair, past shoulders, lengthy",
        "Very Long Hair": "very long hair, waist length or longer",
        "Buzz Cut": "buzz cut, very short clipper cut, military style",
        "Crew Cut": "crew cut, short sides, slightly longer top",
        "Undercut": "undercut hairstyle, shaved sides, longer top",
        "Asymmetric Cut": "asymmetric haircut, uneven lengths, edgy style",
        # Straight Styles
        "Straight Sleek": "straight sleek hair, smooth glossy, polished",
        "Straight Blunt": "straight blunt cut, sharp even ends",
        "Straight Layered": "straight layered hair, face-framing layers",
        "Straight Center Part": "straight hair with center part, symmetrical",
        "Straight Side Part": "straight hair with side part, classic style",
        "Straight Feathered": "straight feathered hair, soft wispy ends",
        # Wavy Styles
        "Beach Waves": "beach waves, tousled wavy hair, effortless texture",
        "Soft Waves": "soft waves, gentle undulating hair, romantic",
        "Hollywood Waves": "Hollywood glamour waves, vintage finger waves",
        "Loose Waves": "loose waves, relaxed wavy texture",
        "Body Wave": "body wave hair, voluminous waves, bouncy",
        "S-Waves": "S-wave hair pattern, defined wave shape",
        # Curly Styles
        "Tight Curls": "tight curls, springy coiled hair, defined curls",
        "Loose Curls": "loose curls, relaxed spiral curls, bouncy",
        "Ringlets": "ringlet curls, spiral corkscrew curls, defined",
        "Natural Curls": "natural curly hair, textured curls, authentic",
        "Spiral Curls": "spiral curls, twisted coils, defined pattern",
        "Afro": "afro hairstyle, natural voluminous curls, rounded shape",
        "Afro Puffs": "afro puffs, gathered natural hair, cute style",
        "Coils": "coiled hair texture, tight natural coils, kinky",
        "Twist Out": "twist out style, defined stretched curls",
        "Wash and Go": "wash and go curls, natural curl pattern defined",
        # Updos
        "High Bun": "high bun, hair gathered on top of head, sleek or messy",
        "Low Bun": "low bun, hair gathered at nape, elegant",
        "Messy Bun": "messy bun, casual undone updo, relaxed",
        "Sleek Bun": "sleek bun, smooth polished updo, refined",
        "Chignon": "chignon, elegant low twisted bun, formal",
        "French Twist": "French twist updo, rolled sophisticated style",
        "Top Knot": "top knot, high gathered bun, casual chic",
        "Ballerina Bun": "ballerina bun, tight smooth high bun, dancer style",
        "Gibson Tuck": "Gibson tuck, vintage rolled updo, elegant",
        "Beehive": "beehive updo, voluminous height, retro 60s style",
        # Ponytails
        "High Ponytail": "high ponytail, hair gathered high on head, youthful",
        "Low Ponytail": "low ponytail, hair gathered at nape, sleek",
        "Side Ponytail": "side ponytail, asymmetric gathered style",
        "Sleek Ponytail": "sleek ponytail, smooth polished, refined",
        "Bubble Ponytail": "bubble ponytail, segmented with elastics, fun",
        "Curly Ponytail": "curly ponytail, textured gathered curls",
        "Braided Ponytail": "braided ponytail, braid incorporated, detailed",
        "Wrapped Ponytail": "wrapped ponytail, hair wrapped around elastic, polished",
        # Braids
        "French Braid": "French braid, woven from crown, classic",
        "Dutch Braid": "Dutch braid, inverted French braid, raised pattern",
        "Fishtail Braid": "fishtail braid, intricate woven pattern",
        "Box Braids": "box braids, sectioned protective style, geometric",
        "Cornrows": "cornrows, tight braids along scalp, patterns",
        "Crown Braid": "crown braid, braided around head, halo style",
        "Side Braid": "side braid, single braid to one side",
        "Double Braids": "double braids, two braids, pigtail style",
        "Milkmaid Braids": "milkmaid braids, wrapped around head, folk style",
        "Waterfall Braid": "waterfall braid, cascading woven style",
        "Goddess Braids": "goddess braids, large cornrows, regal style",
        "Knotless Braids": "knotless braids, feed-in technique, seamless",
        "Micro Braids": "micro braids, tiny thin braids, intricate",
        "Twist Braids": "twist braids, two-strand twisted style",
        "Locs": "locs hairstyle, matured dreadlocks, natural",
        "Faux Locs": "faux locs, temporary loc style, protective",
        # Bangs & Fringe
        "Blunt Bangs": "blunt straight bangs, thick fringe, bold",
        "Side Swept Bangs": "side swept bangs, angled fringe, soft",
        "Curtain Bangs": "curtain bangs, parted fringe, 70s style, face framing",
        "Wispy Bangs": "wispy bangs, thin soft fringe, delicate",
        "Micro Bangs": "micro bangs, very short baby bangs, edgy",
        "Long Bangs": "long bangs, eye-covering fringe, mysterious",
        "Bottleneck Bangs": "bottleneck bangs, shorter center longer sides",
        "Birkin Bangs": "Birkin bangs, full slightly parted, French style",
        # Textured & Specialty
        "Shag Haircut": "shag haircut, layered choppy, 70s rock style",
        "Wolf Cut": "wolf cut, shaggy mullet hybrid, edgy layers",
        "Mullet": "mullet hairstyle, short front long back, retro",
        "Mohawk": "mohawk, shaved sides, center strip of hair",
        "Faux Hawk": "faux hawk, styled mohawk look, less extreme",
        "Textured Crop": "textured crop, short with texture, modern",
        "Curtained Hair": "curtained hair, middle part, face framing",
        "Feathered Layers": "feathered layers, Farrah Fawcett style, flipped",
        "Razor Cut": "razor cut, textured choppy ends, edgy",
        "Blowout": "blowout style, voluminous bouncy, salon fresh",
        # Vintage Styles
        "Victory Rolls": "victory rolls, 1940s pin-up style, rolled sections",
        "Finger Waves": "finger waves, 1920s marcel waves, gatsby style",
        "Pin Curls": "pin curl set, vintage curled style, retro",
        "Bouffant": "bouffant style, teased volume, 60s glamour",
        "Flip Hairstyle": "flip hairstyle, ends flipped outward, 60s style",
        "Pageboy": "pageboy cut, smooth with inward curved ends",
        # Hair Accessories
        "Hair with Headband": "hair with headband accessory, styled with band",
        "Hair with Bow": "hair with bow accessory, ribbon or bow detail",
        "Hair with Clips": "hair with decorative clips, barrettes, pins",
        "Hair with Flowers": "hair adorned with flowers, floral accessory",
        "Hair with Tiara": "hair with tiara, crown accessory, regal",
        "Hair with Scarf": "hair wrapped with scarf, bohemian style",
    }

    # Clothing Styles
    CLOTHING_STYLES = {
        "None": "",
        # Dresses
        "Evening Gown": "elegant evening gown, floor-length formal dress, glamorous",
        "Cocktail Dress": "cocktail dress, knee-length party dress, sophisticated",
        "Little Black Dress": "little black dress, LBD, classic elegant, timeless",
        "Maxi Dress": "maxi dress, floor-length casual dress, flowing",
        "Mini Dress": "mini dress, short hemline, youthful playful",
        "Midi Dress": "midi dress, mid-calf length, elegant modern",
        "Bodycon Dress": "bodycon dress, form-fitting, figure-hugging",
        "A-Line Dress": "A-line dress, fitted bodice, flared skirt",
        "Wrap Dress": "wrap dress, crossover front, flattering silhouette",
        "Slip Dress": "slip dress, minimalist satin, delicate straps",
        "Shirt Dress": "shirt dress, button-front, casual elegant",
        "Sundress": "sundress, light summer dress, casual cheerful",
        "Ball Gown": "ball gown, full dramatic skirt, princess style",
        "Mermaid Dress": "mermaid dress, fitted to knee then flared, dramatic",
        "Sheath Dress": "sheath dress, straight fitted silhouette, professional",
        # Tops
        "Blouse": "elegant blouse, dressy top, feminine",
        "Button-Up Shirt": "button-up shirt, collared, classic",
        "T-Shirt": "casual t-shirt, basic tee, relaxed",
        "Tank Top": "tank top, sleeveless, casual summer",
        "Crop Top": "crop top, midriff-baring, trendy",
        "Turtleneck": "turtleneck sweater, high neck, sophisticated",
        "Off-Shoulder Top": "off-shoulder top, exposed shoulders, romantic",
        "Halter Top": "halter top, neck-tied, open back",
        "Camisole": "camisole, delicate straps, layering piece",
        "Bodysuit": "bodysuit, fitted one-piece top, sleek",
        "Corset Top": "corset top, structured boning, dramatic",
        "Bustier": "bustier top, strapless structured, evening",
        "Peasant Blouse": "peasant blouse, bohemian, gathered details",
        "Peplum Top": "peplum top, flared waist detail, feminine",
        # Bottoms
        "Skinny Jeans": "skinny jeans, tight-fitting denim, modern",
        "Wide-Leg Pants": "wide-leg pants, flowing silhouette, elegant",
        "High-Waisted Pants": "high-waisted pants, elongating, retro-modern",
        "Palazzo Pants": "palazzo pants, extremely wide leg, dramatic",
        "Culottes": "culottes, wide cropped pants, fashion-forward",
        "Pencil Skirt": "pencil skirt, fitted knee-length, professional",
        "Mini Skirt": "mini skirt, short hemline, youthful",
        "Maxi Skirt": "maxi skirt, floor-length, flowing",
        "Pleated Skirt": "pleated skirt, accordion pleats, feminine",
        "A-Line Skirt": "A-line skirt, fitted waist flared hem",
        "Leather Pants": "leather pants, edgy sleek, rock style",
        "Cargo Pants": "cargo pants, utilitarian pockets, casual",
        "Shorts": "shorts, casual short pants, summer",
        "Hot Pants": "hot pants, very short shorts, daring",
        # Suits & Formal
        "Pantsuit": "tailored pantsuit, professional matching set",
        "Skirt Suit": "skirt suit, professional matching set, feminine",
        "Blazer": "structured blazer, professional jacket, sharp",
        "Tuxedo": "women's tuxedo, formal masculine-inspired, elegant",
        "Power Suit": "power suit, bold shoulders, commanding",
        # Outerwear
        "Leather Jacket": "leather jacket, edgy biker style, cool",
        "Denim Jacket": "denim jacket, casual classic, versatile",
        "Trench Coat": "trench coat, classic belted, sophisticated",
        "Blazer Jacket": "blazer jacket, structured tailored, professional",
        "Cardigan": "cardigan sweater, cozy layering, casual",
        "Fur Coat": "fur coat, luxurious dramatic, glamorous",
        "Faux Fur": "faux fur coat, ethical glamour, plush",
        "Cape": "cape outerwear, dramatic flowing, statement",
        "Bomber Jacket": "bomber jacket, sporty casual, trendy",
        "Puffer Jacket": "puffer jacket, quilted warm, casual",
        # Swimwear
        "Bikini": "bikini swimsuit, two-piece, beach style",
        "One-Piece Swimsuit": "one-piece swimsuit, classic swimming, elegant",
        "High-Waisted Bikini": "high-waisted bikini, retro style, flattering",
        "String Bikini": "string bikini, minimal coverage, daring",
        "Tankini": "tankini, tank top style swimwear, modest",
        "Monokini": "monokini, cutout one-piece, dramatic",
        # Lingerie & Intimate
        "Lace Lingerie": "delicate lace lingerie, feminine intimate, romantic",
        "Silk Robe": "silk robe, luxurious loungewear, elegant",
        "Bodysuit Lingerie": "bodysuit lingerie, one-piece intimate, sleek",
        "Bralette": "bralette, soft unstructured bra, comfortable",
        "Corset Lingerie": "corset lingerie, structured boning, dramatic",
        # Athleisure & Casual
        "Yoga Outfit": "yoga outfit, stretchy athletic wear, comfortable",
        "Sports Bra": "sports bra, athletic support top, fitness",
        "Leggings": "leggings, fitted stretch pants, versatile",
        "Hoodie": "hoodie sweatshirt, casual comfortable, relaxed",
        "Sweatpants": "sweatpants, comfortable casual, loungewear",
        "Athleisure Set": "matching athleisure set, sporty chic, coordinated",
        # Cultural & Traditional
        "Kimono": "Japanese kimono, traditional wrapped garment, elegant",
        "Sari": "Indian sari, draped fabric, traditional elegant",
        "Cheongsam": "Chinese cheongsam, fitted traditional dress, elegant",
        "Hanbok": "Korean hanbok, traditional dress, colorful elegant",
        "Dirndl": "German dirndl, traditional dress, folk style",
        "Flamenco Dress": "flamenco dress, ruffled dramatic, Spanish style",
        # Vintage Eras
        "1920s Flapper": "1920s flapper dress, beaded fringe, gatsby era",
        "1950s Pin-Up": "1950s pin-up style, full skirt, cinched waist",
        "1960s Mod": "1960s mod fashion, geometric, bold patterns",
        "1970s Bohemian": "1970s bohemian style, flowing fabrics, earthy",
        "1980s Power": "1980s power dressing, bold shoulders, dramatic",
        "1990s Grunge": "1990s grunge style, flannel, distressed",
        # Fantasy & Costume
        "Ball Gown Princess": "princess ball gown, fairy tale dress, magical",
        "Medieval Gown": "medieval gown, historical dress, renaissance",
        "Goddess Dress": "goddess dress, draped Grecian style, ethereal",
        "Fairy Costume": "fairy costume, whimsical wings, magical",
        "Witch Costume": "witch costume, dark dramatic, mystical",
        # Bridal
        "Wedding Dress": "wedding dress, bridal gown, white elegant",
        "Bridal Gown": "formal bridal gown, traditional wedding, elaborate",
        "Wedding Jumpsuit": "bridal jumpsuit, modern wedding, chic",
    }

    # Eye Styles
    EYE_STYLES = {
        "None": "",
        # Eye Colors
        "Blue Eyes": "bright blue eyes, clear azure iris, striking",
        "Green Eyes": "green eyes, emerald iris, captivating",
        "Brown Eyes": "warm brown eyes, chocolate iris, deep",
        "Hazel Eyes": "hazel eyes, mixed brown green gold, changeable",
        "Amber Eyes": "amber eyes, golden honey iris, warm",
        "Gray Eyes": "gray eyes, cool silver iris, mysterious",
        "Black Eyes": "dark black eyes, deep onyx iris, intense",
        "Violet Eyes": "violet eyes, purple iris, rare exotic",
        "Heterochromia": "heterochromia, two different colored eyes, unique",
        "Ice Blue Eyes": "ice blue eyes, pale icy blue, piercing",
        "Steel Gray Eyes": "steel gray eyes, metallic cool, intense",
        "Honey Eyes": "honey colored eyes, golden brown, warm",
        "Olive Eyes": "olive green eyes, earthy green tone",
        "Teal Eyes": "teal eyes, blue-green mix, oceanic",
        "Golden Eyes": "golden eyes, yellow-gold iris, striking",
        # Eye Shapes
        "Almond Eyes": "almond shaped eyes, tapered ends, elegant",
        "Round Eyes": "round eyes, circular shape, youthful innocent",
        "Hooded Eyes": "hooded eyes, heavy lid, mysterious",
        "Monolid Eyes": "monolid eyes, no crease, smooth lid",
        "Upturned Eyes": "upturned eyes, outer corners lifted, cat-like",
        "Downturned Eyes": "downturned eyes, outer corners down, soft",
        "Deep-Set Eyes": "deep-set eyes, set back in socket, intense",
        "Protruding Eyes": "protruding eyes, prominent, expressive",
        "Close-Set Eyes": "close-set eyes, near bridge of nose",
        "Wide-Set Eyes": "wide-set eyes, far apart, open face",
        "Large Eyes": "large expressive eyes, doe eyes, striking",
        "Small Eyes": "small delicate eyes, refined, subtle",
        "Cat Eyes": "cat-like eyes, feline shape, exotic",
        "Doe Eyes": "doe eyes, large innocent, Bambi-like",
        "Bedroom Eyes": "bedroom eyes, heavy-lidded, sultry",
        # Eye Makeup Styles
        "Natural Makeup": "natural eye makeup, subtle enhancement, fresh",
        "Smoky Eye": "smoky eye makeup, dramatic blended shadow, sultry",
        "Cat Eye Liner": "cat eye liner, winged eyeliner, feline flick",
        "Cut Crease": "cut crease eye makeup, defined crease, dramatic",
        "Glitter Eyes": "glitter eye makeup, sparkly shimmer, festive",
        "Graphic Liner": "graphic eyeliner, artistic lines, bold",
        "No Makeup Look": "no makeup look, bare natural, fresh-faced",
        "Soft Glam": "soft glam eye makeup, subtle glamour, elegant",
        "Bold Color": "bold colorful eye makeup, vivid pigments, artistic",
        "Nude Eye": "nude eye makeup, neutral tones, subtle",
        "Bronze Eye": "bronze eye makeup, warm metallic, sun-kissed",
        "Blue Eyeshadow": "blue eyeshadow, cool toned, striking",
        "Green Eyeshadow": "green eyeshadow, earthy or vibrant, unique",
        "Purple Eyeshadow": "purple eyeshadow, dramatic romantic, rich",
        "Red Eyeshadow": "red eyeshadow, bold daring, editorial",
        "Gold Eyeshadow": "gold eyeshadow, metallic warm, luxurious",
        "Silver Eyeshadow": "silver eyeshadow, cool metallic, futuristic",
        "Black Eyeshadow": "black eyeshadow, dramatic gothic, intense",
        "Pink Eyeshadow": "pink eyeshadow, soft feminine, romantic",
        # Eyeliner Styles
        "Thin Liner": "thin eyeliner, subtle definition, natural",
        "Thick Liner": "thick eyeliner, bold dramatic, statement",
        "Winged Liner": "winged eyeliner, classic flick, elegant",
        "Double Wing": "double winged liner, graphic modern, edgy",
        "Smudged Liner": "smudged eyeliner, soft smoky, rock style",
        "White Liner": "white eyeliner, brightening, modern",
        "Colored Liner": "colored eyeliner, bold hue, playful",
        "Floating Liner": "floating crease liner, graphic trend, artistic",
        "Tight Line": "tightlined eyes, waterline liner, subtle definition",
        "No Liner": "no eyeliner, soft natural, bare",
        # Eyelashes
        "Natural Lashes": "natural eyelashes, subtle length, understated",
        "Long Lashes": "long eyelashes, extended length, dramatic",
        "Thick Lashes": "thick lashes, full volume, bold",
        "False Lashes": "false eyelashes, glamorous volume, dramatic",
        "Wispy Lashes": "wispy lashes, feathery natural, soft",
        "Cat Eye Lashes": "cat eye lashes, longer outer corner, feline",
        "Doll Lashes": "doll lashes, rounded full, innocent",
        "Spiky Lashes": "spiky lashes, separated defined, modern",
        "Bottom Lashes": "emphasized bottom lashes, doe-eyed, 60s style",
        "Colored Lashes": "colored lashes, bold hue, artistic",
        "Feather Lashes": "feather lashes, dramatic costume, avant-garde",
        # Eyebrows
        "Natural Brows": "natural eyebrows, untamed shape, authentic",
        "Thick Brows": "thick bold eyebrows, full bushy, statement",
        "Thin Brows": "thin eyebrows, shaped refined, vintage",
        "Arched Brows": "arched eyebrows, high arch, dramatic",
        "Straight Brows": "straight eyebrows, horizontal shape, youthful",
        "Feathered Brows": "feathered eyebrows, brushed up, natural",
        "Laminated Brows": "laminated brows, slicked up, trendy",
        "Ombre Brows": "ombre brows, gradient fill, soft defined",
        "Bleached Brows": "bleached eyebrows, pale blonde, editorial",
        "Colored Brows": "colored eyebrows, bold hue, artistic",
        # Eye Expressions
        "Wide Open Eyes": "wide open eyes, alert surprised, expressive",
        "Half-Closed Eyes": "half-closed eyes, dreamy sultry, relaxed",
        "Squinting Eyes": "squinting eyes, narrowed, suspicious or smiling",
        "Winking": "winking, one eye closed, playful",
        "Looking Up": "eyes looking upward, hopeful dreamy",
        "Looking Down": "eyes looking downward, demure thoughtful",
        "Side Glance": "side glance, looking to side, coy",
        "Direct Gaze": "direct eye contact, confident engaging, bold",
        "Distant Gaze": "distant gaze, looking far away, contemplative",
        "Teary Eyes": "teary eyes, glistening emotional, vulnerable",
        "Sparkling Eyes": "sparkling eyes, bright joyful, alive",
        "Intense Stare": "intense staring eyes, piercing focused, powerful",
        "Soft Gaze": "soft gentle gaze, warm welcoming, kind",
    }

    # Fashion Styles
    FASHION_STYLES = {
        "None": "",
        # Classic Styles
        "Classic Elegant": "classic elegant fashion, timeless refined style, sophisticated",
        "Preppy": "preppy style, collegiate clean-cut, polished traditional",
        "Business Professional": "business professional attire, corporate formal, executive",
        "Business Casual": "business casual style, smart relaxed, office appropriate",
        "Smart Casual": "smart casual fashion, polished yet relaxed, versatile",
        "Old Money": "old money aesthetic, quiet luxury, understated wealth",
        "Quiet Luxury": "quiet luxury fashion, subtle expensive, no logos",
        "Minimalist": "minimalist fashion, simple clean lines, understated",
        "Monochromatic": "monochromatic outfit, single color scheme, cohesive",
        "Neutral Palette": "neutral palette fashion, beige tan cream, sophisticated",
        # Trendy & Modern
        "Streetwear": "streetwear fashion, urban casual, trendy comfortable",
        "Athleisure": "athleisure style, sporty casual, gym-to-street",
        "Y2K": "Y2K fashion, early 2000s style, nostalgic trendy",
        "Cottagecore": "cottagecore aesthetic, rural romantic, pastoral",
        "Dark Academia": "dark academia style, scholarly vintage, literary",
        "Light Academia": "light academia fashion, scholarly bright, classic",
        "Coastal Grandmother": "coastal grandmother style, relaxed elegant, Nancy Meyers",
        "Clean Girl": "clean girl aesthetic, minimal fresh, effortless",
        "Barbiecore": "Barbiecore fashion, hot pink, playful feminine",
        "Balletcore": "balletcore style, ballet-inspired, soft feminine",
        "Mob Wife": "mob wife aesthetic, leopard print, gold, dramatic",
        # Edgy & Alternative
        "Grunge": "grunge fashion, distressed layers, 90s alternative",
        "Punk": "punk style, rebellious DIY, safety pins chains",
        "Goth": "gothic fashion, all black, dark romantic",
        "Emo": "emo style, dark emotional, band tees skinny jeans",
        "Rock Chic": "rock chic fashion, leather edgy, concert style",
        "Biker": "biker style, leather motorcycle, tough cool",
        "Edgy Modern": "edgy modern fashion, avant-garde, boundary-pushing",
        "Industrial": "industrial fashion, utilitarian dark, hardware details",
        "Cyber Goth": "cyber goth style, neon black, futuristic dark",
        "Soft Grunge": "soft grunge fashion, pastel dark mix, Tumblr era",
        # Bohemian & Free Spirit
        "Bohemian": "bohemian fashion, free-spirited eclectic, flowing fabrics",
        "Boho Chic": "boho chic style, refined bohemian, festival elegant",
        "Hippie": "hippie style, 60s-70s peace love, tie-dye fringe",
        "Festival Fashion": "festival fashion, Coachella style, bold fun",
        "Gypsy": "gypsy style, traveling bohemian, layered jewelry",
        "Earthy": "earthy fashion, natural tones, organic materials",
        "Free Spirit": "free spirit fashion, unconventional, expressive",
        # Glamorous & Luxe
        "Hollywood Glamour": "Hollywood glamour, red carpet style, dramatic elegant",
        "Old Hollywood": "old Hollywood fashion, vintage glamour, golden age",
        "Haute Couture": "haute couture fashion, high fashion, designer runway",
        "Luxury Fashion": "luxury fashion, designer labels, expensive refined",
        "Red Carpet": "red carpet fashion, celebrity gala, show-stopping",
        "Evening Glamour": "evening glamour style, formal elegant, sophisticated",
        "Disco Glam": "disco glamour, 70s sparkle, Studio 54 style",
        "Showgirl": "showgirl fashion, Vegas glamour, sequins feathers",
        # Romantic & Feminine
        "Romantic": "romantic fashion, soft feminine, ruffles lace",
        "Feminine": "feminine fashion, girly pretty, soft details",
        "Victorian": "Victorian fashion, historical romantic, modest elaborate",
        "Fairycore": "fairycore aesthetic, whimsical magical, nature-inspired",
        "Princesscore": "princesscore fashion, regal feminine, pretty pink",
        "Regencycore": "Regencycore fashion, Bridgerton-inspired, empire waist",
        "Coquette": "coquette style, flirty feminine, bows ribbons",
        "Dollette": "dollette fashion, doll-like, porcelain pretty",
        # Sporty & Active
        "Sporty": "sporty fashion, athletic casual, active lifestyle",
        "Athletic": "athletic wear, performance clothing, gym style",
        "Tenniscore": "tenniscore fashion, tennis-inspired, preppy sporty",
        "Gorpcore": "gorpcore style, outdoor hiking, technical fashion",
        "Surf Style": "surf style fashion, beach casual, coastal cool",
        "Skater": "skater fashion, skateboard culture, casual cool",
        # Cultural & Global
        "Japanese Street": "Japanese street fashion, Harajuku style, creative bold",
        "Korean Fashion": "Korean fashion, K-fashion, trendy modern",
        "Parisian Chic": "Parisian chic, French style, effortless elegant",
        "Italian Style": "Italian fashion, la bella figura, luxurious",
        "Scandinavian": "Scandinavian fashion, Nordic minimalist, functional",
        "British Style": "British fashion, tailored classic, heritage",
        "American Classic": "American classic fashion, all-American, timeless",
        # Vintage Decades
        "1920s Fashion": "1920s fashion, flapper era, art deco glamour",
        "1930s Fashion": "1930s fashion, Hollywood golden age, elegant bias-cut",
        "1940s Fashion": "1940s fashion, wartime elegance, tailored utility",
        "1950s Fashion": "1950s fashion, full skirts, Dior New Look",
        "1960s Fashion": "1960s fashion, mod mini, youthquake style",
        "1970s Fashion": "1970s fashion, bohemian disco, eclectic",
        "1980s Fashion": "1980s fashion, power shoulders, bold colors",
        "1990s Fashion": "1990s fashion, minimalist grunge, slip dress",
        "2000s Fashion": "2000s fashion, low-rise Y2K, Paris Hilton era",
        # Occasion-Based
        "Resort Wear": "resort wear, vacation fashion, tropical elegant",
        "Beach Style": "beach style fashion, coastal casual, swimwear cover-ups",
        "Cocktail Attire": "cocktail attire, semi-formal, party dress",
        "Black Tie": "black tie fashion, formal evening, gala appropriate",
        "Garden Party": "garden party attire, floral feminine, outdoor elegant",
        "Brunch Style": "brunch style fashion, casual chic, weekend wear",
        # Unique & Niche
        "Avant-Garde": "avant-garde fashion, experimental artistic, unconventional",
        "Deconstructed": "deconstructed fashion, raw edges, asymmetric",
        "Gender Fluid": "gender fluid fashion, unisex androgynous, non-binary",
        "Maximalist": "maximalist fashion, more is more, bold layers",
        "Eclectic": "eclectic fashion, mixed styles, personal unique",
        "Normcore": "normcore fashion, deliberately ordinary, anti-fashion",
        "Ugly Chic": "ugly chic fashion, intentionally unfashionable, ironic",
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

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        """Define ComfyUI input types."""
        return {
            "required": {
                "prompt": ("STRING", {"forceInput": True, "tooltip": "Input prompt - can be basic text (for Expand mode) or detailed prompt (for Modify modes)"}),
                "llm_model": ("LLM_MODEL",),
            },
            "optional": {
                "processing_mode": (["Expand", "Single Pass", "Section by Section"], {"default": "Single Pass", "tooltip": "Expand: generate detailed prompt from basic text. Single Pass/Section by Section: modify existing detailed prompt"}),
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

        system_prompt = """You are an expert prompt engineer for AI image generation (Stable Diffusion, Flux, Z-Image).

Your task is to modify an existing image prompt based on specific instructions and style presets.

RULES:
1. PRESERVE the overall structure and flow of the original prompt
2. Apply section modifications as specified
3. Integrate photography templates, effects, and photographer styles naturally
4. Keep unmentioned sections UNCHANGED
5. Maintain concrete, visual, descriptive writing style
6. Output ONLY the modified prompt - no explanations or commentary
7. Write as a single flowing paragraph"""

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

            preset_system = """You are an expert prompt engineer for AI image generation.

Your task is to integrate photography style presets into an existing prompt.

RULES:
1. Integrate the style presets naturally into the prompt
2. Preserve the subject and core content of the original prompt
3. Enhance lighting, camera, and mood to match the styles
4. Output ONLY the modified prompt - no explanations
5. Write as a single flowing paragraph"""

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

RULES:
1. ONLY modify the specified section
2. Keep ALL other parts of the prompt EXACTLY the same
3. Maintain the same writing style
4. Output ONLY the complete modified prompt - no explanations
5. Write as a single flowing paragraph"""

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

        system_prompt = """You are an expert prompt engineer for AI image generation (Stable Diffusion, Flux, Midjourney).

Your task is to expand a basic concept or short description into a comprehensive, detailed image generation prompt.

STRUCTURE YOUR OUTPUT WITH THESE SECTIONS (write as flowing prose, not bullet points):

1. **Subject**: Detailed description of the main subject - physical features, age, ethnicity, body type, facial features, hair style/color/texture, skin tone and texture

2. **Clothing & Accessories**: Complete outfit description - garments, fabrics, colors, fit, style, jewelry, accessories

3. **Pose & Expression**: Body position, stance, gesture, facial expression, eye direction, mood conveyed

4. **Environment**: Setting, location, background elements, props, atmosphere, time of day

5. **Lighting**: Light quality (soft/hard), direction, color temperature, shadows, highlights, mood

6. **Camera**: Shot type (close-up, full body, etc.), angle, framing, depth of field, focus

RULES:
1. Be specific and visual - describe what can be SEEN, not abstract concepts
2. Use concrete descriptors (not "beautiful" but "high cheekbones, full lips, almond-shaped eyes")
3. Include colors, textures, materials whenever possible
4. Write as a single flowing paragraph or connected paragraphs
5. Output ONLY the expanded prompt - no explanations, headers, or meta-commentary
6. Aim for 200-400 words of rich, detailed description"""

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

        # Output should be at least as long as input, plus 50% buffer for modifications
        # This ensures we don't truncate prompts during modification
        buffer_multiplier = 1.5
        calculated_tokens = int(estimated_input_tokens * buffer_multiplier)

        # Set minimum (for short prompts) and maximum bounds
        min_tokens = 1500  # Generous minimum for detailed prompt modification
        max_tokens = getattr(llm_model, 'max_tokens', 6000)

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
