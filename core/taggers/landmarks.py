"""Landmark Tagger - Detect famous landmarks, monuments, and natural wonders.

Uses CLIP zero-shot classification with comprehensive landmark vocabulary.

Categories:
- Monuments & Towers: Eiffel Tower, Statue of Liberty, Big Ben, etc.
- Ancient & Historic: Pyramids, Colosseum, Machu Picchu, etc.
- Natural Wonders: Grand Canyon, Niagara Falls, Mt Fuji, etc.
- Religious Sites: Notre Dame, Taj Mahal, Angkor Wat, etc.
- Modern Architecture: Burj Khalifa, Sydney Opera House, etc.
- Bridges: Golden Gate, Tower Bridge, Brooklyn Bridge, etc.
"""

import torch
from PIL import Image
from typing import List, Optional, Dict

from .base import BaseTagger, TagItem, TaggerResult


# Comprehensive landmark vocabulary organized by category
LANDMARK_VOCABULARY = {
    "monuments_towers": [
        # Europe
        "Eiffel Tower in Paris",
        "Big Ben clock tower in London",
        "Leaning Tower of Pisa",
        "Brandenburg Gate in Berlin",
        "Tower of London",
        "Atomium in Brussels",
        "Sagrada Familia in Barcelona",
        # Americas
        "Statue of Liberty in New York",
        "Empire State Building",
        "Chrysler Building",
        "One World Trade Center",
        "Space Needle in Seattle",
        "Gateway Arch in St Louis",
        "Christ the Redeemer statue in Rio",
        "CN Tower in Toronto",
        # Asia
        "Tokyo Tower",
        "Tokyo Skytree",
        "Petronas Towers in Kuala Lumpur",
        "Taipei 101",
        "Oriental Pearl Tower in Shanghai",
        "Marina Bay Sands in Singapore",
        # Middle East
        "Burj Khalifa in Dubai",
        "Burj Al Arab hotel",
    ],
    "ancient_historic": [
        # Egypt
        "Great Pyramid of Giza",
        "Sphinx of Giza",
        "Abu Simbel temples",
        # Europe
        "Colosseum in Rome",
        "Roman Forum",
        "Parthenon in Athens",
        "Acropolis of Athens",
        "Stonehenge",
        "Pompeii ruins",
        # Americas
        "Machu Picchu in Peru",
        "Chichen Itza pyramid",
        "Teotihuacan pyramids",
        "Easter Island moai statues",
        # Asia
        "Great Wall of China",
        "Terracotta Army in China",
        "Forbidden City in Beijing",
        "Potala Palace in Tibet",
    ],
    "natural_wonders": [
        # Mountains
        "Mount Fuji in Japan",
        "Mount Everest",
        "Matterhorn mountain",
        "Table Mountain in Cape Town",
        "Uluru Ayers Rock in Australia",
        "Half Dome in Yosemite",
        "Mount Kilimanjaro",
        # Water features
        "Niagara Falls",
        "Victoria Falls",
        "Iguazu Falls",
        "Angel Falls in Venezuela",
        # Canyons & Formations
        "Grand Canyon",
        "Antelope Canyon",
        "Bryce Canyon",
        "Monument Valley",
        "Giant's Causeway in Ireland",
        "Cappadocia rock formations",
        "Zhangjiajie pillars in China",
        # Coastal
        "Great Barrier Reef",
        "Cliffs of Moher",
        "White Cliffs of Dover",
        "Ha Long Bay in Vietnam",
        "Phi Phi Islands in Thailand",
        "Santorini island in Greece",
        # Other
        "Northern Lights aurora borealis",
        "Amazon Rainforest",
        "Sahara Desert dunes",
        "Dead Sea",
    ],
    "religious_sites": [
        # Europe
        "Notre Dame Cathedral in Paris",
        "St Peter's Basilica in Vatican",
        "Sistine Chapel",
        "Sagrada Familia",
        "St Basil's Cathedral in Moscow",
        "Westminster Abbey",
        "Cologne Cathedral",
        "Milan Cathedral",
        # Asia
        "Taj Mahal in India",
        "Angkor Wat in Cambodia",
        "Shwedagon Pagoda in Myanmar",
        "Borobudur temple in Indonesia",
        "Golden Temple in Amritsar",
        "Sensoji Temple in Tokyo",
        "Fushimi Inari shrine in Kyoto",
        "Meiji Shrine in Tokyo",
        # Middle East
        "Dome of the Rock in Jerusalem",
        "Western Wall in Jerusalem",
        "Blue Mosque in Istanbul",
        "Hagia Sophia",
        "Sheikh Zayed Grand Mosque",
        "Al-Aqsa Mosque",
    ],
    "modern_architecture": [
        "Sydney Opera House",
        "Guggenheim Museum Bilbao",
        "Walt Disney Concert Hall",
        "The Shard in London",
        "The Gherkin in London",
        "Louvre Pyramid in Paris",
        "CCTV Headquarters in Beijing",
        "Bird's Nest stadium in Beijing",
        "Gardens by the Bay in Singapore",
        "Fallingwater house by Frank Lloyd Wright",
        "Casa Mila in Barcelona",
        "Neuschwanstein Castle",
        "Palace of Versailles",
        "Buckingham Palace",
        "Windsor Castle",
        "Edinburgh Castle",
        "Prague Castle",
        "Kremlin in Moscow",
        "White House in Washington DC",
        "US Capitol Building",
        "Lincoln Memorial",
        "Washington Monument",
    ],
    "bridges": [
        "Golden Gate Bridge in San Francisco",
        "Brooklyn Bridge in New York",
        "Tower Bridge in London",
        "Sydney Harbour Bridge",
        "Rialto Bridge in Venice",
        "Charles Bridge in Prague",
        "Ponte Vecchio in Florence",
        "Millau Viaduct in France",
        "Akashi Kaikyo Bridge in Japan",
        "Tsing Ma Bridge in Hong Kong",
    ],
    "cities_skylines": [
        "New York City skyline with skyscrapers",
        "Manhattan skyline",
        "Hong Kong skyline",
        "Shanghai skyline with Oriental Pearl Tower",
        "Dubai skyline with Burj Khalifa",
        "Singapore skyline",
        "London skyline with Big Ben",
        "Paris skyline with Eiffel Tower",
        "Tokyo skyline",
        "Sydney skyline with Opera House",
        "San Francisco skyline",
        "Chicago skyline",
        "Las Vegas Strip",
        "Times Square in New York",
        "Shibuya Crossing in Tokyo",
        "Piccadilly Circus in London",
    ],
}


class LandmarkTagger(BaseTagger):
    """Famous landmark detection using CLIP zero-shot classification."""

    def __init__(
        self,
        threshold: float = 0.20,
        top_k: int = 5,
        categories: Optional[List[str]] = None,
        model_name: str = "openai/clip-vit-base-patch32",
    ):
        """
        Initialize landmark tagger.

        Args:
            threshold: Minimum confidence for tags
            top_k: Maximum landmarks to return
            categories: Which categories to detect (None = all)
            model_name: CLIP model to use
        """
        super().__init__()
        self.threshold = threshold
        self.top_k = top_k
        self.categories = categories or list(LANDMARK_VOCABULARY.keys())
        self.model_name = model_name
        self.model = None
        self.processor = None
        self._device = None
        self._text_embeddings = None

    def load(self) -> None:
        """Load model into memory (required by BaseTagger)."""
        self._load_model()
        self._is_loaded = self.model is not None

    def _load_model(self):
        """Load CLIP model and precompute text embeddings."""
        if self.model is not None:
            return

        try:
            from transformers import CLIPProcessor, CLIPModel

            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            self.model = CLIPModel.from_pretrained(self.model_name)
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            self.model = self.model.to(self._device)
            self.model.eval()

            # Precompute text embeddings
            self._precompute_text_embeddings()

        except Exception as e:
            print(f"[Landmark] Failed to load model: {e}")
            import traceback
            traceback.print_exc()

    def _precompute_text_embeddings(self):
        """Precompute text embeddings for all landmarks."""
        all_prompts = []
        prompt_categories = []
        prompt_landmarks = []

        for category in self.categories:
            if category not in LANDMARK_VOCABULARY:
                continue

            for landmark in LANDMARK_VOCABULARY[category]:
                # Create detection prompt
                prompt = f"a photo of {landmark}"
                all_prompts.append(prompt)
                prompt_categories.append(category)
                prompt_landmarks.append(landmark)

        if not all_prompts:
            return

        # Batch process
        batch_size = 50
        all_embeddings = []

        for i in range(0, len(all_prompts), batch_size):
            batch = all_prompts[i:i + batch_size]
            inputs = self.processor(
                text=batch,
                return_tensors="pt",
                padding=True,
            ).to(self._device)

            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                all_embeddings.append(text_features)

        self._text_embeddings = {
            "landmarks": prompt_landmarks,
            "categories": prompt_categories,
            "embeddings": torch.cat(all_embeddings, dim=0),
        }

    def tag(self, image: Image.Image) -> TaggerResult:
        """Detect landmarks in image."""
        self._load_model()

        if self.model is None or self._text_embeddings is None:
            return TaggerResult(tags=[], metadata={"error": "Model not loaded"})

        try:
            # Prepare image
            if image.mode != "RGB":
                image = image.convert("RGB")

            inputs = self.processor(images=image, return_tensors="pt").to(self._device)

            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Compute similarities
            text_features = self._text_embeddings["embeddings"]
            similarities = (image_features @ text_features.T)[0]

            # Get top matches
            top_k_count = min(self.top_k * 2, len(similarities))
            top_probs, top_indices = torch.topk(similarities, top_k_count)

            tags = []
            seen_landmarks = set()
            detected_categories = set()

            for prob, idx in zip(top_probs, top_indices):
                # Normalize confidence (CLIP similarities typically 0.2-0.4)
                conf = (prob.item() - 0.18) / 0.20
                conf = max(0, min(1, conf))

                if conf < self.threshold:
                    continue

                landmark = self._text_embeddings["landmarks"][idx.item()]
                category = self._text_embeddings["categories"][idx.item()]

                # Skip duplicates
                landmark_key = landmark.lower().split(" in ")[0]
                if landmark_key in seen_landmarks:
                    continue
                seen_landmarks.add(landmark_key)

                # Clean landmark name for tag
                tag_text = self._clean_landmark_name(landmark)

                tags.append(TagItem(
                    text=tag_text,
                    confidence=conf,
                    category=category,
                ))

                detected_categories.add(category)

                if len(tags) >= self.top_k:
                    break

            # Add location meta-tag if landmarks detected
            if tags:
                # Get the primary location
                primary = tags[0].text
                if " in " in self._text_embeddings["landmarks"][top_indices[0].item()]:
                    location = self._text_embeddings["landmarks"][top_indices[0].item()].split(" in ")[-1]
                    tags.append(TagItem(
                        text=f"location: {location}",
                        confidence=tags[0].confidence * 0.9,
                        category="location",
                    ))

            return TaggerResult(
                tags=tags,
                metadata={
                    "model": self.model_name,
                    "categories_detected": list(detected_categories),
                    "count": len(tags),
                }
            )

        except Exception as e:
            print(f"[Landmark] Inference error: {e}")
            import traceback
            traceback.print_exc()
            return TaggerResult(tags=[], metadata={"error": str(e)})

    def _clean_landmark_name(self, landmark: str) -> str:
        """Clean landmark text for use as tag."""
        # Keep the full descriptive name but clean up
        return landmark.replace("  ", " ").strip()

    def unload(self):
        """Unload model from memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        if self._text_embeddings is not None:
            del self._text_embeddings
            self._text_embeddings = None
        torch.cuda.empty_cache()
