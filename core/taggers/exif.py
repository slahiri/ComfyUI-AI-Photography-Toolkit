"""EXIF Metadata Extraction Tagger.

Extracts camera and lens metadata from image EXIF data:
- Focal length (35mm, 50mm, 85mm, etc.)
- Aperture/F-stop (f/1.4, f/2.8, f/8, etc.)
- ISO speed (ISO 100, ISO 3200, etc.)
- Shutter speed (1/250s, 1/60s, 30s, etc.)
- Camera make/model
- Lens model
- Flash status
- Exposure mode
"""

from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from typing import Dict, Optional, Any
from fractions import Fraction

from .base import BaseTagger, TagItem, TaggerResult


# EXIF tag IDs for common camera metadata
EXIF_TAGS = {
    "FocalLength": 37386,
    "FocalLengthIn35mmFilm": 41989,
    "FNumber": 33437,
    "ExposureTime": 33434,
    "ISOSpeedRatings": 34855,
    "Make": 271,
    "Model": 272,
    "LensModel": 42036,
    "LensMake": 42035,
    "Flash": 37385,
    "ExposureProgram": 34850,
    "ExposureMode": 41986,
    "WhiteBalance": 41987,
    "MeteringMode": 37383,
    "DateTimeOriginal": 36867,
    "Software": 305,
    "Orientation": 274,
    "XResolution": 282,
    "YResolution": 283,
}

# Exposure program descriptions
EXPOSURE_PROGRAMS = {
    0: "not defined",
    1: "manual",
    2: "program auto",
    3: "aperture priority",
    4: "shutter priority",
    5: "creative program",
    6: "action program",
    7: "portrait mode",
    8: "landscape mode",
}

# Metering mode descriptions
METERING_MODES = {
    0: "unknown",
    1: "average metering",
    2: "center-weighted metering",
    3: "spot metering",
    4: "multi-spot metering",
    5: "pattern metering",
    6: "partial metering",
}

# Focal length categories
FOCAL_LENGTH_CATEGORIES = {
    (0, 20): "ultra wide angle",
    (20, 35): "wide angle",
    (35, 50): "standard",
    (50, 85): "short telephoto",
    (85, 135): "portrait lens",
    (135, 200): "medium telephoto",
    (200, 400): "telephoto",
    (400, float('inf')): "super telephoto",
}

# Aperture categories
APERTURE_CATEGORIES = {
    (0, 1.8): "very fast lens",
    (1.8, 2.8): "fast lens",
    (2.8, 4.0): "moderate aperture",
    (4.0, 5.6): "standard aperture",
    (5.6, 8.0): "small aperture",
    (8.0, 11.0): "narrow aperture",
    (11.0, float('inf')): "very narrow aperture",
}


class ExifTagger(BaseTagger):
    """Extract camera metadata from EXIF data."""

    def __init__(
        self,
        threshold: float = 0.8,
        include_camera_info: bool = True,
        include_lens_info: bool = True,
        include_exposure_info: bool = True,
    ):
        """Initialize EXIF tagger.

        Args:
            threshold: Confidence for extracted tags (default 0.8 since EXIF is reliable)
            include_camera_info: Include camera make/model
            include_lens_info: Include lens and focal length info
            include_exposure_info: Include aperture, ISO, shutter speed
        """
        super().__init__()
        self.threshold = threshold
        self.include_camera_info = include_camera_info
        self.include_lens_info = include_lens_info
        self.include_exposure_info = include_exposure_info
        self._is_loaded = True  # No model to load

    def load(self) -> None:
        """No model to load for EXIF extraction."""
        self._is_loaded = True

    def _get_exif_data(self, image: Image.Image) -> Dict[str, Any]:
        """Extract EXIF data from image."""
        exif_data = {}

        try:
            # Try to get EXIF from PIL
            exif = image._getexif()
            if exif is None:
                return exif_data

            # Map tag IDs to names
            for tag_id, value in exif.items():
                tag_name = TAGS.get(tag_id, str(tag_id))
                exif_data[tag_name] = value

        except (AttributeError, KeyError, TypeError):
            pass

        return exif_data

    def _format_focal_length(self, focal_length: Any, focal_35mm: Any = None) -> Optional[str]:
        """Format focal length value."""
        try:
            if isinstance(focal_length, tuple):
                fl = focal_length[0] / focal_length[1] if focal_length[1] else focal_length[0]
            else:
                fl = float(focal_length)

            # Prefer 35mm equivalent if available
            if focal_35mm:
                if isinstance(focal_35mm, tuple):
                    fl_35 = focal_35mm[0] / focal_35mm[1] if focal_35mm[1] else focal_35mm[0]
                else:
                    fl_35 = float(focal_35mm)
                return f"{int(fl_35)}mm (35mm equiv)"

            return f"{int(fl)}mm"
        except (TypeError, ValueError, ZeroDivisionError):
            return None

    def _get_focal_length_category(self, focal_length: float) -> str:
        """Get descriptive category for focal length."""
        for (low, high), category in FOCAL_LENGTH_CATEGORIES.items():
            if low <= focal_length < high:
                return category
        return "telephoto"

    def _format_aperture(self, f_number: Any) -> Optional[str]:
        """Format aperture value."""
        try:
            if isinstance(f_number, tuple):
                aperture = f_number[0] / f_number[1] if f_number[1] else f_number[0]
            else:
                aperture = float(f_number)

            # Format nicely
            if aperture == int(aperture):
                return f"f/{int(aperture)}"
            return f"f/{aperture:.1f}"
        except (TypeError, ValueError, ZeroDivisionError):
            return None

    def _get_aperture_category(self, aperture: float) -> str:
        """Get descriptive category for aperture."""
        for (low, high), category in APERTURE_CATEGORIES.items():
            if low <= aperture < high:
                return category
        return "narrow aperture"

    def _format_shutter_speed(self, exposure_time: Any) -> Optional[str]:
        """Format shutter speed value."""
        try:
            if isinstance(exposure_time, tuple):
                speed = exposure_time[0] / exposure_time[1] if exposure_time[1] else exposure_time[0]
            else:
                speed = float(exposure_time)

            if speed >= 1:
                return f"{int(speed)}s"
            else:
                # Express as fraction
                denominator = int(1 / speed)
                return f"1/{denominator}s"
        except (TypeError, ValueError, ZeroDivisionError):
            return None

    def _get_shutter_category(self, speed: float) -> str:
        """Get descriptive category for shutter speed."""
        if speed >= 1:
            return "long exposure"
        elif speed >= 0.1:
            return "slow shutter"
        elif speed >= 1/60:
            return "handheld shutter"
        elif speed >= 1/250:
            return "action stopping"
        elif speed >= 1/1000:
            return "fast shutter"
        else:
            return "high speed"

    def _format_iso(self, iso: Any) -> Optional[str]:
        """Format ISO value."""
        try:
            if isinstance(iso, (list, tuple)):
                iso_val = iso[0]
            else:
                iso_val = int(iso)
            return f"ISO {iso_val}"
        except (TypeError, ValueError, IndexError):
            return None

    def _get_iso_category(self, iso: int) -> str:
        """Get descriptive category for ISO."""
        if iso <= 100:
            return "base ISO"
        elif iso <= 400:
            return "low ISO"
        elif iso <= 1600:
            return "moderate ISO"
        elif iso <= 6400:
            return "high ISO"
        else:
            return "very high ISO"

    def tag(self, image: Image.Image) -> TaggerResult:
        """Extract EXIF metadata as tags."""
        tags = []
        metadata = {"has_exif": False}

        exif_data = self._get_exif_data(image)

        if not exif_data:
            return TaggerResult(
                tags=[],
                metadata={"has_exif": False, "error": "No EXIF data found"}
            )

        metadata["has_exif"] = True

        # Camera information
        if self.include_camera_info:
            make = exif_data.get("Make", "").strip()
            model = exif_data.get("Model", "").strip()

            if make:
                # Clean up common manufacturer names
                make_clean = make.replace("NIKON CORPORATION", "Nikon")
                make_clean = make_clean.replace("Canon Inc.", "Canon")
                make_clean = make_clean.replace("SONY", "Sony")
                make_clean = make_clean.replace("FUJIFILM", "Fujifilm")
                make_clean = make_clean.replace("Panasonic", "Panasonic")
                make_clean = make_clean.replace("OLYMPUS", "Olympus")

                tags.append(TagItem(
                    text=f"{make_clean} camera",
                    confidence=self.threshold,
                    category="camera"
                ))
                metadata["camera_make"] = make_clean

            if model:
                # Clean up model name (remove manufacturer if duplicated)
                model_clean = model
                if make:
                    model_clean = model.replace(make, "").strip()

                tags.append(TagItem(
                    text=model_clean,
                    confidence=self.threshold,
                    category="camera"
                ))
                metadata["camera_model"] = model_clean

        # Lens information
        if self.include_lens_info:
            lens_model = exif_data.get("LensModel", "")
            focal_length = exif_data.get("FocalLength")
            focal_35mm = exif_data.get("FocalLengthIn35mmFilm")

            if lens_model:
                tags.append(TagItem(
                    text=lens_model,
                    confidence=self.threshold,
                    category="lens"
                ))
                metadata["lens_model"] = lens_model

            if focal_length:
                fl_str = self._format_focal_length(focal_length, focal_35mm)
                if fl_str:
                    tags.append(TagItem(
                        text=fl_str,
                        confidence=self.threshold,
                        category="focal_length"
                    ))
                    metadata["focal_length"] = fl_str

                    # Get numeric value for category
                    try:
                        if isinstance(focal_length, tuple):
                            fl_num = focal_length[0] / focal_length[1]
                        else:
                            fl_num = float(focal_length)

                        # Use 35mm equivalent if available
                        if focal_35mm:
                            if isinstance(focal_35mm, tuple):
                                fl_num = focal_35mm[0] / focal_35mm[1]
                            else:
                                fl_num = float(focal_35mm)

                        category = self._get_focal_length_category(fl_num)
                        tags.append(TagItem(
                            text=category,
                            confidence=self.threshold * 0.9,
                            category="focal_length"
                        ))
                        metadata["focal_length_category"] = category
                    except:
                        pass

        # Exposure information
        if self.include_exposure_info:
            f_number = exif_data.get("FNumber")
            exposure_time = exif_data.get("ExposureTime")
            iso = exif_data.get("ISOSpeedRatings")

            # Aperture
            if f_number:
                aperture_str = self._format_aperture(f_number)
                if aperture_str:
                    tags.append(TagItem(
                        text=aperture_str,
                        confidence=self.threshold,
                        category="aperture"
                    ))
                    metadata["aperture"] = aperture_str

                    # Get category
                    try:
                        if isinstance(f_number, tuple):
                            f_num = f_number[0] / f_number[1]
                        else:
                            f_num = float(f_number)

                        category = self._get_aperture_category(f_num)
                        tags.append(TagItem(
                            text=category,
                            confidence=self.threshold * 0.9,
                            category="aperture"
                        ))
                        metadata["aperture_category"] = category
                    except:
                        pass

            # Shutter speed
            if exposure_time:
                speed_str = self._format_shutter_speed(exposure_time)
                if speed_str:
                    tags.append(TagItem(
                        text=speed_str,
                        confidence=self.threshold,
                        category="shutter_speed"
                    ))
                    metadata["shutter_speed"] = speed_str

                    # Get category
                    try:
                        if isinstance(exposure_time, tuple):
                            speed = exposure_time[0] / exposure_time[1]
                        else:
                            speed = float(exposure_time)

                        category = self._get_shutter_category(speed)
                        tags.append(TagItem(
                            text=category,
                            confidence=self.threshold * 0.9,
                            category="shutter_speed"
                        ))
                        metadata["shutter_category"] = category
                    except:
                        pass

            # ISO
            if iso:
                iso_str = self._format_iso(iso)
                if iso_str:
                    tags.append(TagItem(
                        text=iso_str,
                        confidence=self.threshold,
                        category="iso"
                    ))
                    metadata["iso"] = iso_str

                    # Get category
                    try:
                        if isinstance(iso, (list, tuple)):
                            iso_val = int(iso[0])
                        else:
                            iso_val = int(iso)

                        category = self._get_iso_category(iso_val)
                        tags.append(TagItem(
                            text=category,
                            confidence=self.threshold * 0.9,
                            category="iso"
                        ))
                        metadata["iso_category"] = category
                    except:
                        pass

            # Exposure program
            exp_program = exif_data.get("ExposureProgram")
            if exp_program and exp_program in EXPOSURE_PROGRAMS:
                program_name = EXPOSURE_PROGRAMS[exp_program]
                if program_name != "not defined":
                    tags.append(TagItem(
                        text=program_name,
                        confidence=self.threshold * 0.8,
                        category="exposure_mode"
                    ))
                    metadata["exposure_program"] = program_name

            # Metering mode
            metering = exif_data.get("MeteringMode")
            if metering and metering in METERING_MODES:
                metering_name = METERING_MODES[metering]
                if metering_name != "unknown":
                    tags.append(TagItem(
                        text=metering_name,
                        confidence=self.threshold * 0.8,
                        category="metering"
                    ))
                    metadata["metering_mode"] = metering_name

            # Flash
            flash = exif_data.get("Flash")
            if flash is not None:
                # Flash fired is bit 0
                flash_fired = bool(flash & 1)
                flash_text = "flash fired" if flash_fired else "no flash"
                tags.append(TagItem(
                    text=flash_text,
                    confidence=self.threshold,
                    category="flash"
                ))
                metadata["flash"] = flash_text

        return TaggerResult(
            tags=tags,
            metadata=metadata
        )

    def unload(self) -> None:
        """No model to unload."""
        pass
