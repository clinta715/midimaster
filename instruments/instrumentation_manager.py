"""
Instrumentation Manager for comprehensive instrument selection and configuration.

Provides advanced instrument management with timbre and register characteristics,
genre-specific selection algorithms, and arrangement configuration capabilities.
"""

from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import random
from .instrument_categories import (
    InstrumentRegistry, InstrumentPreset, InstrumentCategory,
    InstrumentSubcategory, instrument_registry
)


class TimbreType(Enum):
    """Timbre characteristics for instruments."""
    BRIGHT = "bright"
    WARM = "warm"
    DARK = "dark"
    HARSH = "harsh"
    MELLOW = "mellow"
    BRIGHT_AND_SHARP = "bright_and_sharp"
    WARM_AND_RICH = "warm_and_rich"
    DARK_AND_FULL = "dark_and_full"
    HARSH_AND_AGGRESSIVE = "harsh_and_aggressive"
    MELLOW_AND_SMOOTH = "mellow_and_smooth"


class RegisterType(Enum):
    """Register ranges for instruments."""
    LOW = "low"
    MID_LOW = "mid_low"
    MID = "mid"
    MID_HIGH = "mid_high"
    HIGH = "high"


class ArticulationType(Enum):
    """Articulation capabilities for instruments."""
    SUSTAINED = "sustained"
    STACCATO = "staccato"
    LEGATO = "legato"
    PORTAMENTO = "portamento"
    PIZZICATO = "pizzicato"
    COL_LEGNO = "col_legno"
    HARMONICS = "harmonics"
    FLUTTER_TONGUE = "flutter_tongue"


@dataclass
class InstrumentCharacteristics:
    """Comprehensive characteristics for an instrument."""
    timbre: List[TimbreType] = field(default_factory=list)
    register: List[RegisterType] = field(default_factory=list)
    articulation: List[ArticulationType] = field(default_factory=list)
    dynamic_range: Tuple[float, float] = (0.0, 1.0)  # min_volume, max_volume
    genre_associations: List[str] = field(default_factory=list)
    mood_associations: List[str] = field(default_factory=list)
    complexity_level: str = "medium"  # simple, medium, complex
    spatial_config: Dict[str, float] = field(default_factory=dict)  # pan, reverb_send, etc.


@dataclass
class Arrangement:
    """Represents a complete instrument arrangement."""
    instruments: Dict[str, InstrumentPreset] = field(default_factory=dict)
    spatial_config: Dict[str, Dict[str, float]] = field(default_factory=dict)
    dynamic_config: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    reverb_config: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    arrangement_type: str = "standard"


class InstrumentationManager:
    """Manages instrument selection and configuration with advanced characteristics."""

    def __init__(self):
        self.instrument_library: Dict[str, InstrumentCharacteristics] = {}
        self.arrangement_rules: Dict[str, Dict[str, Any]] = {}
        self._initialize_instrument_characteristics()
        self._initialize_arrangement_rules()

    def _initialize_instrument_characteristics(self):
        """Initialize comprehensive characteristics for all instruments."""

        # Synthetic instruments
        self.instrument_library["808_kick"] = InstrumentCharacteristics(
            timbre=[TimbreType.WARM, TimbreType.DARK],
            register=[RegisterType.LOW],
            articulation=[ArticulationType.STACCATO],
            dynamic_range=(0.7, 1.0),
            genre_associations=["electronic", "hip-hop", "funk"],
            mood_associations=["energetic", "groovy"],
            complexity_level="simple",
            spatial_config={"pan": 0.0, "reverb_send": 0.1, "delay_send": 0.05}
        )

        self.instrument_library["fm_bass"] = InstrumentCharacteristics(
            timbre=[TimbreType.BRIGHT, TimbreType.WARM],
            register=[RegisterType.LOW, RegisterType.MID_LOW],
            articulation=[ArticulationType.SUSTAINED, ArticulationType.STACCATO],
            dynamic_range=(0.3, 1.0),
            genre_associations=["electronic", "techno", "ambient"],
            mood_associations=["energetic", "atmospheric"],
            complexity_level="medium",
            spatial_config={"pan": 0.0, "reverb_send": 0.2, "delay_send": 0.1}
        )

        # Ethnic percussion
        self.instrument_library["djembe"] = InstrumentCharacteristics(
            timbre=[TimbreType.WARM, TimbreType.DARK],
            register=[RegisterType.MID_LOW],
            articulation=[ArticulationType.STACCATO, ArticulationType.LEGATO],
            dynamic_range=(0.4, 1.0),
            genre_associations=["world", "folk", "fusion"],
            mood_associations=["energetic", "rhythmic", "cultural"],
            complexity_level="medium",
            spatial_config={"pan": -0.3, "reverb_send": 0.3, "delay_send": 0.2}
        )

        # Wind instruments
        self.instrument_library["trumpet"] = InstrumentCharacteristics(
            timbre=[TimbreType.BRIGHT, TimbreType.BRIGHT_AND_SHARP],
            register=[RegisterType.MID, RegisterType.MID_HIGH],
            articulation=[ArticulationType.STACCATO, ArticulationType.LEGATO, ArticulationType.SUSTAINED],
            dynamic_range=(0.2, 1.0),
            genre_associations=["jazz", "classical", "pop", "rock"],
            mood_associations=["bright", "energetic", "confident"],
            complexity_level="medium",
            spatial_config={"pan": 0.2, "reverb_send": 0.4, "delay_send": 0.1}
        )

        self.instrument_library["saxophone"] = InstrumentCharacteristics(
            timbre=[TimbreType.WARM, TimbreType.MELLOW],
            register=[RegisterType.MID_LOW, RegisterType.MID],
            articulation=[ArticulationType.LEGATO, ArticulationType.SUSTAINED, ArticulationType.PORTAMENTO],
            dynamic_range=(0.1, 1.0),
            genre_associations=["jazz", "blues", "pop", "soul"],
            mood_associations=["smooth", "emotional", "relaxed"],
            complexity_level="complex",
            spatial_config={"pan": -0.1, "reverb_send": 0.5, "delay_send": 0.2}
        )

        # Add more instruments based on existing registry
        for preset_name in instrument_registry.presets.keys():
            if preset_name not in self.instrument_library:
                # Generate default characteristics based on category
                preset = instrument_registry.presets[preset_name]
                self._generate_default_characteristics(preset_name, preset)

    def _generate_default_characteristics(self, preset_name: str, preset: InstrumentPreset):
        """Generate default characteristics for instruments not explicitly defined."""
        characteristics = InstrumentCharacteristics()

        # Set timbre based on category
        if preset.category == InstrumentCategory.SYNTHETIC:
            if "bright" in preset.tags:
                characteristics.timbre = [TimbreType.BRIGHT]
            elif "warm" in preset.tags:
                characteristics.timbre = [TimbreType.WARM]
            else:
                characteristics.timbre = [TimbreType.WARM, TimbreType.BRIGHT]

        elif preset.category == InstrumentCategory.ETHNIC:
            characteristics.timbre = [TimbreType.WARM, TimbreType.DARK]

        elif preset.category == InstrumentCategory.WIND:
            if preset.subcategory == InstrumentSubcategory.BRASS:
                characteristics.timbre = [TimbreType.BRIGHT, TimbreType.WARM]
            else:
                characteristics.timbre = [TimbreType.WARM, TimbreType.MELLOW]

        # Set register based on instrument type
        if "bass" in preset_name.lower():
            characteristics.register = [RegisterType.LOW, RegisterType.MID_LOW]
        elif "lead" in preset_name.lower() or "high" in preset_name.lower():
            characteristics.register = [RegisterType.MID_HIGH, RegisterType.HIGH]
        else:
            characteristics.register = [RegisterType.MID_LOW, RegisterType.MID]

        # Set articulation
        if preset.category == InstrumentCategory.PERCUSSION:
            characteristics.articulation = [ArticulationType.STACCATO]
        else:
            characteristics.articulation = [ArticulationType.LEGATO, ArticulationType.STACCATO]

        # Set genre associations based on tags
        characteristics.genre_associations = preset.tags[:2] if preset.tags else ["general"]
        characteristics.mood_associations = ["neutral"]

        # Set spatial config
        characteristics.spatial_config = {
            "pan": random.uniform(-0.5, 0.5),
            "reverb_send": random.uniform(0.1, 0.4),
            "delay_send": random.uniform(0.0, 0.2)
        }

        self.instrument_library[preset_name] = characteristics

    def _initialize_arrangement_rules(self):
        """Initialize arrangement rules for different genres and moods."""
        self.arrangement_rules = {
            "electronic": {
                "max_instruments": 8,
                "spatial_distribution": "wide",
                "dynamic_balance": "layered",
                "reverb_profile": "digital_hall"
            },
            "jazz": {
                "max_instruments": 12,
                "spatial_distribution": "ensemble",
                "dynamic_balance": "expressive",
                "reverb_profile": "warm_room"
            },
            "rock": {
                "max_instruments": 6,
                "spatial_distribution": "stereo",
                "dynamic_balance": "powerful",
                "reverb_profile": "plate"
            },
            "classical": {
                "max_instruments": 20,
                "spatial_distribution": "orchestral",
                "dynamic_balance": "nuanced",
                "reverb_profile": "concert_hall"
            }
        }

    def select_instruments_for_genre(self, genre: str, mood: str, complexity: str) -> Dict[str, InstrumentPreset]:
        """Select appropriate instruments for genre, mood, and complexity."""
        available_presets = []

        # Find instruments that match genre and mood
        for preset_name, characteristics in self.instrument_library.items():
            if (genre in characteristics.genre_associations or
                mood in characteristics.mood_associations or
                characteristics.complexity_level == complexity):

                preset = instrument_registry.get_preset(preset_name)
                if preset:
                    available_presets.append(preset)

        # Limit based on genre rules
        max_instruments = self.arrangement_rules.get(genre, {}).get("max_instruments", 8)
        if len(available_presets) > max_instruments:
            available_presets = random.sample(available_presets, max_instruments)

        # Return as dictionary
        return {preset.name: preset for preset in available_presets}

    def configure_arrangement(self, instruments: Dict[str, InstrumentPreset], arrangement_type: str) -> Arrangement:
        """Configure how instruments are arranged in the mix."""
        arrangement = Arrangement(instruments=instruments, arrangement_type=arrangement_type)

        # Configure spatial positioning
        self._configure_spatial_positioning(arrangement)

        # Configure dynamics
        self._configure_dynamics(arrangement)

        # Configure reverb routing
        self._configure_reverb_routing(arrangement)

        return arrangement

    def _configure_spatial_positioning(self, arrangement: Arrangement):
        """Configure stereo positioning for instruments."""
        instruments_list = list(arrangement.instruments.keys())

        for i, instrument_name in enumerate(instruments_list):
            if instrument_name in self.instrument_library:
                characteristics = self.instrument_library[instrument_name]
                base_pan = characteristics.spatial_config.get("pan", 0.0)

                # Adjust pan based on position in arrangement
                if len(instruments_list) > 1:
                    position_factor = (i / (len(instruments_list) - 1)) - 0.5
                    adjusted_pan = base_pan + (position_factor * 0.3)
                    adjusted_pan = max(-1.0, min(1.0, adjusted_pan))
                else:
                    adjusted_pan = base_pan

                arrangement.spatial_config[instrument_name] = {
                    "pan": adjusted_pan,
                    "width": 0.8,
                    "position": i
                }

    def _configure_dynamics(self, arrangement: Arrangement):
        """Configure dynamic range and balance."""
        instruments_list = list(arrangement.instruments.keys())

        for i, instrument_name in enumerate(instruments_list):
            if instrument_name in self.instrument_library:
                characteristics = self.instrument_library[instrument_name]

                # Set volume based on role and characteristics
                base_volume = 0.7
                if "bass" in instrument_name.lower():
                    volume = min(base_volume + 0.1, characteristics.dynamic_range[1])
                elif "lead" in instrument_name.lower():
                    volume = min(base_volume + 0.2, characteristics.dynamic_range[1])
                else:
                    volume = base_volume

                arrangement.dynamic_config[instrument_name] = {
                    "volume": volume,
                    "compression": 0.3,
                    "dynamic_range": characteristics.dynamic_range,
                    "priority": len(instruments_list) - i  # Later instruments have lower priority
                }

    def _configure_reverb_routing(self, arrangement: Arrangement):
        """Configure reverb routing for instruments."""
        instruments_list = list(arrangement.instruments.keys())

        for instrument_name in instruments_list:
            if instrument_name in self.instrument_library:
                characteristics = self.instrument_library[instrument_name]

                arrangement.reverb_config[instrument_name] = {
                    "send_level": characteristics.spatial_config.get("reverb_send", 0.2),
                    "delay_send": characteristics.spatial_config.get("delay_send", 0.1),
                    "early_reflections": 0.3,
                    "decay_time": 1.5
                }

    def get_instrument_characteristics(self, instrument_name: str) -> Optional[InstrumentCharacteristics]:
        """Get characteristics for a specific instrument."""
        return self.instrument_library.get(instrument_name)

    def add_instrument_characteristics(self, instrument_name: str, characteristics: InstrumentCharacteristics):
        """Add or update characteristics for an instrument."""
        self.instrument_library[instrument_name] = characteristics

    def get_arrangement_template(self, genre: str, arrangement_type: str) -> Dict[str, Any]:
        """Get arrangement template for a specific genre and type."""
        return self.arrangement_rules.get(genre, {})


# Global instance
instrumentation_manager = InstrumentationManager()