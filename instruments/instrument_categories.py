"""
Comprehensive instrument category system for diverse instrument support.

This module provides a structured system for organizing instruments into categories,
supporting modern synths, ethnic percussion, wind instruments, and hybrid configurations.
"""

from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum


class InstrumentCategory(Enum):
    """Main instrument categories."""
    SYNTHETIC = "synthetic"
    PERCUSSION = "percussion"
    WIND = "wind"
    STRING = "string"
    ELECTRIC = "electric"
    ETHNIC = "ethnic"
    ACOUSTIC = "acoustic"


class InstrumentSubcategory(Enum):
    """Subcategories within main instrument categories."""

    # Synthetic subcategories
    ANALOG = "analog"
    DIGITAL = "digital"
    FM = "fm"
    SUBTRACTIVE = "subtractive"
    WAVETABLE = "wavetable"

    # Percussion subcategories
    DRUMS = "drums"
    CYMBALS = "cymbals"
    PERCUSSIVE = "percussive"

    # Wind subcategories
    BRASS = "brass"
    WOODWIND = "woodwind"
    REED = "reed"

    # String subcategories
    BOWED = "bowed"
    PLUCKED = "plucked"
    STRUCK = "struck"

    # Electric subcategories
    ELECTRIC_GUITAR = "electric_guitar"
    ELECTRIC_BASS = "electric_bass"
    ELECTRIC_KEYBOARD = "electric_keyboard"

    # Ethnic subcategories
    AFRICAN = "african"
    LATIN = "latin"
    MIDDLE_EASTERN = "middle_eastern"
    ASIAN = "asian"
    EUROPEAN_FOLK = "european_folk"


@dataclass
class InstrumentPreset:
    """Represents a specific instrument preset configuration."""
    name: str
    category: InstrumentCategory
    subcategory: Optional[InstrumentSubcategory] = None
    midi_program: Optional[int] = None
    midi_bank: int = 0
    parameters: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    tags: List[str] = field(default_factory=list)

    # Plugin-specific settings
    plugin_name: Optional[str] = None
    plugin_path: Optional[str] = None
    plugin_parameters: Dict[str, Any] = field(default_factory=dict)

    # Layering support
    layers: List['InstrumentLayer'] = field(default_factory=list)
    is_layered: bool = False

    def __post_init__(self):
        if self.layers and len(self.layers) > 1:
            self.is_layered = True


@dataclass
class InstrumentLayer:
    """Represents a single layer in a layered instrument configuration."""
    preset: Union[str, InstrumentPreset]  # Either preset name or preset object
    volume: float = 1.0
    pan: float = 0.0  # -1.0 to 1.0
    transpose: int = 0
    velocity_curve: str = "linear"  # linear, exponential, logarithmic
    midi_channel: Optional[int] = None
    parameters: Dict[str, Any] = field(default_factory=dict)


class InstrumentRegistry:
    """Central registry for all instrument presets and configurations."""

    def __init__(self):
        self.presets: Dict[str, InstrumentPreset] = {}
        self.categories: Dict[InstrumentCategory, List[str]] = {}
        self._initialize_default_presets()

    def _initialize_default_presets(self):
        """Initialize the default instrument presets."""

        # Modern Synth Presets
        self._add_preset(InstrumentPreset(
            name="808_kick",
            category=InstrumentCategory.SYNTHETIC,
            subcategory=InstrumentSubcategory.ANALOG,
            midi_program=35,  # Electric Bass (closest GM equivalent)
            description="Classic Roland TR-808 kick drum",
            tags=["kick", "808", "analog", "percussive"],
            parameters={
                "attack": 0.0,
                "decay": 0.3,
                "sustain": 0.0,
                "release": 0.1,
                "pitch": 36,
                "velocity_curve": "exponential"
            }
        ))

        self._add_preset(InstrumentPreset(
            name="808_snare",
            category=InstrumentCategory.SYNTHETIC,
            subcategory=InstrumentSubcategory.ANALOG,
            midi_program=40,  # Electric Snare
            description="Classic Roland TR-808 snare drum",
            tags=["snare", "808", "analog", "percussive"],
            parameters={
                "attack": 0.0,
                "decay": 0.2,
                "sustain": 0.0,
                "release": 0.3,
                "pitch": 40,
                "velocity_curve": "exponential"
            }
        ))

        self._add_preset(InstrumentPreset(
            name="fm_bass",
            category=InstrumentCategory.SYNTHETIC,
            subcategory=InstrumentSubcategory.FM,
            midi_program=38,  # Synth Bass 1
            description="FM synthesis bass sound",
            tags=["bass", "fm", "subtractive"],
            parameters={
                "algorithm": 1,
                "feedback": 0.5,
                "operator_ratios": [1.0, 2.0, 3.0, 4.0],
                "envelope_attack": 0.01,
                "envelope_decay": 0.3,
                "envelope_sustain": 0.7,
                "envelope_release": 0.2
            }
        ))

        self._add_preset(InstrumentPreset(
            name="fm_lead",
            category=InstrumentCategory.SYNTHETIC,
            subcategory=InstrumentSubcategory.FM,
            midi_program=80,  # Lead 1 (square)
            description="FM synthesis lead sound",
            tags=["lead", "fm", "bright"],
            parameters={
                "algorithm": 2,
                "feedback": 0.3,
                "operator_ratios": [1.0, 1.5, 2.0, 3.0],
                "envelope_attack": 0.0,
                "envelope_decay": 0.1,
                "envelope_sustain": 0.8,
                "envelope_release": 0.3
            }
        ))

        # Ethnic Percussion
        self._add_preset(InstrumentPreset(
            name="djembe",
            category=InstrumentCategory.ETHNIC,
            subcategory=InstrumentSubcategory.AFRICAN,
            midi_program=116,  # Taiko Drum (closest GM equivalent)
            description="West African djembe drum",
            tags=["djembe", "african", "percussion", "hand_drum"],
            parameters={
                "resonance": 0.6,
                "pitch": 50,
                "decay": 1.2,
                "articulation": "open"  # open, muted, slap
            }
        ))

        self._add_preset(InstrumentPreset(
            name="conga",
            category=InstrumentCategory.ETHNIC,
            subcategory=InstrumentSubcategory.LATIN,
            midi_program=117,  # Melodic Tom (closest)
            description="Cuban conga drum",
            tags=["conga", "latin", "percussion", "cuba"],
            parameters={
                "size": "medium",  # small, medium, large
                "pitch": 45,
                "decay": 0.8,
                "resonance": 0.4
            }
        ))

        self._add_preset(InstrumentPreset(
            name="bongo",
            category=InstrumentCategory.ETHNIC,
            subcategory=InstrumentSubcategory.LATIN,
            midi_program=118,  # Synth Drum
            description="Cuban bongo drums",
            tags=["bongo", "latin", "percussion", "pair"],
            parameters={
                "pitch_high": 58,
                "pitch_low": 50,
                "decay_high": 0.6,
                "decay_low": 0.8,
                "resonance": 0.3
            }
        ))

        self._add_preset(InstrumentPreset(
            name="doumbek",
            category=InstrumentCategory.ETHNIC,
            subcategory=InstrumentSubcategory.MIDDLE_EASTERN,
            midi_program=116,  # Taiko Drum
            description="Middle Eastern doumbek drum",
            tags=["doumbek", "middle_eastern", "percussion", "arabic"],
            parameters={
                "resonance": 0.7,
                "pitch": 48,
                "decay": 1.0,
                "articulation": "open"
            }
        ))

        self._add_preset(InstrumentPreset(
            name="taiko",
            category=InstrumentCategory.ETHNIC,
            subcategory=InstrumentSubcategory.ASIAN,
            midi_program=116,  # Taiko Drum
            description="Japanese taiko drum",
            tags=["taiko", "japanese", "percussion", "powerful"],
            parameters={
                "size": "large",
                "pitch": 42,
                "decay": 1.5,
                "resonance": 0.8
            }
        ))

        # Wind Instruments
        self._add_preset(InstrumentPreset(
            name="trumpet",
            category=InstrumentCategory.WIND,
            subcategory=InstrumentSubcategory.BRASS,
            midi_program=56,  # Trumpet
            description="Brass trumpet",
            tags=["trumpet", "brass", "wind", "bright"],
            parameters={
                "mute": "open",  # open, harmon, cup, bucket
                "pressure": 0.7,
                "breath_noise": 0.2,
                "vibrato_rate": 5.0,
                "vibrato_depth": 0.1
            }
        ))

        self._add_preset(InstrumentPreset(
            name="trombone",
            category=InstrumentCategory.WIND,
            subcategory=InstrumentSubcategory.BRASS,
            midi_program=57,  # Trombone
            description="Brass trombone",
            tags=["trombone", "brass", "wind", "smooth"],
            parameters={
                "slide_position": 1,  # 1-7 for trombone positions
                "pressure": 0.8,
                "breath_noise": 0.3,
                "vibrato_rate": 4.0,
                "vibrato_depth": 0.15
            }
        ))

        self._add_preset(InstrumentPreset(
            name="saxophone",
            category=InstrumentCategory.WIND,
            subcategory=InstrumentSubcategory.REED,
            midi_program=66,  # Tenor Sax
            description="Tenor saxophone",
            tags=["saxophone", "reed", "wind", "smooth"],
            parameters={
                "reed_hardness": 0.6,
                "mouthpiece": "medium",
                "breath_noise": 0.4,
                "vibrato_rate": 6.0,
                "vibrato_depth": 0.12
            }
        ))

        self._add_preset(InstrumentPreset(
            name="flute",
            category=InstrumentCategory.WIND,
            subcategory=InstrumentSubcategory.WOODWIND,
            midi_program=73,  # Flute
            description="Concert flute",
            tags=["flute", "woodwind", "wind", "airy"],
            parameters={
                "breath_noise": 0.3,
                "key_noise": 0.1,
                "vibrato_rate": 7.0,
                "vibrato_depth": 0.08,
                "dynamics": "expressive"
            }
        ))

        # Layered/Hybrid Instruments
        layered_pad = InstrumentPreset(
            name="layered_pad",
            category=InstrumentCategory.SYNTHETIC,
            subcategory=InstrumentSubcategory.DIGITAL,
            description="Layered pad with strings and synth",
            tags=["pad", "layered", "atmospheric", "hybrid"],
            layers=[
                InstrumentLayer(preset="string_ensemble", volume=0.7, pan=-0.3),
                InstrumentLayer(preset="warm_pad", volume=0.8, pan=0.3, transpose=7),
                InstrumentLayer(preset="fx_ambience", volume=0.4, pan=0.0)
            ]
        )
        self._add_preset(layered_pad)

        hybrid_bass = InstrumentPreset(
            name="hybrid_bass",
            category=InstrumentCategory.ELECTRIC,
            subcategory=InstrumentSubcategory.ELECTRIC_BASS,
            description="Electric bass with synth enhancement",
            tags=["bass", "hybrid", "electric", "enhanced"],
            layers=[
                InstrumentLayer(preset="electric_bass", volume=0.9, pan=0.0),
                InstrumentLayer(preset="synth_sub", volume=0.6, pan=0.0, transpose=-12)
            ]
        )
        self._add_preset(hybrid_bass)

    def _add_preset(self, preset: InstrumentPreset):
        """Add a preset to the registry."""
        self.presets[preset.name] = preset

        # Add to category index
        if preset.category not in self.categories:
            self.categories[preset.category] = []
        self.categories[preset.category].append(preset.name)

    def get_preset(self, name: str) -> Optional[InstrumentPreset]:
        """Get a preset by name."""
        return self.presets.get(name)

    def get_presets_by_category(self, category: InstrumentCategory) -> List[InstrumentPreset]:
        """Get all presets in a category."""
        preset_names = self.categories.get(category, [])
        return [self.presets[name] for name in preset_names if name in self.presets]

    def get_presets_by_tag(self, tag: str) -> List[InstrumentPreset]:
        """Get all presets with a specific tag."""
        return [preset for preset in self.presets.values() if tag in preset.tags]

    def search_presets(self, query: str) -> List[InstrumentPreset]:
        """Search presets by name, description, or tags."""
        query_lower = query.lower()
        results = []

        for preset in self.presets.values():
            if (query_lower in preset.name.lower() or
                query_lower in preset.description.lower() or
                any(query_lower in tag.lower() for tag in preset.tags)):
                results.append(preset)

        return results

    def create_layered_preset(self, name: str, layers: List[InstrumentLayer],
                            description: str = "", tags: Optional[List[str]] = None) -> InstrumentPreset:
        """Create a new layered preset."""
        if tags is None:
            tags = ["layered", "custom"]

        preset = InstrumentPreset(
            name=name,
            category=InstrumentCategory.SYNTHETIC,  # Default to synthetic for layered
            subcategory=InstrumentSubcategory.DIGITAL,
            description=description,
            tags=tags,
            layers=layers
        )

        self._add_preset(preset)
        return preset

    def get_midi_program_for_preset(self, preset_name: str) -> Optional[int]:
        """Get MIDI program number for a preset."""
        preset = self.get_preset(preset_name)
        return preset.midi_program if preset else None

    def get_parameters_for_preset(self, preset_name: str) -> Dict[str, Any]:
        """Get parameters for a preset."""
        preset = self.get_preset(preset_name)
        return preset.parameters if preset else {}


# Global registry instance
instrument_registry = InstrumentRegistry()


def get_instrument_categories() -> Dict[str, List[str]]:
    """Get all instrument categories and their presets."""
    return {
        category.value: instrument_registry.categories.get(category, [])
        for category in InstrumentCategory
    }


def get_preset_config(preset_name: str) -> Optional[Dict[str, Any]]:
    """Get complete configuration for a preset."""
    preset = instrument_registry.get_preset(preset_name)
    if not preset:
        return None

    return {
        "name": preset.name,
        "category": preset.category.value,
        "subcategory": preset.subcategory.value if preset.subcategory else None,
        "midi_program": preset.midi_program,
        "midi_bank": preset.midi_bank,
        "parameters": preset.parameters,
        "description": preset.description,
        "tags": preset.tags,
        "plugin_name": preset.plugin_name,
        "plugin_path": preset.plugin_path,
        "plugin_parameters": preset.plugin_parameters,
        "is_layered": preset.is_layered,
        "layers": [
            {
                "preset": layer.preset if isinstance(layer.preset, str) else layer.preset.name,
                "volume": layer.volume,
                "pan": layer.pan,
                "transpose": layer.transpose,
                "velocity_curve": layer.velocity_curve,
                "midi_channel": layer.midi_channel,
                "parameters": layer.parameters
            }
            for layer in preset.layers
        ] if preset.layers else []
    }