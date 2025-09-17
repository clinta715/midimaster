"""
Enhanced Plugin Parameter Enumeration System

This module provides advanced parameter enumeration capabilities for VST/CLAP plugins,
including detailed parameter metadata, categorization, and structured access.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum


class ParameterType(Enum):
    """Enumeration of different parameter types."""
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    BOOLEAN = "boolean"
    ENUMERATION = "enumeration"
    STRING = "string"
    GROUP = "group"


class ParameterCategory(Enum):
    """Standard parameter categories for automatic classification."""
    GENERIC = "generic"
    OSCILLATOR = "oscillator"
    FILTER = "filter"
    AMPLITUDE = "amplitude"
    ENVELOPE = "envelope"
    LFO = "lfo"
    EFFECTS = "effects"
    MIXER = "mixer"
    REVERB = "reverb"
    DELAY = "delay"
    DISTORTION = "distortion"
    EQ = "eq"
    MODULATION = "modulation"
    SYNTHESIS = "synthesis"
    MICROPHONE = "microphone"
    KEYBOARD = "keyboard"
    TRANSPORT = "transport"


@dataclass
class ParameterMetadata:
    """Extended metadata for plugin parameters."""
    name: str
    description: str = ""
    units: str = ""
    category: ParameterCategory = ParameterCategory.GENERIC
    subcategory: str = ""
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    default_value: Optional[float] = None
    is_automatable: bool = False
    parameter_type: ParameterType = ParameterType.CONTINUOUS
    enum_values: List[str] = field(default_factory=list)
    group_name: str = ""
    tags: List[str] = field(default_factory=list)
    preset_exclusions: List[str] = field(default_factory=list)


@dataclass
class ParameterInfo:
    """Complete parameter information including current value."""
    metadata: ParameterMetadata
    current_value: Any
    normalized_value: Optional[float] = None
    index: int = 0


class ParameterCollection:
    """Manages a collection of plugin parameters with enhanced access methods."""

    def __init__(self):
        self.parameters: Dict[str, ParameterInfo] = {}
        self.plugin_name: str = ""
        self.plugin_version: str = ""
        self.plugin_type: str = ""

    def add_parameter(self, param_info: ParameterInfo) -> None:
        """Add a parameter to the collection."""
        self.parameters[param_info.metadata.name] = param_info

    def get_parameter(self, name: str) -> Optional[ParameterInfo]:
        """Get a parameter by name."""
        return self.parameters.get(name)

    def get_parameters_by_category(self, category: ParameterCategory) -> List[ParameterInfo]:
        """Get all parameters in a specific category."""
        return [param for param in self.parameters.values()
                if param.metadata.category == category]

    def get_parameters_by_type(self, param_type: ParameterType) -> List[ParameterInfo]:
        """Get all parameters of a specific type."""
        return [param for param in self.parameters.values()
                if param.metadata.parameter_type == param_type]

    def get_categorized_parameters(self) -> Dict[str, List[ParameterInfo]]:
        """Get parameters organized by category."""
        categorized: Dict[str, List[ParameterInfo]] = {}
        for category in ParameterCategory:
            params = self.get_parameters_by_category(category)
            if params:
                categorized[category.value] = params
        return categorized

    def get_parameter_names(self) -> List[str]:
        """Get list of all parameter names."""
        return list(self.parameters.keys())

    def get_parameter_values(self) -> Dict[str, Any]:
        """Get current values of all parameters."""
        return {name: param.current_value for name, param in self.parameters.items()}

    def update_parameter_value(self, name: str, value: Any) -> bool:
        """Update the current value of a parameter."""
        if name in self.parameters:
            self.parameters[name].current_value = value
            return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the parameter collection to a dictionary."""
        return {
            "plugin_name": self.plugin_name,
            "plugin_version": self.plugin_version,
            "plugin_type": self.plugin_type,
            "parameters": {
                name: {
                    "metadata": {
                        "name": param.metadata.name,
                        "description": param.metadata.description,
                        "units": param.metadata.units,
                        "category": param.metadata.category.value,
                        "subcategory": param.metadata.subcategory,
                        "min_value": param.metadata.min_value,
                        "max_value": param.metadata.max_value,
                        "default_value": param.metadata.default_value,
                        "is_automatable": param.metadata.is_automatable,
                        "parameter_type": param.metadata.parameter_type.value,
                        "enum_values": param.metadata.enum_values,
                        "group_name": param.metadata.group_name,
                        "tags": param.metadata.tags,
                        "preset_exclusions": param.metadata.preset_exclusions
                    },
                    "current_value": param.current_value,
                    "normalized_value": param.normalized_value,
                    "index": param.index
                }
                for name, param in self.parameters.items()
            }
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ParameterCollection':
        """Deserialize a parameter collection from a dictionary."""
        collection = cls()
        collection.plugin_name = data.get("plugin_name", "")
        collection.plugin_version = data.get("plugin_version", "")
        collection.plugin_type = data.get("plugin_type", "")

        for name, param_data in data.get("parameters", {}).items():
            metadata_data = param_data["metadata"]
            metadata = ParameterMetadata(
                name=metadata_data["name"],
                description=metadata_data.get("description", ""),
                units=metadata_data.get("units", ""),
                category=ParameterCategory(metadata_data.get("category", "generic")),
                subcategory=metadata_data.get("subcategory", ""),
                min_value=metadata_data.get("min_value"),
                max_value=metadata_data.get("max_value"),
                default_value=metadata_data.get("default_value"),
                is_automatable=metadata_data.get("is_automatable", False),
                parameter_type=ParameterType(metadata_data.get("parameter_type", "continuous")),
                enum_values=metadata_data.get("enum_values", []),
                group_name=metadata_data.get("group_name", ""),
                tags=metadata_data.get("tags", []),
                preset_exclusions=metadata_data.get("preset_exclusions", [])
            )
            param_info = ParameterInfo(
                metadata=metadata,
                current_value=param_data["current_value"],
                normalized_value=param_data.get("normalized_value"),
                index=param_data.get("index", 0)
            )
            collection.parameters[name] = param_info

        return collection


class ParameterEnumerator:
    """Handles enumeration and metadata extraction for plugin parameters."""

    def __init__(self):
        self.category_keywords = {
            ParameterCategory.OSCILLATOR: ["osc", "wave", "frequency", "pitch", "detune"],
            ParameterCategory.FILTER: ["filter", "cutoff", "resonance", "q", "sweep"],
            ParameterCategory.AMPLITUDE: ["volume", "gain", "amplitude", "level"],
            ParameterCategory.ENVELOPE: ["attack", "decay", "sustain", "release", "adsr"],
            ParameterCategory.LFO: ["lfo", "modulation", "tremolo", "vibrato"],
            ParameterCategory.EFFECTS: ["effect", "fx", "delay", "reverb", "chorus"],
            ParameterCategory.EQ: ["eq", "equalizer", "high", "low", "mid"],
            ParameterCategory.SYNTHESIS: ["synth", "harmonic", "sawtooth", "square"]
        }

    def extract_metadata_from_parameter(self, name: str, value: Any,
                                       index: int = 0) -> ParameterMetadata:
        """Extract metadata from a parameter name and value."""
        metadata = ParameterMetadata(name=name)

        # Extract description from name
        metadata.description = self._generate_description(name)

        # Determine parameter type
        metadata.parameter_type = self._determine_parameter_type(value)

        # Categorize parameter
        metadata.category = self._categorize_parameter(name)

        # Extract units if possible
        metadata.units = self._extract_units(name)

        # Set automatable flag (most VST parameters are automatable)
        metadata.is_automatable = True

        return metadata

    def _generate_description(self, name: str) -> str:
        """Generate a human-readable description from parameter name."""
        cleaned = name.lower().replace("_", " ").replace("-", " ")
        # Add more sophisticated description generation
        if "freq" in cleaned or "pitch" in cleaned:
            return f"Controls the frequency/pitch setting"
        elif "gain" in cleaned or "volume" in cleaned:
            return f"Adjusts the audio level/gain"
        elif "cutoff" in cleaned:
            return f"Sets the filter cutoff frequency"
        elif "reson" in cleaned:
            return f"Controls filter resonance"
        elif "attack" in cleaned:
            return f"Sets the envelope attack time"
        elif "decay" in cleaned:
            return f"Sets the envelope decay time"
        elif "sustain" in cleaned:
            return f"Sets the envelope sustain level"
        elif "release" in cleaned:
            return f"Sets the envelope release time"
        else:
            return f"Parameter: {cleaned.title()}"

    def _determine_parameter_type(self, value: Any) -> ParameterType:
        """Determine parameter type based on value."""
        if isinstance(value, bool):
            return ParameterType.BOOLEAN
        elif isinstance(value, (int, float)):
            # Check if it's likely an enumeration (small positive integers)
            if isinstance(value, int) and 0 <= value <= 10:
                return ParameterType.ENUMERATION
            return ParameterType.CONTINUOUS
        elif isinstance(value, str):
            return ParameterType.STRING
        else:
            return ParameterType.CONTINUOUS

    def _categorize_parameter(self, name: str) -> ParameterCategory:
        """Categorize a parameter based on its name."""
        name_lower = name.lower()
        for category, keywords in self.category_keywords.items():
            for keyword in keywords:
                if keyword in name_lower:
                    return category
        return ParameterCategory.GENERIC

    def _extract_units(self, name: str) -> str:
        """Extract units from parameter name."""
        name_lower = name.lower()
        if "freq" in name_lower or "pitch" in name_lower:
            return "Hz"
        elif "level" in name_lower or "gain" in name_lower:
            return "dB"
        elif "time" in name_lower or "delay" in name_lower:
            return "ms"
        elif "rate" in name_lower:
            return "Hz"
        return ""

    def enumerate_parameters(self, plugin, plugin_name: str = "") -> ParameterCollection:
        """Enumerate all parameters from a plugin and create ParameterCollection."""
        collection = ParameterCollection()
        collection.plugin_name = plugin_name or getattr(plugin, 'name', 'Unknown Plugin')

        if hasattr(plugin, 'parameters'):
            try:
                params = dict(plugin.parameters)
                for index, (name, value) in enumerate(params.items()):
                    metadata = self.extract_metadata_from_parameter(name, value, index)
                    param_info = ParameterInfo(
                        metadata=metadata,
                        current_value=value,
                        index=index
                    )
                    collection.add_parameter(param_info)
            except Exception as e:
                print(f"Error enumerating parameters: {e}")

        return collection

    def get_category_hierarchy(self) -> Dict[str, List[str]]:
        """Get the hierarchy of parameter categories."""
        return {
            "audio_processing": [ParameterCategory.EQ.value, ParameterCategory.FILTER.value],
            "modulators": [ParameterCategory.ENVELOPE.value, ParameterCategory.LFO.value],
            "sound_generation": [ParameterCategory.OSCILLATOR.value, ParameterCategory.SYNTHESIS.value],
            "effects": [ParameterCategory.EFFECTS.value, ParameterCategory.REVERB.value,
                       ParameterCategory.DELAY.value, ParameterCategory.DISTORTION.value],
            "other": [ParameterCategory.GENERIC.value, ParameterCategory.MIXER.value]
        }