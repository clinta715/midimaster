"""
Preset management system for instrument configurations.

This module provides functionality for:
- Loading and saving instrument presets from/to files
- Managing user-defined presets
- Preset validation and migration
- Preset sharing and import/export
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import asdict
import hashlib

from .instrument_categories import (
    InstrumentPreset,
    InstrumentLayer,
    InstrumentCategory,
    InstrumentSubcategory,
    instrument_registry
)


class PresetManager:
    """Manages loading, saving, and organizing instrument presets."""

    def __init__(self, preset_dir: Optional[str] = None):
        self.preset_dir = Path(preset_dir) if preset_dir else Path("instruments/presets")
        self.preset_dir.mkdir(parents=True, exist_ok=True)
        self.user_presets: Dict[str, InstrumentPreset] = {}
        self._load_user_presets()

    def _load_user_presets(self):
        """Load user-defined presets from disk."""
        preset_files = list(self.preset_dir.glob("*.json"))
        for preset_file in preset_files:
            try:
                with open(preset_file, 'r', encoding='utf-8') as f:
                    preset_data = json.load(f)
                    preset = self._deserialize_preset(preset_data)
                    if preset:
                        self.user_presets[preset.name] = preset
                        # Add to global registry
                        instrument_registry._add_preset(preset)
            except Exception as e:
                print(f"Error loading preset {preset_file}: {e}")

    def _deserialize_preset(self, data: Dict[str, Any]) -> Optional[InstrumentPreset]:
        """Deserialize a preset from JSON data."""
        try:
            # Convert string enums back to enum objects
            category = InstrumentCategory(data['category'])
            subcategory = InstrumentSubcategory(data['subcategory']) if data.get('subcategory') else None

            # Deserialize layers if present
            layers = []
            if data.get('layers'):
                for layer_data in data['layers']:
                    layer_preset = layer_data['preset']
                    # If preset is a string, keep as string; if dict, would need recursive deserialization
                    layers.append(InstrumentLayer(
                        preset=layer_preset,
                        volume=layer_data.get('volume', 1.0),
                        pan=layer_data.get('pan', 0.0),
                        transpose=layer_data.get('transpose', 0),
                        velocity_curve=layer_data.get('velocity_curve', 'linear'),
                        midi_channel=layer_data.get('midi_channel'),
                        parameters=layer_data.get('parameters', {})
                    ))

            return InstrumentPreset(
                name=data['name'],
                category=category,
                subcategory=subcategory,
                midi_program=data.get('midi_program'),
                midi_bank=data.get('midi_bank', 0),
                parameters=data.get('parameters', {}),
                description=data.get('description', ''),
                tags=data.get('tags', []),
                plugin_name=data.get('plugin_name'),
                plugin_path=data.get('plugin_path'),
                plugin_parameters=data.get('plugin_parameters', {}),
                layers=layers
            )
        except Exception as e:
            print(f"Error deserializing preset: {e}")
            return None

    def save_preset(self, preset: InstrumentPreset, overwrite: bool = False) -> bool:
        """Save a preset to disk."""
        if preset.name in self.user_presets and not overwrite:
            return False  # Preset exists and not overwriting

        try:
            preset_data = self._serialize_preset(preset)
            preset_file = self.preset_dir / f"{preset.name}.json"

            with open(preset_file, 'w', encoding='utf-8') as f:
                json.dump(preset_data, f, indent=2, ensure_ascii=False)

            self.user_presets[preset.name] = preset
            return True
        except Exception as e:
            print(f"Error saving preset {preset.name}: {e}")
            return False

    def _serialize_preset(self, preset: InstrumentPreset) -> Dict[str, Any]:
        """Serialize a preset to JSON-compatible data."""
        data = {
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
            "layers": []
        }

        # Serialize layers
        for layer in preset.layers:
            layer_data = {
                "preset": layer.preset if isinstance(layer.preset, str) else layer.preset.name,
                "volume": layer.volume,
                "pan": layer.pan,
                "transpose": layer.transpose,
                "velocity_curve": layer.velocity_curve,
                "midi_channel": layer.midi_channel,
                "parameters": layer.parameters
            }
            data["layers"].append(layer_data)

        return data

    def delete_preset(self, preset_name: str) -> bool:
        """Delete a user-defined preset."""
        if preset_name not in self.user_presets:
            return False

        try:
            preset_file = self.preset_dir / f"{preset_name}.json"
            if preset_file.exists():
                preset_file.unlink()

            # Remove from user presets and global registry
            del self.user_presets[preset_name]
            if preset_name in instrument_registry.presets:
                del instrument_registry.presets[preset_name]

            return True
        except Exception as e:
            print(f"Error deleting preset {preset_name}: {e}")
            return False

    def import_preset(self, preset_file: Path) -> Optional[InstrumentPreset]:
        """Import a preset from a file."""
        try:
            with open(preset_file, 'r', encoding='utf-8') as f:
                preset_data = json.load(f)

            preset = self._deserialize_preset(preset_data)
            if preset:
                self.save_preset(preset, overwrite=True)
                return preset
        except Exception as e:
            print(f"Error importing preset from {preset_file}: {e}")

        return None

    def export_preset(self, preset_name: str, export_file: Path) -> bool:
        """Export a preset to a file."""
        preset = instrument_registry.get_preset(preset_name)
        if not preset:
            return False

        try:
            preset_data = self._serialize_preset(preset)
            with open(export_file, 'w', encoding='utf-8') as f:
                json.dump(preset_data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Error exporting preset {preset_name}: {e}")
            return False

    def create_preset_from_template(self, template_name: str, new_name: str,
                                  modifications: Dict[str, Any] = {}) -> Optional[InstrumentPreset]:
        """Create a new preset based on an existing template."""
        template = instrument_registry.get_preset(template_name)
        if not template:
            return None

        # Create a copy of the template
        preset_dict = self._serialize_preset(template)
        preset_dict['name'] = new_name

        # Apply modifications to parameters
        if modifications:
            preset_dict.setdefault('parameters', {})
            preset_dict['parameters'].update(modifications)

        new_preset = self._deserialize_preset(preset_dict)
        if new_preset and self.save_preset(new_preset):
            return new_preset

        return None

    def get_preset_hash(self, preset_name: str) -> Optional[str]:
        """Get a hash of a preset for change detection."""
        preset = instrument_registry.get_preset(preset_name)
        if not preset:
            return None

        preset_data = self._serialize_preset(preset)
        # Remove name and description from hash as they're not functional changes
        hash_data = {k: v for k, v in preset_data.items() if k not in ['name', 'description']}
        data_str = json.dumps(hash_data, sort_keys=True)
        return hashlib.md5(data_str.encode()).hexdigest()

    def validate_preset(self, preset: InstrumentPreset) -> List[str]:
        """Validate a preset and return any issues found."""
        issues = []

        # Check required fields
        if not preset.name:
            issues.append("Preset name is required")
        if not preset.category:
            issues.append("Preset category is required")

        # Check MIDI program range
        if preset.midi_program is not None and not (0 <= preset.midi_program <= 127):
            issues.append("MIDI program must be between 0 and 127")

        # Check bank range
        if not (0 <= preset.midi_bank <= 127):
            issues.append("MIDI bank must be between 0 and 127")

        # Check layer volumes
        for i, layer in enumerate(preset.layers):
            if not (0.0 <= layer.volume <= 1.0):
                issues.append(f"Layer {i} volume must be between 0.0 and 1.0")
            if not (-1.0 <= layer.pan <= 1.0):
                issues.append(f"Layer {i} pan must be between -1.0 and 1.0")

        # Check for circular layer references
        if self._has_circular_reference(preset, set()):
            issues.append("Circular reference detected in layered preset")

        return issues

    def _has_circular_reference(self, preset: InstrumentPreset, visited: set) -> bool:
        """Check for circular references in layered presets."""
        if preset.name in visited:
            return True

        visited.add(preset.name)

        for layer in preset.layers:
            if isinstance(layer.preset, str):
                layer_preset = instrument_registry.get_preset(layer.preset)
                if layer_preset and self._has_circular_reference(layer_preset, visited.copy()):
                    return True
            elif isinstance(layer.preset, InstrumentPreset):
                if self._has_circular_reference(layer.preset, visited.copy()):
                    return True

        return False

    def get_user_presets(self) -> List[InstrumentPreset]:
        """Get all user-defined presets."""
        return list(self.user_presets.values())

    def get_preset_info(self, preset_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a preset."""
        preset = instrument_registry.get_preset(preset_name)
        if not preset:
            return None

        return {
            "name": preset.name,
            "category": preset.category.value,
            "subcategory": preset.subcategory.value if preset.subcategory else None,
            "description": preset.description,
            "tags": preset.tags,
            "midi_program": preset.midi_program,
            "is_layered": preset.is_layered,
            "layer_count": len(preset.layers),
            "has_plugin": bool(preset.plugin_name),
            "is_user_preset": preset_name in self.user_presets,
            "validation_issues": self.validate_preset(preset)
        }

    def search_user_presets(self, query: str) -> List[InstrumentPreset]:
        """Search user presets by name, description, or tags."""
        query_lower = query.lower()
        results = []

        for preset in self.user_presets.values():
            if (query_lower in preset.name.lower() or
                query_lower in preset.description.lower() or
                any(query_lower in tag.lower() for tag in preset.tags)):
                results.append(preset)

        return results


# Global preset manager instance
preset_manager = PresetManager()