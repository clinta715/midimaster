"""
Instrument-aware MIDI output system.

This module extends the basic MIDI output functionality to support:
- Modern synths (808, FM, analog, digital)
- Ethnic percussion instruments
- Wind instruments (brass, woodwind, reed)
- Layered and hybrid instrument configurations
- Instrument-specific parameter handling
"""

import random
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from collections import defaultdict

try:
    import mido
    MIDO_AVAILABLE = True
except ImportError:
    MIDO_AVAILABLE = False
    print("Warning: mido library not available. MIDI output will not work.")

from .instrument_categories import (
    instrument_registry,
    InstrumentPreset,
    InstrumentLayer,
    get_preset_config,
    InstrumentCategory
)
from ..structures.data_structures import Note, Chord, Pattern, PatternType
from ..structures.song_skeleton import SongSkeleton
from ..genres.genre_rules import GenreRules


class InstrumentMidiOutput:
    """Enhanced MIDI output with instrument system integration."""

    def __init__(self):
        if not MIDO_AVAILABLE:
            raise ImportError("mido library is required for MIDI output")

        # Track current instruments per channel for layered presets
        self.channel_instruments: Dict[int, InstrumentPreset] = {}
        self.layer_channels: Dict[str, List[int]] = {}  # preset_name -> [channels]

    def assign_instrument_to_pattern(self, pattern_type: PatternType,
                                   preferred_instruments: Optional[List[str]] = None) -> str:
        """
        Assign an appropriate instrument preset to a pattern type.

        Args:
            pattern_type: The type of pattern (melody, harmony, bass, rhythm)
            preferred_instruments: Optional list of preferred instrument names

        Returns:
            Name of the assigned instrument preset
        """
        if preferred_instruments:
            for instrument_name in preferred_instruments:
                if instrument_registry.get_preset(instrument_name):
                    return instrument_name

        # Default instrument assignments based on pattern type
        defaults = {
            PatternType.MELODY: ["fm_lead", "warm_pad", "electric_guitar"],
            PatternType.HARMONY: ["layered_pad", "string_ensemble", "piano"],
            PatternType.BASS: ["808_kick", "fm_bass", "hybrid_bass"],
            PatternType.RHYTHM: ["808_snare", "drum_machine", "conga"]
        }

        available_defaults = defaults.get(pattern_type, ["grand_piano"])

        for instrument_name in available_defaults:
            if instrument_registry.get_preset(instrument_name):
                return instrument_name

        # Fallback to first available preset
        if instrument_registry.presets:
            return next(iter(instrument_registry.presets.keys()))

        return "grand_piano"  # Ultimate fallback

    def setup_instrument_for_channel(self, track, channel: int,
                                   instrument_name: str, current_time: int = 0) -> int:
        """
        Set up an instrument preset for a specific MIDI channel.

        Args:
            track: MIDI track to add setup messages to
            channel: MIDI channel (0-15)
            instrument_name: Name of instrument preset to use
            current_time: Current time position in ticks

        Returns:
            Updated current time after setup messages
        """
        preset = instrument_registry.get_preset(instrument_name)
        if not preset:
            print(f"Warning: Instrument preset '{instrument_name}' not found")
            return current_time

        # Store instrument assignment for this channel
        self.channel_instruments[channel] = preset

        # Add program change
        if preset.midi_program is not None:
            track.append(mido.Message(
                'program_change',
                channel=channel,
                program=preset.midi_program,
                time=current_time
            ))
            current_time = 0  # Reset for subsequent messages

        # Add bank select if needed
        if preset.midi_bank > 0:
            track.append(mido.Message(
                'control_change',
                channel=channel,
                control=0,  # Bank select MSB
                value=preset.midi_bank // 128,
                time=current_time
            ))
            current_time = 0

            track.append(mido.Message(
                'control_change',
                channel=channel,
                control=32,  # Bank select LSB
                value=preset.midi_bank % 128,
                time=0
            ))

        # Add instrument-specific parameter setup
        current_time = self._setup_instrument_parameters(track, channel, preset, current_time)

        return current_time

    def _setup_instrument_parameters(self, track, channel: int,
                                   preset: InstrumentPreset, current_time: int) -> int:
        """Set up instrument-specific parameters via MIDI CC messages."""
        params = preset.parameters

        # Common parameter mappings (CC numbers)
        param_mappings = {
            "attack": 73,      # Attack time
            "decay": 75,       # Decay time
            "sustain": 79,     # Sustain level
            "release": 72,     # Release time
            "cutoff": 74,      # Filter cutoff
            "resonance": 71,   # Filter resonance
            "volume": 7,       # Channel volume
            "pan": 10,         # Pan position
            "reverb": 91,      # Reverb send
            "chorus": 93,      # Chorus send
            "modulation": 1,   # Modulation wheel
        }

        for param_name, value in params.items():
            if param_name in param_mappings:
                cc_number = param_mappings[param_name]

                # Normalize value to 0-127 range if needed
                if isinstance(value, float):
                    if 0.0 <= value <= 1.0:
                        midi_value = int(value * 127)
                    else:
                        midi_value = int(value)
                else:
                    midi_value = int(value)

                midi_value = max(0, min(127, midi_value))

                track.append(mido.Message(
                    'control_change',
                    channel=channel,
                    control=cc_number,
                    value=midi_value,
                    time=current_time
                ))
                current_time = 0

        return current_time

    def setup_layered_instrument(self, track, base_channel: int,
                               preset_name: str, current_time: int = 0) -> Tuple[List[int], int]:
        """
        Set up a layered instrument across multiple channels.

        Args:
            track: MIDI track to add setup messages to
            base_channel: Starting MIDI channel for layers
            preset_name: Name of layered instrument preset
            current_time: Current time position in ticks

        Returns:
            Tuple of (list of channels used, updated current_time)
        """
        preset = instrument_registry.get_preset(preset_name)
        if not preset or not preset.layers:
            # Fall back to single channel setup
            current_time = self.setup_instrument_for_channel(track, base_channel, preset_name, current_time)
            return [base_channel], current_time

        channels_used = []

        for i, layer in enumerate(preset.layers):
            channel = base_channel + i
            if channel > 15:  # MIDI channel limit
                print(f"Warning: Exceeded MIDI channel limit for layered instrument {preset_name}")
                break

            channels_used.append(channel)

            # If layer is a string, assume it's a preset name
            if isinstance(layer.preset, str):
                layer_preset_name = layer.preset
            elif hasattr(layer.preset, 'name'):
                layer_preset_name = layer.preset.name
            else:
                continue

            # Set up the layer instrument
            current_time = self.setup_instrument_for_channel(track, channel, layer_preset_name, current_time)

            # Apply layer-specific adjustments
            if layer.volume != 1.0:
                volume_value = max(0, min(127, int(layer.volume * 127)))
                track.append(mido.Message(
                    'control_change',
                    channel=channel,
                    control=7,  # Volume
                    value=volume_value,
                    time=current_time
                ))
                current_time = 0

            if layer.pan != 0.0:
                # Convert from -1.0..1.0 to 0..127
                pan_value = max(0, min(127, int((layer.pan + 1.0) * 63.5)))
                track.append(mido.Message(
                    'control_change',
                    channel=channel,
                    control=10,  # Pan
                    value=pan_value,
                    time=current_time
                ))
                current_time = 0

        # Store channel mapping for this layered preset
        self.layer_channels[preset_name] = channels_used

        return channels_used, current_time

    def get_channel_for_pattern(self, pattern: Pattern, base_channel: int = 0) -> Tuple[int, str]:
        """
        Determine the appropriate channel and instrument for a pattern.

        Args:
            pattern: The pattern to analyze
            base_channel: Base channel to start from

        Returns:
            Tuple of (channel, instrument_name)
        """
        pattern_type = getattr(pattern, 'pattern_type', PatternType.MELODY)
        instrument_name = self.assign_instrument_to_pattern(pattern_type)

        # Check if this is a layered instrument that needs multiple channels
        preset = instrument_registry.get_preset(instrument_name)
        if preset and preset.is_layered:
            # Layered instruments get a range of channels
            return base_channel, instrument_name

        return base_channel, instrument_name

    def add_pattern_with_instrument(self, track, pattern: Pattern,
                                  base_channel: int, current_time: int,
                                  genre_rules: Optional[GenreRules] = None) -> int:
        """
        Add a pattern to track with appropriate instrument setup.

        Args:
            track: MIDI track to add to
            pattern: Pattern to add
            base_channel: Base MIDI channel to use
            current_time: Current time position in ticks
            genre_rules: Optional genre rules for swing etc.

        Returns:
            Updated current time
        """
        channel, instrument_name = self.get_channel_for_pattern(pattern, base_channel)

        # Set up instrument for this pattern
        preset = instrument_registry.get_preset(instrument_name)
        if preset and preset.is_layered:
            channels_used, current_time = self.setup_layered_instrument(
                track, channel, instrument_name, current_time
            )
            # For layered instruments, distribute notes across channels
            current_time = self._add_pattern_to_layered_track(
                track, pattern, channels_used, current_time, genre_rules
            )
        else:
            current_time = self.setup_instrument_for_channel(
                track, channel, instrument_name, current_time
            )
            current_time = self._add_pattern_to_single_channel(
                track, pattern, channel, current_time, genre_rules
            )

        return current_time

    def _add_pattern_to_single_channel(self, track, pattern: Pattern,
                                     channel: int, current_time: int,
                                     genre_rules: Optional[GenreRules] = None) -> int:
        """Add pattern to a single MIDI channel."""
        # This would contain the logic from the original MidiOutput._add_pattern_to_track
        # but simplified for single channel operation
        # For now, return current_time unchanged
        return current_time

    def _add_pattern_to_layered_track(self, track, pattern: Pattern,
                                    channels: List[int], current_time: int,
                                    genre_rules: Optional[GenreRules] = None) -> int:
        """Add pattern to multiple channels for layered instrument."""
        # Distribute pattern elements across available channels
        # This is a simplified implementation - in practice you'd want
        # more sophisticated distribution logic based on the layer configuration
        channel_index = 0

        # Process notes
        for note in getattr(pattern, 'notes', []) or []:
            channel_to_use = channels[channel_index % len(channels)]
            # Add note to specific channel
            # (Note: actual implementation would need the full note processing logic)
            channel_index += 1

        # Process chords
        for chord in getattr(pattern, 'chords', []) or []:
            channel_to_use = channels[channel_index % len(channels)]
            # Add chord to specific channel
            # (Note: actual implementation would need the full chord processing logic)
            channel_index += 1

        return current_time

    def get_instrument_info(self, preset_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about an instrument preset."""
        return get_preset_config(preset_name)


# Convenience functions for easy integration
def get_available_instruments(category: Optional[str] = None) -> List[str]:
    """Get list of available instrument presets, optionally filtered by category."""
    if category:
        try:
            cat_enum = InstrumentCategory(category)
            presets = instrument_registry.get_presets_by_category(cat_enum)
            return [p.name for p in presets]
        except ValueError:
            return []

    return list(instrument_registry.presets.keys())


def get_instrument_categories() -> Dict[str, List[str]]:
    """Get all instrument categories and their presets."""
    from .instrument_categories import get_instrument_categories
    return get_instrument_categories()


def validate_instrument_setup(preset_name: str) -> List[str]:
    """Validate an instrument preset setup."""
    preset = instrument_registry.get_preset(preset_name)
    if not preset:
        return [f"Preset '{preset_name}' not found"]

    from .preset_manager import preset_manager
    return preset_manager.validate_preset(preset)