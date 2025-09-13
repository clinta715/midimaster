"""
Plugin Hosting Module for MIDI Master

This module provides functionality to host VST/CLAP audio plugins using DawDreamer for instruments and Pedalboard for effects.

NOTE: This module requires the 'dawdreamer' and 'pedalboard' libraries.
Install with: pip install dawdreamer pedalboard
"""

from typing import List, Dict, Any, Optional, Tuple
import os
import numpy as np
import soundfile as sf

try:
    from pedalboard import Plugin
    from pedalboard._pedalboard import Pedalboard, load_plugin
    PEDALBOARD_AVAILABLE = True
except ImportError:
    PEDALBOARD_AVAILABLE = False
    print("Warning: pedalboard library not available. Plugin hosting will not work.")

try:
    import dawdreamer as daw
    DAWDREAMER_AVAILABLE = True
except ImportError:
    DAWDREAMER_AVAILABLE = False
    print("Warning: dawdreamer library not available. Instrument rendering will not work.")

# Import mido for MIDI message handling
try:
    import mido
    import mido.messages
    MIDO_AVAILABLE = True
except ImportError:
    MIDO_AVAILABLE = False
    print("Warning: mido library not available. MIDI instrument rendering will not work.")

# Import enhanced enumeration system components
try:
    from .plugin_enumeration import (
        ParameterCollection,
        ParameterEnumerator,
        ParameterCategory,
        ParameterType
    )
    ENHANCED_ENUMERATION_AVAILABLE = True
except ImportError:
    ENHANCED_ENUMERATION_AVAILABLE = False
    print("Warning: Enhanced enumeration system not available.")


class PluginHost:
    """Manages loading, enumerating, and interacting with VST/CLAP plugins using DawDreamer for instruments and Pedalboard for effects."""

    def __init__(self):
        if not PEDALBOARD_AVAILABLE:
            raise ImportError("pedalboard library is required for plugin hosting.")
        self.board = Pedalboard()  # For effects chain
        self.loaded_plugin: Optional[Plugin] = None
        self.instrument_processors: List[Any] = []  # List of DawDreamer instrument processors
        self.effect_plugins: List[Plugin] = []  # List of loaded effect plugins
        self.engine = daw.RenderEngine(sample_rate=44100, block_size=512) if DAWDREAMER_AVAILABLE else None

    def classify_plugin_type(self, plugin_path: str) -> str:
        """
        Classify a plugin as an instrument or effect based on its path and name.

        Args:
            plugin_path: Path to the plugin file.

        Returns:
            "instrument" or "effect"
        """
        plugin_name = os.path.basename(plugin_path).lower()

        # Common instrument plugin keywords
        instrument_keywords = [
            "vital", "serum", "massive", "kontakt", "symphobia", "harmor",
            "tal", "tyrell", "nexus", "reaktor", "absynth", "fm8",
            "sylenth", "z3ta", "retrologue", "monark", "scarbee",
            "piano", "organ", "synth", "sampler", "drum", "bass",
            "guitar", "violin", "strings", "brass", "woodwind"
        ]

        # Common effect plugin keywords
        effect_keywords = [
            "reverb", "delay", "chorus", "flanger", "phaser", "distortion",
            "overdrive", "compressor", "limiter", "eq", "filter", "gate",
            "modulation", "pitch", "time", "dynamics", "spatial", "fx"
        ]

        # Check for instrument keywords
        for keyword in instrument_keywords:
            if keyword in plugin_name:
                return "instrument"

        # Check for effect keywords
        for keyword in effect_keywords:
            if keyword in plugin_name:
                return "effect"

        # Default to effect if unclear (safer for Pedalboard integration)
        return "effect"

    def load_plugin(self, plugin_path: str, plugin_type: Optional[str] = None) -> bool:
        """
        Loads a VST/CLAP plugin from the specified path.

        Args:
            plugin_path: Absolute path to the plugin file (.vst3, .clap, .dll, .so, .dylib)
            plugin_type: Optional override for plugin type ("instrument" or "effect")

        Returns:
            True if the plugin was loaded successfully, False otherwise.
        """
        if not os.path.exists(plugin_path):
            print(f"Error: Plugin file not found at {plugin_path}")
            return False

        # Determine plugin type if not specified
        if plugin_type is None:
            plugin_type = self.classify_plugin_type(plugin_path)

        try:
            if plugin_type == "instrument":
                if not DAWDREAMER_AVAILABLE or not self.engine:
                    print(f"Error: DawDreamer not available for instrument loading: {plugin_path}")
                    return False
                # Load instrument with DawDreamer
                processor = self.engine.make_plugin_processor("plugin_id", plugin_path)
                self.instrument_processors.append(processor)
                print(f"Successfully loaded instrument plugin with DawDreamer: {plugin_path}")
            else:  # effect
                # Load effect with Pedalboard
                plugin = load_plugin(plugin_path)
                self.effect_plugins.append(plugin)
                self.board.append(plugin)  # Add to effects chain
                print(f"Successfully loaded effect plugin: {plugin_path}")
                # Keep track of the last loaded plugin for backward compatibility
                self.loaded_plugin = plugin

            return True

        except Exception as e:
            print(f"Error loading plugin {plugin_path}: {e}")
            return False

    def scan_for_plugins(self, search_paths: Optional[List[str]] = None) -> List[str]:
        """
        Scans specified directories for VST/CLAP plugin files.

        Args:
            search_paths: List of directories to scan. If None, uses common default paths.

        Returns:
            A list of absolute paths to found plugin files.
        """
        found_plugins = []
        if search_paths is None:
            # Common default VST/CLAP plugin paths (platform-dependent)
            search_paths = [
                "C:\\Program Files\\VSTPlugins", # Windows
                "C:\\Program Files\\Common Files\\VST3", # Windows
                "/Library/Audio/Plug-Ins/VST", # macOS
                "/Library/Audio/Plug-Ins/VST3", # macOS
                "/usr/local/lib/vst", # Linux
                "/usr/lib/vst" # Linux
            ]

        for path in search_paths:
            if os.path.isdir(path):
                for root, _, files in os.walk(path):
                    for file in files:
                        # Pedalboard supports VST3, VST, AU (macOS), CLAP
                        if file.lower().endswith((".dll", ".vst3", ".vst", ".component", ".clap", ".so", ".dylib")):
                            full_path = os.path.join(root, file)
                            found_plugins.append(full_path)
        return found_plugins

    def get_plugin_parameters(self) -> Optional[Dict[str, Any]]:
        """
        Returns a dictionary of parameters for the currently loaded plugin.

        Returns:
            A dictionary where keys are parameter names and values are their current values,
            or None if no plugin is loaded.
        """
        if self.loaded_plugin:
            plugin_name = getattr(self.loaded_plugin, 'name', 'unknown')  # type: ignore[attr-defined]
            print(f"Debug: Accessing parameters on plugin {plugin_name}")
            if hasattr(self.loaded_plugin, 'parameters'):
                print(f"Debug: Parameters type: {type(self.loaded_plugin.parameters)}")  # type: ignore[attr-defined]
                try:
                    params = dict(self.loaded_plugin.parameters)  # type: ignore[attr-defined]
                    print(f"Debug: Retrieved {len(params)} parameters")
                    return params
                except Exception as e:
                    print(f"Debug: Error accessing parameters: {e}")
                    return None
            else:
                print("Debug: No parameters attribute")
                return {}
        return None

    def get_plugin_presets(self) -> Optional[List[str]]:
        """
        Returns a list of preset names for the currently loaded plugin.
        Pedalboard does not directly expose plugin presets in a generic way.
        This method is a placeholder.

        Returns:
            A list of preset names, or None if no plugin is loaded or presets are not supported.
        """
        print("Warning: Pedalboard does not directly expose plugin presets. This method is a placeholder.")
        return [] # Return empty list as presets are not directly accessible

    def load_plugin_preset(self, preset_name: str) -> bool:
        """
        Loads a specific preset by name for the currently loaded plugin.
        Pedalboard does not directly expose plugin presets in a generic way.
        This method is a placeholder.

        Args:
            preset_name: The name of the preset to load.

        Returns:
            True if the preset was loaded successfully, False otherwise.
        """
        print(f"Warning: Pedalboard does not directly support loading presets by name. Attempted to load: {preset_name}")
        return False # Cannot directly load presets by name

    def get_detailed_parameters(self) -> Optional["ParameterCollection"]:
        """
        Returns a detailed ParameterCollection for the currently loaded plugin.
        Uses the enhanced enumeration system to provide metadata and categorization.

        Returns:
            A ParameterCollection object with detailed parameter information,
            or None if no plugin is loaded or enhanced enumeration is not available.
        """
        if not ENHANCED_ENUMERATION_AVAILABLE:
            print("Warning: Enhanced enumeration system not available.")
            return None

        if not self.loaded_plugin:
            return None

        enumerator = ParameterEnumerator()
        plugin_name = getattr(self.loaded_plugin, 'name', 'unknown')
        return enumerator.enumerate_parameters(self.loaded_plugin, plugin_name)

    def get_parameters_by_category(self, category: str) -> List[str]:
        """
        Returns a list of parameter names filtered by category.
        Legacy method for backward compatibility.

        Args:
            category: The category to filter by.

        Returns:
            List of parameter names in the specified category.
        """
        detailed = self.get_detailed_parameters()
        if not detailed:
            return []

        try:
            category_enum = ParameterCategory(category.lower())
            params = detailed.get_parameters_by_category(category_enum)
            return [p.metadata.name for p in params]
        except ValueError:
            print(f"Warning: Unknown parameter category '{category}'")
            return []

    def get_enumeration_interface(self) -> Optional["ParameterEnumerator"]:
        """
        Returns the ParameterEnumerator instance for advanced enumeration capabilities.

        Returns:
            A ParameterEnumerator instance or None if enhanced enumeration is not available.
        """
        if not ENHANCED_ENUMERATION_AVAILABLE:
            print("Warning: Enhanced enumeration system not available.")
            return None
        return ParameterEnumerator()

    def render_instrument_audio(self, midi_notes: List[Dict[str, Any]], duration_seconds: float, sample_rate: int = 44100) -> np.ndarray:
        """
        Render audio from instrument plugin using DawDreamer with MIDI notes.

        Args:
            midi_notes: List of note dictionaries with keys: 'note', 'velocity', 'start_time', 'duration'
            duration_seconds: Total duration to render
            sample_rate: Sample rate for audio

        Returns:
            Audio buffer as numpy array
        """
        if not DAWDREAMER_AVAILABLE or not self.engine or not self.instrument_processors:
            print("Error: DawDreamer not available or no instrument processors loaded.")
            return np.zeros((int(duration_seconds * sample_rate), 2), dtype=np.float32)

        try:
            # Clear previous MIDI notes
            synth = self.instrument_processors[0]  # Use first instrument

            # Add MIDI notes
            for note in midi_notes:
                synth.add_midi_note(
                    note['note'],
                    note['velocity'],
                    note['start_time'],
                    note['start_time'] + note['duration']
                )

            # Render audio
            self.engine.render(duration_seconds)
            audio = self.engine.get_audio()

            # Convert to numpy array and ensure stereo
            audio_array = np.array(audio)
            if audio_array.ndim == 1:
                # Convert mono to stereo
                audio_array = np.column_stack([audio_array, audio_array])
            elif audio_array.shape[1] == 1:
                # Convert mono to stereo
                audio_array = np.column_stack([audio_array[:, 0], audio_array[:, 0]])

            return audio_array

        except Exception as e:
            print(f"Error rendering instrument audio with DawDreamer: {e}")
            # Return silence as fallback
            return np.zeros((int(duration_seconds * sample_rate), 2), dtype=np.float32)

    def render_midi_to_audio(self, midi_notes: List[Dict[str, Any]], output_audio_path: str,
                            duration_seconds: float, sample_rate: int = 44100) -> bool:
        """
        Renders MIDI notes to an audio file using loaded instrument and effect plugins.

        Args:
            midi_notes: List of note dictionaries with keys: 'note', 'velocity', 'start_time', 'duration'
            output_audio_path: Path for the output audio file (.wav).
            duration_seconds: Duration to render in seconds.
            sample_rate: Sample rate for audio output

        Returns:
            True if rendering was successful, False otherwise.
        """
        if not PEDALBOARD_AVAILABLE:
            print("Error: pedalboard not available for rendering.")
            return False

        if not self.instrument_processors:
            print("Error: No instrument processors loaded for rendering.")
            return False

        try:
            # Render instrument audio with DawDreamer
            instrument_audio = self.render_instrument_audio(midi_notes, duration_seconds, sample_rate)

            # Apply effects chain if any effects are loaded
            if self.effect_plugins:
                print(f"Applying {len(self.effect_plugins)} effect plugins...")
                final_audio = self.board(instrument_audio, sample_rate)
            else:
                final_audio = instrument_audio

            # Ensure we have the right shape (stereo)
            if final_audio.ndim == 1:
                # Convert mono to stereo
                final_audio = np.column_stack([final_audio, final_audio])
            elif final_audio.shape[1] == 1:
                # Convert mono to stereo
                final_audio = np.column_stack([final_audio[:, 0], final_audio[:, 0]])

            # Save the processed audio to a WAV file
            sf.write(output_audio_path, final_audio, sample_rate)

            print(f"Successfully rendered MIDI to audio: {output_audio_path}")
            return True

        except Exception as e:
            print(f"Error during MIDI to audio rendering: {e}")
            return False


# Example Usage (requires dawdreamer, pedalboard and a VST/CLAP plugin)
if __name__ == "__main__":
    print("PluginHost Module Test")
    print("=" * 40)

    host = PluginHost()

    # Replace with actual plugin paths on your system
    # For Windows: "C:\\Program Files\\VSTPlugins\\Synth.dll"
    # For macOS: "/Library/Audio/Plug-Ins/VST/Synth.vst"
    # For Linux: "/usr/lib/vst/Synth.so"
    example_plugin_path = "/path/to/your/synth.vst3" # Placeholder

    # Scan for plugins
    print("\nScanning for plugins...")
    found_plugins = host.scan_for_plugins()
    if found_plugins:
        print(f"Found {len(found_plugins)} plugins:")
        for p in found_plugins:
            print(f"- {p}")
        example_plugin_path = found_plugins[0] # Try to load the first found plugin
    else:
        print("No plugins found in common paths. Please specify a path manually.")

    # Load a plugin
    if host.load_plugin(example_plugin_path):
        print("Plugin loaded successfully.")

        # Get parameters (only works for effects loaded with pedalboard)
        if host.loaded_plugin:
            params = host.get_plugin_parameters()
            if params:
                print("\nPlugin Parameters:")
                for name, value in params.items():
                    print(f"- {name}: {value}")

        # Get presets (will show warning)
        presets = host.get_plugin_presets()
        if presets:
            print("\nPlugin Presets:")
            for preset in presets:
                print(f"- {preset}")

            # Load a preset (will show warning)
            if host.load_plugin_preset(presets[0]):
                print(f"Loaded preset: {presets[0]}")

        # Render MIDI to audio
        midi_notes = [
            {"note": 60, "velocity": 100, "start_time": 0.0, "duration": 1.0},  # C4
            {"note": 64, "velocity": 100, "start_time": 1.0, "duration": 1.0},  # E4
        ]
        output_audio_file = "rendered_audio.wav"
        print(f"\nAttempting to render audio to {output_audio_file}...")
        success = host.render_midi_to_audio(midi_notes, output_audio_file, duration_seconds=2.0)
        if success:
            print("Audio rendering successful!")
        else:
            print("Audio rendering failed.")
    else:
        print("Failed to load plugin.")

    print("\nPluginHost module test complete.")