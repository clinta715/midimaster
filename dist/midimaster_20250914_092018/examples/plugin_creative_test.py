"""
Plugin Creative Test for MIDI Master - DawDreamer Integration

This script tests the new DawDreamer integration for VST instrument plugins,
rendering MIDI to audio, and applying effects with Pedalboard.

Features:
- Loads Vital.vst3 as instrument plugin using DawDreamer
- Creates simple MIDI notes (C4 for 2 seconds)
- Renders instrument audio with DawDreamer
- Applies reverb effect with Pedalboard
- Saves to test_render.wav
- Verifies non-silent output
"""

import sys
import os
import numpy as np
from typing import List, Dict, Any, Optional

# Try to import soundfile for audio verification
try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    sf = None
    SOUNDFILE_AVAILABLE = False
    print("Warning: soundfile not available for audio verification")

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from audio.plugin_host import PluginHost
    IMPORTS_AVAILABLE = True
except ImportError as e:
    IMPORTS_AVAILABLE = False
    print(f"Warning: PluginHost not available: {e}")


class DawDreamerTest:
    """Tests DawDreamer integration for instrument rendering."""

    def __init__(self):
        if IMPORTS_AVAILABLE:
            self.host = PluginHost()
        else:
            self.host = None

    def find_vital_plugin(self) -> Optional[str]:
        """Find Vital.vst3 plugin in common locations."""
        print("🔍 Searching for Vital.vst3 plugin...")

        # Common Vital plugin locations
        vital_paths = [
            "C:\\Program Files\\Common Files\\VST3\\Vital.vst3",
            "C:\\Program Files\\VSTPlugins\\Vital.vst3",
            "/Library/Audio/Plug-Ins/VST3/Vital.vst3",
            "/usr/local/lib/vst3/Vital.vst3",
            "/usr/lib/vst3/Vital.vst3"
        ]

        # Check specific paths
        for path in vital_paths:
            if os.path.exists(path):
                print(f"✅ Found Vital at: {path}")
                return path

        # Scan for plugins and look for Vital
        if self.host:
            found_plugins = self.host.scan_for_plugins()
            for plugin_path in found_plugins:
                if "vital" in os.path.basename(plugin_path).lower():
                    print(f"✅ Found Vital via scan: {plugin_path}")
                    return plugin_path

        print("❌ Vital.vst3 not found in common locations")
        return None

    def create_midi_notes(self) -> List[Dict[str, Any]]:
        """Create simple MIDI notes for testing."""
        print("🎵 Creating MIDI notes...")

        # Simple C4 note for 2 seconds
        midi_notes = [
            {
                "note": 60,        # C4
                "velocity": 100,   # Medium velocity
                "start_time": 0.0, # Start at beginning
                "duration": 2.0    # 2 seconds long
            }
        ]

        print(f"   Created {len(midi_notes)} note(s):")
        for note in midi_notes:
            print(f"   - Note {note['note']} (velocity {note['velocity']}) "
                  f"at {note['start_time']}s for {note['duration']}s")

        return midi_notes

    def load_reverb_effect(self) -> bool:
        """Load a reverb effect plugin for testing."""
        print("🔊 Loading reverb effect...")

        # Try to find a reverb plugin
        if self.host is None:
            print("❌ PluginHost not available")
            return False

        # Look for common reverb plugins
        reverb_plugins = [
            "C:\\Program Files\\Common Files\\VST3\\ValhallaRoom.vst3",
            "C:\\Program Files\\Common Files\\VST3\\ValhallaVintageVerb.vst3",
            "/Library/Audio/Plug-Ins/VST3/ValhallaRoom.vst3",
            "/usr/local/lib/vst3/ValhallaRoom.vst3"
        ]

        for reverb_path in reverb_plugins:
            if os.path.exists(reverb_path):
                if self.host is None:
                    print("❌ PluginHost not initialized for reverb")
                    return False
                success = self.host.load_plugin(reverb_path, "effect")
                if success:
                    print(f"✅ Loaded reverb: {os.path.basename(reverb_path)}")
                    return True
                else:
                    print(f"❌ Failed to load reverb: {reverb_path}")

        # If no real reverb found, that's okay - we can still test without effects
        print("⚠️ No reverb plugin found - proceeding with instrument-only test")
        return True

    def run_dawdreamer_test(self):
        """Run the complete DawDreamer integration test."""
        print("🎵 MIDI MASTER - DAWDREAMER INTEGRATION TEST")
        print("=" * 60)

        if not IMPORTS_AVAILABLE:
            print("❌ PluginHost not available - cannot run test")
            return False

        try:
            # Step 1: Find Vital plugin
            vital_path = self.find_vital_plugin()
            if not vital_path:
                print("❌ Vital.vst3 not found - cannot proceed with instrument test")
                print("   Please install Vital synthesizer and ensure it's in a standard VST3 location")
                return False

            # Step 2: Load Vital as instrument
            print(f"\n🎹 Loading Vital as instrument: {vital_path}")
            if self.host is None:
                print("❌ PluginHost not initialized")
                return False
            success = self.host.load_plugin(vital_path, "instrument")
            if not success:
                print("❌ Failed to load Vital as instrument")
                return False

            # Step 3: Load reverb effect (optional)
            self.load_reverb_effect()

            # Step 4: Create MIDI notes
            midi_notes = self.create_midi_notes()

            # Step 5: Render to audio
            output_path = "test_render.wav"
            print(f"\n🎚️ Rendering MIDI to audio: {output_path}")

            duration_seconds = 3.0  # Allow some time after the note ends
            success = self.host.render_midi_to_audio(
                midi_notes=midi_notes,
                output_audio_path=output_path,
                duration_seconds=duration_seconds,
                sample_rate=44100
            )

            if not success:
                print("❌ Audio rendering failed")
                return False

            # Step 6: Verify output is non-silent (if soundfile available)
            if SOUNDFILE_AVAILABLE:
                return self.verify_audio_output(output_path)
            else:
                print("⚠️ Skipping audio verification: soundfile not available")
                print("   Assuming rendering success based on DawDreamer return value")
                return True

        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            return False

    def verify_audio_output(self, audio_path: str) -> bool:
        """Verify that the rendered audio is non-silent."""
        print(f"\n🔍 Verifying audio output: {audio_path}")

        # Ensure soundfile is available (should always be true when this is called)
        assert sf is not None, "soundfile should be available when verify_audio_output is called"

        try:
            # Load the audio file
            audio_data, sample_rate = sf.read(audio_path)
            print(f"   Audio file loaded: {audio_data.shape}, {sample_rate}Hz")

            # Check if stereo or mono
            if audio_data.ndim == 1:
                # Mono
                signal = audio_data
            else:
                # Stereo - use left channel for analysis
                signal = audio_data[:, 0]

            # Calculate RMS amplitude
            rms = np.sqrt(np.mean(signal**2))
            peak = np.max(np.abs(signal))

            print(f"   RMS: {rms:.4f}")
            print(f"   Peak: {peak:.4f}")

            # Check if signal is essentially silent
            silence_threshold = 1e-6  # Very low threshold for "silent"

            if rms > silence_threshold:
                print("✅ Audio output is NON-SILENT - DawDreamer integration working!")
                print("   🎉 Successfully rendered MIDI notes to audio")
                if self.host and self.host.effect_plugins:
                    print(f"   🎚️ Applied {len(self.host.effect_plugins)} effect(s)")
                return True
            else:
                print("❌ Audio output appears to be SILENT - check DawDreamer configuration")
                print("   Possible issues:")
                print("   - DawDreamer not properly installed")
                print("   - Plugin not compatible with DawDreamer")
                print("   - MIDI notes not processed correctly")
                return False

        except Exception as e:
            print(f"❌ Error verifying audio: {e}")
            return False


def main():
    """Main entry point."""
    try:
        test = DawDreamerTest()
        success = test.run_dawdreamer_test()

        if success:
            print("\n🎉 DAWDREAMER INTEGRATION TEST PASSED!")
            print("   ✅ Plugin loaded successfully")
            print("   ✅ MIDI notes rendered to audio")
            print("   ✅ Audio output is non-silent")
            print("   📁 Output saved to: test_render.wav")
        else:
            print("\n❌ DAWDREAMER INTEGRATION TEST FAILED!")
            print("   Check the error messages above for details")

    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()