"""
Audio Output Module for MIDI Master

This module is responsible for rendering MIDI data to audio files
using a loaded VST/CLAP plugin via the PluginHost.
"""

from typing import List
from structures.song_skeleton import SongSkeleton
from audio.plugin_host import PluginHost
import os


class AudioOutput:
    """
    Handles rendering of generated MIDI music to audio files.

    This class takes a SongSkeleton and a PluginHost instance to
    render the musical content into an audio file (e.g., WAV).
    """

    def __init__(self, plugin_host: PluginHost):
        """
        Initialize the AudioOutput.

        Args:
            plugin_host: An initialized PluginHost instance with a loaded plugin.
        """
        self.plugin_host = plugin_host

    def render_song_to_audio(self, song_skeleton: SongSkeleton, output_audio_path: str, midi_temp_path: str = "temp_midi_for_audio.mid") -> bool:
        """
        Renders the entire song (from SongSkeleton) to an audio file.

        Args:
            song_skeleton: The SongSkeleton object containing the musical content.
            output_audio_path: The desired path for the output audio file (e.g., "output.wav").
            midi_temp_path: Temporary path to save MIDI before rendering (default: temp_midi_for_audio.mid).

        Returns:
            True if rendering was successful, False otherwise.
        """
        if not self.plugin_host.loaded_plugin:
            print("Error: No plugin loaded in PluginHost. Cannot render audio.")
            return False

        # First, save the SongSkeleton to a temporary MIDI file
        # This requires MidiOutput class to be accessible or integrated
        # For simplicity, let's assume MidiOutput is available and can save to a temp file.
        from output.midi_output import MidiOutput
        midi_output = MidiOutput()
        try:
            midi_output.save_to_midi(song_skeleton, midi_temp_path)
            print(f"Temporary MIDI saved to: {midi_temp_path}")
        except Exception as e:
            print(f"Error saving temporary MIDI for audio rendering: {e}")
            return False

        # Determine the total duration of the song for rendering
        # This is a rough estimate. A more accurate duration would come from analyzing the MIDI file.
        total_duration_beats = 0
        for section_type, patterns in song_skeleton.sections.items():
            for pattern in patterns:
                for note in pattern.notes:
                    total_duration_beats = max(total_duration_beats, note.start_time + note.duration)
                for chord in pattern.chords:
                    for note in chord.notes:
                        total_duration_beats = max(total_duration_beats, note.start_time + note.duration)
        
        # Convert beats to seconds (assuming 120 BPM for a rough estimate if tempo not used)
        # A more accurate conversion would use the song_skeleton.tempo
        tempo_bpm = song_skeleton.tempo if song_skeleton.tempo > 0 else 120
        duration_seconds = (total_duration_beats / tempo_bpm) * 60
        # Add a little padding
        duration_seconds += 2.0

        # Now, use the PluginHost to render this temporary MIDI to audio
        success = self.plugin_host.render_midi_to_audio(midi_temp_path, output_audio_path, duration_seconds)

        # Clean up temporary MIDI file
        if os.path.exists(midi_temp_path):
            os.remove(midi_temp_path)
            print(f"Removed temporary MIDI file: {midi_temp_path}")

        return success


# Example Usage (requires a loaded plugin in PluginHost and a SongSkeleton)
if __name__ == "__main__":
    print("AudioOutput Module Test")
    print("=" * 40)

    # This test requires a running PluginHost with a loaded plugin
    # and a SongSkeleton object.
    # This part cannot be fully automated without a real plugin and MIDI data.
    print("This module requires external setup for full testing.")
    print("Please ensure dawdreamer is installed and a plugin is loaded in PluginHost.")

    # Example of how it would be used in main.py:
    # from audio.plugin_host import PluginHost
    # from output.audio_output import AudioOutput
    # from structures.song_skeleton import SongSkeleton
    #
    # plugin_host = PluginHost()
    # if plugin_host.load_plugin("/path/to/your/plugin.vst3"):
    #     audio_output = AudioOutput(plugin_host)
    #     # Assuming song_skeleton is already generated
    #     # song_skeleton = ...
    #     audio_output.render_song_to_audio(song_skeleton, "my_song.wav")
    # else:
    #     print("Failed to load plugin for audio output.")