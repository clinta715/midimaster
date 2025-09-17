#!/usr/bin/env python3
"""
Pop music generation demo for MIDI Master.

This script demonstrates how to generate a pop song with various moods.
"""

import sys
import os

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generators.pattern_orchestrator import PatternOrchestrator as PatternGenerator
from output.midi_output import MidiOutput
from structures.song_skeleton import SongSkeleton
from genres.genre_factory import GenreFactory
from gui.config_manager import ConfigManager
from generators.generator_context import GeneratorContext


def generate_pop_song(mood="happy", tempo=120, filename="pop_song.mid", user_key=None, user_mode=None):
    """
    Generate a pop song with the specified mood and tempo.

    Args:
        mood: Mood of the song ('happy', 'sad', 'energetic', 'calm')
        tempo: Tempo in BPM
        filename: Output MIDI file name
        user_key: User-specified key (e.g., 'A')
        user_mode: User-specified mode (e.g., 'dorian')
    """
    print(f"Generating pop song with {mood} mood at {tempo} BPM...")
    if user_key and user_mode:
        print(f"Using user-specified key/mode: {user_key} {user_mode}")

    # Load configuration settings
    config_manager = ConfigManager()
    temp_settings = config_manager.load_temp_settings()

    # Override with provided key/mode if specified
    if user_key:
        temp_settings['user_key'] = user_key
    if user_mode:
        temp_settings['user_mode'] = user_mode

    # Create genre-specific rules
    genre_rules = GenreFactory.create_genre_rules('pop')

    # Create generator context and set user key/mode if specified
    context = GeneratorContext(
        genre_rules=genre_rules,
        mood=mood,
        note_density=0.5,
        rhythm_density=0.5,
        chord_density=0.5,
        bass_density=0.5
    )
    if temp_settings.get('user_key') and temp_settings.get('user_mode'):
        context.set_user_key_mode(temp_settings['user_key'], temp_settings['user_mode'])

    # Create song skeleton
    song_skeleton = SongSkeleton('pop', tempo, mood)

    # Generate patterns
    pattern_generator = PatternGenerator(genre_rules, mood, context=context)
    patterns = pattern_generator.generate_patterns(song_skeleton, 32)  # 32 bars

    # Build song arrangement
    song_skeleton.build_arrangement(patterns)

    # Output MIDI file
    midi_output = MidiOutput()
    midi_output.save_to_midi(song_skeleton, filename)

    print(f"Successfully generated {filename}")


if __name__ == "__main__":
    # Generate different pop songs with various moods
    generate_pop_song("happy", 120, "pop_happy.mid")
    generate_pop_song("sad", 80, "pop_sad.mid")
    generate_pop_song("energetic", 140, "pop_energetic.mid")
    generate_pop_song("calm", 90, "pop_calm.mid")

    # Generate pop song with A dorian key/mode for verification
    generate_pop_song("happy", 120, "pop_a_dorian.mid", user_key='A', user_mode='dorian')