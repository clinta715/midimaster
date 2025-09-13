#!/usr/bin/env python3
"""
Rock music generation demo for MIDI Master.

This script demonstrates how to generate a rock song with various moods.
"""

import sys
import os

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generators.pattern_orchestrator import PatternOrchestrator as PatternGenerator
from output.midi_output import MidiOutput
from structures.song_skeleton import SongSkeleton
from genres.genre_factory import GenreFactory


def generate_rock_song(mood="energetic", tempo=120, filename="rock_song.mid"):
    """
    Generate a rock song with the specified mood and tempo.
    
    Args:
        mood: Mood of the song ('happy', 'sad', 'energetic', 'calm')
        tempo: Tempo in BPM
        filename: Output MIDI file name
    """
    print(f"Generating rock song with {mood} mood at {tempo} BPM...")
    
    # Create genre-specific rules
    genre_rules = GenreFactory.create_genre_rules('rock')
    
    # Create song skeleton
    song_skeleton = SongSkeleton('rock', tempo, mood)
    
    # Generate patterns
    pattern_generator = PatternGenerator(genre_rules, mood)
    patterns = pattern_generator.generate_patterns(song_skeleton, 32)  # 32 bars
    
    # Build song arrangement
    song_skeleton.build_arrangement(patterns)
    
    # Output MIDI file
    midi_output = MidiOutput()
    midi_output.save_to_midi(song_skeleton, filename)
    
    print(f"Successfully generated {filename}")


if __name__ == "__main__":
    # Generate different rock songs with various moods
    generate_rock_song("energetic", 120, "rock_energetic.mid")
    generate_rock_song("sad", 80, "rock_sad.mid")
    generate_rock_song("happy", 140, "rock_happy.mid")
    generate_rock_song("calm", 90, "rock_calm.mid")