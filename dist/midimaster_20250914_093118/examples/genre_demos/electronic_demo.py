#!/usr/bin/env python3
"""
Electronic music generation demo for MIDI Master.

This script demonstrates how to generate an electronic song with various moods.
"""

import sys
import os

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generators.pattern_orchestrator import PatternOrchestrator as PatternGenerator
from output.midi_output import MidiOutput
from structures.song_skeleton import SongSkeleton
from genres.genre_factory import GenreFactory


def generate_electronic_song(mood="energetic", tempo=128, filename="electronic_song.mid"):
    """
    Generate an electronic song with the specified mood and tempo.
    
    Args:
        mood: Mood of the song ('happy', 'sad', 'energetic', 'calm')
        tempo: Tempo in BPM
        filename: Output MIDI file name
    """
    print(f"Generating electronic song with {mood} mood at {tempo} BPM...")
    
    # Create genre-specific rules
    genre_rules = GenreFactory.create_genre_rules('electronic')
    
    # Create song skeleton
    song_skeleton = SongSkeleton('electronic', tempo, mood)
    
    # Generate patterns
    pattern_generator = PatternGenerator(genre_rules, mood)
    patterns = pattern_generator.generate_patterns(song_skeleton, 64)  # 64 bars
    
    # Build song arrangement
    song_skeleton.build_arrangement(patterns)
    
    # Output MIDI file
    midi_output = MidiOutput()
    midi_output.save_to_midi(song_skeleton, filename)
    
    print(f"Successfully generated {filename}")


if __name__ == "__main__":
    # Generate different electronic songs with various moods
    generate_electronic_song("energetic", 128, "electronic_energetic.mid")
    generate_electronic_song("calm", 100, "electronic_calm.mid")
    generate_electronic_song("happy", 130, "electronic_happy.mid")
    generate_electronic_song("sad", 90, "electronic_sad.mid")