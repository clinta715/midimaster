#!/usr/bin/env python3
"""
Classical music generation demo for MIDI Master.

This script demonstrates how to generate a classical song with various moods.
"""

import sys
import os

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generators.pattern_orchestrator import PatternOrchestrator as PatternGenerator
from output.midi_output import MidiOutput
from structures.song_skeleton import SongSkeleton
from genres.genre_factory import GenreFactory


def generate_classical_song(mood="calm", tempo=100, filename="classical_song.mid"):
    """
    Generate a classical song with the specified mood and tempo.
    
    Args:
        mood: Mood of the song ('happy', 'sad', 'energetic', 'calm')
        tempo: Tempo in BPM
        filename: Output MIDI file name
    """
    print(f"Generating classical song with {mood} mood at {tempo} BPM...")
    
    # Create genre-specific rules
    genre_rules = GenreFactory.create_genre_rules('classical')
    
    # Create song skeleton
    song_skeleton = SongSkeleton('classical', tempo, mood)
    
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
    # Generate different classical songs with various moods
    generate_classical_song("calm", 100, "classical_calm.mid")
    generate_classical_song("happy", 120, "classical_happy.mid")
    generate_classical_song("sad", 80, "classical_sad.mid")
    generate_classical_song("energetic", 140, "classical_energetic.mid")