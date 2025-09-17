#!/usr/bin/env python3
"""
Jazz music generation demo for MIDI Master.

This script demonstrates how to generate a jazz song with various moods.
"""

import sys
import os

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generators.pattern_orchestrator import PatternOrchestrator as PatternGenerator
from output.midi_output import MidiOutput
from structures.song_skeleton import SongSkeleton
from genres.genre_factory import GenreFactory


def generate_jazz_song(mood="calm", tempo=120, filename="jazz_song.mid"):
    """
    Generate a jazz song with the specified mood and tempo.
    
    Args:
        mood: Mood of the song ('happy', 'sad', 'energetic', 'calm')
        tempo: Tempo in BPM
        filename: Output MIDI file name
    """
    print(f"Generating jazz song with {mood} mood at {tempo} BPM...")
    
    # Create genre-specific rules
    genre_rules = GenreFactory.create_genre_rules('jazz')
    
    # Create song skeleton
    song_skeleton = SongSkeleton('jazz', tempo, mood)
    
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
    # Generate different jazz songs with various moods
    generate_jazz_song("calm", 120, "jazz_calm.mid")
    generate_jazz_song("happy", 140, "jazz_happy.mid")
    generate_jazz_song("sad", 80, "jazz_sad.mid")
    generate_jazz_song("energetic", 160, "jazz_energetic.mid")