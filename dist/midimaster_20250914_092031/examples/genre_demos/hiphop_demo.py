#!/usr/bin/env python3
"""
Hip-hop music generation demo for MIDI Master.

This script demonstrates how to generate a hip-hop song with various moods.
"""

import sys
import os

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generators.pattern_orchestrator import PatternOrchestrator as PatternGenerator
from output.midi_output import MidiOutput
from structures.song_skeleton import SongSkeleton
from genres.genre_factory import GenreFactory


def generate_hiphop_song(mood="energetic", tempo=90, filename="hiphop_song.mid"):
    """
    Generate a hip-hop song with the specified mood and tempo.
    
    Args:
        mood: Mood of the song ('happy', 'sad', 'energetic', 'calm')
        tempo: Tempo in BPM
        filename: Output MIDI file name
    """
    print(f"Generating hip-hop song with {mood} mood at {tempo} BPM...")
    
    # Create genre-specific rules
    genre_rules = GenreFactory.create_genre_rules('hip-hop')
    
    # Create song skeleton
    song_skeleton = SongSkeleton('hip-hop', tempo, mood)
    
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
    # Generate different hip-hop songs with various moods
    generate_hiphop_song("energetic", 90, "hiphop_energetic.mid")
    generate_hiphop_song("calm", 80, "hiphop_calm.mid")
    generate_hiphop_song("happy", 95, "hiphop_happy.mid")
    generate_hiphop_song("sad", 75, "hiphop_sad.mid")