#!/usr/bin/env python3
"""
Simple test for DnB genre implementation.
"""

import sys
import os

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generators.pattern_orchestrator import PatternOrchestrator as PatternGenerator
from output.midi_output import MidiOutput
from structures.song_skeleton import SongSkeleton
from genres.genre_factory import GenreFactory


def test_dnb():
    """Test DnB genre generation."""
    print("Testing DnB genre implementation...")

    try:
        # Create DnB rules
        genre_rules = GenreFactory.create_genre_rules('dnb')
        print("✅ Successfully created DnB rules")

        # Test the methods
        scales = genre_rules.get_scales()
        print(f"✅ Scales: {scales[:3]}...")  # Show first 3

        progressions = genre_rules.get_chord_progressions()
        print(f"✅ Chord progressions: {len(progressions)} available")

        rhythm_patterns = genre_rules.get_rhythm_patterns()
        print(f"✅ Rhythm patterns: {len(rhythm_patterns)} available")
        print(f"  - Pattern names: {[p['name'] for p in rhythm_patterns]}")

        beat_chars = genre_rules.get_beat_characteristics()
        print(f"✅ Beat characteristics: tempo {beat_chars['tempo_range']} BPM")

        # Generate a short DnB song
        print("\nGenerating short DnB MIDI...")
        generate_song('dnb', 'energetic', 170, 16, 'test_dnb.mid')

        print("✅ DnB test completed successfully!")

    except Exception as e:
        print(f"❌ DnB test failed: {e}")
        import traceback
        traceback.print_exc()


def generate_song(genre, mood="energetic", tempo=170, bars=16, filename="test_dnb.mid"):
    """
    Generate a song with the specified parameters.
    """
    print(f"Generating {genre} song with {mood} mood at {tempo} BPM...")

    # Create genre-specific rules
    genre_rules = GenreFactory.create_genre_rules(genre)

    # Create song skeleton
    song_skeleton = SongSkeleton(genre, tempo, mood)

    # Generate patterns
    pattern_generator = PatternGenerator(genre_rules, mood)
    patterns = pattern_generator.generate_patterns(song_skeleton, bars)

    # Build song arrangement
    song_skeleton.build_arrangement(patterns)

    # Output MIDI file
    midi_output = MidiOutput()
    midi_output.save_to_midi(song_skeleton, filename)

    print(f"Successfully generated {filename}")


if __name__ == "__main__":
    test_dnb()