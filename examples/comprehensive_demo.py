#!/usr/bin/env python3
"""
Comprehensive demo for MIDI Master.

This script demonstrates how to generate songs in all supported genres
with different moods and parameters.
"""

import sys
import os

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generators.pattern_orchestrator import PatternOrchestrator as PatternGenerator
from output.midi_output import MidiOutput
from structures.song_skeleton import SongSkeleton
from genres.genre_factory import GenreFactory


def generate_song(genre, mood="happy", tempo=120, bars=32, filename=None, separate_files=False):
    """
    Generate a song with the specified parameters.

    Args:
        genre: Music genre ('pop', 'rock', 'jazz', 'electronic', 'hip-hop', 'classical')
        mood: Mood of the song ('happy', 'sad', 'energetic', 'calm')
        tempo: Tempo in BPM
        bars: Number of bars to generate
        filename: Output MIDI file name (auto-generated if None)
        separate_files: If True, save patterns to separate MIDI files per instrument
    """
    if filename is None:
        filename = f"{genre}_{mood}.mid"
    
    print(f"Generating {genre} song with {mood} mood at {tempo} BPM...")
    if separate_files:
        print("  (Will save to separate MIDI files per instrument)")

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
    midi_output.save_to_midi(song_skeleton, filename, genre_rules, separate_files=separate_files)

    if separate_files:
        base_name = filename.replace('.mid', '')
        print(f"Successfully generated separate MIDI files:")
        print(f"  - {base_name}_melody.mid")
        print(f"  - {base_name}_harmony.mid")
        print(f"  - {base_name}_bass.mid")
        print(f"  - {base_name}_rhythm.mid")
    else:
        print(f"Successfully generated {filename}")


def main():
    """Generate example songs in all genres with different moods."""
    print("MIDI Master Comprehensive Demo")
    print("=" * 40)
    
    # Define genres and their typical parameters
    genres = [
        ('pop', 120, 32),
        ('rock', 120, 32),
        ('jazz', 120, 32),
        ('electronic', 128, 64),
        ('hip-hop', 90, 32),
        ('classical', 100, 64)
    ]
    
    # Define moods to generate for each genre
    moods = ['happy', 'sad', 'energetic', 'calm']
    
    # Generate songs for each genre and mood combination
    for genre, tempo, bars in genres:
        for mood in moods:
            filename = f"{genre}_{mood}.mid"
            try:
                generate_song(genre, mood, tempo, bars, filename)
            except Exception as e:
                print(f"Error generating {filename}: {e}")

    # Demonstrate separate files feature with jazz example
    print("\n" + "=" * 40)
    print("DEMONSTRATING SEPARATE FILES FEATURE")
    print("=" * 40)
    try:
        generate_song('jazz', 'energetic', 120, 32, 'jazz_demo_separate.mid', separate_files=True)
    except Exception as e:
        print(f"Error generating separate files demo: {e}")

    print("\nDemo completed! Check the generated MIDI files.")
    print("Note: The separate files feature creates individual MIDI files for each instrument:")
    print("  - melody.mid (melody line)")
    print("  - harmony.mid (chord accompaniment)")
    print("  - bass.mid (bass line)")
    print("  - rhythm.mid (percussion/rhythm)")


if __name__ == "__main__":
    main()