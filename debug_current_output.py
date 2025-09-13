#!/usr/bin/env python3
"""
Debug script to analyze current music generation output.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from generators.pattern_orchestrator import PatternOrchestrator as PatternGenerator
from structures.song_skeleton import SongSkeleton
from genres.genre_factory import GenreFactory
from music_theory import MusicTheory

def analyze_current_output():
    """Analyze the current pattern generation to identify issues."""
    print("=== ANALYZING CURRENT PATTERN GENERATION ===")

    # Create genre rules
    genre_rules = GenreFactory.create_genre_rules('pop')
    all_genre_rules = genre_rules.get_rules()
    print(f"Genre rules available: {list(all_genre_rules.keys())}")
    print(f"Available scales: {all_genre_rules['scales'][:5]}...")
    print(f"Available chord progressions: {all_genre_rules['chord_progressions'][:3]}...")

    # Create song skeleton
    song_skeleton = SongSkeleton('pop', 120, 'happy')

    # Generate patterns with debug output
    pattern_generator = PatternGenerator(genre_rules, 'happy')
    patterns = pattern_generator.generate_patterns(song_skeleton, 4)

    # Print debug info about key/scale configuration after pattern generation
    print(f"Context initialized - Key: {pattern_generator.context.current_key}, Scale: {pattern_generator.context.current_scale}, Pitch count: {len(pattern_generator.context.scale_pitches)}")

    # Get the actual key and scale used
    actual_key = pattern_generator.current_key
    actual_scale = pattern_generator.current_scale
    actual_scale_pitches = pattern_generator.scale_pitches

    print(f"\nGenerated in key: {actual_key} {actual_scale}")
    print(f"Scale pitches used: {actual_scale_pitches}")

    print(f"\nGenerated {len(patterns)} patterns:")
    for i, pattern in enumerate(patterns):
        print(f"Pattern {i+1}: {pattern}")
        print(f"  Notes: {len(pattern.notes)}")
        print(f"  Chords: {len(pattern.chords)}")

        if pattern.notes:
            pitches = [note.pitch for note in pattern.notes]
            print(f"  Note pitches: {pitches[:10]}{'...' if len(pitches) > 10 else ''}")

            # Check if pitches are in the actual scale being used
            if actual_scale_pitches:
                # Convert to pitch classes (mod 12) for comparison
                scale_pitch_classes = [p % 12 for p in actual_scale_pitches]
                in_scale = [p for p in pitches if p % 12 in scale_pitch_classes]
                out_of_scale = [p for p in pitches if p % 12 not in scale_pitch_classes]
                print(f"  Notes in {actual_key} {actual_scale} scale: {len(in_scale)}/{len(pitches)}")
                if out_of_scale:
                    print(f"  Out of scale notes: {out_of_scale[:5]}{'...' if len(out_of_scale) > 5 else ''}")
            else:
                print(f"  No scale established for validation")

        if pattern.chords:
            chord_pitches = []
            for chord in pattern.chords:
                chord_pitches.append([note.pitch for note in chord.notes])
            print(f"  Chord pitches: {chord_pitches[:3]}{'...' if len(chord_pitches) > 3 else ''}")

        print()

if __name__ == "__main__":
    analyze_current_output()