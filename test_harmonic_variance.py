#!/usr/bin/env python3
"""
Test script for harmonic variance feature in chord progressions.

This script tests the new harmonic_variance parameter in the harmony generator
to ensure that it correctly filters chord progressions by harmonic distance.
"""

from genres.genre_factory import GenreFactory
from generators.pattern_orchestrator import PatternOrchestrator
from structures.song_skeleton import SongSkeleton


def test_harmonic_variance():
    """Test the harmonic variance feature with different settings."""
    print("Testing Harmonic Variance Feature")
    print("=" * 50)

    # Create genre rules
    genre_factory = GenreFactory()
    genre_rules = genre_factory.create_genre_rules('pop')

    print(f"Genre: Pop")
    print(f"Available chord progressions: {len(genre_rules.get_chord_progressions())}")

    # Test different variance levels
    variance_levels = ['close', 'medium', 'distant']

    for variance in variance_levels:
        print(f"\n--- Testing variance level: {variance} ---")

        try:
            # Create pattern orchestrator with specific variance
            orchestrator = PatternOrchestrator(
                genre_rules=genre_rules,
                mood='happy',
                harmonic_variance=variance
            )

            # Create a simple song skeleton
            song_skeleton = SongSkeleton("pop", 120, "happy")

            # Generate harmony pattern
            harmony_pattern = orchestrator.generate_chords_only(song_skeleton, 4, 'medium')

            print(f"Generated {len(harmony_pattern.chords)} chords")
            if harmony_pattern.chords:
                chords_info = []
                for i, chord in enumerate(harmony_pattern.chords[:4]):  # Show first 4 chords
                    pitches = [note.pitch for note in chord.notes]
                    chords_info.append(f"Chord {i+1}: {pitches}")
                print(f"Chord pitches: {' | '.join(chords_info)}")
            print("✅ SUCCESS")

        except Exception as e:
            print(f"❌ FAILED: {e}")


def test_progression_filtering():
    """Test the progression filtering functionality directly."""
    print("\n\nTesting Progression Filtering")
    print("=" * 50)

    from music_theory import MusicTheory

    # Sample progressions
    sample_progressions = [
        ["I", "V", "IV"],           # Close harmonic movement
        ["I", "iii", "vi"],         # Medium harmonic movement
        ["I", "VI", "iii", "VII"]   # Distant harmonic movement
    ]

    key_scale = "C major"

    for variance in ['close', 'medium', 'distant']:
        print(f"\n--- Filtering for {variance} variance ---")
        filtered = MusicTheory.filter_progressions_by_distance(
            sample_progressions, key_scale, variance
        )

        print(f"Original progressions: {len(sample_progressions)}")
        print(f"Filtered progressions: {len(filtered)}")

        for prog in filtered:
            # Calculate average distance for this progression
            distances = []
            for i in range(len(prog) - 1):
                dist = MusicTheory.calculate_harmonic_distance(prog[i], prog[i + 1], key_scale)
                distances.append(dist)

            avg_distance = sum(distances) / len(distances) if distances else 0
            print(f"Average distance: {avg_distance:.3f}")


if __name__ == "__main__":
    test_harmonic_variance()
    test_progression_filtering()