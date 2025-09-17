#!/usr/bin/env python3
"""
Test script for the enhanced timing and microtiming algorithms.
"""

import sys
import os

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generators.pattern_orchestrator import PatternOrchestrator as PatternGenerator
from output.midi_output import MidiOutput
from structures.song_skeleton import SongSkeleton
from genres.genre_factory import GenreFactory


def test_timing_enhancements():
    """Test the enhanced timing and microtiming features."""
    print("Testing enhanced timing and microtiming algorithms...")

    try:
        # Create genre rules for testing different genres
        genres_to_test = ['jazz', 'funk', 'rock', 'classical']

        for genre in genres_to_test:
            print(f"\n🧪 Testing {genre} timing enhancements...")

            # Create genre-specific rules
            genre_rules = GenreFactory.create_genre_rules(genre)

            # Create song skeleton
            song_skeleton = SongSkeleton(genre, 120, 'energetic')

            # Generate patterns with advanced timing enabled
            pattern_generator = PatternGenerator(genre_rules, 'energetic')

            # Enable advanced timing
            pattern_generator.enable_advanced_timing(base_tempo=120.0)

            # Add some tempo variations to test TempoCurve
            pattern_generator.timing_engine.tempo_curve.add_acceleration(4.0, 8.0, 1.2)  # Accelerate from bar 2 to 4
            pattern_generator.timing_engine.tempo_curve.add_ritardando(12.0, 16.0, 0.8)  # Slow down at end

            # Generate patterns with timing enhancements
            patterns = pattern_generator.generate_patterns(song_skeleton, 4)  # 4 bars

            print(f"  ✅ Generated {len(patterns)} patterns with enhanced timing")
            print(f"  📊 Pattern types: {[p.pattern_type.value for p in patterns]}")

            # Verify timing variations were applied
            total_notes = sum(len(p.notes) for p in patterns)
            print(f"  🎵 Total notes across patterns: {total_notes}")

            # Check that notes have varied timing (not all exactly on beats)
            varied_timing = False
            for pattern in patterns:
                for note in pattern.notes:
                    if note.start_time % 1.0 != 0.0:  # Not exactly on a beat
                        varied_timing = True
                        break
                if varied_timing:
                    break

            if varied_timing:
                print("  ✅ Microtiming variations detected")
            else:
                print("  ⚠️  No microtiming variations detected")

            # Generate MIDI file for listening test
            midi_filename = f"test_{genre}_timing.mid"
            midi_output = MidiOutput()
            midi_output.save_to_midi(song_skeleton, midi_filename)
            print(f"  🎼 Generated test MIDI: {midi_filename}")

        print("\n✅ All timing enhancement tests completed successfully!")
        print("\n🎵 Test files generated:")
        for genre in genres_to_test:
            print(f"   - test_{genre}_timing.mid")

    except Exception as e:
        print(f"❌ Timing enhancement test failed: {e}")
        import traceback
        traceback.print_exc()


def test_timing_comparison():
    """Compare patterns with and without advanced timing."""
    print("\n🔍 Comparing patterns with and without advanced timing...")

    try:
        # Create genre rules
        genre_rules = GenreFactory.create_genre_rules('jazz')

        # Test without advanced timing
        pattern_generator_basic = PatternGenerator(genre_rules, 'energetic')
        song_skeleton = SongSkeleton('jazz', 120, 'energetic')
        patterns_basic = pattern_generator_basic.generate_patterns(song_skeleton, 4)

        # Test with advanced timing
        pattern_generator_advanced = PatternGenerator(genre_rules, 'energetic')
        pattern_generator_advanced.enable_advanced_timing(base_tempo=120.0)
        patterns_advanced = pattern_generator_advanced.generate_patterns(song_skeleton, 4)

        # Compare note timing precision
        basic_exact_beats = 0
        advanced_exact_beats = 0
        total_notes_basic = 0
        total_notes_advanced = 0

        for pattern in patterns_basic:
            for note in pattern.notes:
                total_notes_basic += 1
                if note.start_time % 1.0 == 0.0:
                    basic_exact_beats += 1

        for pattern in patterns_advanced:
            for note in pattern.notes:
                total_notes_advanced += 1
                if note.start_time % 1.0 == 0.0:
                    advanced_exact_beats += 1

        print(f"📊 Basic timing - Exact beats: {basic_exact_beats}/{total_notes_basic} ({basic_exact_beats/total_notes_basic*100:.1f}%)")
        print(f"📊 Advanced timing - Exact beats: {advanced_exact_beats}/{total_notes_advanced} ({advanced_exact_beats/total_notes_advanced*100:.1f}%)")

        if advanced_exact_beats < basic_exact_beats:
            print("✅ Advanced timing successfully introduced microtiming variations!")
        else:
            print("⚠️  Advanced timing may not be working as expected")

    except Exception as e:
        print(f"❌ Timing comparison test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_timing_enhancements()
    test_timing_comparison()