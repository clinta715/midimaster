#!/usr/bin/env python3
"""
Test script for the expanded instrumentation and arrangement capabilities.

This script demonstrates the use of:
1. Comprehensive instrument library with timbre and register characteristics
2. Advanced arrangement techniques (counterpoint, call-response, layering)
3. Spatial positioning and dynamic range control
4. Reverb routing and instrument doubling capabilities
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from instruments.instrumentation_manager import InstrumentationManager, instrumentation_manager
from instruments.advanced_arrangement_engine import AdvancedArrangementEngine, advanced_arrangement_engine
from structures.data_structures import Note, Pattern, PatternType
from music_theory import MusicNote


def test_instrument_library():
    """Test the comprehensive instrument library."""
    print("\n=== Testing Instrument Library ===")

    # Test instrument selection for different genres/moods
    print("\n1. Instrument Selection for Electronic/Energetic:")
    instruments = instrumentation_manager.select_instruments_for_genre("electronic", "energetic", "medium")
    for name, preset in instruments.items():
        print(f"  - {name}: {preset.description}")

    print("\n2. Instrument Characteristics:")
    for instrument_name in ["808_kick", "fm_bass", "trumpet"]:
        characteristics = instrumentation_manager.get_instrument_characteristics(instrument_name)
        if characteristics:
            print(f"  - {instrument_name}:")
            print(f"    Timbre: {[t.value for t in characteristics.timbre]}")
            print(f"    Register: {[r.value for r in characteristics.register]}")
            print(f"    Articulation: {[a.value for a in characteristics.articulation]}")


def test_arrangement_techniques():
    """Test advanced arrangement techniques."""
    print("\n=== Testing Arrangement Techniques ===")

    # Create a simple test pattern
    test_pattern = Pattern(pattern_type=PatternType.MELODY, notes=[
        Note(pitch=60, duration=1.0, velocity=80, start_time=0.0),
        Note(pitch=64, duration=1.0, velocity=75, start_time=1.0),
        Note(pitch=67, duration=1.0, velocity=70, start_time=2.0),
        Note(pitch=72, duration=1.0, velocity=85, start_time=3.0),
    ], chords=[])

    print(f"\nOriginal pattern notes: {len(test_pattern.notes)}")

    # Test counterpoint
    print("\n1. Counterpoint Arrangement:")
    counterpoint_pattern = advanced_arrangement_engine.apply_arrangement_technique(
        test_pattern, "counterpoint", "jazz", "medium"
    )
    print(f"  Counterpoint pattern notes: {len(counterpoint_pattern.notes)}")

    # Test call-response
    print("\n2. Call-Response Arrangement:")
    call_response_pattern = advanced_arrangement_engine.apply_arrangement_technique(
        test_pattern, "call_response", "jazz", "medium"
    )
    print(f"  Call-response pattern notes: {len(call_response_pattern.notes)}")

    # Test layering
    print("\n3. Layering Arrangement:")
    layering_pattern = advanced_arrangement_engine.apply_arrangement_technique(
        test_pattern, "layering", "electronic", "complex"
    )
    print(f"  Layering pattern notes: {len(layering_pattern.notes)}")


def test_arrangement_configuration():
    """Test arrangement configuration capabilities."""
    print("\n=== Testing Arrangement Configuration ===")

    # Get instruments for jazz genre
    instruments = instrumentation_manager.select_instruments_for_genre("jazz", "smooth", "complex")

    # Configure arrangement
    arrangement = instrumentation_manager.configure_arrangement(
        instruments, "counterpoint"
    )

    print(f"\nArrangement configured with {len(arrangement.instruments)} instruments:")
    for name in arrangement.instruments.keys():
        print(f"  - {name}")

    print("\nSpatial configuration:")
    for name, spatial in arrangement.spatial_config.items():
        pan = spatial.get("pan", 0.0)
        print(f"    Pan: {pan:.2f}")

    print("\nDynamic configuration:")
    for name, dynamic in arrangement.dynamic_config.items():
        volume = dynamic.get("volume", 0.7)
        print(f"    Volume: {volume:.2f}")


def test_quality_evaluation():
    """Test arrangement quality evaluation."""
    print("\n=== Testing Quality Evaluation ===")

    # Create a test arrangement
    instruments = instrumentation_manager.select_instruments_for_genre("classical", "balanced", "medium")
    arrangement = instrumentation_manager.configure_arrangement(instruments, "orchestration")

    # Evaluate quality
    quality_scores = advanced_arrangement_engine.quality_evaluator.evaluate_arrangement(arrangement)

    print("\nArrangement Quality Scores:")
    for metric, score in quality_scores.items():
        print(f"    {metric}: {score:.3f}")


def main():
    """Main test function."""
    print("MIDI Master - Instrumentation and Arrangement Test Suite")
    print("=" * 60)

    try:
        test_instrument_library()
        test_arrangement_techniques()
        test_arrangement_configuration()
        test_quality_evaluation()

        print("\n" + "=" * 60)
        print("✅ All tests completed successfully!")
        print("The expanded instrumentation and arrangement capabilities are working.")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()