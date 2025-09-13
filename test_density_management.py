#!/usr/bin/env python3
"""
Simple test to verify PatternGenerator now accepts density parameters.
"""

import sys
sys.path.append('.')

# Test PatternGenerator constructor directly
try:
    from generators.density_manager import DensityManager
    from generators.pattern_generator import PatternGenerator

    # Mock genre rules for testing
    mock_genre_rules = {
        'scales': ['C major'],
        'chord_progressions': [['I', 'IV', 'V', 'I']],
        'rhythm_patterns': [{'name': 'basic', 'pattern': [0.5, 0.5]}]
    }

    # Test with density parameters (the main issue that was reported)
    pg = PatternGenerator(
        genre_rules=mock_genre_rules,
        mood='happy',
        note_density=0.3,      # This should now work
        rhythm_density=0.6,    # This should now work
        chord_density=0.4,     # This should now work
        bass_density=0.2       # This should now work
    )

    print("✅ SUCCESS: PatternGenerator now accepts all density parameters!")
    print(f"✅ DensityManager created: {pg.density_manager.get_density_settings()}")

except Exception as e:
    print(f"❌ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
#!/usr/bin/env python3
"""
Test script for the Density Management System in MIDI Master.

This script tests the density management functionality by generating music
with different density settings and verifying that the system works correctly.
"""

import sys
import os

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from generators.pattern_generator import PatternGenerator
from generators.density_manager import DensityManager, create_density_manager_from_preset
from output.midi_output import MidiOutput
from structures.song_skeleton import SongSkeleton
from genres.genre_factory import GenreFactory


def test_density_manager():
    """Test the DensityManager class functionality."""
    print("Testing DensityManager...")

    # Test basic density manager
    dm = DensityManager(note_density=0.3, rhythm_density=0.6, chord_density=0.4, bass_density=0.2)
    print(f"✅ Created DensityManager with settings: {dm.get_density_settings()}")

    # Test preset creation
    preset_dm = create_density_manager_from_preset('sparse')
    print(f"✅ Created preset DensityManager: {preset_dm.get_density_settings()}")

    # Test density calculations
    probability = dm.calculate_note_probability()
    print(f"✅ Note placement probability: {probability:.2f}")

    durations = dm.get_available_durations(0.5)
    print(f"✅ Available durations for medium density: {durations}")

    voicing_size = dm.get_chord_voicing_size(4)
    print(f"✅ Chord voicing size: {voicing_size}")

    complexity = dm.get_rhythm_pattern_complexity()
    print(f"✅ Rhythm complexity level: {complexity}")

    bass_notes = dm.get_bass_note_count(4)
    print(f"✅ Bass notes per bar: {bass_notes}")

    print("✅ DensityManager tests passed!\n")


def test_pattern_generation_with_density():
    """Test pattern generation with different density settings."""
    print("Testing pattern generation with density control...")

    # Create genre rules
    genre_rules = GenreFactory.create_genre_rules('pop')

    # Test different density settings
    density_settings = [
        {'name': 'minimal', 'settings': {'note_density': 0.1, 'rhythm_density': 0.2, 'chord_density': 0.1, 'bass_density': 0.1}},
        {'name': 'balanced', 'settings': {'note_density': 0.5, 'rhythm_density': 0.5, 'chord_density': 0.5, 'bass_density': 0.5}},
        {'name': 'dense', 'settings': {'note_density': 0.8, 'rhythm_density': 0.7, 'chord_density': 0.9, 'bass_density': 0.8}}
    ]

    for density_config in density_settings:
        print(f"Testing {density_config['name']} density...")

        # Create pattern generator with density settings
        generator = PatternGenerator(
            genre_rules=genre_rules,
            mood='happy',
            **density_config['settings']
        )

        # Create song skeleton
        skeleton = SongSkeleton('pop', 120, 'happy')

        # Generate patterns
        patterns = generator.generate_patterns(skeleton, 4)

        # Analyze generated patterns
        total_notes = sum(len(pattern.notes) for pattern in patterns)
        total_chords = sum(len(pattern.chords) for pattern in patterns)

        print(f"  Generated {len(patterns)} patterns")
        print(f"  Total notes: {total_notes}")
        print(f"  Total chords: {total_chords}")

        # Verify patterns have expected types
        pattern_types = [p.pattern_type.value for p in patterns]
        expected_types = ['melody', 'harmony', 'bass', 'rhythm']
        assert all(pt in pattern_types for pt in expected_types), f"Missing pattern types: {pattern_types}"

        print(f"  Pattern types: {pattern_types}")
        print("  ✅ Pattern generation successful!\n")


def test_midi_output_with_density():
    """Test MIDI output with density-controlled patterns."""
    print("Testing MIDI output with density control...")

    # Create genre rules
    genre_rules = GenreFactory.create_genre_rules('jazz')

    # Create pattern generator with sparse density for jazz
    generator = PatternGenerator(
        genre_rules=genre_rules,
        mood='calm',
        note_density=0.3,    # Sparse melody
        rhythm_density=0.8,  # Complex rhythm (jazz swing)
        chord_density=0.6,   # Medium chord complexity
        bass_density=0.4     # Moderate bass
    )

    # Create song skeleton
    skeleton = SongSkeleton('jazz', 120, 'calm')

    # Generate patterns
    patterns = generator.generate_patterns(skeleton, 8)

    # Build arrangement
    skeleton.build_arrangement(patterns)

    # Output to MIDI
    midi_output = MidiOutput()
    output_file = 'test_density_jazz.mid'
    midi_output.save_to_midi(skeleton, output_file)

    print(f"✅ Successfully generated {output_file} with density control")
    print(f"  File size: {os.path.getsize(output_file)} bytes\n")


def test_density_presets():
    """Test density presets for different musical styles."""
    print("Testing density presets...")

    presets = ['minimal', 'sparse', 'balanced', 'dense', 'complex']

    for preset in presets:
        print(f"Testing {preset} preset...")

        # Create density manager from preset
        dm = create_density_manager_from_preset(preset)
        settings = dm.get_density_settings()

        print(f"  Settings: {settings}")

        # Verify all settings are within valid range
        for key, value in settings.items():
            assert 0.0 <= value <= 1.0, f"Invalid {key} value: {value}"

        print("  ✅ Preset valid!\n")


def main():
    """Run all density management tests."""
    print("🎵 MIDI Master Density Management Test Suite")
    print("=" * 50)

    try:
        test_density_manager()
        test_pattern_generation_with_density()
        test_midi_output_with_density()
        test_density_presets()

        print("🎉 All density management tests passed!")
        print("\nDensity management features:")
        print("• Control note density (sparse to dense)")
        print("• Adjust rhythm complexity")
        print("• Modify chord voicing size")
        print("• Configure bass line density")
        print("• Use presets for common styles")
        print("• Maintain backward compatibility")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()