#!/usr/bin/env python3
"""
Test script to verify independent key/mode support in MidiMaster.

This script creates a GeneratorContext with user-specified key/mode,
generates a short MIDI sequence, and verifies that the correct pitches
are used for the specified key/mode combination.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from generators.generator_context import GeneratorContext
from generators.melody_generator import MelodyGenerator
from generators.harmony_generator import HarmonyGenerator
from generators.bass_generator import BassGenerator
from genres.genre_factory import GenreFactory


def test_key_mode_override():
    """Test that user key/mode overrides work correctly."""
    print("🎼 MIDI MASTER - KEY/MODE OVERRIDE TEST")
    print("=" * 50)

    # Create genre rules for pop
    genre_factory = GenreFactory()
    genre_rules = genre_factory.create_genre_rules('pop')

    # Create context with user key/mode override
    context = GeneratorContext(
        genre_rules=genre_rules,
        mood='energetic'
    )

    # Set user key/mode - F# phrygian (for scale adherence testing)
    try:
        context.set_user_key_mode('F#', 'phrygian')
        print("✅ Set user key/mode: F# phrygian")
    except ValueError as e:
        print(f"❌ Failed to set key/mode: {e}")
        return False

    # Generate patterns
    print("\n🎵 Generating patterns...")

    melody_gen = MelodyGenerator(context)
    harmony_gen = HarmonyGenerator(context)
    bass_gen = BassGenerator(context)

    # Generate short patterns (4 bars)
    melody_pattern = melody_gen.generate(4)
    harmony_pattern = harmony_gen.generate(4)
    bass_pattern = bass_gen.generate(4)

    print("✅ Generated all patterns")

    # Verify scale pitches are correct for F# phrygian
    # F# phrygian: F#(6), G(7), A(9), B(11), C#(1), D(2), E(4) -> pitches 66,67,69,71,73,74,76
    expected_fsharp_phrygian = [66, 67, 69, 71, 73, 74, 76]  # F#, G, A, B, C#, D, E (octave 4-5)
    actual_scale = context.scale_pitches[:7]  # First octave

    print("\n🔍 VERIFICATION:")
    print(f"Expected F# phrygian scale: {expected_fsharp_phrygian}")
    print(f"Actual scale pitches: {actual_scale}")

    if actual_scale == expected_fsharp_phrygian:
        print("✅ Scale pitches match F# phrygian!")
    else:
        print("❌ Scale pitches do not match F# phrygian")
        return False

    # Check melody pitches are in scale (F# phrygian pitch classes: F#=6, G=7, A=9, B=11, C#=1, D=2, E=4)
    melody_pitches = [note.pitch for note in melody_pattern.notes]
    fsharp_phrygian_classes = {6, 7, 9, 11, 1, 2, 4}
    invalid_melody = [p for p in melody_pitches if p % 12 not in fsharp_phrygian_classes]

    if not invalid_melody:
        print("✅ All melody pitches are in F# phrygian scale")
    else:
        print(f"❌ Found invalid melody pitches: {invalid_melody}")
        return False

    # Check all generated pitches are in scale_pitches
    all_pitches = []
    all_pitches.extend(melody_pitches)
    for chord in harmony_pattern.chords:
        all_pitches.extend([note.pitch for note in chord.notes])
    for note in bass_pattern.notes:
        all_pitches.append(note.pitch)

    invalid_pitches = [p for p in all_pitches if p not in context.scale_pitches]
    if not invalid_pitches:
        print("✅ All generated pitches are in scale_pitches")
    else:
        print(f"❌ Found pitches outside scale: {invalid_pitches}")
        return False

    print("✅ Key/mode override test passed!")

    print("\n🎉 TEST SUCCESSFUL!")
    print("Key/mode independent support is working correctly.")
    print("The generated patterns use F# phrygian scale as expected.")

    return True


def main():
    """Run the key/mode test."""
    try:
        success = test_key_mode_override()
        return success
    except Exception as e:
        print(f"❌ Test crashed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)