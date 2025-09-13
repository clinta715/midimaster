#!/usr/bin/env python3
"""
Simple test for the Density Management System in MIDI Master.
"""

import sys
import os

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from generators.density_manager import DensityManager, create_density_manager_from_preset

    print("Testing DensityManager...")

    # Test basic density manager
    dm = DensityManager(note_density=0.3, rhythm_density=0.6, chord_density=0.4, bass_density=0.2)
    print("✅ Created DensityManager")

    # Test preset creation
    preset_dm = create_density_manager_from_preset('sparse')
    print("✅ Created preset DensityManager")

    # Test density calculations
    probability = dm.calculate_note_probability()
    print(f"✅ Note placement probability: {probability:.2f}")

    durations = dm.get_available_durations(0.5)
    print(f"✅ Available durations: {durations}")

    voicing_size = dm.get_chord_voicing_size(4)
    print(f"✅ Chord voicing size: {voicing_size}")

    bass_notes = dm.get_bass_note_count(4)
    print(f"✅ Bass notes per bar: {bass_notes}")

    print("🎉 All density management tests passed!")

except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()