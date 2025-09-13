#!/usr/bin/env python3
"""
Test script for per-track time signature functionality.

This script tests the new per-track time signature feature by:
1. Creating a SongSkeleton with different time signatures for each pattern type
2. Generating simple patterns for each type
3. Saving to separate MIDI files
4. Verifying that each file has the correct time signature
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from structures.song_skeleton import SongSkeleton
from structures.data_structures import Pattern, PatternType, Note, Chord
from output.midi_output import MidiOutput
from genres.genre_factory import GenreFactory
import mido


def create_test_pattern(pattern_type: PatternType, num_bars: int = 4) -> Pattern:
    """Create a simple test pattern with a few notes."""
    notes = []
    chords = []

    # Create some basic notes for testing
    if pattern_type == PatternType.MELODY:
        # Simple melody: C D E C
        pitches = [60, 62, 64, 60]  # C D E C
    elif pattern_type == PatternType.HARMONY:
        # Simple harmony: C major chord
        pitches = [60, 64, 67]  # C E G
    elif pattern_type == PatternType.BASS:
        # Simple bass: C
        pitches = [36]  # C2
    elif pattern_type == PatternType.RHYTHM:
        # Simple rhythm: kick and snare pattern
        pitches = [36, 38]  # Kick and snare
    else:
        pitches = [60]

    for i, pitch in enumerate(pitches):
        start_time = i * 1.0  # 1 beat apart
        note = Note(pitch, 0.5, velocity=80, start_time=start_time)
        notes.append(note)

    return Pattern(pattern_type, notes, chords)


def test_per_track_time_signatures():
    """Test per-track time signature functionality."""
    print("Testing per-track time signatures...")

    # Create song skeleton
    skeleton = SongSkeleton('pop', 120, 'happy')

    # Set different time signatures for each pattern type
    skeleton.set_time_signature(PatternType.MELODY, 4, 4)    # 4/4 for melody
    skeleton.set_time_signature(PatternType.HARMONY, 3, 4)   # 3/4 for harmony
    skeleton.set_time_signature(PatternType.BASS, 6, 8)      # 6/8 for bass
    skeleton.set_time_signature(PatternType.RHYTHM, 4, 4)    # 4/4 for rhythm

    # Create test patterns
    patterns = []
    for pattern_type in [PatternType.MELODY, PatternType.HARMONY, PatternType.BASS, PatternType.RHYTHM]:
        pattern = create_test_pattern(pattern_type)
        patterns.append(pattern)
        skeleton.add_pattern(pattern)

    # Save to separate files
    midi_output = MidiOutput()
    base_filename = "test_per_track_time_signatures"
    midi_output.save_to_separate_midi_files(skeleton, base_filename)

    print("Generated separate MIDI files.")

    # Verify time signatures in each file
    expected_time_sigs = {
        'melody': (4, 4),
        'harmony': (3, 4),
        'bass': (6, 8),
        'rhythm': (4, 4)
    }

    all_correct = True
    for track_name, (expected_num, expected_den) in expected_time_sigs.items():
        filename = f"{base_filename}_{track_name}.mid"
        if os.path.exists(filename):
            midi_file = mido.MidiFile(filename)
            time_sig_found = False
            for track in midi_file.tracks:
                for msg in track:
                    if msg.type == 'time_signature':
                        if msg.numerator == expected_num and msg.denominator == expected_den:
                            print(f"✅ {track_name}: Correct time signature {msg.numerator}/{msg.denominator}")
                            time_sig_found = True
                        else:
                            print(f"❌ {track_name}: Wrong time signature {msg.numerator}/{msg.denominator}, expected {expected_num}/{expected_den}")
                            all_correct = False
                        break
                if time_sig_found:
                    break
            if not time_sig_found:
                print(f"❌ {track_name}: No time signature found")
                all_correct = False
        else:
            print(f"❌ {track_name}: File {filename} not found")
            all_correct = False

    if all_correct:
        print("\n🎉 All per-track time signatures are correct!")
        return True
    else:
        print("\n❌ Some time signatures are incorrect or missing.")
        return False


if __name__ == "__main__":
    success = test_per_track_time_signatures()
    sys.exit(0 if success else 1)