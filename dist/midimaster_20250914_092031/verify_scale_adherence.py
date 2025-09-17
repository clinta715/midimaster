#!/usr/bin/env python3
"""
Verify that all pitches in split MIDI files are within the specified scale.
"""

import mido
import os
import sys

def get_pitches_from_midi(filename):
    """Extract all note pitches from a MIDI file."""
    if not os.path.exists(filename):
        print(f"File {filename} does not exist.")
        return []

    mid = mido.MidiFile(filename)
    pitches = []

    for track in mid.tracks:
        for msg in track:
            if msg.type == 'note_on' and msg.velocity > 0:
                pitches.append(msg.note)

    return pitches

def main():
    if len(sys.argv) < 3:
        print("Usage: python verify_scale_adherence.py <scale_pitches> <midi_files...>")
        print("Example: python verify_scale_adherence.py '69,71,72,74,76,78,79' file1.mid file2.mid")
        sys.exit(1)

    # Parse scale pitches
    scale_str = sys.argv[1]
    try:
        scale_pitches = set(int(x.strip()) for x in scale_str.split(','))
    except:
        print("Invalid scale pitches format. Use comma-separated integers.")
        sys.exit(1)

    print(f"Scale pitches: {sorted(scale_pitches)}")

    all_good = True

    for filename in sys.argv[2:]:
        pitches = get_pitches_from_midi(filename)
        if not pitches:
            print(f"⚠️  {filename}: No pitches found")
            continue

        invalid_pitches = [p for p in pitches if p not in scale_pitches]
        if invalid_pitches:
            print(f"❌ {filename}: Found {len(invalid_pitches)} pitches outside scale: {invalid_pitches}")
            all_good = False
        else:
            print(f"✅ {filename}: All {len(pitches)} pitches are in scale")

    if all_good:
        print("\n🎉 All pitches are in scale!")
        return 0
    else:
        print("\n❌ Some pitches are outside the scale!")
        return 1

if __name__ == "__main__":
    sys.exit(main())