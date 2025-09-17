import mido
import os

# Test MIDI file validity
midi_file_path = "output/pop_happy_120_4-4_20250915_160831.mid"

if os.path.exists(midi_file_path):
    try:
        midi_file = mido.MidiFile(midi_file_path)
        print(f"MIDI file loaded successfully: {midi_file_path}")
        print(f"Type: {midi_file.type}")
        print(f"Length: {midi_file.length:.2f} seconds")
        print(f"Ticks per beat: {midi_file.ticks_per_beat}")
        print(f"Number of tracks: {len(midi_file.tracks)}")

        for i, track in enumerate(midi_file.tracks):
            print(f"Track {i}: {len(track)} messages")
            note_on_count = sum(1 for msg in track if msg.type == 'note_on' and msg.velocity > 0)
            note_off_count = sum(1 for msg in track if msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0))
            print(f"  Note on: {note_on_count}, Note off: {note_off_count}")

        print("MIDI file is valid and playable")
    except Exception as e:
        print(f"MIDI file invalid: {e}")
else:
    print(f"MIDI file not found: {midi_file_path}")

# Check stem files
stems = ['bass', 'harmony', 'melody', 'rhythm']
for stem in stems:
    stem_path = f"output/pop_happy_120_4-4_20250915_160845_{stem}.mid"
    if os.path.exists(stem_path):
        try:
            midi_file = mido.MidiFile(stem_path)
            print(f"Stem {stem} valid: {len(midi_file.tracks)} tracks")
        except Exception as e:
            print(f"Stem {stem} invalid: {e}")
    else:
        print(f"Stem {stem} not found")