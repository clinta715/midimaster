import mido
import os

files = ['test_separate_melody.mid', 'test_separate_harmony.mid', 'test_separate_bass.mid', 'test_separate_rhythm.mid']
instruments = ['melody', 'harmony', 'bass', 'rhythm']

for file, instr in zip(files, instruments):
    if not os.path.exists(file):
        print(f"{file} does not exist.")
        continue
    
    mid = mido.MidiFile(file)
    print(f"\n=== Inspection for {file} ({instr}) ===")
    print(f"Number of tracks: {len(mid.tracks)}")
    
    if len(mid.tracks) != 1:
        print("WARNING: More than one track!")
    
    track = mid.tracks[0]
    print(f"Messages in track: {len(track)}")
    
    program_set = False
    channel = None
    notes = []
    tempo = None
    time_sig = None
    
    for msg in track:
        if msg.type == 'program_change':
            if program_set:
                print("WARNING: Multiple program changes!")
            program_set = True
            channel = msg.channel
            print(f"Program change: {msg.program} on channel {channel}")
        elif msg.type == 'note_on' and msg.velocity > 0:
            notes.append((msg.note, msg.velocity, msg.time))
        elif msg.type == 'set_tempo':
            tempo = mido.tempo2bpm(msg.tempo)
            print(f"Tempo: {tempo} BPM")
        elif msg.type == 'time_signature':
            time_sig = f"{msg.numerator}/{msg.denominator}"
            print(f"Time signature: {time_sig}")
    
    if channel is None:
        print("WARNING: No program change found, assuming channel 0")
        channel = 0
    
    print(f"Channel used: {channel}")
    print(f"Number of notes: {len(notes)}")
    if notes:
        print(f"Sample note: {notes[0]} (note, velocity, time)")
    
    if not program_set:
        print("WARNING: No program change!")
    
    print("Tempo and time sig preserved: " + ("Yes" if tempo and time_sig else "No"))