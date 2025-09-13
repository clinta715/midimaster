import mido
import os
from collections import defaultdict

file = 'test_combined.mid'
if not os.path.exists(file):
    print(f"{file} does not exist.")
else:
    mid = mido.MidiFile(file)
    print(f"\n=== Inspection for {file} (combined) ===")
    print(f"Number of tracks: {len(mid.tracks)}")
    
    if len(mid.tracks) != 1:
        print("WARNING: More than one track!")
    
    track = mid.tracks[0]
    print(f"Messages in track: {len(track)}")
    
    tempo = None
    time_sig = None
    channels_used = set()
    notes_per_channel = defaultdict(int)
    program_changes = defaultdict(list)
    
    for msg in track:
        if msg.type == 'set_tempo':
            tempo = mido.tempo2bpm(msg.tempo)
            print(f"Tempo: {tempo} BPM")
        elif msg.type == 'time_signature':
            time_sig = f"{msg.numerator}/{msg.denominator}"
            print(f"Time signature: {time_sig}")
        elif msg.type == 'program_change':
            program_changes[msg.channel].append(msg.program)
            channels_used.add(msg.channel)
        elif msg.type == 'note_on' and msg.velocity > 0:
            notes_per_channel[msg.channel] += 1
            channels_used.add(msg.channel)
    
    print(f"Channels used: {sorted(channels_used)}")
    print("Expected channels: 0 (melody), 1 (harmony), 2 (bass), 9 (rhythm)")
    print(f"Notes per channel: {dict(notes_per_channel)}")
    
    for ch, progs in program_changes.items():
        print(f"Program changes on channel {ch}: {progs}")
    
    print("Tempo and time sig preserved: " + ("Yes" if tempo and time_sig else "No"))
    print("No errors: " + ("Yes" if 'note_off' in [m.type for m in track] and len(channels_used) >= 3 else "Check channels/notes"))