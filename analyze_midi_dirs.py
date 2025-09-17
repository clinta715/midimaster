import mido
import os
from collections import defaultdict

def analyze_midi_directory(directory_path):
    """
    Analyze MIDI files in a directory and return aggregated statistics.
    """
    if not os.path.exists(directory_path):
        print(f"Directory {directory_path} does not exist.")
        return {}

    files = [f for f in os.listdir(directory_path) if f.endswith('.mid')]
    if not files:
        print(f"No MIDI files found in {directory_path}.")
        return {}

    # Data structures
    keys = set()
    bpms = set()
    time_sigs = set()
    instruments = defaultdict(int)
    songs = defaultdict(list)  # song_prefix -> list of stems
    bpm_ranges = []
    tempo_msgs = []

    for file in files:
        filepath = os.path.join(directory_path, file)
        try:
            mid = mido.MidiFile(filepath)

            # Parse filename for metadata
            parts = file.replace('.mid', '').split('_')
            if len(parts) >= 4:
                # Find the part with 'bpm'
                bpm_index = next((i for i, p in enumerate(parts) if 'bpm' in p), -1)
                if bpm_index >= 0:
                    song_prefix = '_'.join(parts[:bpm_index - 1]) if bpm_index > 1 else parts[0]
                    key = parts[bpm_index - 1]
                    bpm_str = parts[bpm_index].replace('bpm', '')
                    instrument = parts[bpm_index + 1] if bpm_index + 1 < len(parts) else 'Unknown'
                else:
                    # Fallback for files without bpm
                    song_prefix = '_'.join(parts[:-2]) if len(parts) > 2 else parts[0]
                    key = parts[-2] if len(parts) > 1 else 'Unknown'
                    bpm_str = 'Unknown'
                    instrument = parts[-1] if len(parts) > 0 else 'Unknown'

                if key and 'bpm' not in key:
                    keys.add(key)
                try:
                    bpm_int = int(bpm_str)
                    bpms.add(bpm_int)
                    bpm_ranges.append(bpm_int)
                except ValueError:
                    bpm_int = None

                if instrument != 'Unknown':
                    instruments[instrument] += 1

                # Group stems by song
                songs[song_prefix].append({
                    'file': file,
                    'instrument': instrument,
                    'key': key,
                    'bpm': bpm_int
                })

            # Extract from MIDI
            tempo = None
            time_sig = None
            for track in mid.tracks:
                for msg in track:
                    if msg.type == 'set_tempo':
                        tempo = mido.tempo2bpm(msg.tempo)
                        tempo_msgs.append(tempo)
                    elif msg.type == 'time_signature':
                        time_sig = f"{msg.numerator}/{msg.denominator}"
                        time_sigs.add(time_sig)

        except Exception as e:
            print(f"Error processing {file}: {e}")

    analysis = {
        'keys': sorted(list(keys)),
        'bpm_range': f"{min(bpm_ranges)}-{max(bpm_ranges)} BPM" if bpm_ranges else "N/A",
        'bpms': sorted(list(bpms)),
        'time_signatures': sorted(list(time_sigs)),
        'instrument_counts': dict(instruments),
        'songs': dict(songs),
        'tempo_stats': {
            'from_filename': bpm_ranges,
            'from_midi': tempo_msgs
        },
        'total_files': len(files),
        'total_songs': len(songs)
    }

    return analysis

def print_analysis(directory_name, analysis):
    print(f"\n=== Analysis for {directory_name} ===")
    print(f"Total MIDI files: {analysis['total_files']}")
    print(f"Total songs: {analysis['total_songs']}")
    print(f"Keys: {', '.join(analysis['keys'])}")
    print(f"BPM range: {analysis['bpm_range']}")
    print(f"Time signatures: {', '.join(analysis['time_signatures'])}")

    print("\nInstrumentation breakdown:")
    for instr, count in sorted(analysis['instrument_counts'].items()):
        print(f"  {instr}: {count} files")

    print("\nSample song stems (first 3 songs):")
    for i, (song, stems) in enumerate(analysis['songs'].items()):
        if i >= 3: break
        print(f"  {song}: {len(stems)} stems")
        for stem in stems[:5]:  # Show first 5 stems
            print(f"    - {stem['instrument']} ({stem['key']}, {stem['bpm']} BPM)")

if __name__ == "__main__":
    midi2_path = "reference_midis/midi2"
    midi3_path = "reference_midis/midi3"

    analysis2 = analyze_midi_directory(midi2_path)
    analysis3 = analyze_midi_directory(midi3_path)

    if analysis2:
        print_analysis("midi2", analysis2)

    if analysis3:
        print_analysis("midi3", analysis3)

    # Compare directories
    if analysis2 and analysis3:
        print("\n=== Comparison between midi2 and midi3 ===")

        keys2 = set(analysis2['keys'])
        keys3 = set(analysis3['keys'])
        print(f"Keys in midi2 but not midi3: {keys2 - keys3}")
        print(f"Keys in midi3 but not midi2: {keys3 - keys2}")

        bpms2 = set(analysis2['bpms'])
        bpms3 = set(analysis3['bpms'])
        print(f"BPMs in midi2 but not midi3: {bpms2 - bpms3}")
        print(f"BPMs in midi3 but not midi2: {bpms3 - bpms2}")

        instr2 = set(analysis2['instrument_counts'].keys())
        instr3 = set(analysis3['instrument_counts'].keys())
        print(f"Instruments in midi2 but not midi3: {instr2 - instr3}")
        print(f"Instruments in midi3 but not midi2: {instr3 - instr2}")

        print(f"Average stems per song - midi2: {analysis2['total_files']/analysis2['total_songs']:.1f}")
        print(f"Average stems per song - midi3: {analysis3['total_files']/analysis3['total_songs']:.1f}")