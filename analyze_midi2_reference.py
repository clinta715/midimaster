#!/usr/bin/env python3
"""
Analyze MIDI files in reference_midis/midi2 directory
"""

import mido
import os
from collections import defaultdict
import glob

def analyze_midi_file(filepath):
    """Analyze a single MIDI file and return key information"""
    try:
        mid = mido.MidiFile(filepath)
        filename = os.path.basename(filepath)

        analysis = {
            'filename': filename,
            'tracks': len(mid.tracks),
            'tempo': None,
            'time_signature': None,
            'channels_used': set(),
            'program_changes': defaultdict(list),
            'notes_per_channel': defaultdict(int),
            'total_notes': 0,
            'duration_ticks': sum(len(track) for track in mid.tracks),
            'key_from_filename': None,
            'bpm_from_filename': None,
            'instrument_from_filename': None
        }

        # Parse filename for metadata
        parts = filename.replace('.mid', '').split('_')
        if len(parts) >= 4:
            analysis['key_from_filename'] = parts[2]
            analysis['bpm_from_filename'] = parts[3]
            if len(parts) > 4:
                analysis['instrument_from_filename'] = '_'.join(parts[4:])

        # Analyze MIDI content
        for track_idx, track in enumerate(mid.tracks):
            for msg in track:
                if msg.type == 'set_tempo':
                    analysis['tempo'] = mido.tempo2bpm(msg.tempo)
                elif msg.type == 'time_signature':
                    analysis['time_signature'] = f"{msg.numerator}/{msg.denominator}"
                elif msg.type == 'program_change':
                    analysis['program_changes'][msg.channel].append(msg.program)
                    analysis['channels_used'].add(msg.channel)
                elif msg.type == 'note_on' and msg.velocity > 0:
                    analysis['notes_per_channel'][msg.channel] += 1
                    analysis['total_notes'] += 1
                    analysis['channels_used'].add(msg.channel)

        return analysis

    except Exception as e:
        return {
            'filename': os.path.basename(filepath),
            'error': str(e)
        }

def main():
    midi_dir = 'reference_midis/midi2'
    midi_files = glob.glob(os.path.join(midi_dir, '*.mid'))

    if not midi_files:
        print(f"No MIDI files found in {midi_dir}")
        return

    print("=== MIDI Reference Files Analysis ===")
    print(f"Found {len(midi_files)} MIDI files in {midi_dir}\n")

    # Group by track number
    tracks = defaultdict(list)
    for midi_file in sorted(midi_files):
        analysis = analyze_midi_file(midi_file)

        # Group by track number (first part of filename)
        track_num = analysis['filename'].split('_')[0]
        tracks[track_num].append(analysis)

    # Analyze each track group
    for track_num in sorted(tracks.keys()):
        print(f"\n--- Track {track_num} ---")
        track_files = tracks[track_num]

        # Find the key and BPM from the first file
        first_file = track_files[0]
        key = first_file.get('key_from_filename')
        bpm = first_file.get('bpm_from_filename')

        print(f"Key: {key}, BPM: {bpm}")
        print(f"Files in this track: {len(track_files)}")

        # Analyze each instrument in the track
        for file_analysis in track_files:
            if 'error' in file_analysis:
                print(f"  ERROR in {file_analysis['filename']}: {file_analysis['error']}")
                continue

            instr = file_analysis.get('instrument_from_filename', 'Unknown')
            tempo = file_analysis.get('tempo')
            time_sig = file_analysis.get('time_signature')
            notes = file_analysis.get('total_notes')
            channels = sorted(file_analysis.get('channels_used', []))

            print(f"  {instr}:")
            print(f"    Tempo: {tempo} BPM, Time Sig: {time_sig}")
            print(f"    Notes: {notes}, Channels: {channels}")

            if file_analysis.get('program_changes'):
                for ch, programs in file_analysis['program_changes'].items():
                    print(f"    Channel {ch}: Program {programs}")

        # Summary for the track
        total_notes = sum(f.get('total_notes', 0) for f in track_files if 'error' not in f)
        print(f"  Total notes in track: {total_notes}")

if __name__ == '__main__':
    main()