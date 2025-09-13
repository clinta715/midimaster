#!/usr/bin/env python3
"""
MIDI Splitter Utility

Splits multi-channel MIDI files into separate files for each instrument part:
- Melody (channel 0)
- Harmony (channel 1)
- Bass (channel 2)
- Rhythm (channel 9)

Meta events (tempo, time signature, etc.) are copied to all output files.
Handles cases where channels have no notes.

Usage:
    python midi_splitter.py input.mid output_directory

Dependencies: mido
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List

try:
    import mido
    from mido import MidiFile, MidiTrack, Message, MetaMessage
except ImportError:
    print("Error: mido library is required. Install with: pip install mido")
    exit(1)


class MidiSplitter:
    """Handles splitting of multi-channel MIDI files."""

    # Channel mapping for different instrument parts
    CHANNEL_MAP = {
        'melody': 0,
        'harmony': 1,
        'bass': 2,
        'rhythm': 9
    }

    @staticmethod
    def split_midi_file(input_file: str, output_dir: str) -> None:
        """
        Split a MIDI file into separate files for each instrument channel.

        Args:
            input_file: Path to the input MIDI file
            output_dir: Directory to save the split MIDI files
        """
        # Create output directory if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Load the MIDI file
        try:
            midi_file = MidiFile(input_file)
        except Exception as e:
            raise ValueError(f"Failed to load MIDI file {input_file}: {e}")

        # Get base name for output files
        input_basename = Path(input_file).stem

        # Separate events by channel and collect meta events
        channel_events: Dict[int, List] = {0: [], 1: [], 2: [], 9: []}
        meta_events: List = []

        # Process all tracks
        for track in midi_file.tracks:
            for msg in track:
                if msg.is_meta:
                    # Meta events go to all tracks
                    meta_events.append(msg)
                elif hasattr(msg, 'channel'):
                    # Regular MIDI messages with channel
                    if msg.channel in channel_events:
                        channel_events[msg.channel].append(msg)
                else:
                    # Messages without channel (like sysex) - skip or handle as needed
                    pass

        # Create separate MIDI files for each channel
        for part_name, channel in MidiSplitter.CHANNEL_MAP.items():
            events = channel_events[channel]

            # Skip if no events for this channel
            if not events:
                continue

            # Create new MIDI file
            output_file = MidiFile()
            output_track = MidiTrack()
            output_file.tracks.append(output_track)

            # Copy meta events to the track
            for meta_msg in meta_events:
                output_track.append(meta_msg.copy())

            # Add channel-specific events
            for msg in events:
                # Create a copy of the message to avoid modifying the original
                msg_copy = msg.copy()
                output_track.append(msg_copy)

            # Save the file
            filename = f"{input_basename}_{part_name}.mid"
            output_filepath = output_path / filename
            output_file.save(str(output_filepath))
            print(f"Created: {output_filepath}")


def main():
    """Main function for command line interface."""
    parser = argparse.ArgumentParser(
        description="Split multi-channel MIDI files into separate instrument parts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python midi_splitter.py input.mid ./output/
  python midi_splitter.py /path/to/song.mid /path/to/output_dir/

The script will create separate MIDI files for:
- melody (channel 0)
- harmony (channel 1)
- bass (channel 2)
- rhythm (channel 9)
        """
    )

    parser.add_argument(
        'input_file',
        help='Path to the input MIDI file'
    )

    parser.add_argument(
        'output_dir',
        help='Directory to save the split MIDI files'
    )

    args = parser.parse_args()

    # Validate input file
    if not os.path.isfile(args.input_file):
        print(f"Error: Input file '{args.input_file}' does not exist.")
        exit(1)

    if not args.input_file.lower().endswith('.mid'):
        print(f"Warning: Input file '{args.input_file}' does not have .mid extension.")

    # Create output directory if it doesn't exist
    output_path = Path(args.output_dir)
    try:
        output_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Error: Cannot create output directory '{args.output_dir}': {e}")
        exit(1)

    # Split the MIDI file
    try:
        splitter = MidiSplitter()
        splitter.split_midi_file(args.input_file, args.output_dir)
        print("MIDI file splitting completed successfully!")
    except Exception as e:
        print(f"Error during MIDI splitting: {e}")
        exit(1)


if __name__ == "__main__":
    main()