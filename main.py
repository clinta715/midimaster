#!/usr/bin/env python3
"""
Main entry point for the MIDI Master music generation program.

This module serves as the command-line interface for the MIDI Master application.
It parses user arguments, orchestrates the music generation process, and outputs
the resulting composition as a MIDI file.

Usage:
    python main.py [options]

Example:
    python main.py --genre jazz --tempo 120 --mood calm --density sparse --output my_song.mid
"""

import argparse
import sys
import os

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import core modules
from generators.pattern_orchestrator import PatternOrchestrator as PatternGenerator
from output.midi_output import MidiOutput
from structures.song_skeleton import SongSkeleton
from structures.data_structures import PatternType
from genres.genre_factory import GenreFactory
from generators.density_manager import create_density_manager_from_preset
from audio.plugin_host import PluginHost
from output.audio_output import AudioOutput
from gui.main_window import run_gui
from gui.config_manager import ConfigManager
from generators.generator_context import GeneratorContext


def parse_time_signature(time_sig_str: str) -> tuple[int, int]:
    """
    Parse a time signature string like "4/4" or "3/4" into numerator and denominator.

    Args:
        time_sig_str: Time signature string in format "numerator/denominator"

    Returns:
        Tuple of (numerator, denominator)

    Raises:
        ValueError: If the time signature format is invalid
    """
    try:
        num, den = time_sig_str.split('/')
        return int(num), int(den)
    except ValueError:
        raise ValueError(f"Invalid time signature format: {time_sig_str}. Expected format: 'numerator/denominator' (e.g., '4/4')")


def apply_time_signatures_to_skeleton(song_skeleton: SongSkeleton, args) -> None:
    """
    Apply time signature settings from command-line arguments to the song skeleton.

    Args:
        song_skeleton: The SongSkeleton to configure
        args: Parsed command-line arguments
    """
    # Parse and set time signatures for each pattern type
    time_sig_mappings = {
        PatternType.MELODY: args.melody_time_signature,
        PatternType.HARMONY: args.harmony_time_signature,
        PatternType.BASS: args.bass_time_signature,
        PatternType.RHYTHM: args.rhythm_time_signature
    }

    for pattern_type, time_sig_str in time_sig_mappings.items():
        try:
            numerator, denominator = parse_time_signature(time_sig_str)
            song_skeleton.set_time_signature(pattern_type, numerator, denominator)
            print(f"Set {pattern_type.value} time signature to {numerator}/{denominator}")
        except ValueError as e:
            print(f"Warning: {e}. Using default 4/4 for {pattern_type.value}")
            song_skeleton.set_time_signature(pattern_type, 4, 4)


def main():
    """
    Main function that parses command-line arguments and generates music.

    This function:
    1. Parses command-line arguments
    2. Creates genre-specific rules
    3. Initializes the song structure
    4. Generates musical patterns
    5. Builds the song arrangement
    6. Outputs the result as a MIDI file (and optionally audio)
    """
    # Set up command-line argument parser with description and example usage
    parser = argparse.ArgumentParser(
        description='Generate music using MIDI Master',
        epilog='Example: python main.py --genre jazz --tempo 120 --mood calm --density sparse --output my_song.mid'
    )

    # Define command-line arguments with help text and default values
    parser.add_argument(
        '--genre',
        choices=['pop', 'rock', 'jazz', 'electronic', 'hip-hop', 'classical'],
        default='pop',
        help='Music genre to generate (default: pop)'
    )

    parser.add_argument(
        '--tempo',
        type=int,
        default=120,
        help='Tempo in BPM (default: 120)'
    )

    parser.add_argument(
        '--mood',
        choices=['happy', 'sad', 'energetic', 'calm'],
        default='happy',
        help='Mood of the music (default: happy)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='output.mid',
        help='Output MIDI file name (default: output.mid)'
    )

    parser.add_argument(
        '--bars',
        type=int,
        default=16,
        help='Number of bars to generate (default: 16)'
    )
    parser.add_argument(
        '--density',
        choices=['minimal', 'sparse', 'balanced', 'dense', 'complex'],
        default='balanced',
        help='Overall note density preset (default: balanced)'
    )

    parser.add_argument(
        '--render-audio',
        action='store_true',
        help='Enable rendering to audio file (requires --plugin-path and --audio-output)'
    )

    parser.add_argument(
        '--plugin-path',
        type=str,
        help='Path to the VST/CLAP plugin for audio rendering (e.g., C:/Plugins/Synth.dll)'
    )

    parser.add_argument(
        '--audio-output',
        type=str,
        help='Output audio file name (e.g., output.wav). Required if --render-audio is used.'
    )

    parser.add_argument(
        '--gui',
        action='store_true',
        help='Launch the graphical user interface (GUI) for music generation.'
    )

    parser.add_argument(
        '--separate-files',
        action='store_true',
        help='Save patterns to separate MIDI files per instrument type.'
    )

    # Time signature options for per-track configuration
    parser.add_argument(
        '--melody-time-signature',
        type=str,
        default='4/4',
        help='Time signature for melody track (e.g., 4/4, 3/4, 6/8). Default: 4/4'
    )

    parser.add_argument(
        '--harmony-time-signature',
        type=str,
        default='4/4',
        help='Time signature for harmony track (e.g., 4/4, 3/4, 6/8). Default: 4/4'
    )

    parser.add_argument(
        '--bass-time-signature',
        type=str,
        default='4/4',
        help='Time signature for bass track (e.g., 4/4, 3/4, 6/8). Default: 4/4'
    )

    parser.add_argument(
        '--rhythm-time-signature',
        type=str,
        default='4/4',
        help='Time signature for rhythm track (e.g., 4/4, 3/4, 6/8). Default: 4/4'
    )

    # Parse command-line arguments into args object
    args = parser.parse_args()

    # If --gui is present, launch the GUI and exit
    if args.gui:
        print("Launching GUI...")
        run_gui()
        return # Exit after GUI is closed

    try:
        # Load configuration settings
        config_manager = ConfigManager()
        temp_settings = config_manager.load_temp_settings()

        # Create genre-specific rules based on user selection
        # This provides the musical constraints and characteristics for the selected genre
        genre_rules = GenreFactory.create_genre_rules(args.genre)

        # Create generator context and set user key/mode if specified
        context = GeneratorContext(genre_rules, args.mood)
        if temp_settings.get('user_key') and temp_settings.get('user_mode'):
            context.set_user_key_mode(temp_settings['user_key'], temp_settings['user_mode'])
            print(f"Using user-specified key/mode: {temp_settings['user_key']} {temp_settings['user_mode']}")

        # Create song skeleton with user parameters
        # The song skeleton holds the overall structure of the composition
        song_skeleton = SongSkeleton(args.genre, args.tempo, args.mood)

        # Apply per-track time signature settings if separate files are requested
        if args.separate_files:
            apply_time_signatures_to_skeleton(song_skeleton, args)

        # Create density manager from preset
        density_manager = create_density_manager_from_preset(args.density)

        # Initialize pattern generator with genre rules, mood, and density settings
        # The pattern generator creates the actual musical content based on genre rules
        pattern_generator = PatternGenerator(
            genre_rules,
            args.mood,
            note_density=density_manager.note_density,
            rhythm_density=density_manager.rhythm_density,
            chord_density=density_manager.chord_density,
            bass_density=density_manager.bass_density,
            context=context
        )

        # Generate musical patterns for the song
        # This creates melody, harmony, bass, and rhythm patterns
        patterns = pattern_generator.generate_patterns(song_skeleton, args.bars)

        # Build song arrangement from generated patterns
        # This organizes the patterns into sections like verse, chorus, etc.
        song_skeleton.build_arrangement(patterns)

        # Output the generated song as a MIDI file
        # This converts the internal representation to a standard MIDI file
        midi_output = MidiOutput()
        midi_output.save_to_midi(song_skeleton, args.output, genre_rules, separate_files=args.separate_files)

        # Print success message to console
        print(f"Successfully generated {args.output}")

        # Handle audio rendering if requested
        if args.render_audio:
            if not args.plugin_path or not args.audio_output:
                print("Error: --plugin-path and --audio-output are required for audio rendering.", file=sys.stderr)
                sys.exit(1)

            print(f"Attempting to render audio to {args.audio_output} using plugin {args.plugin_path}...")
            try:
                plugin_host = PluginHost()
                if plugin_host.load_plugin(args.plugin_path):
                    audio_output = AudioOutput(plugin_host)
                    if audio_output.render_song_to_audio(song_skeleton, args.audio_output):
                        print(f"Successfully rendered audio to {args.audio_output}")
                    else:
                        print(f"Failed to render audio to {args.audio_output}", file=sys.stderr)
                        sys.exit(1)
                else:
                    print(f"Failed to load plugin from {args.plugin_path}. Audio rendering aborted.", file=sys.stderr)
                    sys.exit(1)
            except Exception as e:
                print(f"Error during audio rendering: {str(e)}", file=sys.stderr)
                import traceback
                traceback.print_exc()
                sys.exit(1)

    except Exception as e:
        # Handle any errors during the generation process
        # Print error message to stderr and exit with error code
        print(f"Error generating music: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Entry point for the script - only runs when script is executed directly
    main()
