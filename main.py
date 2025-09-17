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
from generators import PatternOrchestrator as PatternGenerator
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
from config.parameter_config import ParameterConfig, Genre, Mood, Density, HarmonicVariance

import logging

# New imports for config/presets/template resolution
from core.settings_preset_manager import SettingsPresetManager
from core.rhythms_db_resolver import RhythmsDbResolver
from core.config_loader import resolve_effective_settings
from core.filename_templater import validate_template

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


def apply_time_signatures_to_skeleton(song_skeleton: SongSkeleton, config: ParameterConfig) -> None:
    """
    Apply time signature settings from configuration to the song skeleton.

    Args:
        song_skeleton: The SongSkeleton to configure
        config: ParameterConfig with time signature settings
    """
    # Parse and set time signatures for each pattern type
    time_sig_mappings = {
        PatternType.MELODY: config.melody_time_signature,
        PatternType.HARMONY: config.harmony_time_signature,
        PatternType.BASS: config.bass_time_signature,
        PatternType.RHYTHM: config.rhythm_time_signature
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
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    """
    # Set up command-line argument parser with description and example usage
    parser = argparse.ArgumentParser(
        description='Generate music using MIDI Master',
        epilog='Example: python main.py --genre jazz --tempo 120 --mood calm --density sparse --output my_song.mid'
    )

    # Define command-line arguments with help text and default values
    parser.add_argument(
        '--genre',
        choices=[e.value for e in Genre],
        default=Genre.POP.value,
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
        choices=[e.value for e in Mood],
        default=Mood.HAPPY.value,
        help='Mood of the music (default: happy)'
    )
    parser.add_argument(
        '--subgenre',
        type=str,
        help='Optional subgenre/style within the selected genre (e.g., deep_house, dub_techno, drill, liquid)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output MIDI file name (default: auto-generate unique filename)'
    )

    parser.add_argument(
        '--bars',
        type=int,
        default=16,
        help='Number of bars to generate (default: 16)'
    )
    parser.add_argument(
        '--density',
        choices=[e.value for e in Density],
        default=Density.BALANCED.value,
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

    parser.add_argument(
        '--harmonic-variance',
        choices=[e.value for e in HarmonicVariance],
        default=HarmonicVariance.MEDIUM.value,
        help='Level of harmonic movement between chords (default: medium)'
    )

    parser.add_argument(
        '--pattern-strength',
        type=float,
        default=1.0,
        choices=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        help='Controls rhythm pattern adherence (0.0-1.0). How strongly to maintain original rhythm patterns (0.0=very loose, 1.0=strict). Default: 1.0'
    )

    parser.add_argument(
        '--swing-percent',
        type=float,
        default=0.5,
        choices=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        help='Amount of swing feel in rhythm (0.0=straight, 1.0=maximum swing). Default: 0.5'
    )

    parser.add_argument(
        '--fill-frequency',
        type=float,
        default=0.25,
        choices=[0.0, 0.1, 0.25, 0.33, 0.5],
        help='Frequency of rhythmic fills (0.0-0.5). How often to add rhythmic fills (0.0=never, 0.5=every 2 bars). Default: 0.25'
    )

    parser.add_argument(
        '--ghost-note-level',
        type=float,
        default=1.0,
        choices=[0.0, 0.3, 0.5, 1.0, 1.5, 2.0],
        help='Intensity of ghost notes (0.0=none, 2.0=very prominent). Default: 1.0'
    )

    # Additional CLI flags (non-breaking)
    parser.add_argument(
        '--filename-template',
        type=str,
        help='Filename template for output. Placeholders: {genre},{mood},{tempo},{bars},{stem},{timestamp},{run_index},{unique_id}'
    )
    parser.add_argument(
        '--rhythms-db',
        type=str,
        help='Path to rhythms MIDI database directory (overrides env/config)'
    )
    parser.add_argument(
        '--load-preset',
        type=str,
        help='Load a named settings preset before generation (CLI flags override loaded values)'
    )
    parser.add_argument(
        '--save-preset',
        type=str,
        help='Save the resolved settings to a named preset after merging'
    )
    parser.add_argument(
        '--list-presets',
        action='store_true',
        help='List available preset names and exit'
    )

    # Parse command-line arguments into args object
    args = parser.parse_args()

    # If --gui is present, launch the GUI and exit
    if args.gui:
        print("Launching GUI...")
        run_gui()
        return # Exit after GUI is closed

    # Handle preset listing early and exit cleanly
    if getattr(args, "list_presets", False):
        preset_dir = os.environ.get("MIDIMASTER_PRESETS_DIR", "configs/presets")
        spm = SettingsPresetManager(preset_dir=preset_dir)
        names = spm.list_presets()
        for n in names:
            print(n)
        sys.exit(0)

    try:
        # Resolve effective settings from CLI/config/presets (non-breaking defaults)
        effective_settings, filename_template, rhythms_db_path = resolve_effective_settings(args)

        # Validate template early if provided
        if filename_template:
            ok, reason = validate_template(filename_template)
            if not ok:
                print(f"Invalid filename template: {reason}", file=sys.stderr)
                sys.exit(1)

        # If CLI explicitly provided rhythms DB path, validate it
        if getattr(args, "rhythms_db", None):
            resolver = RhythmsDbResolver(settings_dir="configs")
            chosen = resolver.get_rhythms_db_path(override=args.rhythms_db)
            is_ok, msg = resolver.validate_path(chosen)
            if not is_ok:
                print(f"Invalid --rhythms-db: {msg}", file=sys.stderr)
                sys.exit(1)

        # Optionally save preset after merging
        if getattr(args, "save_preset", None):
            preset_dir = os.environ.get("MIDIMASTER_PRESETS_DIR", "configs/presets")
            spm = SettingsPresetManager(preset_dir=preset_dir)
            saved = spm.save_preset(args.save_preset, effective_settings)
            if not saved:
                print(f"Failed to save preset '{args.save_preset}'. Please ensure required fields are valid.", file=sys.stderr)
                sys.exit(1)
            else:
                print(f"Preset '{args.save_preset}' saved.")

        # Compute effective generation parameters (presets may have overridden some)
        eff_genre = effective_settings.get('genre', args.genre)
        eff_mood = effective_settings.get('mood', args.mood)
        eff_tempo = int(effective_settings.get('tempo', args.tempo))
        eff_bars = int(effective_settings.get('bars', args.bars))

        # Create ParameterConfig from resolved values (backward-compatible)
        config = ParameterConfig(
            genre=Genre(eff_genre),
            tempo=eff_tempo,
            mood=Mood(eff_mood),
            subgenre=args.subgenre,
            output=args.output,
            bars=eff_bars,
            density=Density(args.density),
            render_audio=args.render_audio,
            plugin_path=args.plugin_path,
            audio_output=args.audio_output,
            separate_files=args.separate_files,
            melody_time_signature=args.melody_time_signature,
            harmony_time_signature=args.harmony_time_signature,
            bass_time_signature=args.bass_time_signature,
            rhythm_time_signature=args.rhythm_time_signature,
            harmonic_variance=HarmonicVariance(args.harmonic_variance),
            pattern_strength=args.pattern_strength,
            swing_percent=args.swing_percent,
            fill_frequency=args.fill_frequency,
            ghost_note_level=args.ghost_note_level
        )

        # Load configuration settings for user key/mode
        config_manager = ConfigManager()
        temp_settings = config_manager.load_temp_settings()

        if temp_settings.get('user_key') and temp_settings.get('user_mode'):
            config.user_key = temp_settings['user_key']
            config.user_mode = temp_settings['user_mode']
            print(f"Using user-specified key/mode: {config.user_key} {config.user_mode}")

        # Create genre-specific rules based on user selection
        # This provides the musical constraints and characteristics for the selected genre
        genre_rules = GenreFactory.create_genre_rules(config.genre)

        # Create generator context and set user key/mode if specified
        context = GeneratorContext(genre_rules, config.mood, subgenre=config.subgenre)
        if config.user_key and config.user_mode:
            try:
                context.set_user_key_mode(config.user_key, config.user_mode)
            except ValueError as e:
                logging.warning(f"Invalid key/mode: {e}. Falling back to C major.")
                config.user_key = 'C'
                config.user_mode = 'major'
                context.set_user_key_mode('C', 'major')

        # Create song skeleton with user parameters
        # The song skeleton holds the overall structure of the composition
        song_skeleton = SongSkeleton(config.genre, config.tempo, config.mood)

        # Apply per-track time signature settings if separate files are requested
        if config.separate_files:
            apply_time_signatures_to_skeleton(song_skeleton, config)

        # Create density manager from preset
        density_manager = create_density_manager_from_preset(config.density)

        # Initialize pattern generator with genre rules, mood, and density settings
        # The pattern generator creates the actual musical content based on genre rules
        pattern_generator = PatternGenerator(
            genre_rules,
            config.mood,
            note_density=density_manager.note_density,
            rhythm_density=density_manager.rhythm_density,
            chord_density=density_manager.chord_density,
            bass_density=density_manager.bass_density,
            harmonic_variance=config.harmonic_variance,
            subgenre=config.subgenre,
            context=context
        )

        # Generate musical patterns for the song
        # This creates melody, harmony, bass, and rhythm patterns
        patterns = pattern_generator.generate_patterns(song_skeleton, config.bars)

        # Build song arrangement from generated patterns
        # This organizes the patterns into sections like verse, chorus, etc.
        song_skeleton.build_arrangement(patterns)

        # Output the generated song as a MIDI file
        # This converts the internal representation to a standard MIDI file
        midi_output = MidiOutput()
        midi_output.save_to_midi(
            song_skeleton,
            config.output,
            genre_rules,
            separate_files=config.separate_files,
            context=context,
            genre=config.genre,
            mood=config.mood,
            tempo=config.tempo,
            time_signature=config.rhythm_time_signature,
            filename_template=filename_template,
            template_settings={
                "genre": str(config.genre.value if hasattr(config.genre, "value") else config.genre),
                "mood": str(config.mood.value if hasattr(config.mood, "value") else config.mood),
                "tempo": int(config.tempo),
                "bars": int(config.bars),
            } if filename_template else None,
            base_output_dir="output"
        )

        # Print success message to console
        print(f"Successfully generated music")

        # Handle audio rendering if requested
        if config.render_audio:
            if not config.plugin_path or not config.audio_output:
                print("Error: --plugin-path and --audio-output are required for audio rendering.", file=sys.stderr)
                sys.exit(1)

            print(f"Attempting to render audio to {config.audio_output} using plugin {config.plugin_path}...")
            try:
                plugin_host = PluginHost()
                if plugin_host.load_plugin(config.plugin_path):
                    audio_output = AudioOutput(plugin_host)
                    if audio_output.render_song_to_audio(song_skeleton, config.audio_output):
                        print(f"Successfully rendered audio to {config.audio_output}")
                    else:
                        print(f"Failed to render audio to {config.audio_output}", file=sys.stderr)
                        sys.exit(1)
                else:
                    print(f"Failed to load plugin from {config.plugin_path}. Audio rendering aborted.", file=sys.stderr)
                    sys.exit(1)
            except Exception as e:
                print(f"Error during audio rendering: {str(e)}", file=sys.stderr)
                import traceback
                traceback.print_exc()
                sys.exit(1)

    except ValueError as e:
        print(f"Validation error: {str(e)}", file=sys.stderr)
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
