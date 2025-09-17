#!/usr/bin/env python3
"""
Per-Track Time Signatures Demo

This example demonstrates the new per-track time signature functionality in MIDI Master.
You can now specify different time signatures for melody, harmony, bass, and rhythm tracks
when generating separate MIDI files.

This is useful for creating more complex rhythmic structures where different instruments
play in different meters, creating polyrhythmic or polymetric compositions.

Usage:
    python examples/per_track_time_signatures_demo.py

Or with custom time signatures:
    python main.py --genre jazz --tempo 120 --mood happy --bars 16 --separate-files \
                   --melody-time-signature 4/4 \
                   --harmony-time-signature 3/4 \
                   --bass-time-signature 6/8 \
                   --rhythm-time-signature 4/4 \
                   --output jazz_polymetric.mid
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from structures.song_skeleton import SongSkeleton
from structures.data_structures import PatternType
from output.midi_output import MidiOutput
from generators.pattern_orchestrator import PatternOrchestrator as PatternGenerator
from genres.genre_factory import GenreFactory
from generators.density_manager import create_density_manager_from_preset


def create_polymetric_jazz_song():
    """
    Create a jazz song with polymetric elements.

    This example creates a jazz piece where:
    - Melody plays in 4/4 (straight time)
    - Harmony plays in 3/4 (waltz feel for chord changes)
    - Bass plays in 6/8 (swing feel)
    - Rhythm plays in 4/4 (standard jazz swing)
    """
    print("🎵 Creating polymetric jazz song with per-track time signatures...")

    # Create genre rules and song skeleton
    genre_rules = GenreFactory.create_genre_rules('jazz')
    song_skeleton = SongSkeleton('jazz', 120, 'happy')

    # Set different time signatures for each track
    song_skeleton.set_time_signature(PatternType.MELODY, 4, 4)    # Straight 4/4 for melody
    song_skeleton.set_time_signature(PatternType.HARMONY, 3, 4)   # 3/4 for harmony (waltz feel)
    song_skeleton.set_time_signature(PatternType.BASS, 6, 8)      # 6/8 for bass (swing)
    song_skeleton.set_time_signature(PatternType.RHYTHM, 4, 4)    # 4/4 for rhythm (standard swing)

    print("Time signatures set:")
    print(f"  Melody: {song_skeleton.get_time_signature(PatternType.MELODY)}")
    print(f"  Harmony: {song_skeleton.get_time_signature(PatternType.HARMONY)}")
    print(f"  Bass: {song_skeleton.get_time_signature(PatternType.BASS)}")
    print(f"  Rhythm: {song_skeleton.get_time_signature(PatternType.RHYTHM)}")

    # Create density manager
    density_manager = create_density_manager_from_preset('balanced')

    # Initialize pattern generator
    pattern_generator = PatternGenerator(
        genre_rules,
        'happy',
        note_density=density_manager.note_density,
        rhythm_density=density_manager.rhythm_density,
        chord_density=density_manager.chord_density,
        bass_density=density_manager.bass_density
    )

    # Generate patterns
    patterns = pattern_generator.generate_patterns(song_skeleton, 16)
    song_skeleton.build_arrangement(patterns)

    # Save to separate MIDI files
    midi_output = MidiOutput()
    base_filename = "jazz_polymetric"
    midi_output.save_to_separate_midi_files(song_skeleton, base_filename, genre_rules)

    print(f"✅ Generated separate MIDI files: {base_filename}_*.mid")
    print("\nEach track now has its own time signature:")
    print("- jazz_polymetric_melody.mid: 4/4 time")
    print("- jazz_polymetric_harmony.mid: 3/4 time")
    print("- jazz_polymetric_bass.mid: 6/8 time")
    print("- jazz_polymetric_rhythm.mid: 4/4 time")
    print("\n💡 Tip: Import these files into your DAW and align them by tempo, not time!")
    print("   This creates interesting polymetric effects where instruments play")
    print("   in different meters but at the same tempo.")


def create_rock_with_complex_rhythms():
    """
    Create a rock song with complex rhythmic elements.

    This example demonstrates how different time signatures can create
    rhythmic complexity in rock music.
    """
    print("\n🎸 Creating rock song with complex rhythms...")

    genre_rules = GenreFactory.create_genre_rules('rock')
    song_skeleton = SongSkeleton('rock', 140, 'energetic')

    # Different time signatures for rhythmic interest
    song_skeleton.set_time_signature(PatternType.MELODY, 4, 4)    # Straight rock rhythm
    song_skeleton.set_time_signature(PatternType.HARMONY, 4, 4)   # Standard chords
    song_skeleton.set_time_signature(PatternType.BASS, 7, 8)      # Complex bass line
    song_skeleton.set_time_signature(PatternType.RHYTHM, 4, 4)    # Standard rock beat

    print("Time signatures set:")
    print(f"  Melody: {song_skeleton.get_time_signature(PatternType.MELODY)}")
    print(f"  Harmony: {song_skeleton.get_time_signature(PatternType.HARMONY)}")
    print(f"  Bass: {song_skeleton.get_time_signature(PatternType.BASS)}")
    print(f"  Rhythm: {song_skeleton.get_time_signature(PatternType.RHYTHM)}")

    # Generate and save
    density_manager = create_density_manager_from_preset('dense')
    pattern_generator = PatternGenerator(
        genre_rules,
        'energetic',
        note_density=density_manager.note_density,
        rhythm_density=density_manager.rhythm_density,
        chord_density=density_manager.chord_density,
        bass_density=density_manager.bass_density
    )

    patterns = pattern_generator.generate_patterns(song_skeleton, 8)
    song_skeleton.build_arrangement(patterns)

    midi_output = MidiOutput()
    base_filename = "rock_complex_rhythms"
    midi_output.save_to_separate_midi_files(song_skeleton, base_filename, genre_rules)

    print(f"✅ Generated separate MIDI files: {base_filename}_*.mid")


def main():
    """Run the per-track time signatures demo."""
    print("🎼 MIDI Master - Per-Track Time Signatures Demo")
    print("=" * 50)

    try:
        create_polymetric_jazz_song()
        create_rock_with_complex_rhythms()

        print("\n🎉 Demo completed successfully!")
        print("\n📚 Key Concepts Demonstrated:")
        print("- Per-track time signature configuration")
        print("- Polymetric composition techniques")
        print("- Separate MIDI file generation")
        print("- Complex rhythmic structures in different genres")

        print("\n🔧 Command-line usage:")
        print("python main.py --genre <genre> --separate-files \\")
        print("               --melody-time-signature <num>/<den> \\")
        print("               --harmony-time-signature <num>/<den> \\")
        print("               --bass-time-signature <num>/<den> \\")
        print("               --rhythm-time-signature <num>/<den>")

    except Exception as e:
        print(f"❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()