#!/usr/bin/env python3
"""
DnB (Drum and Bass) Song Generation Demo

This script generates a separated-parts DnB song based on the detailed plan in docs/dnb_song_plan.md.
It creates a 64-bar DnB track with the following structure:
- Intro (8 bars): Atmospheric buildup, sparse elements
- Buildup (8 bars): Increasing intensity, layering elements
- Drop 1 (16 bars): Main energetic section with full DnB power
- Breakdown (8 bars): Reduction in intensity, atmospheric focus
- Drop 2 (16 bars): Second main section with variations
- Outro (8 bars): Outro with fading elements

Features:
- Tempo: 165 BPM
- Key: A minor (natural minor scale)
- Density: Dense
- Parts: Rhythm (breakbeats), Bass (sub-bass with Reese), Melody (atmospheric), Harmony (modal pads)

Output:
- Combined MIDI: output/dnb_song_combined.mid
- Separate parts: output/dnb_parts/ (rhythm.mid, bass.mid, melody.mid, harmony.mid)
"""

import sys
import os
import subprocess

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from genres.genre_factory import GenreFactory
from generators.pattern_orchestrator import PatternOrchestrator
from structures.song_skeleton import SongSkeleton
from output.midi_output import MidiOutput


def main():
    """Generate a DnB song with separated parts."""

    print("Generating DnB song...")

    try:
        # Create DnB genre rules
        rules = GenreFactory.create_genre_rules('dnb')
        print("✅ Created DnB genre rules")

        # Create PatternOrchestrator with dense settings for energetic DnB
        orchestrator = PatternOrchestrator(
            rules,
            mood='energetic',
            note_density=0.8,    # Dense melody
            rhythm_density=0.8,  # Dense rhythm with high syncopation
            chord_density=0.8,   # Dense harmony with modal pads
            bass_density=0.8     # Dense bass with sub-bass and Reese elements
        )
        print("✅ Created pattern orchestrator with dense settings")

        # Create SongSkeleton with DnB specifications
        # The dynamic structure for 'electronic' genre matches the plan:
        # Intro -> Pre-chorus (Buildup) -> Chorus (Drop) -> Bridge (Breakdown) -> Chorus (Drop) -> Outro
        skeleton = SongSkeleton('electronic', 165, 'energetic')
        print("✅ Created song skeleton (64 bars, 165 BPM, A minor key)")

        # Generate patterns for 64 bars
        patterns = orchestrator.generate_patterns(skeleton, 64)
        print(f"✅ Generated {len(patterns)} patterns (melody, harmony, bass, rhythm)")

        # Build song arrangement using dynamic structure
        skeleton.build_arrangement(patterns)
        print("✅ Built song arrangement with DnB structure")

        # Save combined MIDI file
        midi_output = MidiOutput()
        combined_filename = 'output/dnb_song_combined.mid'
        midi_output.save_to_midi(skeleton, combined_filename)
        print(f"✅ Saved combined MIDI: {combined_filename}")

        # Split into separate parts using midi_splitter.py
        output_dir = 'output/dnb_parts'
        try:
            result = subprocess.run([
                sys.executable, 'output/midi_splitter.py',
                combined_filename, output_dir
            ], capture_output=True, text=True, check=True)
            print(f"✅ Split MIDI into parts in: {output_dir}")
            print("Generated parts: rhythm.mid, bass.mid, melody.mid, harmony.mid")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error splitting MIDI: {e}")
            print(f"stderr: {e.stderr}")
            return

        print("\n🎵 DnB song generation complete!")
        print(f"Combined MIDI: {combined_filename}")
        print(f"Separate parts: {output_dir}/")

    except Exception as e:
        print(f"❌ Error generating DnB song: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()