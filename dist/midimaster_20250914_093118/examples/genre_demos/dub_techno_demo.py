from genres.genre_factory import GenreFactory
from generators.pattern_orchestrator import PatternOrchestrator as PatternGenerator
from structures.song_skeleton import SongSkeleton
from output.midi_output import MidiOutput

if __name__ == "__main__":
    # Create electronic genre rules (techno subgenre not directly supported, using electronic)
    rules = GenreFactory.create_genre_rules('electronic')

    # Initialize PatternOrchestrator with parameters optimized for dub techno style (sparse, groove)
    orchestrator = PatternGenerator(
        rules,
        mood='calm',
        note_density=0.2,
        rhythm_density=0.3,
        chord_density=0.2,
        bass_density=0.2,
        subgenre='dub_techno'
    )

    # Create SongSkeleton with tempo for dub techno
    skeleton = SongSkeleton('electronic', 125, 'calm')

    # Generate patterns for 32 bars
    patterns = orchestrator.generate_patterns(skeleton, 32)

    # Build song arrangement
    skeleton.build_arrangement(patterns)

    # Save the output as MIDI file
    midi_output = MidiOutput()
    midi_output.save_to_midi(skeleton, 'dub_techno_output.mid')

    print("Dub techno-inspired MIDI generated: dub_techno_output.mid")