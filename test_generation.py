from generators.pattern_orchestrator import PatternOrchestrator as PatternGenerator
from output.midi_output import MidiOutput
from structures.song_skeleton import SongSkeleton
from genres.genre_factory import GenreFactory
from generators.density_manager import create_density_manager_from_preset
from generators.generator_context import GeneratorContext

# Test valid generation
genre = 'pop'
mood = 'happy'
tempo = 120
bars = 4  # small for test
density = 'balanced'

print(f"Testing generation: {genre} {mood} {tempo} BPM {bars} bars")

try:
    # Create genre rules
    genre_rules = GenreFactory.create_genre_rules(genre)

    # Create song skeleton
    song_skeleton = SongSkeleton(genre, tempo, mood)

    # Create density manager
    density_manager = create_density_manager_from_preset(density)

    # Create context
    context = GeneratorContext(
        genre_rules=genre_rules,
        mood=mood,
        note_density=density_manager.note_density,
        rhythm_density=density_manager.rhythm_density,
        chord_density=density_manager.chord_density,
        bass_density=density_manager.bass_density
    )

    # Create generator
    pattern_generator = PatternGenerator(
        genre_rules,
        mood,
        note_density=density_manager.note_density,
        rhythm_density=density_manager.rhythm_density,
        chord_density=density_manager.chord_density,
        bass_density=density_manager.bass_density,
        context=context
    )

    # Generate patterns
    patterns = pattern_generator.generate_patterns(song_skeleton, bars)

    # Build arrangement
    song_skeleton.build_arrangement(patterns)

    # Output MIDI
    midi_output = MidiOutput()
    output_path = midi_output.generate_output_filename(genre, mood, tempo, "4/4")
    output_path = midi_output.get_unique_filename(output_path)

    midi_output.save_to_midi(
        song_skeleton,
        output_path,
        genre_rules,
        separate_files=False,
        context=context,
        genre=genre,
        mood=mood,
        tempo=tempo,
        time_signature="4/4"
    )

    print(f"Success: Generated {output_path}")

    # Check if file exists
    import os
    if os.path.exists(output_path):
        print("File created successfully")
    else:
        print("File not found")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()