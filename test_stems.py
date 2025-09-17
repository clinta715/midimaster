from generators.pattern_orchestrator import PatternOrchestrator as PatternGenerator
from output.midi_output import MidiOutput
from structures.song_skeleton import SongSkeleton
from genres.genre_factory import GenreFactory
from generators.density_manager import create_density_manager_from_preset
from generators.generator_context import GeneratorContext

# Test separate stems generation
genre = 'pop'
mood = 'happy'
tempo = 120
bars = 4

print(f"Testing separate stems: {genre} {mood} {tempo} BPM {bars} bars")

try:
    genre_rules = GenreFactory.create_genre_rules(genre)
    song_skeleton = SongSkeleton(genre, tempo, mood)
    density_manager = create_density_manager_from_preset('balanced')

    context = GeneratorContext(
        genre_rules=genre_rules,
        mood=mood,
        note_density=density_manager.note_density,
        rhythm_density=density_manager.rhythm_density,
        chord_density=density_manager.chord_density,
        bass_density=density_manager.bass_density
    )

    pattern_generator = PatternGenerator(
        genre_rules, mood,
        note_density=density_manager.note_density,
        rhythm_density=density_manager.rhythm_density,
        chord_density=density_manager.chord_density,
        bass_density=density_manager.bass_density,
        context=context
    )

    patterns = pattern_generator.generate_patterns(song_skeleton, bars)
    song_skeleton.build_arrangement(patterns)

    midi_output = MidiOutput()
    base_filename = midi_output.generate_output_filename(genre, mood, tempo, "4/4").replace('.mid', '')
    midi_output.save_to_separate_midi_files(song_skeleton, base_filename, genre_rules, context)

    print("Separate stems generated")

    # Check if stem files exist
    import os
    stems = ['bass', 'harmony', 'melody', 'rhythm']
    for stem in stems:
        filename = f"{base_filename}_{stem}.mid"
        unique_filename = midi_output.get_unique_filename(filename)
        if os.path.exists(unique_filename):
            print(f"Stem {stem}: {unique_filename} exists")
        else:
            print(f"Stem {stem}: {unique_filename} not found")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()