from genres.genre_factory import GenreFactory
from music_theory import MusicTheory

# Setup
factory = GenreFactory()
pop_rules = factory.create_genre_rules('pop')

# Test C major
transposed_c = pop_rules.get_transposed_chord_progressions('C', 'major')
print("C major progressions:")
for progression in transposed_c:
    print(f"  Progression: {progression}")
    for chord in progression:
        pitches = MusicTheory.get_chord_pitches_from_roman(chord, 'C major')
        print(f"    {chord}: {pitches}")

# Test A dorian
transposed_a = pop_rules.get_transposed_chord_progressions('A', 'dorian')
print("\nA dorian progressions:")
for progression in transposed_a:
    print(f"  Progression: {progression}")
    for chord in progression:
        pitches = MusicTheory.get_chord_pitches_from_roman(chord, 'A dorian')
        print(f"    {chord}: {pitches}")