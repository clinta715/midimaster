from genres.genre_rules import GenreRules
from music_theory import MusicTheory

# Create GenreRules instance
genre_rules = GenreRules()

# Pop progression I-IV-V
progression = ['I', 'IV', 'V']

# Test C major
transposed_c = genre_rules.get_transposed_chord_progressions('C', 'major')
print("C major I-IV-V transposed:")
for progression in transposed_c:
    print(f"C major progression: {progression}")
    for chord in progression:
        pitches = MusicTheory.get_chord_pitches_from_roman(chord, 'C major')
        print(f"  {chord}: pitches {pitches}")

# Check if I-IV-V is included
pop_progression = ['I', 'IV', 'V']
expected_c = ['C', 'F', 'G']
expected_a = ['Am', 'Dm', 'Em']
print("\nExpected for C major I-IV-V: ", expected_c)
print("Expected for A dorian I-IV-V: ", expected_a)
if any(prog == expected_c for prog in transposed_c):
    print("C major transposition matches expected for I-IV-V")
else:
    print("C major transposition does not match expected")

# Test A dorian
transposed_a = genre_rules.get_transposed_chord_progressions('A', 'dorian')
print("\nA dorian I-IV-V transposed:")
for progression in transposed_a:
    print(f"A dorian progression: {progression}")
    for chord in progression:
        pitches = MusicTheory.get_chord_pitches_from_roman(chord, 'A dorian')
        print(f"  {chord}: pitches {pitches}")
if any(prog == expected_a for prog in transposed_a):
    print("A dorian transposition matches expected for I-IV-V")
else:
    print("A dorian transposition does not match expected")