from generators.generator_utils import initialize_key_and_scale
from generators.generator_context import GeneratorContext
from genres.genre_factory import GenreFactory

# Create context without user_key/mode
genre_rules = GenreFactory().create_genre_rules('jazz')
mood = 'energetic'
context = GeneratorContext(genre_rules, mood, note_density=0.5, rhythm_density=0.5, chord_density=0.5, bass_density=0.5)
context.user_key = None
context.user_mode = None

print("Before initialization:")
print(f"user_key: {context.user_key}, user_mode: {context.user_mode}")
print(f"current_key: {context.current_key}, current_scale: {context.current_scale}")
print(f"scale_pitches: {len(context.scale_pitches)} pitches")

# Call the function
initialize_key_and_scale(context)

print("\nAfter initialization (fallback to genre random):")
print(f"user_key: {context.user_key}, user_mode: {context.user_mode}")
print(f"current_key: {context.current_key}, current_scale: {context.current_scale}")
print(f"scale_pitches: {len(context.scale_pitches)} pitches")
print(f"Selected scale: {context.current_key} {context.current_scale}")
print(f"Genre scales available: {context.genre_rules.get_scales()}")

# Run multiple times to verify random selection
print("\nRunning 3 times to show random selection:")
for i in range(3):
    test_context = GeneratorContext(genre_rules, mood, note_density=0.5, rhythm_density=0.5, chord_density=0.5, bass_density=0.5)
    test_context.user_key = None
    test_context.user_mode = None
    initialize_key_and_scale(test_context)
    print(f"Run {i+1}: {test_context.current_key} {test_context.current_scale}")