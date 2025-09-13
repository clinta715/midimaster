from generators.generator_context import GeneratorContext
from music_theory import MusicTheory
from genres.genre_factory import GenreFactory

genre_rules = GenreFactory().create_genre_rules('pop')
mood = 'happy'

# Test valid case
context = GeneratorContext(genre_rules, mood)
try:
    context.set_user_key_mode("C", "major")
    print("Valid case (C major): Success - user_key='C', user_mode='major'")
    print(f"Set: {context.user_key}, {context.user_mode}")
except ValueError as e:
    print(f"Valid case failed: {e}")

# Test invalid key
context = GeneratorContext(genre_rules, mood)
try:
    context.set_user_key_mode("Z", "major")
    print("Invalid key case: Unexpected success")
except ValueError as e:
    print(f"Invalid key (Z major): Success - Error: {e}")

# Test invalid mode
context = GeneratorContext(genre_rules, mood)
try:
    context.set_user_key_mode("C", "invalid_mode")
    print("Invalid mode case: Unexpected success")
except ValueError as e:
    print(f"Invalid mode (C invalid_mode): Success - Error: {e}")

# Test flat/sharp
context = GeneratorContext(genre_rules, mood)
try:
    context.set_user_key_mode("F#", "dorian")
    print("Valid sharp (F# dorian): Success")
except ValueError as e:
    print(f"Valid sharp failed: {e}")