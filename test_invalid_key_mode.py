from generators.generator_context import GeneratorContext
from genres.genre_factory import GenreFactory

genre_rules = GenreFactory.create_genre_rules('pop')
context = GeneratorContext(genre_rules, 'happy')

print("Testing invalid key/mode")
try:
    context.set_user_key_mode("Z", "major")
    print("Should have failed")
except ValueError as e:
    print(f"Correctly rejected invalid key: {e}")

context2 = GeneratorContext(genre_rules, 'happy')
try:
    context2.set_user_key_mode("C", "invalid")
    print("Should have failed")
except ValueError as e:
    print(f"Correctly rejected invalid mode: {e}")

print("Testing fallback")
context3 = GeneratorContext(genre_rules, 'happy')
# No set_user_key_mode, should use fallback
from generators.generator_utils import initialize_key_and_scale
try:
    initialize_key_and_scale(context3)
    print(f"Fallback worked: {context3.current_key} {context3.current_scale}")
except Exception as e:
    print(f"Fallback error: {e}")