from config.parameter_config import ParameterConfig, Genre

# Test tempo clamping with pop genre (range 90-140)
try:
    config = ParameterConfig(genre=Genre.POP, tempo=50)
    print(f"Tempo 50 clamped to: {config.tempo} (should be 90)")
except Exception as e:
    print(f"Error: {e}")

try:
    config = ParameterConfig(genre=Genre.POP, tempo=150)
    print(f"Tempo 150 clamped to: {config.tempo} (should be 140)")
except Exception as e:
    print(f"Error: {e}")

try:
    config = ParameterConfig(genre=Genre.POP, tempo=120)
    print(f"Tempo 120 remains: {config.tempo}")
except Exception as e:
    print(f"Error: {e}")