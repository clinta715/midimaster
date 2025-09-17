"""
Debug script to test HarmonyGenerator call in PatternOrchestrator.
This will trigger the generation and show debug logs before the AttributeError.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from generators.pattern_orchestrator import PatternOrchestrator
from structures.song_skeleton import SongSkeleton
from genres.genre_factory import GenreFactory
from generators.density_manager import create_density_manager_from_preset

# Minimal setup
genre = 'pop'
mood = 'happy'
tempo = 120
bars = 16
density = 'balanced'

# Create genre rules
genre_rules = GenreFactory.create_genre_rules(genre)

# Create song skeleton
song_skeleton = SongSkeleton(genre, tempo, mood)

# Create density manager
density_manager = create_density_manager_from_preset(density)

# Create orchestrator
orchestrator = PatternOrchestrator(
    genre_rules=genre_rules,
    mood=mood,
    note_density=density_manager.note_density,
    rhythm_density=density_manager.rhythm_density,
    chord_density=density_manager.chord_density,
    bass_density=density_manager.bass_density
)

print("Starting generation...")
print(f"Orchestrator class: {orchestrator.__class__.__name__}")
# type: ignore[reportAttributeAccessIssue]
print(f"Has generate_patterns: {hasattr(orchestrator, 'generate_patterns')}")
if hasattr(orchestrator, 'generate_patterns'):
    print(f"Method type: {type(orchestrator.generate_patterns)}")
print("Available methods:", [m for m in dir(orchestrator) if not m.startswith('_')])
try:
    patterns = orchestrator.generate_patterns(song_skeleton, bars)
    print("Generation succeeded unexpectedly.")
except AttributeError as e:
    print(f"Caught expected AttributeError: {e}")
    print("Debug logs should have printed above.")

print("Debug script complete.")