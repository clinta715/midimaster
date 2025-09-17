import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from typing import List
from structures.data_structures import Pattern
from structures.song_skeleton import SongSkeleton
from generators.pattern_orchestrator import PatternOrchestrator
from genres.genre_factory import GenreFactory
from generators.generator_context import GeneratorContext

def create_simple_skeleton() -> SongSkeleton:
    """Create a simple song skeleton for testing."""
    return SongSkeleton(genre='pop', mood='energetic', tempo=120)

def test_pattern_strength(strength: float, genre_rules, mood: str = 'energetic'):
    """Test rhythm generation with specific pattern_strength."""
    print(f"\n=== Testing pattern_strength = {strength} ===")
    
    # Create context and orchestrator
    context = GeneratorContext(
        genre_rules=genre_rules,
        mood=mood,
        pattern_strength=strength
    )
    orchestrator = PatternOrchestrator(
        genre_rules=genre_rules,
        mood=mood,
        pattern_strength=strength
    )
    
    # Enable loop-based for refinement testing
    orchestrator.enable_loop_based_generation(max_iterations=2)
    
    # Generate rhythm
    skeleton = create_simple_skeleton()
    rhythm_pattern: Pattern = orchestrator.generate_beats_only(skeleton, num_bars=4)
    
    # Extract velocities
    velocities = [note.velocity for note in rhythm_pattern.notes]
    if velocities:
        avg_vel = sum(velocities) / len(velocities)
        min_vel = min(velocities)
        max_vel = max(velocities)
        print(f"Rhythm notes count: {len(velocities)}")
        print(f"Average velocity: {avg_vel:.2f}")
        print(f"Min velocity: {min_vel}, Max velocity: {max_vel}")
        print(f"Expected scaling: base ~64 * {strength} = ~{64 * strength:.2f}")
    else:
        print("No notes generated!")

def main():
    """Run pattern_strength tests."""
    print("Pattern Strength Verification Test")
    print("=" * 40)
    
    # Use GenreFactory for testing
    genre_rules = GenreFactory.create_genre_rules('pop')
    mood = 'energetic'
    
    strengths = [0.5, 1.0, 1.5]
    
    for strength in strengths:
        test_pattern_strength(strength, genre_rules, mood)
    
    print("\nTest complete. Check velocities for scaling effect.")

if __name__ == "__main__":
    main()