"""
Generator Utility Functions

This module contains utility functions used by the various music generators
for common operations like velocity determination and key/scale initialization.
"""

import random
from generators.generator_context import GeneratorContext


def get_velocity_for_mood(mood: str) -> int:
    """Get appropriate velocity based on mood.

    Maps the selected mood to an appropriate velocity value that
    affects the perceived volume and intensity of the notes.

    Args:
        mood: The selected mood ('happy', 'sad', 'energetic', 'calm')

    Returns:
        int: Velocity value (0-127) based on the selected mood
    """
    # Define velocity mappings for different moods
    mood_velocities = {
        'happy': 80,      # Medium-high velocity for happy mood
        'sad': 50,        # Low velocity for sad mood
        'energetic': 100, # High velocity for energetic mood
        'calm': 60        # Medium-low velocity for calm mood
    }

    # Return the velocity for the selected mood, with a default of 70
    return mood_velocities.get(mood, 70)


def initialize_key_and_scale(context: GeneratorContext) -> None:
    """Establish the key and scale for the generator context.

    This function prioritizes user-specified key/mode, falling back to
    genre-based random selection if none provided. The resulting scale
    forms the foundation for all melodic and harmonic content.

    Args:
        context: The GeneratorContext to initialize with key and scale
    """
    # Check for user-specified key/mode override
    if context.user_key and context.user_mode:
        selected_scale = f"{context.user_key} {context.user_mode}"
        print(f"Using user-specified scale: {selected_scale}")
    else:
        # Get scales from genre rules (fallback to C major if none specified)
        scales = context.genre_rules.get_scales()
        if not scales:
            scales = ['C major']
        # Randomly select one scale from the available options
        selected_scale = random.choice(scales)
        print(f"Selected scale from genre rules: {selected_scale}")

    # Use the MusicTheory convenience method to get scale pitches
    try:
        context.scale_pitches = context.music_theory.get_scale_pitches_from_string(
            selected_scale, octave_range=2
        )
        print(f"Generated scale pitches: {context.scale_pitches}")
    except Exception as e:
        print(f"Error generating scale: {e}")
        # Fallback to C major if scale generation fails
        context.scale_pitches = context.music_theory.get_scale_pitches_from_string(
            "C major", octave_range=2
        )

    # Parse the scale string for key and scale type (always update for consistency)
    parts = selected_scale.split()
    if len(parts) >= 2:
        context.current_key = parts[0]
        context.current_scale = ' '.join(parts[1:])
    else:
        context.current_key = 'C'
        context.current_scale = 'major'