"""
Generator Utility Functions

This module contains utility functions used by the various music generators
for common operations like velocity determination and key/scale initialization.
"""

import random
from typing import List, Any, Optional
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

import math
from typing import Iterable

def _convert_repo_pattern_to_durations(pattern_obj: Any, length_beats: float, subdivision: int) -> List[float]:
    """
    Convert repository pattern_json into the internal durations list expected by RhythmGenerator.

    Accepted pattern_obj shapes:
    - List[float|int]: 
        a) Durations in beats (sum ~= length_beats) -> returned as-is (filtered to > 0)
        b) Step grid (0/1 or truthy values) across the bar -> convert truthy steps to onsets and return inter-onset durations
    - List[dict]: Event list with keys like 'onset_beats' (or 'onset'/'start'/'time' variants)
        -> sort by onset within [0, length_beats) and compute inter-onset durations

    Args:
        pattern_obj: Parsed JSON from repository (already a Python object)
        length_beats: Target pattern length in beats (e.g., 4.0)
        subdivision: Pulses per beat (typical values: 2,3,4,6,8,12,16,24). Used for step grid inference.

    Returns:
        List[float]: durations in beats summing to approximately length_beats. Empty on irrecoverable input.
    """
    try:
        L = float(length_beats) if length_beats and length_beats > 0 else 4.0
    except Exception:
        L = 4.0

    if not isinstance(pattern_obj, list) or len(pattern_obj) == 0:
        return []

    # Helper to finalize durations: filter tiny/negative, normalize final sum to <= L but never zero
    def finalize(durs: Iterable[float]) -> List[float]:
        cleaned: List[float] = [float(x) for x in durs if isinstance(x, (int, float)) and float(x) > 1e-6]
        if not cleaned:
            return []
        total = sum(cleaned)
        # If the total is way larger than L (e.g., multiple bars), reduce to single-bar cycle
        if total > 0 and total > L + 1e-3:
            # Normalize proportionally to fit in L preserving ratios
            scale = L / total
            cleaned = [max(1e-4, x * scale) for x in cleaned]
        # If the total is slightly less than L, add final tail to complete the bar
        total = sum(cleaned)
        if total < L - 1e-3:
            cleaned.append(max(1e-4, L - total))
        return cleaned

    # Case 1: list of numbers
    if all(isinstance(x, (int, float)) for x in pattern_obj):
        nums = [float(x) for x in pattern_obj]

        # Heuristic A: durations if sum close to L or within [0.5*L, 2*L]
        total = sum(max(0.0, n) for n in nums)
        nonpos = sum(1 for n in nums if n <= 0.0)

        # Heuristic B: step grid if values are mostly 0/1 or booleans
        unique_vals = {int(v) if float(v).is_integer() else v for v in nums}
        looks_like_steps = unique_vals.issubset({0, 1}) or all(0.0 <= v <= 1.0 for v in nums) and nonpos < len(nums)

        # If it plausibly encodes durations
        if total > 0.0 and (abs(total - L) <= 0.5 or (0.5 * L) <= total <= (2.0 * L)) and not looks_like_steps:
            durs = [n for n in nums if n > 0.0]
            return finalize(durs)

        # Otherwise treat as step grid over the bar
        N = len(nums)
        if N <= 0:
            return []
        step = L / float(N)
        onsets = [i * step for i, v in enumerate(nums) if v and float(v) > 0.0]
        if not onsets:
            # No active steps; fallback to a single whole-bar hit
            return [L]
        onsets.sort()
        durs: List[float] = []
        for i, t in enumerate(onsets):
            nxt = onsets[(i + 1) % len(onsets)]
            dur = (nxt - t) if i < len(onsets) - 1 else (L - t + onsets[0])
            # Protect against numerical issues
            durs.append(max(1e-4, dur))
        return finalize(durs)

    # Case 2: list of dict events
    if all(isinstance(x, dict) for x in pattern_obj):
        # Accept multiple possible keys for onset
        onset_keys = ("onset_beats", "onset", "start_beats", "start", "time_beats", "time")
        onsets: List[float] = []
        for ev in pattern_obj:
            if not isinstance(ev, dict):
                continue
            onset_val: Optional[float] = None
            for k in onset_keys:
                if k in ev:
                    try:
                        onset_val = float(ev[k])
                        break
                    except Exception:
                        pass
            if onset_val is None:
                continue
            # Wrap/clamp into the bar
            t = onset_val % L
            if 0.0 <= t < L + 1e-6:
                onsets.append(t)

        if not onsets:
            return []

        onsets = sorted(onsets)
        durs: List[float] = []
        for i, t in enumerate(onsets):
            nxt = onsets[(i + 1) % len(onsets)]
            dur = (nxt - t) if i < len(onsets) - 1 else (L - t + onsets[0])
            durs.append(max(1e-4, dur))
        return finalize(durs)

    # Unknown shape -> cannot convert
    return []