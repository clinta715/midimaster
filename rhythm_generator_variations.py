"""
Variation Engine for Rhythm Generator

This module extends the RhythmGenerator with dynamic variation features:
- Bar-level A/B pattern switching
- Fills every 4/8 bars with increased ghost snares
- Ghost-note probability variations
- Hat thinning during micro-drops
- Micro-timing variations for human feel

Add this to your rhythm_generator.py to enable dynamic rhythm variations.
"""

import random
from typing import Dict, List, Any

class RhythmVariationEngine:
    """Engine for adding dynamic variations to rhythm patterns."""

    def __init__(self, pattern_strength: float = 1.0, swing_percent: float = 0.5,
                 fill_frequency: float = 0.25, ghost_note_level: float = 1.0):
        """
        Initialize the variation engine.

        Args:
            pattern_strength: How strongly to maintain the original pattern (0.0-1.0)
            swing_percent: Amount of swing feel (0.0-1.0)
            fill_frequency: How often to add fills (0.0-0.5, higher = more fills)
            ghost_note_level: Ghost note intensity multiplier (0.0-2.0)
        """
        self.pattern_strength = pattern_strength
        self.swing_percent = swing_percent
        self.fill_frequency = fill_frequency
        self.ghost_note_level = ghost_note_level

    def should_switch_pattern(self, bar: int, current_pattern: str) -> bool:
        """Determine if we should switch to alternate pattern."""
        # Switch patterns every 4 bars if pattern_strength allows variation
        if bar > 0 and bar % 4 == 0 and self.pattern_strength < 1.0:
            variation_chance = 1.0 - self.pattern_strength
            return random.random() < variation_chance
        return False

    def should_add_fill(self, bar: int) -> bool:
        """Determine if this bar should have fill variations."""
        if self.fill_frequency <= 0:
            return False

        # Add fills every N bars based on fill_frequency
        fill_interval = max(1, int(1.0 / self.fill_frequency))
        return (bar + 1) % fill_interval == 0

    def should_micro_drop(self, bar: int) -> bool:
        """Determine if this bar should have reduced intensity."""
        # Micro-drops every 8 bars
        return bar > 0 and bar % 8 == 0

    def modify_ghost_probability(self, voice: str, is_fill_bar: bool, base_probability: float) -> float:
        """Modify ghost note probability based on variation settings."""
        if voice != 'ghost_snare':
            return base_probability

        # Increase ghost snares during fills
        if is_fill_bar:
            return min(1.0, base_probability * 1.5 * self.ghost_note_level)

        # Apply ghost note level multiplier
        return min(1.0, base_probability * self.ghost_note_level)

    def modify_hat_density(self, voice: str, is_micro_drop: bool, base_probability: float) -> float:
        """Modify hat density during micro-drops."""
        if voice not in ('ch', 'oh'):
            return base_probability

        # Reduce hat density during micro-drops
        if is_micro_drop:
            return base_probability * 0.3  # Reduce to 30% density

        return base_probability

    def apply_swing_timing(self, start_time: float, voice: str, step_position: int) -> float:
        """Apply swing timing variations."""
        if abs(self.swing_percent - 0.5) < 0.01:
            return start_time  # No swing

        # Apply swing to even-numbered steps (16th notes)
        if step_position % 2 == 0:  # Even steps get delayed
            swing_delay = (self.swing_percent - 0.5) * 0.1  # Max 10% of beat
            return start_time + swing_delay
        else:  # Odd steps get early
            swing_early = (0.5 - self.swing_percent) * 0.1
            return max(0, start_time + swing_early)

        return start_time

    def generate_fill_pattern(self, voices: Dict[str, List[int]], steps_per_bar: int) -> Dict[str, List[int]]:
        """Generate a fill pattern by adding extra ghost snares and variations."""
        fill_voices = voices.copy()

        # Add extra ghost snares for fills
        if 'ghost_snare' in fill_voices:
            existing_ghosts = fill_voices['ghost_snare']
            # Add ghost snares on off-beats that aren't already occupied
            for step in range(steps_per_bar):
                if step % 2 == 1 and step not in existing_ghosts and random.random() < 0.4:
                    existing_ghosts.append(step)
            fill_voices['ghost_snare'] = sorted(existing_ghosts)

        # Add occasional snare hits for fill energy
        if 'snare' in fill_voices:
            existing_snares = fill_voices['snare']
            # Add extra snare on beat 3.5 (step 14 in 16-step pattern)
            if 14 not in existing_snares and random.random() < 0.6:
                existing_snares.append(14)
                fill_voices['snare'] = sorted(existing_snares)

        return fill_voices


# Example usage in rhythm generator:
"""
# In your rhythm_generator.py generate method:

variation_engine = RhythmVariationEngine(
    pattern_strength=pattern_strength,
    swing_percent=swing_percent,
    fill_frequency=fill_frequency,
    ghost_note_level=ghost_note_level
)

for bar in range(num_bars):
    # Check for pattern switching
    if variation_engine.should_switch_pattern(bar, current_template.get('name')):
        # Switch to alternate pattern
        current_template = next_template

    # Check for fills
    is_fill_bar = variation_engine.should_add_fill(bar)

    # Check for micro-drops
    is_micro_drop = variation_engine.should_micro_drop(bar)

    # Apply variations to voices
    bar_voices = current_template['voices'].copy()

    if is_fill_bar:
        bar_voices = variation_engine.generate_fill_pattern(bar_voices, steps_per_bar)

    # Process each voice with variations
    for v, steps in bar_voices.items():
        for s in steps:
            # Modify ghost probability
            base_prob = calculate_base_probability(v)
            ghost_prob = variation_engine.modify_ghost_probability(v, is_fill_bar, base_prob)
            hat_prob = variation_engine.modify_hat_density(v, is_micro_drop, base_prob)

            final_prob = min(ghost_prob, hat_prob)

            if random.random() < final_prob:
                start_time = bar_start_beats + s * step_len_beats
                start_time = variation_engine.apply_swing_timing(start_time, v, s)
                # ... rest of note generation
"""