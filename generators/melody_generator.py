"""
Melody Generator Module
 
This module contains the MelodyGenerator class responsible for generating
melodic patterns based on genre rules and musical context.
"""

import random
from typing import TYPE_CHECKING, List, Optional, Dict

from structures.data_structures import Note, Pattern, PatternType
from generators.generator_utils import get_velocity_for_mood, initialize_key_and_scale

if TYPE_CHECKING:
    from generators.generator_context import GeneratorContext


class MelodyGenerator:
    """Generates melodic patterns using scale-based pitch selection.
 
    The MelodyGenerator creates single-note melodic lines using pitches
    from the established scale following typical melodic principles.
    """
 
    def __init__(self, context: 'GeneratorContext'):
        """
        Initialize the MelodyGenerator.
 
        Args:
            context: Shared GeneratorContext containing music theory and configuration
        """
        self.context = context
        self.last_pitch: Optional[int] = None
 
    def _choose_next_pitch(self, contour: str,
                           interval_weights: Dict[int, float],
                           min_pitch: Optional[int],
                           max_pitch: Optional[int]) -> int:
        """Choose the next pitch based on melodic contour, interval weights, and register limits.
 
        Args:
            contour: The melodic contour ('rising', 'falling', 'arc', 'valley')
            interval_weights: Weighted probabilities for diatonic step sizes
            min_pitch: Optional absolute MIDI min bound
            max_pitch: Optional absolute MIDI max bound
 
        Returns:
            The next pitch for the melody
        """
        # Establish usable scale
        if not self.context.scale_pitches:
            scale = [60, 62, 64, 65, 67, 69, 71]
        else:
            scale = self.context.scale_pitches
 
        # Filter scale by register bounds if provided
        def in_register(p: int) -> bool:
            if min_pitch is not None and p < min_pitch:
                return False
            if max_pitch is not None and p > max_pitch:
                return False
            return True
 
        constrained_scale = [p for p in scale if in_register(p)] or scale
 
        if self.last_pitch is None:
            return random.choice(constrained_scale)
 
        # Directionality based on contour
        if contour == 'rising':
            direction_probabilities = {'up': 0.7, 'down': 0.2, 'same': 0.1}
        elif contour == 'falling':
            direction_probabilities = {'up': 0.2, 'down': 0.7, 'same': 0.1}
        else:  # arc, valley, or other
            direction_probabilities = {'up': 0.45, 'down': 0.45, 'same': 0.1}
 
        # Build candidate next pitches with combined weights
        possible_pitches: List[tuple[int, float]] = []
        for interval, prob in interval_weights.items():
            for direction, dir_prob in direction_probabilities.items():
                if direction == 'up':
                    next_pitch = self.last_pitch + interval
                elif direction == 'down':
                    next_pitch = self.last_pitch - interval
                else:
                    next_pitch = self.last_pitch
 
                if next_pitch in constrained_scale:
                    possible_pitches.append((next_pitch, prob * dir_prob))
 
        if not possible_pitches:
            # Fallback if constraints leave no options
            return random.choice(constrained_scale)
 
        pitches, weights = zip(*possible_pitches)
        return random.choices(pitches, weights=weights, k=1)[0]
 
    def generate(self, num_bars: int) -> Pattern:
        """Generate a melody pattern.
 
        Creates a single-note melodic line using pitches from the established scale.
        The melody follows typical melodic principles with stepwise motion and
        occasional leaps, with note durations and velocities influenced by the mood.
 
        Args:
            num_bars: Number of bars to generate
 
        Returns:
            Pattern object containing the melody notes
        """
        # Ensure key and scale are established
        if not self.context.scale_pitches:
            initialize_key_and_scale(self.context)
        # Scale validation
        if not self.context.scale_pitches or len(self.context.scale_pitches) < 7:
            print("Warning: Invalid scale detected. Falling back to C major scale.")
            self.context.scale_pitches = [60, 62, 64, 65, 67, 69, 71, 72, 74, 76, 77, 79, 81, 83]  # C major over 2 octaves
 
        notes = []
        chords = []
 
        # Fetch subgenre-aware melody style from genre rules
        subgenre = getattr(self.context, 'subgenre', None)
        style = self.context.genre_rules.get_melody_style(subgenre)
        # Fallback for missing melody style configurations
        if not style or 'interval_weights' not in style or not style['interval_weights']:
            print("Warning: Missing melody style configuration. Using default interval weights.")
            style = {
                'contour_weights': {'rising': 0.25, 'falling': 0.25, 'arc': 0.25, 'valley': 0.25},
                'interval_weights': {1: 0.4, 2: 0.3, 3: 0.15, 4: 0.1, 5: 0.05},
                'min_pitch': 48,
                'max_pitch': 84
            }
 
        contour_weights = style.get('contour_weights', {'rising': 0.25, 'falling': 0.25, 'arc': 0.25, 'valley': 0.25})
        interval_weights = style.get('interval_weights', {1: 0.4, 2: 0.3, 3: 0.15, 4: 0.1, 5: 0.05})
        min_pitch = style.get('min_pitch', 48)
        max_pitch = style.get('max_pitch', 84)
 
        start_time = 0.0
        self.last_pitch = None
 
        # Choose a melodic contour for this pattern using weights
        contour_names = list(contour_weights.keys())
        contour_probs = list(contour_weights.values())
        contour = random.choices(contour_names, weights=contour_probs, k=1)[0]
 
        # Generate melody notes from the established scale
        # 4 beats per bar is the default time signature
        total_beats = num_bars * 4
        while start_time < total_beats:
            if self.context.density_manager.should_place_note(start_time, total_beats):
                # Use density manager to get available durations
                available_durations = self.context.density_manager.get_available_durations(
                    self.context.density_manager.note_density
                )
                duration = random.choice(available_durations)
 
                if start_time + duration > total_beats:
                    duration = total_beats - start_time
 
                pitch = self._choose_next_pitch(contour, interval_weights, min_pitch, max_pitch)
                self.last_pitch = pitch
 
                # Adjust velocity based on mood and metric position
                base_velocity = get_velocity_for_mood(self.context.mood)
                 
                # Emphasize notes on strong beats (e.g., beat 1 and 3 in 4/4)
                beat_in_bar = start_time % 4 # Assuming 4 beats per bar
                if abs(beat_in_bar - 0) < 0.01: # Strong beat 1
                    velocity = min(127, base_velocity + 20)
                elif abs(beat_in_bar - 2) < 0.01: # Medium strong beat 3
                    velocity = min(127, base_velocity + 10)
                else: # Weak beats
                    velocity = base_velocity
 
                # Add a small random variation for humanization
                velocity = max(0, min(127, velocity + random.randint(-5, 5)))
 
                # Ensure pitch is in scale (should already be due to _choose_next_pitch logic)
                assert pitch in self.context.scale_pitches, f"Melody pitch {pitch} not in scale {self.context.scale_pitches}"
                note = Note(pitch, duration, velocity, start_time)
                notes.append(note)
 
                start_time += duration
            else:
                # Rest for one beat when no note is placed
                start_time += 1.0
 
        # Return the melody pattern
        return Pattern(PatternType.MELODY, notes, chords)