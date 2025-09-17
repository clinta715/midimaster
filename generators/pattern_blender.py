"""
Pattern Blender Module

This module provides the PatternBlender class for combining elements from multiple
musical patterns. It supports rhythmic, harmonic, and melodic blending, along with
cross-fading and layering techniques. The class integrates with existing Pattern
data structures and uses the PatternVariationEngine for smooth transitions.

Key features:
- Rhythmic blending: Combine drum elements (kick, snare, hi-hat) from different patterns
- Harmonic blending: Merge chord progressions with smooth transitions
- Melodic blending: Blend pitch contours and scales
- Cross-fading: Temporal transitions between patterns
- Layering: Combine complementary patterns with balance controls

Usage:
    blender = PatternBlender()
    blended = blender.blend_rhythms([pat1, pat2], weights=[0.6, 0.4])
"""

from typing import List, Optional, Dict, Tuple
from structures.data_structures import Pattern, Note, Chord, PatternType
from generators.pattern_variations import PatternVariationEngine
from music_theory import MusicTheory
import random
import copy
import math

class PatternBlender:
    """
    Class for blending musical patterns from multiple sources.

    Provides methods to combine rhythmic elements, merge harmonic progressions,
    blend melodic contours, perform cross-fades, and layer patterns. Integrates
    with the PatternVariationEngine for applying variations during transitions.

    Args:
        variation_engine: Optional PatternVariationEngine instance for transitions.
                          Defaults to a new instance.
    """

    # Standard drum pitches for rhythmic blending
    DRUM_PITCHES = {
        36: 'kick',      # Bass Drum
        38: 'snare',     # Acoustic Snare
        42: 'hihat_closed',  # Closed Hi-Hat
        46: 'hihat_open',    # Open Hi-Hat
        49: 'crash',     # Crash Cymbal
        51: 'ride'       # Ride Cymbal
    }

    def __init__(self, variation_engine: Optional[PatternVariationEngine] = None):
        """
        Initialize the PatternBlender.

        Args:
            variation_engine: The engine for applying variations (default: new instance)
        """
        self.variation_engine = variation_engine or PatternVariationEngine()

    def _normalize_weights(self, weights: Optional[List[float]], num_patterns: int) -> List[float]:
        """
        Normalize weights to sum to 1.0. If None, use equal weights.

        Args:
            weights: List of weights or None
            num_patterns: Number of patterns

        Returns:
            Normalized weights list
        """
        if weights is None:
            return [1.0 / num_patterns] * num_patterns
        total = sum(weights)
        if total == 0:
            return [0.0] * num_patterns
        return [w / total for w in weights]

    def _get_notes_by_type(self, pattern: Pattern, note_type: str) -> List[Note]:
        """
        Extract notes of a specific type (e.g., 'kick') from a rhythm pattern.

        Args:
            pattern: The rhythm Pattern
            note_type: Drum type (e.g., 'kick')

        Returns:
            List of matching notes
        """
        pitch = next((p for p, t in self.DRUM_PITCHES.items() if t == note_type), None)
        if pitch is None:
            return []
        all_notes = pattern.get_all_notes()
        return [n for n in all_notes if n.pitch == pitch]

    def blend_rhythms(self, patterns: List[Pattern], weights: Optional[List[float]] = None,
                      blend_type: str = 'hybrid') -> Pattern:
        """
        Blend rhythmic elements from multiple patterns.

        Supports combining kick from one, hi-hat from another, etc. for hybrid rhythms.

        Args:
            patterns: List of rhythm Patterns to blend
            weights: Optional weights for each pattern (automatic equal if None)
            blend_type: 'hybrid' (combine by element), 'merge' (union), 'interpolate' (weighted average)

        Returns:
            Blended rhythm Pattern
        """
        if not all(p.pattern_type == PatternType.RHYTHM for p in patterns):
            raise ValueError("All patterns must be of type RHYTHM")

        weights = self._normalize_weights(weights, len(patterns))
        blended_notes = []

        if blend_type == 'hybrid':
            # Combine specific elements: e.g., kicks from highest weight, hi-hats from next
            sorted_patterns = sorted(enumerate(patterns), key=lambda x: weights[x[0]], reverse=True)
            element_order = ['kick', 'snare', 'hihat_closed', 'hihat_open', 'crash', 'ride']

            for elem in element_order:
                # Select from pattern with highest remaining weight for this element
                for idx, pat in sorted_patterns:
                    notes = self._get_notes_by_type(pat, elem)
                    if notes:
                        # Weighted selection or copy
                        blended_notes.extend(notes)
                        break
        elif blend_type == 'merge':
            # Union of all notes, remove duplicates by time/pitch
            all_notes = []
            for pat in patterns:
                all_notes.extend(pat.get_all_notes())
            # Sort and remove overlaps (simple: keep first)
            all_notes.sort(key=lambda n: (n.start_time, n.pitch))
            unique_notes = []
            for n in all_notes:
                if not any(un.start_time == n.start_time and un.pitch == n.pitch for un in unique_notes):
                    unique_notes.append(n)
            blended_notes = unique_notes
        elif blend_type == 'interpolate':
            # Weighted average positions/velocities
            max_time = max(max(n.start_time + n.duration for n in p.get_all_notes()) for p in patterns if p.get_all_notes())
            time_steps = [i * 0.25 for i in range(int(max_time / 0.25))]  # Quarter note steps

            for t in time_steps:
                total_vel = 0
                count = 0
                for i, pat in enumerate(patterns):
                    notes_at_t = [n for n in pat.get_all_notes() if abs(n.start_time - t) < 0.1]
                    if notes_at_t:
                        avg_vel = sum(n.velocity for n in notes_at_t) / len(notes_at_t)
                        total_vel += avg_vel * weights[i]
                        count += weights[i]
                if count > 0:
                    avg_vel = total_vel / count
                    # Create note at t with average vel, random pitch from patterns
                    pitches = [random.choice([n.pitch for n in p.get_all_notes() if abs(n.start_time - t) < 0.1])
                               for p in patterns if any(abs(n.start_time - t) < 0.1 for n in p.get_all_notes())]
                    if pitches:
                        pitch = random.choice(pitches)
                        blended_notes.append(Note(pitch, 0.25, int(avg_vel), t))

        # Apply variation for smoothness
        if blended_notes:
            temp_pat = Pattern(PatternType.RHYTHM, blended_notes, [])
            blended = self.variation_engine.rhythmic_density_adjust(temp_pat, 1.0, 0.3)
        else:
            blended = Pattern(PatternType.RHYTHM, [], [])

        return blended

    def blend_harmonies(self, patterns: List[Pattern], weights: Optional[List[float]] = None,
                        transition_smoothness: float = 0.5) -> Pattern:
        """
        Blend harmonic progressions from multiple patterns.

        Merges chords, ensuring compatibility via common tones and smooth voice leading.

        Args:
            patterns: List of harmony Patterns
            weights: Optional weights (automatic equal)
            transition_smoothness: Intensity for variation engine transitions (0.0-1.0)

        Returns:
            Blended harmony Pattern
        """
        if not all(p.pattern_type == PatternType.HARMONY for p in patterns):
            raise ValueError("All patterns must be of type HARMONY")

        weights = self._normalize_weights(weights, len(patterns))
        all_chords = []

        # Collect all chords, weighted by pattern
        for i, pat in enumerate(patterns):
            for chord in pat.chords:
                # Weight by duplicating or scaling duration/velocity
                weighted_chord = copy.deepcopy(chord)
                weighted_chord.notes = [Note(n.pitch, n.duration * weights[i], n.velocity, n.start_time) for n in chord.notes]
                all_chords.append(weighted_chord)

        # Sort by start_time
        all_chords.sort(key=lambda c: c.start_time)

        # Merge overlapping chords using common tones
        merged_chords = []
        current_chord = None
        for chord in all_chords:
            if current_chord is None or chord.start_time > current_chord.start_time + current_chord.duration():
                if current_chord:
                    merged_chords.append(current_chord)
                current_chord = chord
            else:
                # Merge: add pitches not already present
                existing_pitches = {n.pitch for n in current_chord.notes}
                for n in chord.notes:
                    if n.pitch not in existing_pitches:
                        current_chord.add_note(Note(n.pitch, current_chord.duration(), n.velocity, current_chord.start_time))

        if current_chord:
            merged_chords.append(current_chord)

        # Apply transitions for smoothness
        if len(merged_chords) > 1 and transition_smoothness > 0:
            temp_pat = Pattern(PatternType.HARMONY, [], merged_chords)
            blended = self.variation_engine.velocity_randomization(temp_pat, 15, transition_smoothness)
        else:
            blended = Pattern(PatternType.HARMONY, [], merged_chords)

        # Ensure scale compatibility
        scale = "C major"  # Assume or from context
        for chord in blended.chords:
            pitches = [n.pitch for n in chord.notes]
            scale_pitches = set(MusicTheory.get_scale_pitches_from_string(scale))
            chord.notes = [n for n in chord.notes if n.pitch in scale_pitches]

        return blended

    def blend_melodies(self, patterns: List[Pattern], weights: Optional[List[float]] = None,
                       blend_type: str = 'contour') -> Pattern:
        """
        Blend melodic contours from multiple patterns.

        Interpolates pitches and rhythms for seamless transitions.

        Args:
            patterns: List of melody Patterns
            weights: Optional weights (automatic equal)
            blend_type: 'contour' (pitch interpolation), 'merge' (select notes), 'scale' (common scale)

        Returns:
            Blended melody Pattern
        """
        if not all(p.pattern_type == PatternType.MELODY for p in patterns):
            raise ValueError("All patterns must be of type MELODY")

        weights = self._normalize_weights(weights, len(patterns))
        blended_notes = []

        if blend_type == 'contour':
            # Align by time, interpolate pitches
            max_time = max(max((n.start_time + n.duration for n in p.notes), default=0) for p in patterns)
            time_steps = sorted(set(n.start_time for p in patterns for n in p.notes))

            for t in time_steps:
                pitches = []
                durations = []
                velocities = []
                for i, pat in enumerate(patterns):
                    notes_at_t = [n for n in pat.notes if abs(n.start_time - t) < 0.125]
                    if notes_at_t:
                        note = random.choice(notes_at_t)  # Or closest
                        pitches.append(note.pitch)
                        durations.append(note.duration)
                        velocities.append(note.velocity)
                        w = weights[i]
                    else:
                        pitches.append(None)
                        durations.append(0.25)
                        velocities.append(64)
                        w = 0

                if any(p is not None for p in pitches):
                    # Weighted average pitch (quantize to scale if needed)
                    valid_pitches = [(p, weights[i]) for i, p in enumerate(pitches) if p is not None]
                    if valid_pitches:
                        total_weight = sum(w for _, w in valid_pitches)
                        avg_pitch = sum(p * w for p, w in valid_pitches) / total_weight
                        avg_pitch = round(avg_pitch)  # Simple quantize

                        avg_dur = sum(d * weights[i] for i, d in enumerate(durations) if pitches[i] is not None) / sum(weights[i] for i, p in enumerate(pitches) if p is not None)
                        avg_vel = sum(v * weights[i] for i, v in enumerate(velocities) if pitches[i] is not None) / sum(weights[i] for i, p in enumerate(pitches) if p is not None)

                        blended_notes.append(Note(int(avg_pitch), avg_dur, int(avg_vel), t))
        elif blend_type == 'merge':
            # Weighted selection of notes
            all_notes = []
            for i, pat in enumerate(patterns):
                for n in pat.notes:
                    # Duplicate based on weight (approximate)
                    dups = int(1 / weights[i]) if weights[i] > 0 else 1
                    all_notes.extend([copy.deepcopy(n)] * dups)
            if all_notes:
                blended_notes = random.sample(all_notes, min(len(all_notes), len(all_notes)))  # Shuffle/select
        elif blend_type == 'scale':
            # Merge pitch collections, ensure common scale
            all_pitches = set()
            for pat in patterns:
                all_pitches.update(n.pitch for n in pat.notes)
            common_scale = MusicTheory.get_scale_pitches_from_string("C major")  # Assume
            filtered_notes = []
            for pat in patterns:
                for n in pat.notes:
                    if n.pitch in common_scale:
                        filtered_notes.append(n)
            blended_notes = filtered_notes[:int(len(filtered_notes) * sum(weights))]  # Weighted length

        # Apply variation
        if blended_notes:
            temp_pat = Pattern(PatternType.MELODY, blended_notes, [])
            blended = self.variation_engine.pitch_transpose(temp_pat, 0, 0.2)  # Subtle variation
        else:
            blended = Pattern(PatternType.MELODY, [], [])

        return blended

    def cross_fade(self, pattern1: Pattern, pattern2: Pattern, fade_length: float,
                   position: float = 0.5, intensity: float = 1.0) -> Pattern:
        """
        Create a temporal cross-fade between two patterns.

        Fades from pattern1 to pattern2 over the specified length.

        Args:
            pattern1: Starting pattern
            pattern2: Ending pattern
            fade_length: Duration of fade in beats
            position: Where to place the fade (0.0 = start, 1.0 = end)
            intensity: Strength of variation during fade

        Returns:
            Pattern with cross-fade
        """
        # Time-stretch both to fit total length
        total_length = max(pattern1.get_all_notes()[-1].start_time if pattern1.get_all_notes() else 0,
                           pattern2.get_all_notes()[-1].start_time if pattern2.get_all_notes() else 0) + fade_length
        p1_stretched = self.variation_engine.time_stretch(pattern1, total_length / (pattern1.notes[0].start_time if pattern1.notes else 1), intensity * 0.5)
        p2_stretched = self.variation_engine.time_stretch(pattern2, total_length / (pattern2.notes[0].start_time if pattern2.notes else 1), intensity * 0.5)

        fade_start = position * (total_length - fade_length)
        all_notes1 = p1_stretched.get_all_notes()
        all_notes2 = p2_stretched.get_all_notes()

        blended_notes = []
        for n1 in all_notes1:
            if n1.start_time < fade_start:
                blended_notes.append(n1)
            elif n1.start_time < fade_start + fade_length:
                fade_factor = (n1.start_time - fade_start) / fade_length
                n1.velocity = int(n1.velocity * (1 - fade_factor))
                blended_notes.append(n1)
        for n2 in all_notes2:
            if n2.start_time >= fade_start + fade_length:
                blended_notes.append(n2)
            elif n2.start_time >= fade_start:
                fade_factor = (n2.start_time - fade_start) / fade_length
                n2.velocity = int(n2.velocity * fade_factor)
                blended_notes.append(n2)

        # Sort and apply variation
        blended_notes.sort(key=lambda n: n.start_time)
        temp_pat = Pattern(pattern1.pattern_type, blended_notes, p1_stretched.chords + p2_stretched.chords)
        final = self.variation_engine.velocity_randomization(temp_pat, 10, intensity)

        return final

    def layer_patterns(self, patterns: List[Pattern], balances: Optional[List[float]] = None,
                       max_overlap: float = 0.8) -> Pattern:
        """
        Layer multiple complementary patterns with balance controls.

        Adjusts velocities to prevent clipping and handles overlaps.

        Args:
            patterns: List of patterns to layer
            balances: Optional balance weights (0.0-1.0, automatic equal)
            max_overlap: Max velocity overlap factor

        Returns:
            Layered Pattern
        """
        if balances is None:
            balances = [1.0 / len(patterns)] * len(patterns)

        all_notes = []
        all_chords = []

        for i, pat in enumerate(patterns):
            balance = balances[i]
            scaled_notes = []
            for n in pat.notes:
                scaled_n = copy.deepcopy(n)
                scaled_n.velocity = int(n.velocity * balance)
                scaled_notes.append(scaled_n)
            all_notes.extend(scaled_notes)

            scaled_chords = []
            for c in pat.chords:
                scaled_c = copy.deepcopy(c)
                for n in scaled_c.notes:
                    n.velocity = int(n.velocity * balance)
                scaled_chords.append(scaled_c)
            all_chords.extend(scaled_chords)

        # Handle overlaps: reduce velocity if multiple notes at same time
        all_notes.sort(key=lambda n: (n.start_time, n.pitch))
        for i in range(len(all_notes)):
            t = all_notes[i].start_time
            same_time_notes = [all_notes[j] for j in range(i, len(all_notes)) if abs(all_notes[j].start_time - t) < 0.01]
            if len(same_time_notes) > 1:
                overlap_factor = max_overlap / len(same_time_notes)
                for n in same_time_notes:
                    n.velocity = int(n.velocity * overlap_factor)

        # Determine type from first pattern
        p_type = patterns[0].pattern_type if patterns else PatternType.MELODY

        layered = Pattern(p_type, all_notes, all_chords)
        # Apply subtle variation
        layered = self.variation_engine.add_ornamentation(layered, "ghost", 0.2)

        return layered

# Examples
if __name__ == "__main__":
    pass
    # Assume some patterns exist
    # pat1 = Pattern(...)  # Rhythm with kicks
    # pat2 = Pattern(...)  # Rhythm with hi-hats
    # blender = PatternBlender()
    # hybrid_rhythm = blender.blend_rhythms([pat1, pat2], [0.7, 0.3], 'hybrid')
    # print(f"Blended rhythm: {len(hybrid_rhythm.notes)} notes")

    # harmony1 = Pattern(...)  # I-IV-V
    # harmony2 = Pattern(...)  # ii-V-I
    # blended_harmony = blender.blend_harmonies([harmony1, harmony2])
    # print(f"Blended harmony: {len(blended_harmony.chords)} chords")

    # melody1 = Pattern(...)  # Ascending contour
    # melody2 = Pattern(...)  # Descending contour
    # blended_melody = blender.blend_melodies([melody1, melody2], blend_type='contour')
    # print(f"Blended melody: {len(blended_melody.notes)} notes")

    # faded = blender.cross_fade(melody1, melody2, fade_length=2.0)
    # print(f"Cross-faded: {len(faded.notes)} notes")

    # layered = blender.layer_patterns([rhythm, melody, harmony], balances=[0.8, 0.6, 0.4])
    # print(f"Layered: {len(layered.get_all_notes())} total notes")
    # Assume some patterns exist
    # pat1 = Pattern(...)  # Rhythm with kicks
    # pat2 = Pattern(...)  # Rhythm with hi-hats
    # blender = PatternBlender()
    # hybrid_rhythm = blender.blend_rhythms([pat1, pat2], [0.7, 0.3], 'hybrid')
    # print(f"Blended rhythm: {len(hybrid_rhythm.notes)} notes")

    # harmony1 = Pattern(...)  # I-IV-V
    # harmony2 = Pattern(...)  # ii-V-I
    # blended_harmony = blender.blend_harmonies([harmony1, harmony2])
    # print(f"Blended harmony: {len(blended_harmony.chords)} chords")

    # melody1 = Pattern(...)  # Ascending contour
    # melody2 = Pattern(...)  # Descending contour
    # blended_melody = blender.blend_melodies([melody1, melody2], blend_type='contour')
    # print(f"Blended melody: {len(blended_melody.notes)} notes")

    # faded = blender.cross_fade(melody1, melody2, fade_length=2.0)
    # print(f"Cross-faded: {len(faded.notes)} notes")

    # layered = blender.layer_patterns([rhythm, melody, harmony], balances=[0.8, 0.6, 0.4])
    # print(f"Layered: {len(layered.get_all_notes())} total notes")