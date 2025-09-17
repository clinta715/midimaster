"""
Harmony Generator Module

This module contains the HarmonyGenerator class responsible for generating
harmonically rich chord progressions based on genre rules and musical context.
"""

import random
from typing import TYPE_CHECKING, List, Dict, Optional

from structures.data_structures import Note, Pattern, PatternType, Chord
from generators.generator_utils import get_velocity_for_mood
from music_theory import MusicTheory, ScaleType
from generators.generator_context import GeneratorContext

if TYPE_CHECKING:
    from generators.generator_context import GeneratorContext

class HarmonyGenerator:
    """
    Generates harmonically coherent chord progressions and voicings for various genres.
    """
    
    def __init__(self, context: Optional['GeneratorContext'] = None):
        self.context = context
        if context:
            self.key = context.current_key
            self.scale = context.current_scale
            self.scale_str = f"{self.key} {self.scale.lower()}"
            self.genre_rules = context.genre_rules
            self.chord_progressions = self.genre_rules.get_chord_progressions()
            self.mood = context.mood
        else:
            # Fallback for standalone use
            self.key = 'C'
            self.scale = ScaleType.MAJOR
            self.scale_str = 'C major'
            self.genre_rules = None
            self.chord_progressions = self._load_genre_progressions('pop')
            self.mood = 'neutral'
        self.music_theory = MusicTheory()
    
    def _load_genre_progressions(self, genre: str) -> List[List[str]]:
        """
        Load predefined chord progressions for fallback genres.
        """
        progressions = {
            'jazz': [
                ['ii', 'V', 'I'],
                ['vi', 'ii', 'V', 'I'],
                ['I', 'vi', 'ii', 'V'],
            ],
            'pop': [
                ['I', 'V', 'vi', 'IV'],
                ['I', 'IV', 'V'],
                ['vi', 'IV', 'I', 'V'],
            ],
            'rock': [
                ['I', 'IV', 'V'],
                ['vi', 'IV', 'I', 'V'],
            ],
            'electronic': [
                ['I', 'vi', 'IV', 'V'],
                ['i', 'VI', 'III', 'VII'],
            ]
        }
        return progressions.get(genre.lower(), progressions['pop'])
    
    def initialize_harmony(self, genre: str, mood: str, tempo: int = 120):
        """
        Initialize key and scale based on genre and mood using context if available.
        """
        if self.context:
            # Use context's key and scale
            pass  # Already set in __init__
        else:
            # Fallback initialization
            self.key = 'C'
            self.scale = ScaleType.MAJOR
            self.scale_str = 'C major'
    
    def generate(self, num_bars: int, chord_complexity: str = 'medium', harmonic_variance: str = 'medium') -> Pattern:
        """
        Generate a harmony pattern with specified complexity and variance.
        
        Args:
            num_bars: Number of bars for the progression
            chord_complexity: 'simple', 'medium', 'complex' for chord extensions
            harmonic_variance: 'close', 'medium', 'distant' for progression selection
            
        Returns:
            Pattern object with harmony
        """
        if self.context:
            genre = self.context.genre_rules.get_genre_name()
            mood = self.context.mood
        else:
            genre = 'pop'
            mood = 'neutral'
        
        self.initialize_harmony(genre, mood)
        
        # Filter progressions by harmonic variance
        filtered_progressions = self.music_theory.filter_progressions_by_distance(
            self.chord_progressions, self.scale_str, harmonic_variance
        )
        if not filtered_progressions:
            filtered_progressions = self.chord_progressions
        
        # Select a progression template
        progression_template = random.choice(filtered_progressions)
        # Extend to fit num_bars
        full_progression = (progression_template * (num_bars // len(progression_template) + 1))[:num_bars]
        
        chords = []
        for roman_numeral in full_progression:
            # Get base chord pitches using MusicTheory
            base_pitches = self.music_theory.get_chord_pitches_from_roman(roman_numeral, self.scale_str)
            if not base_pitches:
                # Fallback
                base_pitches = [60, 64, 67]  # C major
            
            # Add extensions based on complexity
            pitches = self._add_chord_extensions(base_pitches, chord_complexity, roman_numeral)
            
            # Create notes with duration (whole note for chords)
            velocity = get_velocity_for_mood(self.mood)
            chord_notes = [Note(pitch=p, velocity=velocity, duration=4.0) for p in pitches]
            chord = Chord(notes=chord_notes)
            chords.append(chord)
        
        # Create pattern
        all_notes = []
        for chord in chords:
            all_notes.extend(chord.notes)
        pattern = Pattern(
            pattern_type=PatternType.HARMONY,
            notes=all_notes,
            chords=chords
        )
        return pattern
    
    def _add_chord_extensions(self, base_pitches: List[int], complexity: str, roman_numeral: str) -> List[int]:
        """
        Add extensions to base chord pitches based on complexity.
        """
        if complexity == 'simple':
            return base_pitches[:3]  # Triad only
        elif complexity == 'medium':
            return base_pitches  # As is (may include 7th if in roman)
        elif complexity == 'complex':
            # Add 7th or 9th if not present
            root = base_pitches[0]
            # Determine if major/minor from roman
            is_major = roman_numeral.isupper() and not roman_numeral.lower().startswith('v')
            if is_major:
                # Add maj7 if not present
                maj7 = root + 11
                if maj7 not in base_pitches:
                    base_pitches.append(maj7)
            else:
                # Add min7
                min7 = root + 10
                if min7 not in base_pitches:
                    base_pitches.append(min7)
            # Sort and limit to octave
            base_pitches = sorted(set(p % 12 + (p // 12) * 12 for p in base_pitches))[:5]
            return base_pitches
        return base_pitches
    
    def generate_harmony_pattern(self, genre: str, mood: str, bars: int = 16) -> Pattern:
        """
        Generate a full harmony pattern (legacy method, uses generate internally).
        """
        return self.generate(bars, 'medium', 'medium')
    
    def refine_harmony(self, melody_notes: List[Note], existing_chords: List[Chord]) -> List[Chord]:
        """
        Refine existing chords to better support the melody.
        """
        # Simple implementation: adjust chord voicings to include melody notes if possible
        refined_chords = []
        for chord in existing_chords:
            # Check if melody note fits in chord
            for note in melody_notes:
                if note.pitch in [n.pitch for n in chord.notes]:
                    # Note fits, keep chord
                    break
            else:
                # Adjust closest chord note to melody if possible
                adjusted_notes = chord.notes.copy()
                # Implementation placeholder: keep as is for now
                pass
            refined_chords.append(chord)
        return refined_chords