"""
Song skeleton implementation for the MIDI Master music generation program.

This module defines the SongSkeleton class, which represents the overall
structure of a musical composition. It contains the high-level organization
of a song including its genre, tempo, mood, and arrangement into sections.

The SongSkeleton serves as the container for all musical content and
provides methods for organizing patterns into a coherent song structure.
"""

from typing import List, Dict, Optional, Tuple
from .data_structures import Pattern, SectionType, PatternType
import random


class SongSkeleton:
    """Represents the overall structure of a song.
    
    The SongSkeleton is the central organizing structure for a musical composition.
    It contains:
    - High-level metadata (genre, tempo, mood)
    - Organization into sections (verse, chorus, bridge, etc.)
    - Collections of musical patterns (melody, harmony, bass, rhythm)
    
    The skeleton provides methods for building arrangements and organizing
    musical content into a coherent structure.
    """
    
    def __init__(self, genre: str, tempo: int, mood: str):
        """
        Initialize a SongSkeleton.

        Args:
            genre: Music genre (e.g., 'pop', 'rock', 'jazz')
            tempo: Tempo in BPM (beats per minute)
            mood: Mood of the song (e.g., 'happy', 'sad', 'energetic', 'calm')
        """
        self.genre = genre
        self.tempo = tempo
        self.mood = mood
        # Dictionary mapping section types to lists of patterns
        self.sections: Dict[SectionType, List[Pattern]] = {}
        # List of all patterns in the song (including those in sections)
        self.patterns: List[Pattern] = []
        # Time signatures per pattern type: Dict[PatternType, Tuple[numerator, denominator]]
        # Defaults to 4/4 for all pattern types if not specified
        self.time_signatures: Dict[PatternType, Tuple[int, int]] = {}

    @staticmethod
    def generate_dynamic_structure(genre: str, mood: str) -> List[SectionType]:
        """
        Generates a dynamic song structure based on genre and mood.

        Args:
            genre: The genre of the song.
            mood: The mood of the song.

        Returns:
            A list of SectionType enums representing the song's structure.
        """
        structure = []

        # Basic structure elements
        basic_structure = [
            SectionType.INTRO,
            SectionType.VERSE,
            SectionType.CHORUS,
            SectionType.VERSE,
            SectionType.CHORUS,
            SectionType.BRIDGE,
            SectionType.CHORUS,
            SectionType.OUTRO
        ]

        # Genre-specific adjustments
        if genre == 'pop':
            # Pop often has pre-chorus and post-chorus
            structure.extend([SectionType.INTRO])
            for i, section in enumerate(basic_structure[1:-1]): # Exclude intro/outro
                if section == SectionType.CHORUS:
                    structure.append(SectionType.PRE_CHORUS)
                    structure.append(section)
                    structure.append(SectionType.POST_CHORUS)
                else:
                    structure.append(section)
            structure.extend([SectionType.OUTRO])
        elif genre == 'rock':
            # Rock often has a solo section
            structure.extend(basic_structure[:5]) # Intro, Verse, Chorus, Verse, Chorus
            structure.append(SectionType.SOLO)
            structure.extend(basic_structure[5:]) # Bridge, Chorus, Outro
        elif genre == 'jazz':
            # Jazz often has a head-solos-head structure
            structure = [
                SectionType.INTRO,
                SectionType.VERSE, # Head
                SectionType.SOLO,
                SectionType.SOLO,
                SectionType.VERSE, # Head
                SectionType.OUTRO
            ]
        elif genre == 'electronic':
            # Electronic often has buildup/drop
            structure = [
                SectionType.INTRO,
                SectionType.PRE_CHORUS, # Buildup
                SectionType.CHORUS, # Drop
                SectionType.BRIDGE, # Breakdown
                SectionType.CHORUS, # Drop
                SectionType.OUTRO
            ]
        elif genre == 'hip-hop':
            # Hip-hop similar to pop but with more emphasis on verses
            structure = [
                SectionType.INTRO,
                SectionType.VERSE,
                SectionType.CHORUS,
                SectionType.VERSE,
                SectionType.CHORUS,
                SectionType.BRIDGE,
                SectionType.VERSE,
                SectionType.OUTRO
            ]
        elif genre == 'classical':
            # Classical is more structured, less dynamic for now
            structure = basic_structure
        else:
            structure = basic_structure

        # Mood-specific adjustments
        if mood == 'energetic':
            # Add more choruses for energetic mood
            chorus_indices = [i for i, s in enumerate(structure) if s == SectionType.CHORUS]
            for idx in reversed(chorus_indices):
                structure.insert(idx + 1, SectionType.CHORUS)
        elif mood == 'calm':
            # Reduce choruses, extend verses/bridges
            structure = [s for s in structure if s != SectionType.CHORUS or structure.count(s) <= 1]
            if SectionType.VERSE in structure and structure.count(SectionType.VERSE) < 2:
                structure.insert(structure.index(SectionType.VERSE) + 1, SectionType.VERSE)

        # Add random fills
        final_structure = []
        for i, section in enumerate(structure):
            final_structure.append(section)
            if i < len(structure) - 1 and random.random() < 0.2: # 20% chance for a fill
                final_structure.append(SectionType.FILL)

        return final_structure

    def set_time_signature(self, pattern_type: PatternType, numerator: int, denominator: int):
        """
        Set the time signature for a specific pattern type.

        Args:
            pattern_type: The pattern type (MELODY, HARMONY, RHYTHM, BASS)
            numerator: Time signature numerator (e.g., 4 for 4/4)
            denominator: Time signature denominator (e.g., 4 for 4/4)
        """
        self.time_signatures[pattern_type] = (numerator, denominator)

    def get_time_signature(self, pattern_type: PatternType) -> Tuple[int, int]:
        """
        Get the time signature for a specific pattern type.

        Args:
            pattern_type: The pattern type to get time signature for

        Returns:
            Tuple of (numerator, denominator). Defaults to (4, 4) if not set.
        """
        return self.time_signatures.get(pattern_type, (4, 4))

    def __repr__(self):
        """Return a string representation of the SongSkeleton."""
        return f"SongSkeleton(genre={self.genre}, tempo={self.tempo}, mood={self.mood}, sections={list(self.sections.keys())})"
    
    def add_pattern(self, pattern: Pattern):
        """Add a pattern to the song.
        
        Adds a pattern to the main patterns list. Patterns in this list
        may or may not be organized into sections.
        
        Args:
            pattern: Pattern to add to the song
        """
        self.patterns.append(pattern)
    
    def add_section(self, section_type: SectionType, patterns: List[Pattern]):
        """Add a section with its patterns to the song.
        
        Organizes patterns into named sections (verse, chorus, etc.) that
        form the high-level structure of the song.
        
        Args:
            section_type: Type of section to add (verse, chorus, bridge, etc.)
            patterns: List of patterns for this section
        """
        self.sections[section_type] = patterns
    
    def build_arrangement(self, patterns: List[Pattern]):
        """
        Build a song arrangement based on dynamic structure generation.

        This method creates a song arrangement by organizing patterns
        into dynamically generated song sections, considering genre and mood.

        Args:
            patterns: List of patterns to arrange
        """
        self.sections = {}
        self.patterns = patterns # Store all generated patterns

        dynamic_structure = self.generate_dynamic_structure(self.genre, self.mood)

        # Group patterns by type for easier assignment
        patterns_by_type = {
            PatternType.MELODY: [p for p in patterns if p.pattern_type == PatternType.MELODY],
            PatternType.HARMONY: [p for p in patterns if p.pattern_type == PatternType.HARMONY],
            PatternType.BASS: [p for p in patterns if p.pattern_type == PatternType.BASS],
            PatternType.RHYTHM: [p for p in patterns if p.pattern_type == PatternType.RHYTHM]
        }

        # Assign patterns to sections based on the dynamic structure
        for section_type in dynamic_structure:
            if section_type not in self.sections:
                self.sections[section_type] = []

            # Simple assignment logic for now. Can be expanded for more sophistication.
            if section_type == SectionType.SOLO and patterns_by_type[PatternType.MELODY]:
                # Solo sections can use a melody pattern
                self.sections[section_type].append(patterns_by_type[PatternType.MELODY].pop(0))
            elif section_type == SectionType.FILL and patterns_by_type[PatternType.RHYTHM]:
                # Fills can use a rhythm pattern
                self.sections[section_type].append(patterns_by_type[PatternType.RHYTHM].pop(0))
            # For other section types (INTRO, VERSE, CHORUS, BRIDGE, OUTRO, PRE_CHORUS, POST_CHORUS),
            # we can assign a combination of patterns or specific ones based on more complex rules.
            # For simplicity, let's assign all remaining patterns to the first occurrence of a general section type.
            # This part needs more sophisticated logic for a truly dynamic arrangement.
            elif section_type in [SectionType.INTRO, SectionType.VERSE, SectionType.CHORUS, SectionType.BRIDGE, SectionType.OUTRO, SectionType.PRE_CHORUS, SectionType.POST_CHORUS]:
                # Assign a mix of patterns if available and not yet assigned
                if patterns_by_type[PatternType.MELODY] and not any(p.pattern_type == PatternType.MELODY for p in self.sections[section_type]):
                    self.sections[section_type].append(patterns_by_type[PatternType.MELODY].pop(0))
                if patterns_by_type[PatternType.HARMONY] and not any(p.pattern_type == PatternType.HARMONY for p in self.sections[section_type]):
                    self.sections[section_type].append(patterns_by_type[PatternType.HARMONY].pop(0))
                if patterns_by_type[PatternType.BASS] and not any(p.pattern_type == PatternType.BASS for p in self.sections[section_type]):
                    self.sections[section_type].append(patterns_by_type[PatternType.BASS].pop(0))
                if patterns_by_type[PatternType.RHYTHM] and not any(p.pattern_type == PatternType.RHYTHM for p in self.sections[section_type]):
                    self.sections[section_type].append(patterns_by_type[PatternType.RHYTHM].pop(0))