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
        # Ordered list of (section_type, patterns) preserving multiple occurrences and order
        self.sections: List[Tuple[SectionType, List[Pattern]]] = []
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
        section_types = [s.name for s, _ in self.sections]
        return f"SongSkeleton(genre={self.genre}, tempo={self.tempo}, mood={self.mood}, sections={section_types})"
    
    def add_pattern(self, pattern: Pattern):
        """Add a pattern to the song.
        
        Adds a pattern to the main patterns list. Patterns in this list
        may or may not be organized into sections.
        
        Args:
            pattern: Pattern to add to the song
        """
        self.patterns.append(pattern)
    
    def add_section(self, section_type: SectionType, patterns: List[Pattern]):
        """Add a section with its patterns to the song in order (supports repeats)."""
        self.sections.append((section_type, patterns))
    
    def build_arrangement(self, patterns: List[Pattern]):
        """
        Build a simple ordered song arrangement using the dynamic structure.
    
        This preserves multiple occurrences of the same section type by storing sections
        as an ordered list. It cycles through available patterns if needed.
    
        Args:
            patterns: List of patterns to arrange
        """
        # Reset and store all generated patterns
        self.sections = []
        self.patterns = patterns
    
        dynamic_structure = self.generate_dynamic_structure(self.genre, self.mood)
    
        # Group patterns by type for easier assignment and maintain cyclic indices
        patterns_by_type = {
            PatternType.MELODY: [p for p in patterns if p.pattern_type == PatternType.MELODY],
            PatternType.HARMONY: [p for p in patterns if p.pattern_type == PatternType.HARMONY],
            PatternType.BASS: [p for p in patterns if p.pattern_type == PatternType.BASS],
            PatternType.RHYTHM: [p for p in patterns if p.pattern_type == PatternType.RHYTHM]
        }
        indices = {ptype: 0 for ptype in patterns_by_type.keys()}
    
        def next_pattern(ptype: PatternType) -> Optional[Pattern]:
            if not patterns_by_type[ptype]:
                return None
            i = indices[ptype] % len(patterns_by_type[ptype])
            indices[ptype] += 1
            return patterns_by_type[ptype][i]
    
        # Assign patterns to sections in order
        for section_type in dynamic_structure:
            section_patterns: List[Pattern] = []
            if section_type == SectionType.SOLO:
                mp = next_pattern(PatternType.MELODY)
                if mp:
                    section_patterns.append(mp)
            elif section_type == SectionType.FILL:
                rp = next_pattern(PatternType.RHYTHM)
                if rp:
                    section_patterns.append(rp)
            else:
                # General sections get a blend of available parts
                for ptype in (PatternType.MELODY, PatternType.HARMONY, PatternType.BASS, PatternType.RHYTHM):
                    p = next_pattern(ptype)
                    if p:
                        section_patterns.append(p)
            self.sections.append((section_type, section_patterns))