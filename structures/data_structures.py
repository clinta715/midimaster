"""
Data structures for the MIDI Master music generation program.

This module defines the core data structures used to represent musical elements
in the MIDI Master program. These classes provide the foundation for storing
and manipulating musical data throughout the generation process.

The main classes are:
- Note: Represents a single musical note
- Chord: Represents a collection of notes played simultaneously
- Pattern: Represents a musical pattern (melody, harmony, rhythm, or bass)
"""

from enum import Enum
from typing import List, Optional, Dict, Any
import copy
from dataclasses import dataclass


@dataclass
class Note:
    """Represents a musical note with pitch, duration, velocity, and channel.

    A Note object encapsulates all the essential properties of a musical note:
    - pitch: MIDI pitch value (0-127)
    - duration: Length of the note in beats
    - velocity: Volume/intensity of the note (0-127)
    - start_time: When the note begins in the composition
    - channel: MIDI channel for the note (1-16, default 1)

    This class provides validation for all parameters and useful string representation.
    """
    pitch: int
    duration: float
    velocity: int = 64
    start_time: float = 0.0
    channel: int = 1

    def __post_init__(self):
        """Validate parameters after initialization."""
        # Validate pitch range (MIDI standard: 0-127)
        if not 0 <= self.pitch <= 127:
            raise ValueError("Pitch must be between 0 and 127")

        # Validate duration is positive
        if self.duration <= 0:
            raise ValueError("Duration must be positive")

        # Validate velocity range (MIDI standard: 0-127)
        if not 0 <= self.velocity <= 127:
            raise ValueError("Velocity must be between 0 and 127")

        # Validate channel range (MIDI standard: 1-16)
        if not 1 <= self.channel <= 16:
            raise ValueError("Channel must be between 1 and 16")

    def __repr__(self):
        """Return a string representation of the Note."""
        return f"Note(pitch={self.pitch}, duration={self.duration}, velocity={self.velocity}, start_time={self.start_time}, channel={self.channel})"

    def __eq__(self, other):
        """Check equality with another Note object.

        Two notes are equal if they have the same pitch, duration, velocity, and start_time.

        Args:
            other: Object to compare with

        Returns:
            bool: True if equal, False otherwise
        """
        # Check if other is a Note instance
        if not isinstance(other, Note):
            return False

        # Compare all note properties
        return (self.pitch == other.pitch and
                self.duration == other.duration and
                self.velocity == other.velocity and
                self.start_time == other.start_time and
                self.channel == other.channel)


class Chord:
    """Represents a chord as a collection of Notes played simultaneously.
    
    A Chord object contains multiple Note objects that are played at the same time.
    It ensures all notes in the chord start at the same time and provides utilities
    for working with the chord as a unit.
    """
    
    def __init__(self, notes: List[Note], start_time: float = 0.0):
        """
        Initialize a Chord.
        
        Args:
            notes: List of Notes that make up the chord (must not be empty)
            start_time: Start time in beats from beginning of piece (default 0.0)
            
        Raises:
            ValueError: If notes list is empty
        """
        # Validate that chord contains at least one note
        if not notes:
            raise ValueError("Chord must contain at least one note")
            
        self.notes = notes
        self.start_time = start_time
        
        # Ensure all notes in chord start at the same time
        for note in self.notes:
            note.start_time = start_time
    
    def __repr__(self):
        """Return a string representation of the Chord."""
        return f"Chord(notes={self.notes}, start_time={self.start_time})"
    
    def add_note(self, note: Note):
        """Add a note to the chord.
        
        Args:
            note: Note to add to the chord
        """
        # Set the note's start time to match the chord
        note.start_time = self.start_time
        # Add the note to the chord's note list
        self.notes.append(note)
    
    def duration(self) -> float:
        """Get the duration of the chord (duration of the longest note).
        
        Returns:
            float: Duration of the longest note in the chord
        """
        # Return the maximum duration among all notes in the chord
        return max(note.duration for note in self.notes)


class PatternType(Enum):
    """Types of musical patterns.
    
    This enumeration defines the different types of musical patterns that
    can be generated and used in compositions.
    """
    MELODY = "melody"      # Single-note melodic lines
    HARMONY = "harmony"    # Chord progressions and harmonies
    RHYTHM = "rhythm"      # Percussive and rhythmic patterns
    BASS = "bass"          # Bass lines and low-register patterns


class Pattern:
    """Represents a musical pattern (melody, harmony, rhythm, or bass line).
    
    A Pattern object contains both individual notes and chords, organized by type.
    It serves as a container for musical content that can be arranged into song sections.
    """
    
    def __init__(self, pattern_type: PatternType, notes: List[Note], chords: List[Chord], tempo: Optional[float] = None, swing_factor: Optional[float] = None, syncopation_level: Optional[float] = None):
        """
        Initialize a Pattern.
        
        Args:
            pattern_type: Type of pattern (melody, harmony, rhythm, or bass)
            notes: List of individual notes in the pattern
            chords: List of chords in the pattern
        """
        self.pattern_type = pattern_type
        self.notes = notes
        self.chords = chords
        self.tempo = tempo
        self.swing_factor = swing_factor
        self.syncopation_level = syncopation_level
    
    def __repr__(self):
        """Return a string representation of the Pattern."""
        return f"Pattern(type={self.pattern_type.value}, notes={len(self.notes)}, chords={len(self.chords)})"
    
    def add_note(self, note: Note):
        """Add a note to the pattern.
        
        Args:
            note: Note to add to the pattern
        """
        self.notes.append(note)
    
    def add_chord(self, chord: Chord):
        """Add a chord to the pattern.
        
        Args:
            chord: Chord to add to the pattern
        """
        self.chords.append(chord)
    
    def get_all_notes(self) -> List[Note]:
        """Get all notes from both individual notes and chords.
        
        Returns:
            List[Note]: All notes in the pattern, including those in chords
        """
        # Start with individual notes
        all_notes = self.notes.copy()
        
        # Add notes from all chords
        for chord in self.chords:
            all_notes.extend(chord.notes)
        
        return all_notes


class SectionType(Enum):
    """Types of song sections.
    
    This enumeration defines the different types of sections that can be
    used to structure a song arrangement.
    """
    INTRO = "intro"        # Song introduction
    VERSE = "verse"        # Main verse sections
    PRE_CHORUS = "pre_chorus" # Section leading to chorus
    CHORUS = "chorus"      # Chorus/refrain sections
    POST_CHORUS = "post_chorus" # Section after chorus
    BRIDGE = "bridge"      # Bridge sections
    SOLO = "solo"          # Instrumental solo section
    FILL = "fill"          # Short transitional fill
    OUTRO = "outro"        # Song ending