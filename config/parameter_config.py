from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime
import json
import logging

class Genre(str, Enum):
    POP = "pop"
    ROCK = "rock"
    JAZZ = "jazz"
    ELECTRONIC = "electronic"
    HIP_HOP = "hip-hop"
    CLASSICAL = "classical"

class Mood(str, Enum):
    HAPPY = "happy"
    SAD = "sad"
    ENERGETIC = "energetic"
    CALM = "calm"

class Density(str, Enum):
    MINIMAL = "minimal"
    SPARSE = "sparse"
    BALANCED = "balanced"
    DENSE = "dense"
    COMPLEX = "complex"

class HarmonicVariance(str, Enum):
    CLOSE = "close"
    MEDIUM = "medium"
    DISTANT = "distant"

@dataclass
class ParameterConfig:
    """
    Unified parameter configuration class for music generation.
    Used by both CLI and GUI interfaces.
    """
    genre: Genre = Genre.POP
    tempo: int = 120
    mood: Mood = Mood.HAPPY
    subgenre: Optional[str] = None
    output: Optional[str] = None
    bars: int = 16
    density: Density = Density.BALANCED
    render_audio: bool = False
    plugin_path: Optional[str] = None
    audio_output: Optional[str] = None
    separate_files: bool = False
    melody_time_signature: str = "4/4"
    harmony_time_signature: str = "4/4"
    bass_time_signature: str = "4/4"
    rhythm_time_signature: str = "4/4"
    harmonic_variance: HarmonicVariance = HarmonicVariance.MEDIUM
    pattern_strength: float = 1.0
    swing_percent: float = 0.5
    fill_frequency: float = 0.25
    ghost_note_level: float = 1.0
    user_key: Optional[str] = None
    user_mode: Optional[str] = None

    def __post_init__(self):
        """Validate parameters after initialization."""
        self.validate()

    def validate(self) -> None:
        """
        Validate all parameters against constraints.
        Raises ValueError if invalid.
        """
        tempo_ranges = {
            Genre.POP.value: (90, 140),
            Genre.ROCK.value: (70, 140),
            Genre.JAZZ.value: (60, 200),
            Genre.ELECTRONIC.value: (80, 180),
            Genre.HIP_HOP.value: (70, 110),
            Genre.CLASSICAL.value: (40, 120),
        }
        if self.genre.value in tempo_ranges:
            min_tempo, max_tempo = tempo_ranges[self.genre.value]
            if self.tempo < min_tempo:
                logging.warning(f"Tempo {self.tempo} is below recommended range for {self.genre.value} ({min_tempo}-{max_tempo}). Clamping to {min_tempo}.")
                self.tempo = min_tempo
            elif self.tempo > max_tempo:
                logging.warning(f"Tempo {self.tempo} is above recommended range for {self.genre.value} ({min_tempo}-{max_tempo}). Clamping to {max_tempo}.")
                self.tempo = max_tempo
        # Genre validation
        if self.genre not in Genre:
            raise ValueError(f"Invalid genre: {self.genre}. Must be one of {[e.value for e in Genre]}")

        # Tempo
        if not (40 <= self.tempo <= 220):
            raise ValueError(f"Tempo must be between 40-220 BPM, got {self.tempo}")

        # Mood
        if self.mood not in Mood:
            raise ValueError(f"Invalid mood: {self.mood}. Must be one of {[e.value for e in Mood]}")

        # Bars
        if self.bars < 1:
            raise ValueError(f"Bars must be at least 1, got {self.bars}")

        # Density
        if self.density not in Density:
            raise ValueError(f"Invalid density: {self.density}. Must be one of {[e.value for e in Density]}")

        # Harmonic variance
        if self.harmonic_variance not in HarmonicVariance:
            raise ValueError(f"Invalid harmonic variance: {self.harmonic_variance}. Must be one of {[e.value for e in HarmonicVariance]}")

        # Pattern strength
        valid_strengths = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        if self.pattern_strength not in valid_strengths:
            raise ValueError(f"Pattern strength must be one of {valid_strengths}, got {self.pattern_strength}")

        # Swing percent
        valid_swings = [i/10 for i in range(11)]
        if self.swing_percent not in valid_swings:
            raise ValueError(f"Swing percent must be one of {valid_swings}, got {self.swing_percent}")

        # Fill frequency
        valid_fills = [0.0, 0.1, 0.25, 0.33, 0.5]
        if self.fill_frequency not in valid_fills:
            raise ValueError(f"Fill frequency must be one of {valid_fills}, got {self.fill_frequency}")

        # Ghost note level
        valid_ghosts = [0.0, 0.3, 0.5, 1.0, 1.5, 2.0]
        if self.ghost_note_level not in valid_ghosts:
            raise ValueError(f"Ghost note level must be one of {valid_ghosts}, got {self.ghost_note_level}")

        # Time signatures
        time_sigs = [self.melody_time_signature, self.harmony_time_signature,
                     self.bass_time_signature, self.rhythm_time_signature]
        for ts in time_sigs:
            try:
                num, den = ts.split('/')
                num = int(num)
                den = int(den)
                if not (1 <= num <= 16) or not (1 <= den <= 16) or den not in [1,2,4,8,16]:
                    raise ValueError(f"Invalid time signature: {ts}")
            except ValueError:
                raise ValueError(f"Invalid time signature format: {ts}. Expected 'num/den' e.g., '4/4'")

        # Output file must end with .mid if specified
        if self.output is not None and not self.output.endswith('.mid'):
            raise ValueError(f"Output file must end with .mid, got {self.output}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        # Convert enums to strings
        d['genre'] = self.genre.value
        d['mood'] = self.mood.value
        d['density'] = self.density.value
        d['harmonic_variance'] = self.harmonic_variance.value
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ParameterConfig':
        """
        Create instance from dictionary.
        Validates after creation.
        """
        # Convert string enums to enum values
        if 'genre' in data and isinstance(data['genre'], str):
            data['genre'] = Genre(data['genre'])
        if 'mood' in data and isinstance(data['mood'], str):
            data['mood'] = Mood(data['mood'])
        if 'density' in data and isinstance(data['density'], str):
            data['density'] = Density(data['density'])
        if 'harmonic_variance' in data and isinstance(data['harmonic_variance'], str):
            data['harmonic_variance'] = HarmonicVariance(data['harmonic_variance'])

        instance = cls(**data)
        return instance

    def save_to_file(self, path: str) -> None:
        """
        Save parameters to JSON file.
        Uses ConfigManager format.
        """
        config_data = {
            "version": "1.0",
            "name": "config",
            "description": "Saved parameters",
            "timestamp": datetime.now().isoformat(),
            "parameters": self.to_dict()
        }
        with open(path, 'w') as f:
            json.dump(config_data, f, indent=2)

    @classmethod
    def load_from_file(cls, path: str) -> 'ParameterConfig':
        """
        Load parameters from JSON file.
        """
        with open(path, 'r') as f:
            config_data = json.load(f)
        params = config_data.get('parameters', {})
        return cls.from_dict(params)