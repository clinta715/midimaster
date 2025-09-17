# Density Management System Design

## Overview
The Density Management System provides fine-grained control over the overall density of notes and beats in generated music. This allows users to create music ranging from sparse and minimal to dense and complex arrangements.

## Core Components

### 1. Density Parameters
The system uses four main density parameters:

- **note_density** (0.0-1.0): Controls overall note density across all patterns
  - 0.0 = Sparse (minimal notes, lots of space)
  - 0.5 = Medium (balanced density)
  - 1.0 = Dense (maximum notes, busy texture)

- **rhythm_density** (0.0-1.0): Controls rhythmic complexity
  - 0.0 = Simple rhythms (whole notes, half notes)
  - 0.5 = Medium rhythms (quarter notes, eighth notes)
  - 1.0 = Complex rhythms (sixteenth notes, triplets)

- **chord_density** (0.0-1.0): Controls chord voicing density
  - 0.0 = Minimal voicing (root + fifth)
  - 0.5 = Standard voicing (root + third + fifth)
  - 1.0 = Extended voicing (full chord with extensions)

- **bass_density** (0.0-1.0): Controls bass line complexity
  - 0.0 = Sparse bass (one note per bar)
  - 0.5 = Medium bass (one note per beat)
  - 1.0 = Walking bass (multiple notes per beat)

### 2. Density Manager Class
A utility class that handles density calculations:

```python
class DensityManager:
    def calculate_note_probability(self, density: float) -> float:
        """Calculate probability of placing a note based on density"""

    def get_available_durations(self, density: float) -> List[float]:
        """Get available note durations based on density"""

    def get_chord_voicing_size(self, density: float, max_notes: int) -> int:
        """Determine how many notes to include in chord voicing"""

    def get_rhythm_pattern_complexity(self, density: float) -> int:
        """Determine rhythm pattern complexity level"""
```

### 3. Pattern Generator Integration
The PatternGenerator class will be enhanced to:

- Accept density parameters in constructor
- Use DensityManager for all note placement decisions
- Apply density filtering to generated notes
- Maintain backward compatibility with existing code

## Implementation Strategy

### Phase 1: Core Infrastructure
1. Create DensityManager class
2. Add density parameters to PatternGenerator
3. Implement density-based note filtering

### Phase 2: Pattern-Specific Density Control
1. Update melody generation with note_density
2. Update harmony generation with chord_density
3. Update bass generation with bass_density
4. Update rhythm generation with rhythm_density

### Phase 3: Integration and Testing
1. Update main generation methods
2. Add command-line density parameters
3. Create density presets for common styles
4. Test with various density combinations

## Usage Examples

### Command Line Usage
```bash
# Sparse, minimal music
python main.py --genre jazz --density 0.2

# Dense, complex music
python main.py --genre electronic --density 0.8

# Custom density parameters
python main.py --genre pop --note-density 0.6 --rhythm-density 0.4 --chord-density 0.7 --bass-density 0.5
```

### Programmatic Usage
```python
from generators.pattern_generator import PatternGenerator
from structures.song_skeleton import SongSkeleton

# Create generator with custom density
generator = PatternGenerator(
    genre_rules=pop_rules,
    mood='happy',
    note_density=0.3,      # Sparse overall
    rhythm_density=0.6,    # Medium rhythm complexity
    chord_density=0.4,     # Simple chord voicings
    bass_density=0.2       # Minimal bass
)

patterns = generator.generate_patterns(skeleton, 16)
```

## Density Presets

### Preset Definitions
- **Minimal**: note_density=0.1, rhythm_density=0.2, chord_density=0.1, bass_density=0.1
- **Sparse**: note_density=0.3, rhythm_density=0.4, chord_density=0.3, bass_density=0.3
- **Balanced**: note_density=0.5, rhythm_density=0.5, chord_density=0.5, bass_density=0.5
- **Dense**: note_density=0.7, rhythm_density=0.6, chord_density=0.8, bass_density=0.7
- **Complex**: note_density=0.9, rhythm_density=0.8, chord_density=0.9, bass_density=0.9

### Genre-Specific Defaults
Different genres will have different default density settings:
- **Classical**: Lower density, more space
- **Jazz**: Medium density with complex rhythms
- **Pop**: Medium density, balanced
- **Electronic**: Higher density, more rhythmic complexity
- **Hip-Hop**: Medium density with emphasis on rhythm

## Backward Compatibility
The system maintains full backward compatibility:
- Existing code continues to work unchanged
- Default density values provide current behavior
- New density parameters are optional
- All existing APIs remain functional

## Benefits
1. **Creative Control**: Users can create music with specific density characteristics
2. **Genre Flexibility**: Different genres can have appropriate density defaults
3. **Mood Enhancement**: Density can reinforce musical mood and emotion
4. **Performance Optimization**: Sparse settings can reduce computational load
5. **Educational Value**: Helps users understand musical texture and density concepts