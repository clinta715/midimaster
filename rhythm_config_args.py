"""
Rhythm Configuration Arguments for MIDI Master

Add these command-line arguments to main.py to expose rhythm variation controls:

parser.add_argument(
    '--pattern-strength',
    type=float,
    default=1.0,
    choices=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    help='How strongly to maintain original rhythm patterns (0.0=very loose, 1.0=strict). Default: 1.0'
)

parser.add_argument(
    '--swing-percent',
    type=float,
    default=0.5,
    choices=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    help='Amount of swing feel in rhythm (0.0=straight, 1.0=maximum swing). Default: 0.5'
)

parser.add_argument(
    '--fill-frequency',
    type=float,
    default=0.25,
    choices=[0.0, 0.1, 0.25, 0.33, 0.5],
    help='How often to add rhythmic fills (0.0=never, 0.5=every 2 bars). Default: 0.25'
)

parser.add_argument(
    '--ghost-note-level',
    type=float,
    default=1.0,
    choices=[0.0, 0.3, 0.5, 1.0, 1.5, 2.0],
    help='Intensity of ghost notes (0.0=none, 2.0=very prominent). Default: 1.0'
)

Then pass these to the pattern generator:

pattern_generator = PatternGenerator(
    genre_rules,
    args.mood,
    note_density=density_manager.note_density,
    rhythm_density=density_manager.rhythm_density,
    chord_density=density_manager.chord_density,
    bass_density=density_manager.bass_density,
    harmonic_variance=args.harmonic_variance,
    subgenre=args.subgenre,
    context=context,
    # Add rhythm variation parameters
    pattern_strength=args.pattern_strength,
    swing_percent=args.swing_percent,
    fill_frequency=args.fill_frequency,
    ghost_note_level=args.ghost_note_level
)
"""

# Example usage:
"""
python main.py --genre hip-hop --subgenre trap --pattern-strength 0.8 --swing-percent 0.6 --fill-frequency 0.33 --ghost-note-level 1.2

This creates trap beats with:
- 80% pattern adherence (some variation)
- Moderate swing feel
- Fills every 3 bars
- Enhanced ghost notes for groove
"""