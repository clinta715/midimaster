"""
Genre-specific rules for the MIDI Master music generation program.

This module defines the musical characteristics and rules for different genres.
Each genre class specifies the scales, chord progressions, rhythm patterns,
typical song structures, and instrumentation that define that genre.

The rules in this module are used by the PatternGenerator to create
authentic-sounding patterns for each genre.
"""

from typing import Dict, List, Any, Optional


class GenreRules:
    """Base class for genre-specific rules.

    This abstract base class defines the interface for genre-specific rules.
    All genre rule classes should inherit from this class and implement
    the required methods to define their musical characteristics.
    """

    def get_rules(self) -> Dict[str, Any]:
        """Get the rules for this genre.

        Returns a dictionary containing all the genre-specific rules
        that will be used by the pattern generator.

        Returns:
            Dictionary containing genre rules:
            - 'scales': List of scale strings
            - 'chord_progressions': List of chord progression lists
            - 'rhythm_patterns': List of rhythm pattern dictionaries
            - 'typical_structure': List of section type strings
            - 'instrumentation': List of instrument strings
        """
        return {
            'scales': self.get_scales(),
            'chord_progressions': self.get_chord_progressions(),
            'rhythm_patterns': self.get_rhythm_patterns(),
            'typical_structure': self.get_typical_structure(),
            'instrumentation': self.get_instrumentation()
        }

    def get_scales(self) -> List[str]:
        """Get the scales typically used in this genre.

        Returns:
            List of scale strings (e.g., ['C major', 'G major'])
        """
        return []

    def get_chord_progressions(self) -> List[List[str]]:
        """Get common chord progressions for this genre.

        Chord progressions are expressed using Roman numeral notation.

        Returns:
            List of chord progression lists (e.g., [['I', 'V', 'vi', 'IV']])
        """
        return []

    def get_rhythm_patterns(self) -> List[Dict[str, Any]]:
        """Get typical rhythm patterns for this genre.

        Rhythm patterns are defined as lists of note durations.

        Returns:
            List of rhythm pattern dictionaries with 'name' and 'pattern' keys
        """
        return []

    def get_typical_structure(self) -> List[str]:
        """Get the typical song structure for this genre.

        Returns:
            List of section type strings
        """
        return []

    def get_instrumentation(self) -> List[str]:
        """Get typical instruments used in this genre.

        Returns:
            List of instrument strings
        """
        return []

    def get_subgenres(self) -> List[str]:
        """Get subgenres for this genre.

        Returns:
            List of subgenre strings specific to this genre
        """
        return []

    def get_transposed_chord_progressions(self, key: str, mode: str) -> List[List[str]]:
        """Get chord progressions transposed to the specified key and mode.

        Takes the genre's Roman numeral progressions and transposes them
        to actual chord names in the given key/mode.

        Args:
            key: Root note (e.g., 'A')
            mode: Scale type (e.g., 'dorian')

        Returns:
            List of chord progression lists with actual chord names
        """
        from music_theory import MusicTheory

        progressions = self.get_chord_progressions()
        transposed_progressions = []

        for progression in progressions:
            transposed_progression = []
            key_scale_string = f"{key} {mode}"

            for roman_numeral in progression:
                # Get chord pitches from Roman numeral in the target key/mode
                chord_pitches = MusicTheory.get_chord_pitches_from_roman(
                    roman_numeral, key_scale_string
                )
                if chord_pitches:
                    # Convert pitches back to chord name (simplified approach)
                    chord_name = self._pitches_to_chord_name(chord_pitches, key_scale_string)
                    transposed_progression.append(chord_name)
                else:
                    # Fallback to original Roman numeral if transposition fails
                    transposed_progression.append(roman_numeral)

            transposed_progressions.append(transposed_progression)

        return transposed_progressions

    def _pitches_to_chord_name(self, pitches: List[int], key_scale_string: str) -> str:
        """Convert chord pitches to a chord name string.

        This is a simplified implementation that identifies common chord types
        based on intervals from the root.

        Args:
            pitches: List of MIDI pitch values
            key_scale_string: Key and scale for context

        Returns:
            Chord name string (e.g., 'Am', 'Fmaj7')
        """
        if not pitches:
            return "N"  # No chord

        from music_theory import MusicTheory, Note

        # Get root pitch and normalize to 0-11 range
        root_pitch = pitches[0] % 12

        # Find the note name for the root
        root_name = None
        for note in Note:
            if note.value % 12 == root_pitch:
                root_name = note.name.replace('_', '#').replace('S', '#')
                if root_name.endswith('FLAT'):
                    root_name = root_name.replace('FLAT', 'b')
                break

        if root_name is None:
            return "N"

        # Analyze intervals to determine chord type
        intervals = [(p - pitches[0]) % 12 for p in pitches[1:]]

        if intervals == [3, 7]:  # Minor chord
            return f"{root_name}m"
        elif intervals == [4, 7]:  # Major chord
            return root_name
        elif intervals == [3, 6]:  # Diminished
            return f"{root_name}dim"
        elif intervals == [4, 8]:  # Augmented
            return f"{root_name}aug"
        elif intervals == [4, 7, 10]:  # Dominant 7th
            return f"{root_name}7"
        elif intervals == [4, 7, 11]:  # Major 7th
            return f"{root_name}maj7"
        elif intervals == [3, 7, 10]:  # Minor 7th
            return f"{root_name}m7"
        elif intervals == [3, 6, 9]:  # Diminished 7th
            return f"{root_name}dim7"
        else:
            # Default to major if unrecognized
            return root_name

    def get_beat_characteristics(self, subgenre: Optional[str] = None) -> Dict[str, Any]:
        """Get beat characteristics for this genre, optionally for a specific subgenre.

        Args:
            subgenre: Optional subgenre to get characteristics for

        Returns:
            Dictionary containing beat characteristics:
            - swing_factor: Amount of swing (0.5 for straight, >0.5 for swing)
            - syncopation_level: Level of syncopation (0.0-1.0)
            - emphasis_patterns: List of beat positions to emphasize
            - tempo_range: Tuple of (min_tempo, max_tempo)
        """
        return {
            'swing_factor': 0.5,
            'syncopation_level': 0.0,
            'emphasis_patterns': [],
            'tempo_range': (60, 180)
        }


class PopRules(GenreRules):
    """Rules for pop music.

    Pop music is characterized by catchy melodies, simple chord progressions,
    and a strong rhythmic foundation. It typically uses major keys and
    follows a verse-chorus structure.
    """

    def get_scales(self) -> List[str]:
        """Get the scales typically used in pop music.

        Pop music primarily uses major scales, with some modal mixture.

        Returns:
            List of major scale strings
        """
        return ['C major', 'G major', 'D major', 'A major', 'E major', 'B major',
                'F# major', 'C# major', 'F major', 'Bb major', 'Eb major', 'Ab major']

    def get_chord_progressions(self) -> List[List[str]]:
        """Get common chord progressions for pop music.

        Pop music is known for several classic chord progressions
        that are widely used across the genre.

        Returns:
            List of classic pop chord progressions
        """
        return [
            ['I', 'V', 'vi', 'IV'],  # Classic pop progression
            ['vi', 'IV', 'I', 'V'],  # Minor key variant
            ['I', 'vi', 'IV', 'V'],  # Another common progression
            ['I', 'IV', 'vi', 'V'],   # Classic doo-wop progression
            ['I', 'IV', 'V', 'V'], # Added variation
            ['vi', 'V', 'IV', 'V'] # Added variation
        ]

    def get_rhythm_patterns(self) -> List[Dict[str, Any]]:
        """Get typical rhythm patterns for pop music.

        Pop music uses a variety of rhythm patterns, from straight eighths
        to syncopated patterns that create a driving feel.

        Returns:
            List of pop rhythm patterns
        """
        return [
            {'name': 'straight_eight', 'pattern': [0.5, 0.5, 0.5, 0.5]},
            {'name': 'swing_eight', 'pattern': [0.75, 0.25, 0.75, 0.25]},
            {'name': 'syncopated', 'pattern': [0.25, 0.25, 0.5, 0.5, 0.5]}
        ]

    def get_typical_structure(self) -> List[str]:
        """Get the typical song structure for pop music.

        Pop songs typically follow a verse-chorus structure with
        an intro, bridge, and outro.

        Returns:
            List representing typical pop song structure
        """
        return ['intro', 'verse', 'chorus', 'verse', 'chorus', 'bridge', 'chorus', 'outro']

    def get_instrumentation(self) -> List[str]:
        """Get typical instruments used in pop music.

        Pop music uses a mix of traditional and electronic instruments.

        Returns:
            List of typical pop instruments
        """
        return ['vocals', 'drums', 'bass', 'guitar', 'piano', 'synthesizer']

    def get_subgenres(self) -> List[str]:
        """Get subgenres for pop music.

        Returns:
            List of pop subgenres
        """
        return ['dance_pop', 'power_pop', 'synth_pop', 'indie_pop', 'teen_pop']

    def get_beat_characteristics(self, subgenre: Optional[str] = None) -> Dict[str, Any]:
        """Get beat characteristics for pop music.

        Args:
            subgenre: Optional subgenre to get characteristics for

        Returns:
            Dictionary containing beat characteristics
        """
        # Base characteristics for pop
        characteristics = {
            'swing_factor': 0.55,
            'syncopation_level': 0.3,
            'emphasis_patterns': [1, 3],  # Emphasize beats 1 and 3 (backbeat)
            'tempo_range': (90, 140)
        }

        # Adjust based on subgenre
        if subgenre == 'dance_pop':
            characteristics.update({
                'swing_factor': 0.5,
                'syncopation_level': 0.4,
                'tempo_range': (120, 140)
            })
        elif subgenre == 'power_pop':
            characteristics.update({
                'swing_factor': 0.5,
                'syncopation_level': 0.2,
                'emphasis_patterns': [1, 2, 3, 4]  # More driving rhythm
            })

        return characteristics


class RockRules(GenreRules):
    """Rules for rock music.

    Rock music is characterized by strong rhythms, power chords,
    and a prominent electric guitar sound. It often uses both major
    and minor keys and features guitar-driven arrangements.
    """

    def get_scales(self) -> List[str]:
        """Get the scales typically used in rock music.

        Rock music uses both major and minor scales, with a preference
        for minor keys in many subgenres.

        Returns:
            List of major and minor scale strings
        """
        return ['E minor', 'A minor', 'D minor', 'G minor', 'C minor', 'F minor',
                'Bb minor', 'Eb minor', 'E major', 'A major', 'D major', 'G major']

    def get_chord_progressions(self) -> List[List[str]]:
        """Get common chord progressions for rock music.

        Rock music features both classic progressions and blues-based
        progressions with flattened chords.

        Returns:
            List of classic rock chord progressions
        """
        return [
            ['I', 'IV', 'V', 'I'],     # Classic rock progression
            ['I', 'V', 'vi', 'IV'],    # Pop-rock crossover
            ['vi', 'IV', 'I', 'V'],    # Minor key variant
            ['I', 'bVII', 'IV', 'I'],   # Blues-based progression
            ['i', 'VI', 'III', 'VII'], # Minor key rock progression
            ['I', 'vi', 'ii', 'V'] # Rock ballad
        ]

    def get_rhythm_patterns(self) -> List[Dict[str, Any]]:
        """Get typical rhythm patterns for rock music.

        Rock music emphasizes strong beats and power chord patterns.

        Returns:
            List of rock rhythm patterns
        """
        return [
            {'name': 'power_chord', 'pattern': [1.0, 1.0]},
            {'name': 'eight_bar', 'pattern': [0.5, 0.5, 0.5, 0.5]},
            {'name': 'straight_eight', 'pattern': [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]}
        ]

    def get_typical_structure(self) -> List[str]:
        """Get the typical song structure for rock music.

        Rock songs often include a guitar solo section in addition
        to standard verse-chorus structure.

        Returns:
            List representing typical rock song structure
        """
        return ['intro', 'verse', 'chorus', 'verse', 'chorus', 'solo', 'chorus', 'outro']

    def get_instrumentation(self) -> List[str]:
        """Get typical instruments used in rock music.

        Rock music is defined by its electric instruments and strong rhythm section.

        Returns:
            List of typical rock instruments
        """
        return ['vocals', 'electric_guitar', 'bass_guitar', 'drums', 'keyboards']

    def get_subgenres(self) -> List[str]:
        """Get subgenres for rock music.

        Returns:
            List of rock subgenres
        """
        return ['classic_rock', 'punk_rock', 'alternative_rock', 'hard_rock', 'indie_rock']

    def get_beat_characteristics(self, subgenre: Optional[str] = None) -> Dict[str, Any]:
        """Get beat characteristics for rock music.

        Args:
            subgenre: Optional subgenre to get characteristics for

        Returns:
            Dictionary containing beat characteristics
        """
        # Base characteristics for rock
        characteristics = {
            'swing_factor': 0.5,
            'syncopation_level': 0.2,
            'emphasis_patterns': [1, 3],  # Strong backbeat
            'tempo_range': (100, 160)
        }

        # Adjust based on subgenre
        if subgenre == 'punk_rock':
            characteristics.update({
                'swing_factor': 0.5,
                'syncopation_level': 0.1,
                'tempo_range': (140, 180)
            })
        elif subgenre == 'hard_rock':
            characteristics.update({
                'swing_factor': 0.5,
                'syncopation_level': 0.3,
                'emphasis_patterns': [1, 2, 3, 4]  # More aggressive rhythm
            })

        return characteristics


class JazzRules(GenreRules):
    """Rules for jazz music.

    Jazz music is characterized by complex harmonies, improvisation,
    and sophisticated rhythmic patterns. It uses extended chords and
    modal scales, with a focus on individual expression.
    """

    def get_scales(self) -> List[str]:
        """Get the scales typically used in jazz music.

        Jazz makes extensive use of modes and complex scales.

        Returns:
            List of modal and complex scale strings
        """
        return ['C major', 'D dorian', 'E phrygian', 'F lydian', 'G mixolydian',
                'A aeolian', 'B locrian', 'Bb major', 'Eb major', 'Ab major']

    def get_chord_progressions(self) -> List[List[str]]:
        """Get common chord progressions for jazz music.

        Jazz features complex chord progressions with extended chords
        and chromatic voice leading.

        Returns:
            List of classic jazz chord progressions
        """
        return [
            ['ii', 'V', 'I'],           # Classic ii-V-I progression
            ['iii', 'vi', 'ii', 'V'],   # Extended progression
            ['I', 'vi', 'ii', 'V'],     # Turnaround progression
            ['I', 'IV', 'vii°', 'iii', 'vi', 'ii', 'V', 'I'], # Circle of fifths
            ['I', 'ii', 'iii', 'IV'], # Ascending progression
            ['I', 'V7', 'I', 'V7'] # Basic blues
        ]

    def get_rhythm_patterns(self) -> List[Dict[str, Any]]:
        """Get typical rhythm patterns for jazz music.

        Jazz uses swing rhythms and complex syncopated patterns.

        Returns:
            List of jazz rhythm patterns
        """
        return [
            {'name': 'swing', 'pattern': [0.75, 0.25, 0.75, 0.25]},
            {'name': 'bebop', 'pattern': [0.25, 0.25, 0.25, 0.75]},
            {'name': 'latin', 'pattern': [0.5, 0.25, 0.25, 0.5, 0.5]}
        ]

    def get_typical_structure(self) -> List[str]:
        """Get the typical song structure for jazz music.

        Jazz compositions often follow a head-solos-head structure.

        Returns:
            List representing typical jazz song structure
        """
        return ['intro', 'head', 'solos', 'head', 'outro']

    def get_instrumentation(self) -> List[str]:
        """Get typical instruments used in jazz music.

        Jazz uses a standard combo of melodic and harmonic instruments.

        Returns:
            List of typical jazz instruments
        """
        return ['vocals', 'saxophone', 'trumpet', 'piano', 'double_bass', 'drums']

    def get_subgenres(self) -> List[str]:
        """Get subgenres for jazz music.

        Returns:
            List of jazz subgenres
        """
        return ['bebop', 'cool_jazz', 'free_jazz', 'fusion', 'latin_jazz']

    def get_beat_characteristics(self, subgenre: Optional[str] = None) -> Dict[str, Any]:
        """Get beat characteristics for jazz music.

        Args:
            subgenre: Optional subgenre to get characteristics for

        Returns:
            Dictionary containing beat characteristics
        """
        # Base characteristics for jazz
        characteristics = {
            'swing_factor': 0.66,
            'syncopation_level': 0.6,
            'emphasis_patterns': [1, 3],  # Swing emphasis
            'tempo_range': (120, 200)
        }

        # Adjust based on subgenre
        if subgenre == 'bebop':
            characteristics.update({
                'swing_factor': 0.7,
                'syncopation_level': 0.8,
                'tempo_range': (160, 220)
            })
        elif subgenre == 'cool_jazz':
            characteristics.update({
                'swing_factor': 0.6,
                'syncopation_level': 0.4,
                'tempo_range': (100, 140)
            })

        return characteristics


class ElectronicRules(GenreRules):
    """Rules for electronic music.

    Electronic music is characterized by synthesized sounds, repetitive
    rhythms, and computer-generated patterns. It often uses minor keys
    and emphasizes rhythm over traditional harmonic progressions.
    """

    def get_scales(self) -> List[str]:
        """Get the scales typically used in electronic music.

        Electronic music often uses minor scales for a darker sound.

        Returns:
            List of minor scale strings
        """
        return ['C minor', 'A minor', 'F minor', 'D minor', 'G minor', 'Eb minor']

    def get_chord_progressions(self) -> List[List[str]]:
        """Get common chord progressions for electronic music.

        Electronic music often uses simple, repetitive progressions
        that create a hypnotic effect.

        Returns:
            List of electronic music chord progressions
        """
        return [
            ['i', 'bVII', 'bVI', 'bVII'],  # Popular in EDM
            ['vi', 'IV', 'I', 'V'],        # Pop influence
            ['i', 'bVI', 'bIII', 'bVII'],  # Minor key progression
            ['I', 'V', 'vi', 'IV'],         # Major key progression
            ['i', 'iv', 'v', 'i'] # Classic minor progression
        ]

    def get_rhythm_patterns(self) -> List[Dict[str, Any]]:
        """Get typical rhythm patterns for electronic music.

        Electronic music emphasizes steady, driving rhythms.

        Returns:
            List of electronic music rhythm patterns
        """
        return [
            {'name': 'four_on_floor', 'pattern': [1.0, 1.0, 1.0, 1.0]},
            {'name': 'breakbeat', 'pattern': [0.5, 0.25, 0.25, 0.5, 0.5]},
            {'name': 'syncopated', 'pattern': [0.25, 0.25, 0.5, 0.5, 0.5]}
        ]

    def get_typical_structure(self) -> List[str]:
        """Get the typical song structure for electronic music.

        Electronic music often follows a buildup-drop structure.

        Returns:
            List representing typical electronic music structure
        """
        return ['intro', 'buildup', 'drop', 'breakdown', 'drop', 'outro']

    def get_instrumentation(self) -> List[str]:
        """Get typical instruments used in electronic music.

        Electronic music is created with electronic instruments and software.

        Returns:
            List of typical electronic music instruments
        """
        return ['synthesizer', 'drum_machine', 'sampler', 'sequencer', 'effects_processor']

    def get_subgenres(self) -> List[str]:
        """Get subgenres for electronic music.

        Returns:
            List of electronic subgenres
        """
        return ['house', 'techno', 'ambient', 'dubstep', 'trance']

    def get_beat_characteristics(self, subgenre: Optional[str] = None) -> Dict[str, Any]:
        """Get beat characteristics for electronic music.

        Args:
            subgenre: Optional subgenre to get characteristics for

        Returns:
            Dictionary containing beat characteristics
        """
        # Base characteristics for electronic
        characteristics = {
            'swing_factor': 0.5,
            'syncopation_level': 0.4,
            'emphasis_patterns': [1, 2, 3, 4],  # Steady beat
            'tempo_range': (120, 140)
        }

        # Adjust based on subgenre
        if subgenre == 'techno':
            characteristics.update({
                'swing_factor': 0.5,
                'syncopation_level': 0.2,
                'tempo_range': (130, 150)
            })
        elif subgenre == 'house':
            characteristics.update({
                'swing_factor': 0.55,
                'syncopation_level': 0.5,
                'tempo_range': (120, 135)
            })

        return characteristics


class HipHopRules(GenreRules):
    """Rules for hip-hop music.

    Hip-hop music is characterized by rhythmic speech (rapping) over
    beats, with an emphasis on groove and attitude. It often uses
    samples from other songs and features syncopated rhythms.
    """

    def get_scales(self) -> List[str]:
        """Get the scales typically used in hip-hop music.

        Hip-hop often uses minor scales for a gritty sound.

        Returns:
            List of minor scale strings
        """
        return ['C minor', 'Eb minor', 'F minor', 'G minor', 'A minor', 'Bb minor']

    def get_chord_progressions(self) -> List[List[str]]:
        """Get common chord progressions for hip-hop music.

        Hip-hop often uses simple progressions that support the rhythm
        and vocal elements.

        Returns:
            List of hip-hop chord progressions
        """
        return [
            ['i', 'bVII', 'bVI', 'bVII'],  # Minor progression common in hip-hop
            ['vi', 'IV', 'I', 'V'],        # Pop sample influence
            ['i', 'bVI', 'bIII', 'bVII'],  # Another minor progression
            ['I', 'V', 'vi', 'IV'],         # Major progression for upbeat tracks
            ['i', 'iv', 'v', 'i'] # Classic minor progression
        ]

    def get_rhythm_patterns(self) -> List[Dict[str, Any]]:
        """Get typical rhythm patterns for hip-hop music.

        Hip-hop uses distinctive rhythmic patterns like boom-bap and trap.

        Returns:
            List of hip-hop rhythm patterns
        """
        return [
            {'name': 'boom_bap', 'pattern': [0.5, 0.5, 0.5, 0.5]},
            {'name': 'trap', 'pattern': [0.75, 0.25, 0.75, 0.25]},
            {'name': 'syncopated', 'pattern': [0.25, 0.25, 0.5, 0.5, 0.5]}
        ]

    def get_typical_structure(self) -> List[str]:
        """Get the typical song structure for hip-hop music.

        Hip-hop follows a verse-chorus structure similar to pop.

        Returns:
            List representing typical hip-hop song structure
        """
        return ['intro', 'verse', 'chorus', 'verse', 'chorus', 'bridge', 'verse', 'outro']

    def get_instrumentation(self) -> List[str]:
        """Get typical instruments used in hip-hop music.

        Hip-hop uses electronic instruments and samples.

        Returns:
            List of typical hip-hop instruments
        """
        return ['vocals', 'drum_machine', 'sampler', 'synthesizer', 'bass']

    def get_subgenres(self) -> List[str]:
        """Get subgenres for hip-hop music.

        Returns:
            List of hip-hop subgenres
        """
        return ['west_coast', 'east_coast', 'trap', 'boom_bap', 'conscious']

    def get_beat_characteristics(self, subgenre: Optional[str] = None) -> Dict[str, Any]:
        """Get beat characteristics for hip-hop music.

        Args:
            subgenre: Optional subgenre to get characteristics for

        Returns:
            Dictionary containing beat characteristics
        """
        # Base characteristics for hip-hop
        characteristics = {
            'swing_factor': 0.6,
            'syncopation_level': 0.7,
            'emphasis_patterns': [2, 4],  # Off-beat emphasis
            'tempo_range': (80, 110)
        }

        # Adjust based on subgenre
        if subgenre == 'trap':
            characteristics.update({
                'swing_factor': 0.5,
                'syncopation_level': 0.5,
                'tempo_range': (70, 90)
            })
        elif subgenre == 'boom_bap':
            characteristics.update({
                'swing_factor': 0.55,
                'syncopation_level': 0.6,
                'tempo_range': (85, 100)
            })

        return characteristics


class ClassicalRules(GenreRules):
    """Rules for classical music.

    Classical music is characterized by complex forms, sophisticated
    harmonies, and intricate melodies. It uses a wide range of scales
    and follows formal structures like sonata form.
    """

    def get_scales(self) -> List[str]:
        """Get the scales typically used in classical music.

        Classical music uses both major and minor scales extensively,
        including all keys and modes.

        Returns:
            List of major and minor scale strings in all keys
        """
        return ['C major', 'G major', 'D major', 'A major', 'E major', 'B major',
                'F# major', 'C# major', 'F major', 'Bb major', 'Eb major', 'Ab major',
                'A minor', 'E minor', 'B minor', 'F# minor', 'C# minor', 'G# minor',
                'D# minor', 'A# minor', 'D minor', 'G minor', 'C minor', 'F minor',
                'Bb minor', 'Eb minor', 'Ab minor']

    def get_chord_progressions(self) -> List[List[str]]:
        """Get common chord progressions for classical music.

        Classical music uses traditional cadential progressions
        and sophisticated voice leading.

        Returns:
            List of classical music chord progressions
        """
        return [
            ['I', 'IV', 'V', 'I'],         # Classical cadence
            ['ii', 'V', 'I'],              # Authentic cadence
            ['I', 'vi', 'IV', 'V'],        # Plagal cadence variant
            ['I', 'IV', 'vi', 'V'],         # Deceptive cadence
            ['I', 'V', 'vi', 'iii', 'IV', 'I', 'IV', 'V'] # Pachelbel's Canon
        ]

    def get_rhythm_patterns(self) -> List[Dict[str, Any]]:
        """Get typical rhythm patterns for classical music.

        Classical music uses a variety of rhythmic patterns from different periods.

        Returns:
            List of classical music rhythm patterns
        """
        return [
            {'name': 'waltz', 'pattern': [1.0, 1.0, 1.0]},
            {'name': 'common_time', 'pattern': [1.0, 1.0, 1.0, 1.0]},
            {'name': 'cut_time', 'pattern': [2.0, 2.0]}
        ]

    def get_typical_structure(self) -> List[str]:
        """Get the typical song structure for classical music.

        Classical music follows formal structures like sonata form.

        Returns:
            List representing typical classical music structure
        """
        return ['exposition', 'development', 'recapitulation']

    def get_instrumentation(self) -> List[str]:
        """Get typical instruments used in classical music.

        Classical music uses orchestral instruments in sections.

        Returns:
            List of typical classical music instruments
        """
        return ['orchestra', 'strings', 'woodwinds', 'brass', 'percussion', 'piano']

    def get_subgenres(self) -> List[str]:
        """Get subgenres for classical music.

        Returns:
            List of classical subgenres
        """
        return ['baroque', 'classical', 'romantic', 'modern', 'contemporary']

    def get_beat_characteristics(self, subgenre: Optional[str] = None) -> Dict[str, Any]:
        """Get beat characteristics for classical music.

        Args:
            subgenre: Optional subgenre to get characteristics for

        Returns:
            Dictionary containing beat characteristics
        """
        # Base characteristics for classical
        characteristics = {
            'swing_factor': 0.5,
            'syncopation_level': 0.1,
            'emphasis_patterns': [1],  # Strong downbeat
            'tempo_range': (60, 160)
        }

        # Adjust based on subgenre
        if subgenre == 'baroque':
            characteristics.update({
                'swing_factor': 0.5,
                'syncopation_level': 0.2,
                'tempo_range': (80, 120)
            })
        elif subgenre == 'modern':
            characteristics.update({
                'swing_factor': 0.5,
                'syncopation_level': 0.3,
                'tempo_range': (40, 200)
            })

        return characteristics
class DnBRules(GenreRules):
     """Rules for Drum and Bass music.

     Drum and Bass (DnB) is characterized by fast tempos (160-180 BPM),
     intricate breakbeat rhythms, heavy sub-bass, and atmospheric synths.
     It often uses minor keys with modal interchange and features
     complex rhythmic layering over simple harmonic structures.
     """

     def get_scales(self) -> List[str]:
         """Get the scales typically used in DnB music.

         DnB uses minor scales and modal scales for atmospheric soundscapes.

         Returns:
             List of scale strings
         """
         return ['C minor', 'A minor', 'F minor', 'D minor', 'G minor', 'Eb minor',
                 'Bb minor', 'Ab minor', 'C dorian', 'D phrygian', 'Eb lydian', 'F mixolydian']

     def get_chord_progressions(self) -> List[List[str]]:
         """Get common chord progressions for DnB music.

         DnB uses simple, repetitive progressions that create groove
         and allow focus on rhythm and bass.

         Returns:
             List of DnB chord progressions
         """
         return [
             ['i', 'iv', 'v'],              # Simple minor progression
             ['i', 'bVII', 'bVI', 'bVII'],  # Atmospheric progression
             ['vi', 'IV', 'i', 'V'],        # Modal interchange
             ['i', 'bVI', 'bIII', 'bVII'],  # Liquid DnB style
             ['I', 'V', 'vi', 'IV'],         # Major key for jump-up
             ['i', 'iv', 'i', 'iv'] # Repetitive groove
         ]

     def get_rhythm_patterns(self) -> List[Dict[str, Any]]:
         """Get typical rhythm patterns for DnB music.

         DnB features intricate breakbeat patterns and fast snare rolls.

         Returns:
             List of DnB rhythm patterns
         """
         return [
             {'name': 'amen_break', 'pattern': [0.5, 0.25, 0.25, 0.5, 0.25, 0.25, 0.25, 0.25]},
             {'name': 'double_kick', 'pattern': [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]},
             {'name': 'syncopated_snare', 'pattern': [0.5, 0.5, 0.25, 0.25, 0.5]},
             {'name': 'jungle_pattern', 'pattern': [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]}
         ]

     def get_typical_structure(self) -> List[str]:
         """Get the typical song structure for DnB music.

         DnB follows a build-drop structure with breakdown sections.

         Returns:
             List representing typical DnB song structure
         """
         return ['intro', 'buildup', 'drop', 'breakdown', 'drop', 'atmospheric_section', 'outro']

     def get_instrumentation(self) -> List[str]:
         """Get typical instruments used in DnB music.

         DnB uses electronic production tools and samples.

         Returns:
             List of typical DnB instruments
         """
         return ['synthesizer', 'drum_machine', 'sampler', 'sub_bass', 'effects_processor']

     def get_subgenres(self) -> List[str]:
         """Get subgenres for DnB music.

         Returns:
             List of DnB subgenres
         """
         return ['jungle', 'liquid', 'techstep', 'jump-up', 'neurofunk']

     def get_beat_characteristics(self, subgenre: Optional[str] = None) -> Dict[str, Any]:
         """Get beat characteristics for DnB music.

         Args:
             subgenre: Optional subgenre to get characteristics for

         Returns:
             Dictionary containing beat characteristics
         """
         # Base characteristics for DnB
         characteristics = {
             'swing_factor': 0.5,
             'syncopation_level': 0.8,
             'emphasis_patterns': [1, 2, 3, 4],  # Full beat emphasis with syncopation
             'tempo_range': (160, 180)
         }

         # Adjust based on subgenre
         if subgenre == 'liquid':
             characteristics.update({
                 'swing_factor': 0.52,
                 'syncopation_level': 0.6,
                 'tempo_range': (170, 175)  # Slightly calmer
             })
         elif subgenre == 'jungle':
             characteristics.update({
                 'swing_factor': 0.48,
                 'syncopation_level': 0.9,
                 'tempo_range': (155, 165)
             })
         elif subgenre == 'techstep':
             characteristics.update({
                 'swing_factor': 0.5,
                 'syncopation_level': 0.7,
                 'tempo_range': (160, 180)
             })

         return characteristics

     def get_bass_patterns(self) -> List[Dict[str, Any]]:
         """Get typical bass patterns for DnB music.

         DnB features prominent sub-bass patterns that drive the groove.

         Returns:
             List of DnB bass patterns
         """
         return [
             {'name': 'wobble_bass', 'pattern': [1.0, 0.5, 0.5, 0.25, 0.25, 0.5]},
             {'name': 'sub_bass', 'pattern': [1.0, 1.0, 0.5, 0.5]},
             {'name': 'rolling_bass', 'pattern': [0.25, 0.25, 0.25, 0.25, 0.5, 0.5]}
         ]

     def get_melody_patterns(self) -> List[Dict[str, Any]]:
         """Get typical melody patterns for DnB music.

         DnB melodies are often atmospheric and syncopated.

         Returns:
             List of DnB melody patterns
         """
         return [
             {'name': 'atmospheric_pad', 'pattern': [1.0, 0.5, 0.5, 0.25, 0.75]},
             {'name': 'syncopated_lead', 'pattern': [0.25, 0.25, 0.5, 0.5, 0.5]},
             {'name': 'rolling_melody', 'pattern': [0.5, 0.25, 0.25, 0.5, 0.5]}
         ]

     def get_harmony_patterns(self) -> List[Dict[str, Any]]:
         """Get typical harmony patterns for DnB music.

         DnB harmonies are often simple and repetitive.

         Returns:
             List of DnB harmony patterns
         """
         return [
             {'name': 'modal_chords', 'pattern': [1.0, 1.0, 0.5, 0.5]},
             {'name': 'atmospheric_pad', 'pattern': [2.0, 1.0, 1.0]},
             {'name': 'rhythmic_harmony', 'pattern': [0.5, 0.5, 0.5, 0.5, 1.0]}
         ]