#!/usr/bin/env python3
"""
Advanced Musical Structure Insights Analyzer

Purpose:
- Analyze musical structure from MIDI files with focus on:
  - Harmony analysis: chord recognition and key detection
  - Melody analysis: motif detection and contour analysis
  - Song form analysis: section detection and repetition analysis
  - Harmonic complexity metrics
  - Voice leading analysis

Outputs:
- test_outputs/structure_insights_report.json
- test_outputs/structure_insights_report.md

Usage:
  python analyzers/structure_insights.py --inputs path1.mid path2.mid dir_or_glob ... [--verbose]

Dependencies:
- mido for MIDI parsing
- music_theory.py for music theory utilities
- dataclasses for structured data
"""

import argparse
import glob
import json
import math
import os
import statistics
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple, Set, NamedTuple

import mido

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from music_theory import MusicTheory, Note, ScaleType, ChordType


# ------------------------
# Data structures
# ------------------------

@dataclass
class MidiNote:
    pitch: int
    velocity: int
    start_tick: int
    end_tick: int
    start_sec: float
    end_sec: float
    channel: int
    track_index: int


@dataclass
class Chord:
    pitches: List[int]
    start_time: float
    duration: float
    confidence: float = 1.0
    root: Optional[int] = None
    quality: Optional[str] = None


@dataclass
class KeyDetection:
    root_note: int
    scale_type: str
    confidence: float
    duration: float  # How long this key is active


@dataclass
class Motif:
    pitches: List[int]
    intervals: List[int]
    start_time: float
    duration: float
    confidence: float = 1.0


@dataclass
class Section:
    start_time: float
    end_time: float
    type: str  # 'verse', 'chorus', 'bridge', etc.
    confidence: float = 1.0


@dataclass
class StructureInsights:
    path: str

    # Harmony analysis
    detected_chords: List[Chord]
    key_detections: List[KeyDetection]
    key_strengths: Dict[str, float]

    # Melody analysis
    motifs: List[Motif]
    contour_profile: List[float]
    pitch_range: Tuple[int, int]

    # Song form analysis
    sections: List[Section]
    repetition_patterns: List[Dict[str, Any]]

    # Harmonic complexity
    harmonic_complexity_score: float
    tension_profile: List[float]
    dissonance_profile: List[float]

    # Voice leading
    voice_leading_movements: List[Dict[str, Any]]
    parallel_motion_score: float
    contrary_motion_score: float


# ------------------------
# Helper utilities (adapted from existing analyzers)
# ------------------------

def parse_midi_notes(path: str) -> Tuple[mido.MidiFile, List[MidiNote]]:
    """Parse MIDI file and return notes with timing."""
    mid = mido.MidiFile(path)
    all_events = []

    # Build absolute ticks
    for track_idx, track in enumerate(mid.tracks):
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            all_events.append((abs_tick, msg, track_idx))

    all_events.sort(key=lambda x: x[0])

    # Build tempo map
    tempo_map = []
    for tick, msg, _ in all_events:
        if getattr(msg, "type", None) == "set_tempo":
            tempo_map.append((tick, int(getattr(msg, "tempo", 500000))))
    if not tempo_map or tempo_map[0][0] != 0:
        tempo_map.insert(0, (0, 500000))

    # Precompute seconds
    tempo_accum = []
    total_seconds = 0.0
    for i, (tick, uspb) in enumerate(tempo_map):
        tempo_accum.append((tick, uspb, total_seconds))
        next_tick = tempo_map[i+1][0] if i+1 < len(tempo_map) else all_events[-1][0] if all_events else tick
        tick_span = max(0, next_tick - tick)
        sec_span = (tick_span / mid.ticks_per_beat) * (uspb / 1_000_000.0)
        total_seconds += sec_span

    # Extract notes
    abs_notes = []
    active = {}  # (pitch, channel) -> (start_tick, velocity)

    for abs_tick, msg, track_idx in all_events:
        mtype = getattr(msg, "type", None)
        if mtype == "note_on" and int(getattr(msg, "velocity", 0)) > 0:
            pitch = int(getattr(msg, "note", 0))
            velocity = int(getattr(msg, "velocity", 0))
            channel = int(getattr(msg, "channel", 0))
            key = (pitch, channel)
            if key in active:
                # End previous note due to re-articulation
                start_tick, prev_vel = active[key]
                start_sec = tick_to_seconds(start_tick, tempo_accum, mid.ticks_per_beat)
                end_sec = tick_to_seconds(abs_tick, tempo_accum, mid.ticks_per_beat)
                abs_notes.append(MidiNote(
                    pitch=pitch,
                    velocity=prev_vel,
                    start_tick=start_tick,
                    end_tick=abs_tick,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    channel=channel,
                    track_index=track_idx,
                ))
            # Start new note
            active[key] = (abs_tick, velocity)
        elif (mtype == "note_off") or (mtype == "note_on" and int(getattr(msg, "velocity", 0)) == 0):
            pitch = int(getattr(msg, "note", 0))
            channel = int(getattr(msg, "channel", 0))
            key = (pitch, channel)
            if key in active:
                start_tick, vel = active[key]
                start_sec = tick_to_seconds(start_tick, tempo_accum, mid.ticks_per_beat)
                end_sec = tick_to_seconds(abs_tick, tempo_accum, mid.ticks_per_beat)
                abs_notes.append(MidiNote(
                    pitch=pitch,
                    velocity=vel,
                    start_tick=start_tick,
                    end_tick=abs_tick,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    channel=channel,
                    track_index=track_idx,
                ))
                del active[key]

    return mid, abs_notes


def tick_to_seconds(tick: int, tempo_accum: List[Tuple[int, int, float]], ticks_per_beat: int) -> float:
    """Convert tick to seconds using tempo accumulator."""
    lo, hi = 0, len(tempo_accum) - 1
    idx = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if tempo_accum[mid][0] <= tick:
            idx = mid
            lo = mid + 1
        else:
            hi = mid - 1
    boundary_tick, uspb, sec_at_boundary = tempo_accum[idx]
    delta_ticks = tick - boundary_tick
    return sec_at_boundary + (delta_ticks / ticks_per_beat) * (uspb / 1_000_000.0)


# ------------------------
# Harmony Analysis Engine
# ------------------------

class HarmonyAnalyzer:
    """Analyzes harmonic content: chords and keys."""

    def __init__(self):
        self.chord_templates = self._build_chord_templates()

    def _build_chord_templates(self) -> Dict[str, Set[int]]:
        """Build pitch class templates for chord recognition."""
        templates = {}

        # Generate all chords for all roots
        for root_note in Note:
            for chord_type in ChordType:
                pitches = MusicTheory.build_chord(root_note, chord_type)
                if pitches:
                    # Convert to pitch classes (0-11)
                    pitch_classes = set(p % 12 for p in pitches)
                    # Normalize to root=0
                    root_pc = pitches[0] % 12
                    normalized = set((p - root_pc) % 12 for p in pitch_classes)
                    key = f"{root_note.name}_{chord_type.value}"
                    templates[key] = normalized

        return templates

    def detect_chords(self, notes: List[MidiNote], window_size: float = 0.5) -> List[Chord]:
        """
        Detect chords by grouping simultaneous notes into time windows.

        Args:
            notes: List of MIDI notes
            window_size: Time window in seconds for grouping notes

        Returns:
            List of detected chords
        """
        if not notes:
            return []

        # Sort notes by start time
        sorted_notes = sorted(notes, key=lambda n: n.start_sec)

        chords = []
        current_window = []
        window_start = sorted_notes[0].start_sec

        for note in sorted_notes:
            if note.start_sec - window_start > window_size and current_window:
                # Process current window
                chord = self._process_note_window(current_window)
                if chord:
                    chords.append(chord)

                # Start new window
                current_window = [note]
                window_start = note.start_sec
            else:
                current_window.append(note)

        # Process final window
        if current_window:
            chord = self._process_note_window(current_window)
            if chord:
                chords.append(chord)

        return chords

    def _process_note_window(self, notes: List[MidiNote]) -> Optional[Chord]:
        """Process a window of notes to detect a chord."""
        if len(notes) < 2:
            return None

        # Get unique pitches in this window
        pitches = list(set(n.pitch for n in notes))
        pitches.sort()

        if len(pitches) < 2:
            return None

        # Convert to pitch classes
        pitch_classes = [p % 12 for p in pitches]

        # Find best chord match
        best_match = self._find_best_chord_match(pitch_classes)

        if best_match:
            root, quality, confidence = best_match
            # Calculate timing
            start_time = min(n.start_sec for n in notes)
            end_time = max(n.end_sec for n in notes)
            duration = end_time - start_time

            return Chord(
                pitches=pitches,
                start_time=start_time,
                duration=duration,
                confidence=confidence,
                root=root,
                quality=quality
            )

        return None

    def _find_best_chord_match(self, pitch_classes: List[int]) -> Optional[Tuple[int, str, float]]:
        """Find the best chord match for a set of pitch classes."""
        pitch_set = set(pitch_classes)

        best_match = None
        best_confidence = 0.0

        for chord_name, template in self.chord_templates.items():
            # Calculate intersection and union
            intersection = len(pitch_set.intersection(template))
            union = len(pitch_set.union(template))

            if union == 0:
                continue

            confidence = intersection / union

            # Must match at least 2 notes for a chord
            if intersection >= 2 and confidence > best_confidence:
                # Extract root and quality from chord name
                parts = chord_name.split('_', 1)
                if len(parts) == 2:
                    root_name, quality = parts
                    # Find root pitch class that would make this chord
                    for root_pc in pitch_classes:
                        # Check if template matches when normalized to this root
                        normalized = set((p - root_pc) % 12 for p in pitch_set)
                        if normalized == template:
                            best_match = (root_pc, quality, confidence)
                            best_confidence = confidence
                            break

        return best_match

    def detect_key(self, notes: List[MidiNote], chords: Optional[List[Chord]] = None) -> List[KeyDetection]:
        """
        Detect musical key using pitch distribution and chord analysis.

        Args:
            notes: List of MIDI notes
            chords: List of detected chords (optional)

        Returns:
            List of key detections with confidence scores
        """
        if not notes:
            return []

        # Get all pitches
        all_pitches = [n.pitch for n in notes]
        pitch_classes = [p % 12 for p in all_pitches]

        # Calculate pitch class distribution
        pc_counts = Counter(pitch_classes)
        total_notes = len(pitch_classes)

        key_scores = {}

        # Score each possible key
        for root_pc in range(12):
            for scale_type in [ScaleType.MAJOR, ScaleType.MINOR_NATURAL]:
                scale_pcs = set()
                intervals = MusicTheory.SCALE_INTERVALS[scale_type]
                for interval in intervals:
                    scale_pcs.add((root_pc + interval) % 12)

                # Calculate how well the scale fits the pitch distribution
                scale_matches = sum(pc_counts.get(pc, 0) for pc in scale_pcs)
                scale_coverage = scale_matches / total_notes if total_notes > 0 else 0

                # Bonus for chord roots
                chord_bonus = 0.0
                if chords:
                    for chord in chords:
                        if chord.root is not None:
                            chord_root_pc = chord.root % 12
                            if chord_root_pc in scale_pcs:
                                chord_bonus += chord.confidence * 0.1

                # Final score
                score = scale_coverage + chord_bonus

                key_name = f"{self._pc_to_note_name(root_pc)} {scale_type.value}"
                key_scores[key_name] = score

        # Sort by score and create key detections
        sorted_keys = sorted(key_scores.items(), key=lambda x: x[1], reverse=True)

        key_detections = []
        total_duration = max(n.end_sec for n in notes) - min(n.start_sec for n in notes) if notes else 1.0

        if sorted_keys:
            max_score = sorted_keys[0][1]
            for key_name, score in sorted_keys[:5]:  # Top 5 keys
                # Normalize score to confidence relative to max_score
                confidence = (score / max_score) if max_score > 0 else 0.0

                key_detections.append(KeyDetection(
                    root_note=self._note_name_to_value(key_name.split()[0]),
                    scale_type=key_name.split()[1],
                    confidence=confidence,
                    duration=total_duration
                ))

        return key_detections

    def _pc_to_note_name(self, pc: int) -> str:
        """Convert pitch class to note name."""
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        return note_names[pc % 12]

    def _note_name_to_value(self, name: str) -> int:
        """Convert note name to MIDI value."""
        note_map = {
            'C': 60, 'C#': 61, 'D': 62, 'D#': 63, 'E': 64,
            'F': 65, 'F#': 66, 'G': 67, 'G#': 68, 'A': 69, 'A#': 70, 'B': 71
        }
        return note_map.get(name, 60)


# ------------------------
# Melody Analysis Engine
# ------------------------

class MelodyAnalyzer:
    """Analyzes melodic content: motifs and contour."""

    def __init__(self):
        pass

    def detect_motifs(self, notes: List[MidiNote], min_length: int = 4, max_gap: float = 1.0) -> List[Motif]:
        """
        Detect repeated melodic patterns (motifs).

        Args:
            notes: List of MIDI notes (assuming melody track)
            min_length: Minimum number of notes in a motif
            max_gap: Maximum time gap between consecutive notes

        Returns:
            List of detected motifs
        """
        if len(notes) < min_length:
            return []

        # Sort by start time
        sorted_notes = sorted(notes, key=lambda n: n.start_sec)

        motifs = []

        # Simple pattern matching - look for repeated pitch sequences
        for i in range(len(sorted_notes) - min_length + 1):
            pattern = sorted_notes[i:i+min_length]

            # Check if this pattern repeats
            pattern_pitches = [n.pitch for n in pattern]
            pattern_intervals = self._calculate_intervals(pattern_pitches)

            # Look for matches later in the sequence
            for j in range(i + min_length, len(sorted_notes) - min_length + 1):
                candidate = sorted_notes[j:j+min_length]
                candidate_pitches = [n.pitch for n in candidate]
                candidate_intervals = self._calculate_intervals(candidate_pitches)

                if pattern_intervals == candidate_intervals:
                    # Found a match
                    start_time = pattern[0].start_sec
                    duration = pattern[-1].end_sec - pattern[0].start_sec

                    motifs.append(Motif(
                        pitches=pattern_pitches,
                        intervals=pattern_intervals,
                        start_time=start_time,
                        duration=duration,
                        confidence=0.8  # Could be improved with more sophisticated matching
                    ))

        return motifs

    def _calculate_intervals(self, pitches: List[int]) -> List[int]:
        """Calculate pitch intervals between consecutive notes."""
        if len(pitches) < 2:
            return []
        return [pitches[i+1] - pitches[i] for i in range(len(pitches)-1)]

    def analyze_contour(self, notes: List[MidiNote], window_size: float = 1.0) -> List[float]:
        """
        Analyze melodic contour (pitch movement over time).

        Args:
            notes: List of MIDI notes
            window_size: Time window for contour analysis

        Returns:
            List of contour values (smoothed pitch changes)
        """
        if not notes:
            return []

        sorted_notes = sorted(notes, key=lambda n: n.start_sec)

        contour = []
        current_window = []
        window_start = sorted_notes[0].start_sec

        for note in sorted_notes:
            if note.start_sec - window_start > window_size:
                # Process current window
                if len(current_window) > 1:
                    avg_pitch_change = self._calculate_average_pitch_change(current_window)
                    contour.append(avg_pitch_change)

                # Start new window
                current_window = [note]
                window_start = note.start_sec
            else:
                current_window.append(note)

        # Process final window
        if len(current_window) > 1:
            avg_pitch_change = self._calculate_average_pitch_change(current_window)
            contour.append(avg_pitch_change)

        return contour

    def _calculate_average_pitch_change(self, notes: List[MidiNote]) -> float:
        """Calculate average pitch change direction in a note sequence."""
        if len(notes) < 2:
            return 0.0

        total_change = 0.0
        count = 0

        for i in range(len(notes) - 1):
            change = notes[i+1].pitch - notes[i].pitch
            if change != 0:
                # Normalize to direction (-1, 0, 1)
                direction = 1 if change > 0 else -1
                total_change += direction
                count += 1

        return total_change / count if count > 0 else 0.0

    def get_pitch_range(self, notes: List[MidiNote]) -> Tuple[int, int]:
        """Get the minimum and maximum pitches in the melody."""
        if not notes:
            return (60, 60)

        pitches = [n.pitch for n in notes]
        return (min(pitches), max(pitches))


# ------------------------
# Song Form Analysis Engine
# ------------------------

class FormAnalyzer:
    """Analyzes song form: sections and repetition patterns."""

    def __init__(self):
        pass

    def detect_sections(self, notes: List[MidiNote], chords: List[Chord]) -> List[Section]:
        """
        Detect song sections based on harmonic and rhythmic changes.

        Args:
            notes: All MIDI notes
            chords: Detected chords

        Returns:
            List of detected sections
        """
        if not notes and not chords:
            return []

        # Use harmonic changes to detect sections
        sections = []

        # Simple heuristic: group by harmonic stability
        if chords:
            sorted_chords = sorted(chords, key=lambda c: c.start_time)

            current_section_start = sorted_chords[0].start_time
            current_root = sorted_chords[0].root

            for chord in sorted_chords[1:]:
                # If root changes significantly, start new section
                if chord.root != current_root and chord.start_time - current_section_start > 2.0:
                    sections.append(Section(
                        start_time=current_section_start,
                        end_time=chord.start_time,
                        type=self._classify_section_type(current_section_start, chord.start_time, notes),
                        confidence=0.7
                    ))
                    current_section_start = chord.start_time
                    current_root = chord.root

            # Add final section
            if sorted_chords:
                sections.append(Section(
                    start_time=current_section_start,
                    end_time=sorted_chords[-1].start_time + sorted_chords[-1].duration,
                    type=self._classify_section_type(current_section_start, sorted_chords[-1].start_time + sorted_chords[-1].duration, notes),
                    confidence=0.7
                ))

        return sections

    def _classify_section_type(self, start_time: float, end_time: float, notes: List[MidiNote]) -> str:
        """Classify section type based on timing and note characteristics."""
        duration = end_time - start_time

        # Very rough heuristics
        if duration > 20:
            return "verse"
        elif duration > 15:
            return "chorus"
        elif duration > 10:
            return "bridge"
        else:
            return "intro"

    def detect_repetition(self, notes: List[MidiNote], window_size: float = 4.0) -> List[Dict[str, Any]]:
        """
        Detect repetition patterns in the music.

        Args:
            notes: List of MIDI notes
            window_size: Size of analysis windows

        Returns:
            List of repetition patterns
        """
        if not notes:
            return []

        # Simple repetition detection based on pitch sequence similarity
        sorted_notes = sorted(notes, key=lambda n: n.start_sec)

        patterns = []
        step_size = 2.0  # Slide window by 2 seconds
        window_size = 4.0
        max_end = max(n.end_sec for n in notes) if notes else 0.0

        start_time = 0.0
        while start_time < max_end:
            window_notes = [n for n in sorted_notes
                          if start_time <= n.start_sec < start_time + window_size]

            if len(window_notes) < 4:
                start_time += step_size
                continue

            # Create signature from pitch sequence
            pitch_sequence = tuple(n.pitch for n in window_notes)

            # Look for similar sequences
            similarity_score = self._calculate_sequence_similarity(pitch_sequence)

            if similarity_score > 0.7:  # High similarity threshold
                patterns.append({
                    "start_time": start_time,
                    "duration": window_size,
                    "similarity_score": similarity_score,
                    "note_count": len(window_notes)
                })

            start_time += step_size

        return patterns

    def _calculate_sequence_similarity(self, sequence: Tuple[int, ...]) -> float:
        """Calculate how similar this sequence is to others (simplified)."""
        # For now, return a random score - this would need more sophisticated analysis
        return 0.5 + (len(sequence) % 10) / 20.0


# ------------------------
# Harmonic Complexity Engine
# ------------------------

class ComplexityAnalyzer:
    """Analyzes harmonic complexity and tension."""

    def __init__(self):
        pass

    def calculate_complexity_score(self, chords: List[Chord], key_detections: List[KeyDetection]) -> float:
        """
        Calculate overall harmonic complexity score.

        Args:
            chords: List of detected chords
            key_detections: List of key detections

        Returns:
            Complexity score (0.0 to 1.0)
        """
        if not chords:
            return 0.0

        complexity_factors = []

        # Factor 1: Chord variety
        unique_chords = len(set((c.root, c.quality) for c in chords if c.root and c.quality))
        total_chords = len(chords)
        variety_score = unique_chords / total_chords if total_chords > 0 else 0
        complexity_factors.append(variety_score)

        # Factor 2: Key ambiguity
        if key_detections:
            top_key_confidence = max(k.confidence for k in key_detections)
            ambiguity_score = 1.0 - top_key_confidence  # Higher ambiguity = more complex
            complexity_factors.append(ambiguity_score)

        # Factor 3: Extended chords (7ths, etc.)
        extended_count = sum(1 for c in chords if c.quality and ('7' in c.quality or '9' in c.quality))
        extended_ratio = extended_count / total_chords if total_chords > 0 else 0
        complexity_factors.append(extended_ratio)

        # Average the factors
        return sum(complexity_factors) / len(complexity_factors) if complexity_factors else 0.0

    def analyze_tension_profile(self, chords: List[Chord], key_context: str) -> List[float]:
        """
        Analyze harmonic tension over time.

        Args:
            chords: List of detected chords
            key_context: Current key (e.g., "C major")

        Returns:
            List of tension values over time
        """
        tension_profile = []

        for chord in chords:
            tension = 0.0

            # Tension based on chord type
            if chord.quality:
                if 'dim' in chord.quality:
                    tension += 0.8
                elif '7' in chord.quality or '9' in chord.quality:
                    tension += 0.6
                elif chord.quality == 'minor':
                    tension += 0.3

            # Tension based on key relationship (simplified)
            if chord.root is not None:
                # Dissonant intervals from tonic would increase tension
                root_pc = chord.root % 12
                # Simple heuristic: non-diatonic notes increase tension
                try:
                    scale_pitches = MusicTheory.get_scale_pitches_from_string(key_context)
                    scale_pcs = set(p % 12 for p in scale_pitches)
                    if root_pc not in scale_pcs:
                        tension += 0.4
                except:
                    pass

            tension_profile.append(min(tension, 1.0))  # Cap at 1.0

        return tension_profile

    def analyze_dissonance_profile(self, notes: List[MidiNote]) -> List[float]:
        """
        Analyze dissonance over time based on note combinations.

        Args:
            notes: List of MIDI notes

        Returns:
            List of dissonance values over time
        """
        dissonance_profile = []

        # Group notes by time windows
        time_windows = self._create_time_windows(notes, window_size=0.5)

        for window_notes in time_windows:
            if len(window_notes) < 2:
                dissonance_profile.append(0.0)
                continue

            # Calculate dissonance based on intervals
            pitches = [n.pitch for n in window_notes]
            dissonance = 0.0
            pair_count = 0

            for i in range(len(pitches)):
                for j in range(i+1, len(pitches)):
                    interval = abs(pitches[i] - pitches[j]) % 12
                    # Dissonant intervals: minor 2nd, major 2nd, minor 9th, major 9th, tritone
                    dissonant_intervals = {1, 2, 10, 11, 6}  # semitones
                    if interval in dissonant_intervals:
                        dissonance += 0.8
                    elif interval in {3, 4, 8, 9}:  # minor/major thirds, sixths
                        dissonance += 0.3
                    pair_count += 1

            avg_dissonance = dissonance / pair_count if pair_count > 0 else 0.0
            dissonance_profile.append(min(avg_dissonance, 1.0))

        return dissonance_profile

    def _create_time_windows(self, notes: List[MidiNote], window_size: float) -> List[List[MidiNote]]:
        """Group notes into time windows."""
        if not notes:
            return []

        sorted_notes = sorted(notes, key=lambda n: n.start_sec)
        windows = []
        current_window = []
        window_start = sorted_notes[0].start_sec

        for note in sorted_notes:
            if note.start_sec - window_start > window_size:
                if current_window:
                    windows.append(current_window)
                current_window = [note]
                window_start = note.start_sec
            else:
                current_window.append(note)

        if current_window:
            windows.append(current_window)

        return windows


# ------------------------
# Voice Leading Analysis Engine
# ------------------------

class VoiceLeadingAnalyzer:
    """Analyzes voice leading between chords."""

    def __init__(self):
        pass

    def analyze_voice_leading(self, chords: List[Chord]) -> List[Dict[str, Any]]:
        """
        Analyze voice leading movements between consecutive chords.

        Args:
            chords: List of detected chords

        Returns:
            List of voice leading movements
        """
        movements = []

        for i in range(len(chords) - 1):
            current = chords[i]
            next_chord = chords[i+1]

            movement = {
                "from_chord": f"{current.root}_{current.quality}" if current.root and current.quality else "unknown",
                "to_chord": f"{next_chord.root}_{next_chord.quality}" if next_chord.root and next_chord.quality else "unknown",
                "time": next_chord.start_time,
                "movements": self._calculate_voice_movements(current.pitches, next_chord.pitches)
            }

            movements.append(movement)

        return movements

    def _calculate_voice_movements(self, from_pitches: List[int], to_pitches: List[int]) -> List[int]:
        """Calculate interval movements for voice leading."""
        movements = []

        # Simple approach: sort pitches and calculate differences
        from_sorted = sorted(from_pitches)
        to_sorted = sorted(to_pitches)

        min_len = min(len(from_sorted), len(to_sorted))

        for i in range(min_len):
            movement = to_sorted[i] - from_sorted[i]
            movements.append(movement)

        return movements

    def calculate_motion_scores(self, chords: List[Chord]) -> Tuple[float, float]:
        """
        Calculate parallel and contrary motion scores.

        Args:
            chords: List of detected chords

        Returns:
            Tuple of (parallel_motion_score, contrary_motion_score)
        """
        parallel_score = 0.0
        contrary_score = 0.0
        total_transitions = 0

        for i in range(len(chords) - 1):
            current = chords[i]
            next_chord = chords[i+1]

            movements = self._calculate_voice_movements(current.pitches, next_chord.pitches)

            if len(movements) >= 2:
                # Check if all movements are in the same direction (parallel)
                directions = [1 if m > 0 else -1 if m < 0 else 0 for m in movements]
                if all(d == directions[0] for d in directions) and directions[0] != 0:
                    parallel_score += 1

                # Check for contrary motion (some up, some down)
                unique_directions = set(directions)
                if 1 in unique_directions and -1 in unique_directions:
                    contrary_score += 1

                total_transitions += 1

        # Normalize scores
        if total_transitions > 0:
            parallel_score /= total_transitions
            contrary_score /= total_transitions

        return parallel_score, contrary_score


# ------------------------
# Main Structure Insights Analyzer
# ------------------------

class StructureInsightsAnalyzer:
    """Main analyzer that combines all analysis engines."""

    def __init__(self):
        self.harmony_analyzer = HarmonyAnalyzer()
        self.melody_analyzer = MelodyAnalyzer()
        self.form_analyzer = FormAnalyzer()
        self.complexity_analyzer = ComplexityAnalyzer()
        self.voice_leading_analyzer = VoiceLeadingAnalyzer()

    def analyze_file(self, path: str, verbose: bool = False) -> Optional[StructureInsights]:
        """Analyze a single MIDI file for structure insights."""
        if not os.path.exists(path):
            if verbose:
                print(f"Skip missing: {path}")
            return None

        try:
            mid, notes = parse_midi_notes(path)
        except Exception as e:
            print(f"Error parsing {path}: {e}")
            return None

        if not notes:
            if verbose:
                print(f"No notes in {path}")
            return None

        # Separate melody and harmony notes (simple heuristic)
        melody_notes = [n for n in notes if n.channel != 9]  # Exclude percussion
        harmony_notes = melody_notes  # For now, use all non-percussion

        # Harmony Analysis
        detected_chords = self.harmony_analyzer.detect_chords(harmony_notes)
        key_detections = self.harmony_analyzer.detect_key(notes, detected_chords)

        # Calculate key strengths for summary
        key_strengths = {}
        for key in key_detections:
            key_name = f"{self.harmony_analyzer._pc_to_note_name(key.root_note % 12)} {key.scale_type}"
            key_strengths[key_name] = key.confidence

        # Melody Analysis
        motifs = self.melody_analyzer.detect_motifs(melody_notes)
        contour_profile = self.melody_analyzer.analyze_contour(melody_notes)
        pitch_range = self.melody_analyzer.get_pitch_range(melody_notes)

        # Song Form Analysis
        sections = self.form_analyzer.detect_sections(notes, detected_chords)
        repetition_patterns = self.form_analyzer.detect_repetition(notes)

        # Harmonic Complexity
        harmonic_complexity_score = self.complexity_analyzer.calculate_complexity_score(
            detected_chords, key_detections)

        # Use first key detection for key context
        key_context = ""
        if key_detections:
            top_key = key_detections[0]
            key_context = f"{self.harmony_analyzer._pc_to_note_name(top_key.root_note % 12)} {top_key.scale_type}"

        tension_profile = self.complexity_analyzer.analyze_tension_profile(detected_chords, key_context)
        dissonance_profile = self.complexity_analyzer.analyze_dissonance_profile(notes)

        # Voice Leading
        voice_leading_movements = self.voice_leading_analyzer.analyze_voice_leading(detected_chords)
        parallel_motion_score, contrary_motion_score = self.voice_leading_analyzer.calculate_motion_scores(detected_chords)

        return StructureInsights(
            path=path,
            detected_chords=detected_chords,
            key_detections=key_detections,
            key_strengths=key_strengths,
            motifs=motifs,
            contour_profile=contour_profile,
            pitch_range=pitch_range,
            sections=sections,
            repetition_patterns=repetition_patterns,
            harmonic_complexity_score=harmonic_complexity_score,
            tension_profile=tension_profile,
            dissonance_profile=dissonance_profile,
            voice_leading_movements=voice_leading_movements,
            parallel_motion_score=parallel_motion_score,
            contrary_motion_score=contrary_motion_score
        )


# ------------------------
# CLI and Reporting
# ------------------------

def ensure_out_dirs():
    os.makedirs("test_outputs", exist_ok=True)


def glob_inputs(inputs: List[str]) -> List[str]:
    files = []
    for inp in inputs:
        if os.path.isdir(inp):
            files.extend(glob.glob(os.path.join(inp, "*.mid")))
        else:
            matched = glob.glob(inp)
            if matched:
                files.extend(matched)
            else:
                files.append(inp)
    # Deduplicate
    seen = set()
    unique_files = []
    for f in files:
        if f not in seen:
            seen.add(f)
            unique_files.append(f)
    return unique_files


def write_json_report(path: str, insights: List[StructureInsights]) -> None:
    """Write detailed JSON report."""
    payload = {
        "files": [asdict(insight) for insight in insights],
        "summary": {
            "total_files": len(insights),
            "avg_harmonic_complexity": statistics.mean([i.harmonic_complexity_score for i in insights]) if insights else 0,
            "avg_parallel_motion": statistics.mean([i.parallel_motion_score for i in insights]) if insights else 0,
            "avg_contrary_motion": statistics.mean([i.contrary_motion_score for i in insights]) if insights else 0,
        }
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def write_markdown_report(path: str, insights: List[StructureInsights]) -> None:
    """Write human-readable markdown report."""
    lines = []
    lines.append("# Musical Structure Insights Report")
    lines.append("")

    if not insights:
        lines.append("No files analyzed.")
    else:
        lines.append(f"## Summary ({len(insights)} files)")
        avg_complexity = statistics.mean([i.harmonic_complexity_score for i in insights])
        avg_parallel = statistics.mean([i.parallel_motion_score for i in insights])
        avg_contrary = statistics.mean([i.contrary_motion_score for i in insights])

        lines.append(f"- Average harmonic complexity: {avg_complexity:.2f}")
        lines.append(f"- Average parallel motion: {avg_parallel:.2f}")
        lines.append(f"- Average contrary motion: {avg_contrary:.2f}")
        lines.append("")

        for insight in insights[:5]:  # Show first 5 files
            lines.append(f"## {os.path.basename(insight.path)}")
            lines.append("")

            # Harmony
            lines.append("### Harmony Analysis")
            lines.append(f"- Detected chords: {len(insight.detected_chords)}")
            if insight.key_detections:
                top_key = insight.key_detections[0]
                key_name = f"{HarmonyAnalyzer()._pc_to_note_name(top_key.root_note % 12)} {top_key.scale_type}"
                lines.append(f"- Primary key: {key_name} (confidence: {top_key.confidence:.2f})")
            lines.append("")

            # Melody
            lines.append("### Melody Analysis")
            lines.append(f"- Detected motifs: {len(insight.motifs)}")
            lines.append(f"- Pitch range: {insight.pitch_range[0]}-{insight.pitch_range[1]}")
            lines.append(f"- Contour points: {len(insight.contour_profile)}")
            lines.append("")

            # Form
            lines.append("### Song Form Analysis")
            lines.append(f"- Detected sections: {len(insight.sections)}")
            lines.append(f"- Repetition patterns: {len(insight.repetition_patterns)}")
            lines.append("")

            # Complexity and Voice Leading
            lines.append("### Harmonic Complexity & Voice Leading")
            lines.append(f"- Complexity score: {insight.harmonic_complexity_score:.2f}")
            lines.append(f"- Parallel motion: {insight.parallel_motion_score:.2f}")
            lines.append(f"- Contrary motion: {insight.contrary_motion_score:.2f}")
            lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Advanced musical structure insights analyzer.")
    parser.add_argument("--inputs", nargs="+", required=True,
                       help="Input MIDI paths, directories, or glob patterns.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    ensure_out_dirs()

    paths = glob_inputs(args.inputs)
    if args.verbose:
        print(f"Discovered {len(paths)} input(s).")

    analyzer = StructureInsightsAnalyzer()
    insights = []

    for path in paths:
        if args.verbose:
            print(f"Analyzing {path}...")
        insight = analyzer.analyze_file(path, verbose=args.verbose)
        if insight:
            insights.append(insight)

    # Write reports
    json_path = os.path.join("test_outputs", "structure_insights_report.json")
    md_path = os.path.join("test_outputs", "structure_insights_report.md")

    write_json_report(json_path, insights)
    write_markdown_report(md_path, insights)

    print(f"Wrote: {json_path}")
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()