"""
Ambient Music Characteristics Analyzer

Purpose:
- Analyze MIDI files for ambient/atmospheric music characteristics
- Provide metrics for genre classification enhancement
- Support detection of long notes, sparsity, tempo stability, and atmospheric patterns

Features:
- Average note duration calculation
- Rhythmic sparsity metrics
- Tempo stability analysis
- Atmospheric pattern detection
"""

import mido
from typing import Dict, List, Tuple
from dataclasses import dataclass
from statistics import mean, stdev
import numpy as np


@dataclass
class NoteInfo:
    start_time: float
    end_time: float
    duration: float
    velocity: int
    pitch: int
    is_sustained: bool  # Duration > 1 beat


class AmbientAnalyzer:
    """Analyzer for ambient music characteristics."""

    def __init__(self, default_tempo_bpm: float = 80.0):
        self.default_tempo_bpm = default_tempo_bpm

    def analyze_ambient_characteristics(self, mid: mido.MidiFile) -> Dict[str, float]:
        """
        Analyze ambient characteristics of a MIDI file.

        Args:
            mid: mido.MidiFile object

        Returns:
            Dictionary of ambient metrics as floats (0.0-1.0 normalized where applicable)
        """
        if not mid.tracks:
            return {
                'avg_note_duration_beats': 0.0,
                'sparsity_score': 0.0,
                'tempo_stability': 1.0,
                'atmospheric_pattern_score': 0.0
            }

        # Build tempo map and convert to seconds
        tempo_map = self._build_tempo_map(mid)
        notes = self._extract_all_notes(mid, tempo_map, mid.ticks_per_beat)

        if not notes:
            return {
                'avg_note_duration_beats': 0.0,
                'sparsity_score': 0.0,
                'tempo_stability': 1.0,
                'atmospheric_pattern_score': 0.0
            }

        total_duration = max(n.end_time for n in notes)
        avg_tempo = self._calculate_average_tempo(tempo_map)
        beat_duration = 60.0 / avg_tempo

        # 1. Average note duration (in beats)
        durations_beats = [(n.duration / beat_duration) for n in notes]
        avg_note_duration_beats = mean(durations_beats) if durations_beats else 0.0

        # 2. Rhythmic sparsity metrics
        # - Notes per second (low = sparse)
        notes_per_second = len(notes) / total_duration if total_duration > 0 else 0.0
        # - Silence ratio (high = atmospheric)
        onsets = sorted([n.start_time for n in notes])
        silence_intervals = []
        for i in range(1, len(onsets)):
            silence = onsets[i] - onsets[i-1] - beat_duration * 0.25  # Ignore short gaps
            if silence > 0:
                silence_intervals.append(silence)
        silence_ratio = sum(silence_intervals) / total_duration if total_duration > 0 else 0.0
        sparsity_score = 1.0 - min(notes_per_second / 5.0, 1.0) + min(silence_ratio / (beat_duration * 2), 0.5)  # Normalize 0-1

        # 3. Tempo stability analysis
        tempo_values = [tempo[1] for tempo in tempo_map]  # us per beat
        if len(tempo_values) > 1:
            tempo_bpm_values = [60_000_000.0 / us for us in tempo_values]
            tempo_stability = 1.0 - (stdev(tempo_bpm_values) / mean(tempo_bpm_values)) if stdev(tempo_bpm_values) else 1.0
        else:
            tempo_stability = 1.0
        tempo_stability = max(0.0, min(1.0, tempo_stability))

        # 4. Atmospheric pattern detection
        # - Ratio of sustained notes (duration > 1 beat)
        sustained_notes = [n for n in notes if n.duration > beat_duration]
        sustained_ratio = len(sustained_notes) / len(notes) if notes else 0.0
        # - Low velocity sustained (ambient pads often soft)
        soft_sustained = [n for n in sustained_notes if n.velocity < 64]
        soft_ratio = len(soft_sustained) / len(sustained_notes) if sustained_notes else 0.0
        # - Low note density in higher registers (pads in low-mid)
        low_notes = [n for n in notes if n.pitch < 72]  # Below C5
        low_density = len(low_notes) / len(notes) if notes else 0.0
        atmospheric_score = (sustained_ratio * 0.4 + soft_ratio * 0.3 + low_density * 0.3)  # Weighted 0-1

        return {
            'avg_note_duration_beats': avg_note_duration_beats,
            'sparsity_score': max(0.0, min(1.0, sparsity_score)),
            'tempo_stability': tempo_stability,
            'atmospheric_pattern_score': max(0.0, min(1.0, atmospheric_score))
        }

    def _build_tempo_map(self, mid: mido.MidiFile) -> List[Tuple[int, int]]:
        """Build tempo map as (tick, microseconds_per_beat)."""
        events = []
        for track in mid.tracks:
            abs_tick = 0
            for msg in track:
                abs_tick += msg.time
                if msg.type == 'set_tempo':
                    events.append((abs_tick, msg.tempo))
        events.sort(key=lambda x: x[0])
        if not events:
            events = [(0, 500000)]  # Default 120 BPM
        # Deduplicate consecutive
        tempo_map = [events[0]]
        for tick, tempo in events[1:]:
            if tempo != tempo_map[-1][1]:
                tempo_map.append((tick, tempo))
        return tempo_map

    def _extract_all_notes(self, mid: mido.MidiFile, tempo_map: List[Tuple[int, int]],
                           ticks_per_beat: int) -> List[NoteInfo]:
        """Extract all notes with times in seconds."""
        notes = []
        for track_idx, track in enumerate(mid.tracks):
            active_notes = {}  # pitch -> (start_tick, velocity)
            abs_tick = 0
            for msg in track:
                abs_tick += msg.time
                if msg.type == 'note_on' and msg.velocity > 0:
                    active_notes[msg.note] = (abs_tick, msg.velocity)
                elif msg.type in ('note_off', 'note_on') and msg.velocity == 0:
                    if msg.note in active_notes:
                        start_tick, velocity = active_notes.pop(msg.note)
                        start_sec = self._tick_to_seconds(start_tick, tempo_map, ticks_per_beat)
                        end_sec = self._tick_to_seconds(abs_tick, tempo_map, ticks_per_beat)
                        duration = end_sec - start_sec
                        is_sustained = duration > (60.0 / self.default_tempo_bpm)  # >1 beat approx
                        notes.append(NoteInfo(start_sec, end_sec, duration, velocity, msg.note, is_sustained))
        return notes

    def _tick_to_seconds(self, tick: int, tempo_map: List[Tuple[int, int]],
                         ticks_per_beat: int) -> float:
        """Convert tick to seconds using tempo map."""
        # Find relevant tempo segment
        for i, (t_start, uspb) in enumerate(tempo_map):
            if tick < t_start:
                t_start, uspb = tempo_map[i-1]
                break
            next_t = tempo_map[i+1][0] if i+1 < len(tempo_map) else float('inf')
            if t_start <= tick < next_t:
                break
        else:
            t_start, uspb = tempo_map[-1]
        delta_ticks = tick - t_start
        return (delta_ticks / ticks_per_beat) * (uspb / 1_000_000.0)

    def _calculate_average_tempo(self, tempo_map: List[Tuple[int, int]]) -> float:
        """Calculate average BPM from tempo map."""
        if not tempo_map:
            return self.default_tempo_bpm
        bpms = [60_000_000.0 / uspb for _, uspb in tempo_map]
        return mean(bpms)


def analyze_ambient_characteristics(mid: mido.MidiFile) -> Dict[str, float]:
    """
    Standalone function for ambient characteristics analysis.

    Args:
        mid: mido.MidiFile

    Returns:
        Dict of ambient metrics
    """
    analyzer = AmbientAnalyzer()
    return analyzer.analyze_ambient_characteristics(mid)