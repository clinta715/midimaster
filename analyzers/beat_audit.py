#!/usr/bin/env python3
"""
Beat Audit Analyzer

Purpose:
- Inventory rhythm style definitions and selection logic (reads from code and filenames to tag genres).
- Analyze output MIDI files for:
  - Beat length (by local tempo), rhythm note duration stats and histogram buckets
  - Grid alignment to nearest beat with tolerance (50ms default; 25ms for DnB)
  - Cross-part alignment: non-rhythm onsets near rhythm onsets (±100ms)
  - Diversity: pattern signature from 1–2 bar IOIs quantized to 1/16, with accent flags

Outputs:
- test_outputs/beat_audit_report.md
- test_outputs/beat_audit_metrics.json

Usage:
  python analyzers/beat_audit.py --inputs path1.mid path2.mid dir_or_glob ... [--window-bars 2] [--verbose]

Notes:
- Handles combined files (parts on channels 0=melody,1=harmony,2=bass,9=rhythm) and per-part files.
- Builds a tempo map from all tracks' absolute ticks and set_tempo events.
"""

import argparse
import glob
import json
import math
import os
import re
import statistics
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple, Iterable, Set

import mido


# ------------------------
# Inventory (static, from codebase understanding)
# ------------------------

INVENTORY_SUMMARY = {
    "pop": {
        "patterns": [
            {"name": "straight_eight", "pattern": [0.5, 0.5, 0.5, 0.5]},
            {"name": "swing_eight", "pattern": [0.75, 0.25, 0.75, 0.25]},
            {"name": "syncopated", "pattern": [0.25, 0.25, 0.5, 0.5, 0.5]},
        ],
        "selection": "random choice of listed patterns per generation",
        "swing_factor": 0.55,
        "syncopation_level": 0.3,
        "emphasis_patterns": [1, 3],
        "tempo_range": [90, 140],
    },
    "rock": {
        "patterns": [
            {"name": "power_chord", "pattern": [1.0, 1.0]},
            {"name": "eight_bar", "pattern": [0.5, 0.5, 0.5, 0.5]},
            {"name": "straight_eight", "pattern": [0.5] * 8},
        ],
        "selection": "random choice of listed patterns per generation",
        "swing_factor": 0.5,
        "syncopation_level": 0.2,
        "emphasis_patterns": [1, 3],
        "tempo_range": [100, 160],
    },
    "jazz": {
        "patterns": [
            {"name": "swing", "pattern": [0.75, 0.25, 0.75, 0.25]},
            {"name": "bebop", "pattern": [0.25, 0.25, 0.25, 0.75]},
            {"name": "latin", "pattern": [0.5, 0.25, 0.25, 0.5, 0.5]},
        ],
        "selection": "random choice of listed patterns per generation",
        "swing_factor": 0.66,
        "syncopation_level": 0.6,
        "emphasis_patterns": [1, 3],
        "tempo_range": [120, 200],
    },
    "electronic": {
        "patterns": [
            {"name": "four_on_floor", "pattern": [1.0, 1.0, 1.0, 1.0]},
            {"name": "breakbeat", "pattern": [0.5, 0.25, 0.25, 0.5, 0.5]},
            {"name": "syncopated", "pattern": [0.25, 0.25, 0.5, 0.5, 0.5]},
        ],
        "selection": "random choice of listed patterns per generation",
        "swing_factor": 0.5,
        "syncopation_level": 0.4,
        "emphasis_patterns": [1, 2, 3, 4],
        "tempo_range": [120, 140],
    },
    "hip-hop": {
        "patterns": [
            {"name": "boom_bap", "pattern": [0.5, 0.5, 0.5, 0.5]},
            {"name": "trap", "pattern": [0.75, 0.25, 0.75, 0.25]},
            {"name": "syncopated", "pattern": [0.25, 0.25, 0.5, 0.5, 0.5]},
        ],
        "selection": "random choice of listed patterns per generation",
        "swing_factor": 0.6,
        "syncopation_level": 0.7,
        "emphasis_patterns": [2, 4],
        "tempo_range": [80, 110],
    },
    "classical": {
        "patterns": [
            {"name": "waltz", "pattern": [1.0, 1.0, 1.0]},
            {"name": "common_time", "pattern": [1.0, 1.0, 1.0, 1.0]},
            {"name": "cut_time", "pattern": [2.0, 2.0]},
        ],
        "selection": "random choice of listed patterns per generation",
        "swing_factor": 0.5,
        "syncopation_level": 0.1,
        "emphasis_patterns": [1],
        "tempo_range": [60, 160],
    },
    "dnb": {
        "patterns": [
            {"name": "amen_break", "pattern": [0.5, 0.25, 0.25, 0.5, 0.25, 0.25, 0.25, 0.25]},
            {"name": "double_kick", "pattern": [0.25] * 8},
            {"name": "syncopated_snare", "pattern": [0.5, 0.5, 0.25, 0.25, 0.5]},
            {"name": "jungle_pattern", "pattern": [0.25] * 8},
        ],
        "selection": "random choice of listed patterns per generation",
        "swing_factor": 0.5,
        "syncopation_level": 0.8,
        "emphasis_patterns": [1, 2, 3, 4],
        "tempo_range": [160, 180],
    },
}


# ------------------------
# Data structures
# ------------------------

@dataclass
class Note:
    pitch: int
    velocity: int
    start_tick: int
    end_tick: int
    start_sec: float
    end_sec: float
    channel: int
    track_index: int


@dataclass
class FileMetrics:
    path: str
    genre: str
    tempo_bpm_avg: float
    beat_length_sec_avg: float
    rhythm_track_indices: List[int]
    rhythm_channels: List[int]
    rhythm_note_count: int
    note_duration_stats: Dict[str, float]
    note_duration_hist: Dict[str, int]
    grid_alignment_tolerance_ms: int
    grid_alignment_percent: float
    cross_alignment_percent: float
    diversity_signature: str
    diversity_unique_signatures_genre: int
    variation_index_stddev_s: float
    pass_fail: Dict[str, bool]


# ------------------------
# Helpers: genre detection, IO
# ------------------------

GENRE_HINTS = [
    ("dnb", "dnb"),
    ("drum-and-bass", "dnb"),
    ("pop", "pop"),
    ("rock", "rock"),
    ("jazz", "jazz"),
    ("electronic", "electronic"),
    ("ambient", "ambient"),
    ("atmospheric", "ambient"),
    ("hip-hop", "hip-hop"),
    ("hiphop", "hip-hop"),
    ("classical", "classical"),
]


def detect_genre_from_path(path: str) -> str:
    low = path.replace("\\", "/").lower()
    for key, genre in GENRE_HINTS:
        if key in low:
            return genre
    # Fallback from inventory names in filename prefix
    for genre in INVENTORY_SUMMARY.keys():
        if f"{genre}_" in low or f"_{genre}_" in low:
            return genre
    # Additional ambient check
    if "ambient" in low or "atmos" in low:
        return "ambient"
    return "unknown"


def ensure_out_dirs():
    os.makedirs("test_outputs", exist_ok=True)


def glob_inputs(inputs: List[str]) -> List[str]:
    files: List[str] = []
    for inp in inputs:
        # Accept directory: collect *.mid
        if os.path.isdir(inp):
            files.extend(glob.glob(os.path.join(inp, "*.mid")))
        else:
            matched = glob.glob(inp)
            if matched:
                files.extend(matched)
            else:
                # If looks like a file, include as-is; we will handle existence later
                files.append(inp)
    # Deduplicate, keep stable order
    seen = set()
    unique_files = []
    for f in files:
        if f not in seen:
            seen.add(f)
            unique_files.append(f)
    return unique_files


# ------------------------
# Tempo map and tick<->sec conversion
# ------------------------

def build_event_absolute_ticks(mid: mido.MidiFile) -> List[Tuple[int, mido.Message]]:
    """
    Create a combined list of (absolute_tick, message) across all tracks.
    Absolute tick per track is computed by summing delta ticks in that track;
    then all events are merged and sorted by absolute_tick for a global timeline.
    """
    events: List[Tuple[int, mido.Message]] = []
    for track in mid.tracks:
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            events.append((abs_tick, msg))
    events.sort(key=lambda x: x[0])
    return events


def build_tempo_map(mid: mido.MidiFile) -> List[Tuple[int, int]]:
    """
    Returns a list of (tick, microseconds_per_beat) sorted by tick.
    Ensures an initial tempo at tick 0.
    """
    events = build_event_absolute_ticks(mid)
    tempos: List[Tuple[int, int]] = []
    for tick, msg in events:
        if getattr(msg, "type", None) == "set_tempo":
            tempos.append((tick, int(getattr(msg, "tempo", 500000))))
    if not tempos or tempos[0][0] != 0:
        # default 500000 us/qn at tick 0
        tempos.insert(0, (0, 500000))
    # Deduplicate consecutive identical tempos
    cleaned: List[Tuple[int, int]] = []
    last: Optional[int] = None
    for t, us in tempos:
        if last is None or us != last:
            cleaned.append((t, us))
            last = us
    return cleaned


def precompute_seconds_at_tempo_changes(tempo_map: List[Tuple[int, int]], ticks_per_beat: int, end_tick: int) -> List[Tuple[int, int, float]]:
    """
    Compute cumulative seconds at each tempo change boundary up to end_tick.
    Returns list of (tick, microseconds_per_beat, seconds_at_tick).
    """
    result: List[Tuple[int, int, float]] = []
    total_seconds = 0.0
    for i, (tick, uspb) in enumerate(tempo_map):
        result.append((tick, uspb, total_seconds))
        next_tick = tempo_map[i+1][0] if i+1 < len(tempo_map) else end_tick
        tick_span = max(0, next_tick - tick)
        sec_span = (tick_span / ticks_per_beat) * (uspb / 1_000_000.0)
        total_seconds += sec_span
    return result


def tick_to_seconds(tick: int, tempo_accum: List[Tuple[int, int, float]], ticks_per_beat: int) -> float:
    """
    Convert an absolute tick to seconds using precomputed tempo accumulators.
    """
    # Binary search last tempo boundary <= tick
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


def tick_to_bpm(tick: int, tempo_map: List[Tuple[int, int]]) -> float:
    # Find current uspb at tick in tempo_map
    lo, hi = 0, len(tempo_map) - 1
    idx = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if tempo_map[mid][0] <= tick:
            idx = mid
            lo = mid + 1
        else:
            hi = mid - 1
    uspb = tempo_map[idx][1]
    return 60_000_000.0 / uspb


# ------------------------
# MIDI parsing
# ------------------------

def extract_track_name(track: mido.MidiTrack) -> Optional[str]:
    for msg in track:
        if getattr(msg, "type", None) == "track_name":
            return str(getattr(msg, "name", "")).lower()
    return None


def parse_midi_notes(path: str) -> Tuple[mido.MidiFile, List[Note]]:
    mid = mido.MidiFile(path)
    # Compute end_tick
    track_end_ticks: List[int] = []
    abs_notes: List[Note] = []

    # Build absolute ticks per track, collect tempo map and seconds accum
    # We need maximum absolute tick for end bound
    all_events: List[List[Tuple[int, mido.Message]]] = []
    max_tick = 0
    for ti, track in enumerate(mid.tracks):
        abs_tick = 0
        events = []
        for msg in track:
            abs_tick += msg.time
            events.append((abs_tick, msg))
        all_events.append(events)
        if events:
            max_tick = max(max_tick, events[-1][0])

    tempo_map = build_tempo_map(mid)
    tempo_accum = precompute_seconds_at_tempo_changes(tempo_map, mid.ticks_per_beat, max_tick)

    # Extract notes per track
    for ti, events in enumerate(all_events):
        active: Dict[Tuple[int, int], Tuple[int, int]] = {}  # (pitch, channel) -> (start_tick, velocity)
        for abs_tick, msg in events:
            if getattr(msg, "type", None) == "note_on" and int(getattr(msg, "velocity", 0)) > 0:
                note_obj = getattr(msg, "note", None)
                if not isinstance(note_obj, int):
                    continue
                channel = int(getattr(msg, "channel", 0))
                velocity = int(getattr(msg, "velocity", 0))
                key: Tuple[int, int] = (note_obj, channel)
                active[key] = (abs_tick, velocity)
            elif getattr(msg, "type", None) in ("note_off", "note_on") and int(getattr(msg, "velocity", 0)) == 0:
                note_obj = getattr(msg, "note", None)
                if not isinstance(note_obj, int):
                    continue
                channel = int(getattr(msg, "channel", 0))
                key: Tuple[int, int] = (note_obj, channel)
                if key in active:
                    start_tick, vel = active[key]
                    if abs_tick > start_tick:
                        start_sec = tick_to_seconds(start_tick, tempo_accum, mid.ticks_per_beat)
                        end_sec = tick_to_seconds(abs_tick, tempo_accum, mid.ticks_per_beat)
                        abs_notes.append(
                            Note(
                                pitch=key[0],
                                velocity=vel,
                                start_tick=start_tick,
                                end_tick=abs_tick,
                                start_sec=start_sec,
                                end_sec=end_sec,
                                channel=key[1],
                                track_index=ti,
                            )
                        )
                    del active[key]
    return mid, abs_notes


# ------------------------
# Rhythm track detection and parts grouping
# ------------------------

def select_rhythm_tracks(mid: mido.MidiFile, notes: List[Note]) -> Tuple[List[int], List[int]]:
    """
    Returns (track_indices, channels) considered as rhythm.
    Heuristics:
      - Any track with channel 9 notes (GM percussion)
      - Track name contains 'rhythm' or 'drum'
      - Highest density of percussion-range notes (35-60) if none above

    IMPORTANT: In single-track, multi-channel MIDI (common in this project), do NOT
    treat all channels on that track as rhythm. Restrict rhythm channels to 9
    unless a separate named track (rhythm/drum) exists with non-9 channels.
    """
    track_names = [extract_track_name(t) or "" for t in mid.tracks]

    # Map channels per track
    channels_by_track: Dict[int, Set[int]] = defaultdict(set)
    for n in notes:
        channels_by_track[n.track_index].add(n.channel)

    rhythm_track_indices: Set[int] = set()
    rhythm_channels: Set[int] = set()

    # Channel 9 (zero-based) is percussion; mark tracks containing it
    any_ch9 = False
    for n in notes:
        if n.channel == 9:
            any_ch9 = True
            rhythm_track_indices.add(n.track_index)
    if any_ch9:
        rhythm_channels.add(9)

    # Tracks explicitly named rhythm/drum
    named_rhythm_tracks: Set[int] = set()
    for i, name in enumerate(track_names):
        if "rhythm" in name or "drum" in name:
            named_rhythm_tracks.add(i)
            rhythm_track_indices.add(i)

    # If no rhythm tracks yet, fallback by percussion-range density
    if not rhythm_track_indices:
        counts: Dict[int, int] = Counter()
        for n in notes:
            if 35 <= n.pitch <= 60:
                counts[n.track_index] += 1
        if counts:
            best_track, _ = counts.most_common(1)[0]
            rhythm_track_indices.add(best_track)

    # Channel selection policy:
    # - If we have any channel 9 present anywhere, limit rhythm_channels to {9} by default.
    # - Additionally include channels from explicitly named rhythm/drum tracks (for non-GM kits).
    for ti in named_rhythm_tracks:
        for ch in channels_by_track.get(ti, set()):
            if ch != 9:
                rhythm_channels.add(ch)

    return sorted(list(rhythm_track_indices)), sorted(list(rhythm_channels))


def split_parts(notes: List[Note], rhythm_tracks: List[int], rhythm_channels: List[int]) -> Tuple[List[Note], List[Note]]:
    """
    Return (rhythm_notes, non_rhythm_notes)
    """
    rhythm_set: List[Note] = []
    non_set: List[Note] = []
    for n in notes:
        if (n.track_index in rhythm_tracks) or (n.channel in rhythm_channels) or (n.channel == 9):
            rhythm_set.append(n)
        else:
            non_set.append(n)
    return rhythm_set, non_set


# ------------------------
# Metrics
# ------------------------

def durations_stats(durations: List[float]) -> Dict[str, float]:
    if not durations:
        return {"avg": 0.0, "median": 0.0, "variance": 0.0}
    avg = statistics.mean(durations)
    med = statistics.median(durations)
    var = statistics.pvariance(durations) if len(durations) > 1 else 0.0
    return {"avg": avg, "median": med, "variance": var}


def durations_histogram(durations: List[float]) -> Dict[str, int]:
    buckets = {
        "<=0.125": 0,
        "0.125-0.25": 0,
        "0.25-0.5": 0,
        "0.5-1.0": 0,
        ">1.0": 0,
    }
    for d in durations:
        if d <= 0.125:
            buckets["<=0.125"] += 1
        elif d <= 0.25:
            buckets["0.125-0.25"] += 1
        elif d <= 0.5:
            buckets["0.25-0.5"] += 1
        elif d <= 1.0:
            buckets["0.5-1.0"] += 1
        else:
            buckets[">1.0"] += 1
    return buckets


def compute_alignment(rhythm_onsets: List[float], onset_bpms: List[float], genre: str) -> Tuple[int, float]:
    """
    Grid alignment: nearest rhythmic grid position at local onset time.

    For most genres we align to the beat grid (b = 60/BPM).
    For high-subdivision genres (e.g., DnB), align to sixteenth grid (b/4) to reflect breakbeat density.

    Returns (tolerance_ms, aligned_fraction)
    """
    if not rhythm_onsets or not onset_bpms:
        return (25 if genre == "dnb" else 50), 0.0

    tol_ms = 25 if genre == "dnb" else 50
    tol = tol_ms / 1000.0
    aligned = 0
    total = 0

    # Use origin at 0 seconds
    for t, bpm in zip(rhythm_onsets, onset_bpms):
        b = 60.0 / max(1e-6, bpm)
        # Choose grid resolution: DnB -> sixteenth; else -> beat
        div = 4 if genre == "dnb" else 1
        step = b / max(1, div)
        # nearest grid position to t at chosen resolution
        grid = round(t / step) * step
        if abs(t - grid) <= tol:
            aligned += 1
        total += 1

    frac = (aligned / total) if total else 0.0
    return tol_ms, frac


def compute_cross_alignment(non_rhythm_onsets: List[float], rhythm_onsets: List[float]) -> float:
    """
    For each non-rhythm onset s, find nearest rhythm onset r aligned if |r - s| <= 100ms.
    Returns aligned fraction.
    """
    if not non_rhythm_onsets or not rhythm_onsets:
        return 0.0
    tol = 0.1
    r_sorted = sorted(rhythm_onsets)
    aligned = 0
    for s in non_rhythm_onsets:
        # binary search nearest
        lo, hi = 0, len(r_sorted) - 1
        pos = 0
        while lo <= hi:
            mid = (lo + hi) // 2
            if r_sorted[mid] <= s:
                pos = mid
                lo = mid + 1
            else:
                hi = mid - 1
        candidates = [r_sorted[pos]]
        if pos + 1 < len(r_sorted):
            candidates.append(r_sorted[pos + 1])
        if any(abs(s - c) <= tol for c in candidates):
            aligned += 1
    return aligned / len(non_rhythm_onsets)


def make_pattern_signature(rhythm_notes: List[Note], window_bars: int, bpm_ref: float) -> Tuple[str, float]:
    """
    Build signature using first window (1-2 bars) IOIs quantized to 1/16.
    Returns (signature_string, stddev_ioi_s).
    """
    if bpm_ref <= 1e-6 or not rhythm_notes:
        return ("", 0.0)
    b = 60.0 / bpm_ref
    sixteenth = b / 4.0
    # window length: bars * 4 beats (assuming 4/4)
    window_len = window_bars * 4 * b
    onsets = sorted([n.start_sec for n in rhythm_notes])
    if not onsets:
        return ("", 0.0)
    start0 = onsets[0]
    window_onsets = [t for t in onsets if (t - start0) <= window_len]
    if len(window_onsets) < 2:
        return ("", 0.0)
    iois = [window_onsets[i] - window_onsets[i-1] for i in range(1, len(window_onsets))]
    stddev = statistics.pstdev(iois) if len(iois) > 1 else 0.0
    # quantize
    q = [int(round(i / max(1e-9, sixteenth))) for i in iois]
    # accent flags: velocity > median in window
    velocities = []
    for n in rhythm_notes:
        if n.start_sec in window_onsets or (abs(n.start_sec - min(window_onsets, key=lambda x: abs(x - n.start_sec))) < 1e-6):
            velocities.append(n.velocity)
    v_med = statistics.median(velocities) if velocities else 0
    accents = []
    # For matching velocities to each onset, map by onset time
    onset_to_vels: Dict[float, List[int]] = defaultdict(list)
    for n in rhythm_notes:
        if n.start_sec in window_onsets:
            onset_to_vels[n.start_sec].append(n.velocity)
    # Reduce to per-onset max velocity
    onset_vel = [max(onset_to_vels[t]) if onset_to_vels[t] else 0 for t in window_onsets]
    accents = [1 if v > v_med else 0 for v in onset_vel[1:]]  # skip reference first
    signature = f"q16:{','.join(map(str, q))}|acc:{''.join(map(str, accents))}"
    return signature, stddev


# ------------------------
# Pass/Fail thresholds
# ------------------------

def evaluate_pass_fail(genre: str, tempo_bpm: float, duration_stats: Dict[str, float], grid_align: float,
                       cross_align: float, unique_signatures_genre: int, variation_stddev: float) -> Dict[str, bool]:
    """
    Thresholds as specified:
      - Beat length adequacy (Avg duration) by BPM band
      - Grid alignment: >= 90% (25ms for DnB, 50ms otherwise)
      - Cross-part alignment: >= 85%
      - Diversity: per-genre unique signatures >= 3; variation index stddev IOI > 0.1s for non-minimal styles
    """
    # Beat length adequacy bands:
    dur_avg = duration_stats.get("avg", 0.0)
    dur_var = duration_stats.get("variance", 0.0)

    def band_ok(bpm: float, avg: float, var: float) -> bool:
        if 80 <= bpm < 100:
            return (0.30 <= avg <= 0.60) and (var > 0.10)
        elif 100 <= bpm < 140:
            return (0.20 <= avg <= 0.40) and (var > 0.05)
        elif 140 <= bpm <= 180:
            return (0.15 <= avg <= 0.30) and (var > 0.03)
        else:
            # out-of-band: just require var > 0.02
            return var > 0.02

    beat_len_pass = band_ok(tempo_bpm, dur_avg, dur_var)
    grid_pass = grid_align >= 0.90
    cross_pass = cross_align >= 0.85
    diversity_pass = (unique_signatures_genre >= 3) and (variation_stddev > 0.1)

    # For DnB, apply stricter alignment tolerance already handled when calculating
    return {
        "beat_length": beat_len_pass,
        "grid_alignment": grid_pass,
        "cross_alignment": cross_pass,
        "diversity": diversity_pass,
    }


# ------------------------
# Main analysis per file and aggregation
# ------------------------

def analyze_file(path: str, window_bars: int, diversity_signature_sets: Dict[str, Set[str]], verbose: bool = False) -> Optional[FileMetrics]:
    if not os.path.exists(path):
        if verbose:
            print(f"Skip missing: {path}")
        return None

    try:
        mid, notes = parse_midi_notes(path)
    except Exception as e:
        print(f"Error parsing {path}: {e}")
        return None

    genre = detect_genre_from_path(path)
    # Compute average tempo across rhythm onsets; if none, across all notes
    tempo_map = build_tempo_map(mid)

    rhythm_tracks, rhythm_channels = select_rhythm_tracks(mid, notes)
    rhythm_notes, non_rhythm_notes = split_parts(notes, rhythm_tracks, rhythm_channels)

    # If no rhythm detected, consider entire file as rhythm (fallback)
    if not rhythm_notes:
        rhythm_notes = notes[:]

    # Compute per-onset BPMs for rhythm notes
    onset_bpms = [tick_to_bpm(n.start_tick, tempo_map) for n in rhythm_notes]
    tempo_bpm_avg = statistics.mean(onset_bpms) if onset_bpms else (statistics.mean([tick_to_bpm(n.start_tick, tempo_map) for n in notes]) if notes else 120.0)
    beat_length_sec_avg = 60.0 / max(1e-6, tempo_bpm_avg)

    # Rhythm note durations
    rhythm_durs = [max(0.0, n.end_sec - n.start_sec) for n in rhythm_notes]
    dur_stats = durations_stats(rhythm_durs)
    dur_hist = durations_histogram(rhythm_durs)

    # Grid alignment
    rhythm_onsets = [n.start_sec for n in rhythm_notes]
    tol_ms, grid_frac = compute_alignment(rhythm_onsets, onset_bpms if onset_bpms else [tempo_bpm_avg] * len(rhythm_onsets), genre)

    # Cross-part alignment
    non_onsets = [n.start_sec for n in non_rhythm_notes]
    cross_frac = compute_cross_alignment(non_onsets, rhythm_onsets)

    # Diversity signature
    signature, var_std = make_pattern_signature(rhythm_notes, window_bars=window_bars, bpm_ref=tempo_bpm_avg)
    # Track per-genre unique signatures
    if genre not in diversity_signature_sets:
        diversity_signature_sets[genre] = set()
    if signature:
        diversity_signature_sets[genre].add(signature)
    unique_sig_count = len(diversity_signature_sets[genre])

    # Pass/fail
    pf = evaluate_pass_fail(genre, tempo_bpm_avg, dur_stats, grid_frac, cross_frac, unique_sig_count, var_std)

    return FileMetrics(
        path=path,
        genre=genre,
        tempo_bpm_avg=tempo_bpm_avg,
        beat_length_sec_avg=beat_length_sec_avg,
        rhythm_track_indices=rhythm_tracks,
        rhythm_channels=rhythm_channels,
        rhythm_note_count=len(rhythm_notes),
        note_duration_stats=dur_stats,
        note_duration_hist=dur_hist,
        grid_alignment_tolerance_ms=tol_ms,
        grid_alignment_percent=grid_frac,
        cross_alignment_percent=cross_frac,
        diversity_signature=signature,
        diversity_unique_signatures_genre=unique_sig_count,
        variation_index_stddev_s=var_std,
        pass_fail=pf,
    )


def aggregate_by_genre(file_metrics: List[FileMetrics]) -> Dict[str, Any]:
    by_genre: Dict[str, List[FileMetrics]] = defaultdict(list)
    for fm in file_metrics:
        by_genre[fm.genre].append(fm)
    summary: Dict[str, Any] = {}
    for genre, fms in by_genre.items():
        if not fms:
            continue
        tempos = [fm.tempo_bpm_avg for fm in fms]
        beat_lengths = [fm.beat_length_sec_avg for fm in fms]
        grid = [fm.grid_alignment_percent for fm in fms]
        cross = [fm.cross_alignment_percent for fm in fms]
        var_std = [fm.variation_index_stddev_s for fm in fms]
        # diversity unique count captured per-file as the running count; recompute robustly:
        signatures: Set[str] = set()
        for fm in fms:
            if fm.diversity_signature:
                signatures.add(fm.diversity_signature)

        # Pass/fail counts
        pf_counts = Counter()
        for fm in fms:
            for k, v in fm.pass_fail.items():
                pf_counts[f"{k}_pass" if v else f"{k}_fail"] += 1

        summary[genre] = {
            "files": len(fms),
            "tempo_bpm_avg": statistics.mean(tempos),
            "beat_length_sec_avg": statistics.mean(beat_lengths),
            "grid_alignment_percent_avg": statistics.mean(grid),
            "cross_alignment_percent_avg": statistics.mean(cross),
            "variation_index_stddev_s_avg": statistics.mean(var_std),
            "unique_signatures": len(signatures),
            "pass_fail_counts": dict(pf_counts),
        }
    return summary


# ------------------------
# Reporting
# ------------------------

def write_metrics_json(path: str, files: List[FileMetrics], genres_summary: Dict[str, Any]) -> None:
    payload = {
        "inventory": {
            "per_genre": {
                g: {
                    "style_count": len(INVENTORY_SUMMARY[g]["patterns"]),
                    "style_names": [p["name"] for p in INVENTORY_SUMMARY[g]["patterns"]],
                    "selection": INVENTORY_SUMMARY[g]["selection"],
                    "swing_factor": INVENTORY_SUMMARY[g]["swing_factor"],
                    "syncopation_level": INVENTORY_SUMMARY[g]["syncopation_level"],
                    "emphasis_patterns": INVENTORY_SUMMARY[g]["emphasis_patterns"],
                    "tempo_range": INVENTORY_SUMMARY[g]["tempo_range"],
                } for g in INVENTORY_SUMMARY
            }
        },
        "files": {fm.path: asdict(fm) for fm in files},
        "genres": genres_summary,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_report_md(path: str, files: List[FileMetrics], genres_summary: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# Beat Audit Report")
    lines.append("")
    lines.append("## Inventory Summary")
    lines.append("")
    for g, meta in INVENTORY_SUMMARY.items():
        lines.append(f"- {g}: {len(meta['patterns'])} styles " +
                     f"({', '.join([p['name'] for p in meta['patterns']])}); " +
                     f"selection={meta['selection']}; swing={meta['swing_factor']}; " +
                     f"syncopation={meta['syncopation_level']}; emphasis={meta['emphasis_patterns']}; " +
                     f"tempo_range={meta['tempo_range']}")
    lines.append("")

    # Group by genre and include pass/fail summaries
    by_genre: Dict[str, List[FileMetrics]] = defaultdict(list)
    for fm in files:
        by_genre[fm.genre].append(fm)

    for genre, group in sorted(by_genre.items()):
        lines.append(f"## {genre.upper()}")
        gs = genres_summary.get(genre, {})
        lines.append(f"- Files analyzed: {gs.get('files', 0)}")
        lines.append(f"- Avg tempo: {gs.get('tempo_bpm_avg', 0):.1f} BPM; Avg beat length: {gs.get('beat_length_sec_avg', 0):.3f}s")
        lines.append(f"- Avg grid alignment: {gs.get('grid_alignment_percent_avg', 0)*100:.1f}%")
        lines.append(f"- Avg cross-part alignment: {gs.get('cross_alignment_percent_avg', 0)*100:.1f}%")
        lines.append(f"- Unique pattern signatures: {gs.get('unique_signatures', 0)}")
        lines.append(f"- Variation index (IOI stddev): {gs.get('variation_index_stddev_s_avg', 0):.3f}s")
        pf = gs.get("pass_fail_counts", {})
        lines.append(f"- Pass/Fail: {pf}")
        lines.append("")
        lines.append("Representative files:")
        for fm in group[:5]:
            lines.append(f"  - {os.path.basename(fm.path)}: "
                         f"tempo={fm.tempo_bpm_avg:.1f} BPM, "
                         f"grid={fm.grid_alignment_percent*100:.1f}%, "
                         f"cross={fm.cross_alignment_percent*100:.1f}%, "
                         f"dur_avg={fm.note_duration_stats.get('avg',0):.3f}s, "
                         f"var={fm.note_duration_stats.get('variance',0):.3f}, "
                         f"sig={fm.diversity_signature or 'n/a'}")
        lines.append("")

        # Recommendations section
        lines.append("### Recommendations")
        if gs.get("grid_alignment_percent_avg", 1.0) < 0.90:
            lines.append("- Improve quantization or reduce humanization jitter; ensure beat grid adherence.")
        if genre == "dnb" and gs.get("grid_alignment_percent_avg", 1.0) < 0.90:
            lines.append("- For DnB, enforce tighter alignment (±25ms).")
        if gs.get("unique_signatures", 0) < 3:
            lines.append("- Add more rhythm patterns or vary selection weights for more diversity.")
        if gs.get("variation_index_stddev_s_avg", 0) <= 0.1:
            lines.append("- Increase rhythmic variation (syncopation/density) for non-minimal styles.")
        lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ------------------------
# CLI
# ------------------------

def main():
    parser = argparse.ArgumentParser(description="Beat audit for generated MIDI files.")
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="Input MIDI paths, directories, or glob patterns.")
    parser.add_argument("--window-bars", type=int, default=2,
                        help="Bars to use for diversity signature window (default 2).")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    ensure_out_dirs()

    paths = glob_inputs(args.inputs)
    if args.verbose:
        print(f"Discovered {len(paths)} input(s).")

    diversity_signature_sets: Dict[str, Set[str]] = {}
    file_metrics: List[FileMetrics] = []
    for p in paths:
        fm = analyze_file(p, window_bars=args.window_bars, diversity_signature_sets=diversity_signature_sets, verbose=args.verbose)
        if fm:
            file_metrics.append(fm)

    genres_summary = aggregate_by_genre(file_metrics)

    # Write outputs
    metrics_path = os.path.join("test_outputs", "beat_audit_metrics.json")
    report_path = os.path.join("test_outputs", "beat_audit_report.md")
    write_metrics_json(metrics_path, file_metrics, genres_summary)
    write_report_md(report_path, file_metrics, genres_summary)

    # Minimal console summary
    print(f"Wrote: {metrics_path}")
    print(f"Wrote: {report_path}")


if __name__ == "__main__":
    main()