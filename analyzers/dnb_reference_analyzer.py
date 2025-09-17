#!/usr/bin/env python3
"""
DnB Reference MIDI Analyzer

Goal:
- Analyze real/reference Drum & Bass (DnB) MIDI files to extract actionable, genre-specific rhythmic features:
  - Kick/snare placement histograms on a 16th (and optional 32nd) grid
  - Snare backbeat regularity (2 and 4), snare ghost rate and positions
  - Hi-hat offbeat density, open-hat placement
  - Microtiming (ms) to nearest 16th grid per instrument class
  - Velocity statistics per class and per grid position
  - Repetitiveness and fill detection across bars
- Aggregate across files and emit JSON + Markdown recommendations to calibrate generators.

Usage:
  python analyzers/dnb_reference_analyzer.py --inputs path1.mid path2.mid dir_or_glob ... \
      [--grid 16|32] [--bars 8] [--out-prefix test_outputs/dnb_reference] [--verbose]

Outputs:
  - {out_prefix}_metrics.json
  - {out_prefix}_report.md
"""

import argparse
import glob
import json
import math
import os
import statistics
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple, Iterable, Set

import mido

# ---------------
# Drum class mapping (GM defaults; configurable later if needed)
# ---------------
# GM channel 10 (zero-based channel 9) drum note numbers
DRUM_CLASSES: Dict[str, Set[int]] = {
    "kick": {35, 36},  # Acoustic/Electric Bass Drum
    "snare": {38, 40},  # Acoustic/Electric Snare
    "chh": {42, 44},  # Closed HH, Pedal HH
    "ohh": {46},  # Open HH
    "ride": {51, 59, 53},  # Ride1, RideCymbal2, RideBell (approx)
    "crash": {49, 57, 55, 52},  # Crash1, Crash2, Splash, Chinese
    "tom": {41, 43, 45, 47, 48, 50},  # Toms
    "perc": {37, 39},  # Side stick, Clap
    # Others like shakers/cabasa/cowbell could be added if present
}

CLASS_ORDER = ["kick", "snare", "chh", "ohh", "ride", "crash", "tom", "perc"]

# ---------------
# Data structures
# ---------------

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
    ticks_per_beat: int
    time_sig: Tuple[int, int]  # numerator, denominator
    tempo_bpm_avg: float
    duration_sec: float
    bars: int
    counts_by_class: Dict[str, int]
    kick_pos16_hist: List[int]
    snare_pos16_hist: List[int]
    chh_pos16_hist: List[int]
    ohh_pos16_hist: List[int]
    vel_stats_by_class: Dict[str, Dict[str, float]]
    vel_pos16_avg: Dict[str, List[float]]
    microtiming_ms_by_class: Dict[str, Dict[str, float]]  # mean, std
    snare_backbeat_regular: float
    snare_ghost_rate: float
    snare_ghost_pos16_hist: List[int]
    hat_offbeat_rate: float
    repetitiveness: Dict[str, float]  # identical_bar_fraction per class (kick/snare/hats)
    fill_bars: List[int]  # bar indices flagged as fills


# ---------------
# Helpers: IO
# ---------------

def ensure_out_dir(prefix: str) -> None:
    d = os.path.dirname(prefix)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def glob_inputs(inputs: List[str]) -> List[str]:
    files: List[str] = []
    for inp in inputs:
        if os.path.isdir(inp):
            files.extend(glob.glob(os.path.join(inp, "*.mid")))
        else:
            matched = glob.glob(inp)
            if matched:
                files.extend(matched)
            else:
                files.append(inp)
    # Deduplicate keep order
    seen = set()
    out = []
    for f in files:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


# ---------------
# Tempo/timing utilities (self-contained, adapted from analyzers/beat_audit.py)
# ---------------

def build_event_absolute_ticks(mid: mido.MidiFile) -> List[Tuple[int, mido.Message]]:
    events: List[Tuple[int, mido.Message]] = []
    for track in mid.tracks:
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            events.append((abs_tick, msg))
    events.sort(key=lambda x: x[0])
    return events


def build_tempo_map(mid: mido.MidiFile) -> List[Tuple[int, int]]:
    events = build_event_absolute_ticks(mid)
    tempos: List[Tuple[int, int]] = []
    for tick, msg in events:
        if getattr(msg, "type", None) == "set_tempo":
            tempos.append((tick, int(getattr(msg, "tempo", 500000))))
    if not tempos or tempos[0][0] != 0:
        tempos.insert(0, (0, 500000))
    cleaned: List[Tuple[int, int]] = []
    last: Optional[int] = None
    for t, us in tempos:
        if last is None or us != last:
            cleaned.append((t, us))
            last = us
    return cleaned


def precompute_seconds_at_tempo_changes(tempo_map: List[Tuple[int, int]], ticks_per_beat: int, end_tick: int) -> List[Tuple[int, int, float]]:
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


def get_uspb_at_tick(tick: int, tempo_map: List[Tuple[int, int]]) -> int:
    lo, hi = 0, len(tempo_map) - 1
    idx = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if tempo_map[mid][0] <= tick:
            idx = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return tempo_map[idx][1]


def extract_time_signature(mid: mido.MidiFile) -> Tuple[int, int]:
    # Default 4/4
    num, den = 4, 4
    for track in mid.tracks:
        for msg in track:
            if getattr(msg, "type", None) == "time_signature":
                num = int(getattr(msg, "numerator", num))
                den = int(getattr(msg, "denominator", den))
                return num, den
    return num, den


def parse_midi_notes(path: str) -> Tuple[mido.MidiFile, List[Note], int]:
    """Return (MidiFile, notes[], max_tick) with absolute ticks and seconds computed."""
    mid = mido.MidiFile(path)
    all_events: List[List[Tuple[int, mido.Message]]] = []
    max_tick = 0
    for ti, track in enumerate(mid.tracks):
        abs_tick = 0
        events: List[Tuple[int, mido.Message]] = []
        for msg in track:
            abs_tick += msg.time
            events.append((abs_tick, msg))
        all_events.append(events)
        if events:
            max_tick = max(max_tick, events[-1][0])

    tempo_map = build_tempo_map(mid)
    tempo_accum = precompute_seconds_at_tempo_changes(tempo_map, mid.ticks_per_beat, max_tick)

    abs_notes: List[Note] = []
    for ti, events in enumerate(all_events):
        active: Dict[Tuple[int, int], Tuple[int, int]] = {}
        for abs_tick, msg in events:
            mtype = getattr(msg, "type", None)
            if mtype == "note_on" and int(getattr(msg, "velocity", 0)) > 0:
                note_obj = getattr(msg, "note", None)
                if not isinstance(note_obj, int):
                    continue
                channel = int(getattr(msg, "channel", 0))
                velocity = int(getattr(msg, "velocity", 0))
                active[(note_obj, channel)] = (abs_tick, velocity)
            elif (mtype == "note_off") or (mtype == "note_on" and int(getattr(msg, "velocity", 0)) == 0):
                note_obj = getattr(msg, "note", None)
                if not isinstance(note_obj, int):
                    continue
                channel = int(getattr(msg, "channel", 0))
                key = (note_obj, channel)
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
    return mid, abs_notes, max_tick


def extract_track_name(track: mido.MidiTrack) -> Optional[str]:
    for msg in track:
        if getattr(msg, "type", None) == "track_name":
            return str(getattr(msg, "name", "")).lower()
    return None


def select_rhythm_notes(mid: mido.MidiFile, notes: List[Note]) -> List[Note]:
    """
    Heuristics for rhythm/drum notes:
      - Any channel 9 notes (GM percussion) are included
      - Any track with name containing 'drum' or 'rhythm' (include all channels on that track)
      - Fallback: notes with pitches in typical GM drum ranges (35-60)
    """
    rhythm: List[Note] = []
    named_rhythm_tracks: Set[int] = set()
    for i, track in enumerate(mid.tracks):
        name = extract_track_name(track) or ""
        if "drum" in name or "rhythm" in name or "kit" in name:
            named_rhythm_tracks.add(i)

    any_ch9 = any(n.channel == 9 for n in notes)
    for n in notes:
        if n.channel == 9:
            rhythm.append(n)
        elif n.track_index in named_rhythm_tracks:
            rhythm.append(n)
        elif (not any_ch9) and (35 <= n.pitch <= 60):
            rhythm.append(n)
    # Deduplicate by start_tick/pitch/channel
    rhythm.sort(key=lambda x: (x.start_tick, x.pitch, x.channel))
    return rhythm


def drum_class_for_pitch(pitch: int) -> Optional[str]:
    for cls, ps in DRUM_CLASSES.items():
        if pitch in ps:
            return cls
    # Fallback rough mapping for out-of-GM cases
    if 35 <= pitch <= 36:
        return "kick"
    if 37 <= pitch <= 41:
        return "snare"
    if 42 <= pitch <= 46:
        return "chh" if pitch in (42, 44) else "ohh"
    if 47 <= pitch <= 59:
        return "tom"
    return None


# ---------------
# Core analysis
# ---------------

def analyze_file(path: str, grid: int, max_bars: int, verbose: bool = False) -> Optional[FileMetrics]:
    if not os.path.exists(path):
        if verbose:
            print(f"Skip missing: {path}")
        return None

    try:
        mid, notes, max_tick = parse_midi_notes(path)
    except Exception as e:
        print(f"Error parsing {path}: {e}")
        return None

    if not notes:
        if verbose:
            print(f"No notes in {path}")
        return None

    tpb = mid.ticks_per_beat
    num, den = extract_time_signature(mid)
    # bar length in ticks: numerator beats per bar, each beat is quarter*(4/den)
    bar_ticks = int(round(tpb * num * (4.0 / den)))
    sixteenth_tick = tpb // 4
    grid_tick = sixteenth_tick if grid == 16 else max(1, tpb // 8)  # 32nd for grid=32

    tempo_map = build_tempo_map(mid)
    tempo_accum = precompute_seconds_at_tempo_changes(tempo_map, tpb, max_tick)

    # Duration
    duration_sec = tick_to_seconds(max_tick, tempo_accum, tpb)
    # Average bpm over rhythm onsets later; for now compute over all notes uniformly
    bpm_samples = [tick_to_bpm(n.start_tick, tempo_map) for n in notes]
    tempo_bpm_avg = statistics.mean(bpm_samples) if bpm_samples else 120.0

    # Select rhythm/drum notes
    rnotes = select_rhythm_notes(mid, notes)
    if not rnotes:
        if verbose:
            print(f"No rhythm notes detected, using all notes for {path}")
        rnotes = notes[:]

    total_bars = max(1, int(math.ceil(max_tick / bar_ticks)))  # rough
    bars = min(total_bars, max_bars) if max_bars > 0 else total_bars

    # Per-class structures
    counts_by_class: Dict[str, int] = {k: 0 for k in CLASS_ORDER}
    pos16_hist: Dict[str, List[int]] = {k: [0] * 16 for k in ("kick", "snare", "chh", "ohh")}
    vel_stats_accum: Dict[str, List[int]] = {k: [] for k in CLASS_ORDER}
    vel_pos16_sum: Dict[str, List[int]] = {k: [0] * 16 for k in ("kick", "snare", "chh")}
    vel_pos16_cnt: Dict[str, List[int]] = {k: [0] * 16 for k in ("kick", "snare", "chh")}
    micro_devs_ms: Dict[str, List[float]] = {k: [] for k in CLASS_ORDER}
    snare_vels: List[int] = []
    snare_pos16_all: List[int] = []
    chh_pos16_all: List[int] = []
    ohh_pos16_all: List[int] = []

    # Per-bar signatures for repetitiveness/fills
    per_bar_binary: Dict[str, List[List[int]]] = {
        "kick": [[0] * 16 for _ in range(bars)],
        "snare": [[0] * 16 for _ in range(bars)],
        "chh": [[0] * 16 for _ in range(bars)],
    }
    per_bar_counts_all: List[int] = [0 for _ in range(bars)]  # all rhythm notes per bar

    def nearest_grid_delta_ms(start_tick: int) -> float:
        """Return signed deviation to nearest 16th grid in milliseconds, using local tempo."""
        # Nearest 16th grid tick
        nearest_grid_tick = int(round(start_tick / sixteenth_tick)) * sixteenth_tick
        delta_ticks = start_tick - nearest_grid_tick
        uspb = get_uspb_at_tick(start_tick, tempo_map)
        ms = (delta_ticks / tpb) * (uspb / 1000.0)
        return ms

    for n in rnotes:
        cls = drum_class_for_pitch(n.pitch)
        if cls is None:
            continue
        counts_by_class[cls] = counts_by_class.get(cls, 0) + 1
        vel_stats_accum.setdefault(cls, []).append(n.velocity)

        # Grid position within bar (16th)
        bar_index = min(bars - 1, int((n.start_tick % (bar_ticks * bars)) // bar_ticks)) if bars > 0 else 0
        # For consistent modulo across entire piece (assume repeating bar structure)
        pos16 = int(round((n.start_tick % bar_ticks) / sixteenth_tick)) % 16

        if cls in ("kick", "snare", "chh", "ohh"):
            if cls == "kick":
                pos16_hist["kick"][pos16] += 1
                vel_pos16_sum["kick"][pos16] += n.velocity
                vel_pos16_cnt["kick"][pos16] += 1
                per_bar_binary["kick"][bar_index][pos16] = 1
            elif cls == "snare":
                pos16_hist["snare"][pos16] += 1
                vel_pos16_sum["snare"][pos16] += n.velocity
                vel_pos16_cnt["snare"][pos16] += 1
                per_bar_binary["snare"][bar_index][pos16] = 1
                snare_vels.append(n.velocity)
                snare_pos16_all.append(pos16)
            elif cls == "chh":
                pos16_hist["chh"][pos16] += 1
                vel_pos16_sum["chh"][pos16] += n.velocity
                vel_pos16_cnt["chh"][pos16] += 1
                per_bar_binary["chh"][bar_index][pos16] = 1
                chh_pos16_all.append(pos16)
            elif cls == "ohh":
                pos16_hist["ohh"][pos16] += 1
                ohh_pos16_all.append(pos16)

        # Microtiming (to nearest 16th)
        dev_ms = nearest_grid_delta_ms(n.start_tick)
        micro_devs_ms.setdefault(cls, []).append(dev_ms)

        # Fills: bar counts
        if bar_index < len(per_bar_counts_all):
            per_bar_counts_all[bar_index] += 1

    # Velocity stats per class
    vel_stats_by_class: Dict[str, Dict[str, float]] = {}
    for cls, arr in vel_stats_accum.items():
        if arr:
            vel_stats_by_class[cls] = {
                "mean": float(statistics.mean(arr)),
                "median": float(statistics.median(arr)),
                "std": float(statistics.pstdev(arr)) if len(arr) > 1 else 0.0,
                "min": float(min(arr)),
                "max": float(max(arr)),
                "count": float(len(arr)),
            }
        else:
            vel_stats_by_class[cls] = {"mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "count": 0.0}

    # Velocity per position avg (kick/snare/chh)
    vel_pos16_avg: Dict[str, List[float]] = {}
    for k in ("kick", "snare", "chh"):
        avg_list = []
        for i in range(16):
            c = vel_pos16_cnt[k][i]
            avg_list.append(float(vel_pos16_sum[k][i]) / c if c > 0 else 0.0)
        vel_pos16_avg[k] = avg_list

    # Microtiming stats per class (absolute mean and std, with signed mean also useful)
    microtiming_ms_by_class: Dict[str, Dict[str, float]] = {}
    for cls, devs in micro_devs_ms.items():
        if devs:
            abs_devs = [abs(x) for x in devs]
            microtiming_ms_by_class[cls] = {
                "mean_signed": float(statistics.mean(devs)),
                "mean_abs": float(statistics.mean(abs_devs)),
                "std": float(statistics.pstdev(devs)) if len(devs) > 1 else 0.0,
                "count": float(len(devs)),
            }
        else:
            microtiming_ms_by_class[cls] = {"mean_signed": 0.0, "mean_abs": 0.0, "std": 0.0, "count": 0.0}

    # Snare backbeat regularity: positions near 2 and 4 beats -> indices 4 and 12 on 16th grid
    # Allow tolerance of ±1 sixteenth index
    def near(p: int, center: int, tol: int = 1) -> bool:
        return min((p - center) % 16, (center - p) % 16) <= tol

    if snare_pos16_all:
        on_backbeat = sum(1 for p in snare_pos16_all if near(p, 4) or near(p, 12))
        snare_backbeat_regular = on_backbeat / len(snare_pos16_all)
    else:
        snare_backbeat_regular = 0.0

    # Snare ghosting: define dynamic threshold relative to median
    snare_ghost_rate = 0.0
    snare_ghost_pos16_hist = [0] * 16
    if snare_vels:
        med = statistics.median(snare_vels)
        thr = 0.6 * med  # ghost if velocity < 60% of median snare velocity
        ghosts = 0
        for p, v in zip(snare_pos16_all, snare_vels):
            if v < thr:
                ghosts += 1
                snare_ghost_pos16_hist[p] += 1
        snare_ghost_rate = ghosts / len(snare_vels)

    # Hi-hat offbeat density: offbeats at 8th upbeats -> indices 2,6,10,14 (16th grid)
    hat_total = len(chh_pos16_all)
    hat_offbeats = sum(1 for p in chh_pos16_all if (p % 4) == 2)
    hat_offbeat_rate = (hat_offbeats / hat_total) if hat_total > 0 else 0.0

    # Repetitiveness: identical bar signatures fraction for kick/snare/chh
    repetitiveness: Dict[str, float] = {}
    for k in ("kick", "snare", "chh"):
        bars_vec = per_bar_binary[k]
        if not bars_vec:
            repetitiveness[k] = 0.0
            continue
        # Convert to tuples for hashability
        tuples = [tuple(v) for v in bars_vec]
        # Mode count / bars
        mode_count = 0
        if tuples:
            counts = {}
            for t in tuples:
                counts[t] = counts.get(t, 0) + 1
            mode_count = max(counts.values())
        repetitiveness[k] = mode_count / len(tuples) if tuples else 0.0

    # Fill detection: spike in last bar per 4/8-bar phrase; here: last bar overall compared to avg
    fill_bars: List[int] = []
    if per_bar_counts_all:
        avg_count = statistics.mean(per_bar_counts_all)
        for i, c in enumerate(per_bar_counts_all):
            if avg_count > 0 and c >= 1.5 * avg_count:
                fill_bars.append(i)

    return FileMetrics(
        path=path,
        ticks_per_beat=tpb,
        time_sig=(num, den),
        tempo_bpm_avg=tempo_bpm_avg,
        duration_sec=duration_sec,
        bars=bars,
        counts_by_class=counts_by_class,
        kick_pos16_hist=pos16_hist["kick"],
        snare_pos16_hist=pos16_hist["snare"],
        chh_pos16_hist=pos16_hist["chh"],
        ohh_pos16_hist=pos16_hist["ohh"],
        vel_stats_by_class=vel_stats_by_class,
        vel_pos16_avg=vel_pos16_avg,
        microtiming_ms_by_class=microtiming_ms_by_class,
        snare_backbeat_regular=snare_backbeat_regular,
        snare_ghost_rate=snare_ghost_rate,
        snare_ghost_pos16_hist=snare_ghost_pos16_hist,
        hat_offbeat_rate=hat_offbeat_rate,
        repetitiveness=repetitiveness,
        fill_bars=fill_bars,
    )


# ---------------
# Aggregation and reporting
# ---------------

def aggregate(files: List[FileMetrics]) -> Dict[str, Any]:
    agg: Dict[str, Any] = {}

    def safe_median(vals: List[float], default: float = 0.0) -> float:
        return float(statistics.median(vals)) if vals else default

    # Tempo/duration overview
    agg["tempo_bpm_avg_median"] = safe_median([f.tempo_bpm_avg for f in files])
    agg["duration_sec_median"] = safe_median([f.duration_sec for f in files])
    agg["bars_median"] = safe_median([f.bars for f in files])

    # Positions
    def pos_median(key: str) -> List[float]:
        if not files:
            return [0.0] * 16
        cols = list(zip(*[getattr(f, key) for f in files]))  # 16 tuples
        return [safe_median([float(x) for x in col]) for col in cols]

    agg["kick_pos16_hist_median"] = pos_median("kick_pos16_hist")
    agg["snare_pos16_hist_median"] = pos_median("snare_pos16_hist")
    agg["chh_pos16_hist_median"] = pos_median("chh_pos16_hist")
    agg["ohh_pos16_hist_median"] = pos_median("ohh_pos16_hist")

    # Backbeat, ghosts, hats
    agg["snare_backbeat_regular_median"] = safe_median([f.snare_backbeat_regular for f in files])
    agg["snare_ghost_rate_median"] = safe_median([f.snare_ghost_rate for f in files])
    agg["hat_offbeat_rate_median"] = safe_median([f.hat_offbeat_rate for f in files])

    # Microtiming
    def micro(cls: str, stat: str) -> float:
        arr = [f.microtiming_ms_by_class.get(cls, {}).get(stat, 0.0) for f in files]
        arr = [float(x) for x in arr if x is not None]
        return safe_median(arr)

    agg["micro_mean_signed_chh_ms"] = micro("chh", "mean_signed")
    agg["micro_mean_abs_chh_ms"] = micro("chh", "mean_abs")
    agg["micro_mean_signed_snare_ms"] = micro("snare", "mean_signed")
    agg["micro_mean_abs_snare_ms"] = micro("snare", "mean_abs")
    agg["micro_mean_signed_kick_ms"] = micro("kick", "mean_signed")
    agg["micro_mean_abs_kick_ms"] = micro("kick", "mean_abs")

    # Repetitiveness
    for k in ("kick", "snare", "chh"):
        agg[f"repetitiveness_{k}_median"] = safe_median([f.repetitiveness.get(k, 0.0) for f in files])

    # Prescriptive recommendations (baseline parameter targets)
    rec: List[str] = []
    # Snare backbeat ideally high for 2-step backbone
    bb = agg["snare_backbeat_regular_median"]
    if bb < 0.8:
        rec.append(f"Increase snare backbeat regularity to ~0.85–0.95 (current median {bb:.2f}). Bias snare to 16th indices 4 and 12 with ±1 tolerance.")
    else:
        rec.append(f"Snare backbeat regularity is good (median {bb:.2f}). Maintain strong 2+4 backbeats.")

    # Snare ghosts
    gr = agg["snare_ghost_rate_median"]
    if gr < 0.25:
        rec.append(f"Add snare ghost notes around e/a of beats to reach ~0.30–0.45 ghost rate (current median {gr:.2f}). Use low velocities (30–60% of snare median).")
    else:
        rec.append(f"Snare ghost activity is adequate (median {gr:.2f}).")

    # Hats microtiming
    chh_mean_signed = agg["micro_mean_signed_chh_ms"]
    if chh_mean_signed < -2:
        rec.append(f"Hats tend to be early by {abs(chh_mean_signed):.1f} ms; shift hats slightly late (+6–12 ms) to emphasize groove.")
    elif chh_mean_signed > 2:
        rec.append(f"Hats are late by {chh_mean_signed:.1f} ms; this can work for laid-back feel. Keep within +4–12 ms window.")
    else:
        rec.append("Hats microtiming near grid. Introduce small positive delay (+6–12 ms) for DnB shuffle feel.")

    # Hat offbeats
    hbo = agg["hat_offbeat_rate_median"]
    if hbo < 0.5:
        rec.append(f"Increase offbeat closed-hat hits on 8th upbeats to ~0.6–0.8 fraction (current median {hbo:.2f}).")

    # Repetitiveness/fills
    rep_sn = agg["repetitiveness_snare_median"]
    if rep_sn > 0.7:
        rec.append(f"Snare patterns are highly repetitive (median identical-bar fraction {rep_sn:.2f}). Add 4/8-bar fills (last bar density spike +30–60%).")

    agg["recommendations"] = rec
    return agg


def write_json(path: str, files: List[FileMetrics], aggregate_data: Dict[str, Any]) -> None:
    payload = {
        "files": {fm.path: asdict(fm) for fm in files},
        "aggregate": aggregate_data,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_report_md(path: str, files: List[FileMetrics], agg: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# DnB Reference Analysis Report")
    lines.append("")
    lines.append("## Aggregate Summary")
    lines.append(f"- Median tempo: {agg.get('tempo_bpm_avg_median', 0):.1f} BPM")
    lines.append(f"- Median duration: {agg.get('duration_sec_median', 0):.1f} s")
    lines.append(f"- Median bars analyzed: {agg.get('bars_median', 0):.1f}")
    lines.append(f"- Snare backbeat regularity (median): {agg.get('snare_backbeat_regular_median', 0.0):.2f}")
    lines.append(f"- Snare ghost rate (median): {agg.get('snare_ghost_rate_median', 0.0):.2f}")
    lines.append(f"- Hat offbeat rate (median): {agg.get('hat_offbeat_rate_median', 0.0):.2f}")
    lines.append(f"- Microtiming (ms, median): kick mean_abs={agg.get('micro_mean_abs_kick_ms', 0.0):.1f}, "
                 f"snare mean_abs={agg.get('micro_mean_abs_snare_ms', 0.0):.1f}, "
                 f"chh mean_abs={agg.get('micro_mean_abs_chh_ms', 0.0):.1f}")
    lines.append("")
    lines.append("### Recommendations")
    for r in agg.get("recommendations", []):
        lines.append(f"- {r}")
    lines.append("")

    lines.append("## Position Histograms (Median across files)")
    def row(title: str, arr: List[float]) -> None:
        lines.append(f"### {title}")
        lines.append("| 16th idx | " + " | ".join([str(i) for i in range(16)]) + " |")
        lines.append("|---|" + "|".join(["---"] * 16) + "|")
        vals = [int(round(v)) for v in arr]
        lines.append("| count | " + " | ".join(str(v) for v in vals) + " |")
        lines.append("")

    row("Kick", agg.get("kick_pos16_hist_median", [0.0]*16))
    row("Snare", agg.get("snare_pos16_hist_median", [0.0]*16))
    row("Closed Hi-hat", agg.get("chh_pos16_hist_median", [0.0]*16))
    row("Open Hi-hat", agg.get("ohh_pos16_hist_median", [0.0]*16))

    lines.append("## File Summaries")
    for fm in files[:20]:  # show first 20
        lines.append(f"### {os.path.basename(fm.path)}")
        lines.append(f"- Tempo avg: {fm.tempo_bpm_avg:.1f} BPM; Duration: {fm.duration_sec:.1f} s; Bars: {fm.bars}")
        lines.append(f"- Snare backbeat regularity: {fm.snare_backbeat_regular:.2f}; Ghost rate: {fm.snare_ghost_rate:.2f}; Hat offbeat: {fm.hat_offbeat_rate:.2f}")
        lines.append(f"- Microtiming mean_abs (ms): kick={fm.microtiming_ms_by_class.get('kick',{}).get('mean_abs',0.0):.1f}, "
                     f"snare={fm.microtiming_ms_by_class.get('snare',{}).get('mean_abs',0.0):.1f}, "
                     f"chh={fm.microtiming_ms_by_class.get('chh',{}).get('mean_abs',0.0):.1f}")
        lines.append(f"- Fill bars flagged: {fm.fill_bars if fm.fill_bars else 'none'}")
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ---------------
# CLI
# ---------------

def main():
    parser = argparse.ArgumentParser(description="DnB reference MIDI analyzer for genre-specific rhythmic features.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Input MIDI paths, directories, or glob patterns.")
    parser.add_argument("--grid", type=int, default=16, choices=[16, 32], help="Grid for placement analysis (default 16).")
    parser.add_argument("--bars", type=int, default=8, help="Bars to analyze per file window (0=all detected) (default 8).")
    parser.add_argument("--out-prefix", type=str, default=os.path.join("test_outputs", "dnb_reference"),
                        help="Output prefix for JSON/MD reports (default test_outputs/dnb_reference).")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    ensure_out_dir(args.out_prefix)

    paths = glob_inputs(args.inputs)
    if args.verbose:
        print(f"Discovered {len(paths)} input(s).")

    file_metrics: List[FileMetrics] = []
    for p in paths:
        fm = analyze_file(p, grid=args.grid, max_bars=args.bars, verbose=args.verbose)
        if fm:
            file_metrics.append(fm)

    agg = aggregate(file_metrics)

    json_path = f"{args.out_prefix}_metrics.json"
    md_path = f"{args.out_prefix}_report.md"
    write_json(json_path, file_metrics, agg)
    write_report_md(md_path, file_metrics, agg)
    print(f"Wrote: {json_path}")
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()