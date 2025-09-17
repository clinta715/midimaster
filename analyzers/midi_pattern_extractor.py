import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import mido
from mido import MidiFile, Message, MetaMessage


@dataclass
class NoteEvent:
    """Structured note event."""
    start_time: float  # Absolute time in ticks
    note: int
    velocity: int
    duration: float  # Duration in ticks
    channel: int
    instrument: Optional[str] = None  # From program change or track name


@dataclass
class TempoEvent:
    """Tempo change event."""
    time: float  # Absolute time in ticks
    bpm: float


@dataclass
class TimeSignatureEvent:
    """Time signature change event."""
    time: float  # Absolute time in ticks
    numerator: int
    denominator: int


@dataclass
class MidiPatternData:
    """Structured MIDI pattern data."""
    file_path: str
    ticks_per_beat: int
    tracks: int
    length_ticks: int
    tempos: List[TempoEvent]
    time_signatures: List[TimeSignatureEvent]
    notes: List[NoteEvent]
    track_info: Dict[str, Any]  # Additional track metadata


def get_instrument_name(msg: Message, track_name: str) -> Optional[str]:
    """Determine instrument from program change or track name."""
    if getattr(msg, 'type') == 'program_change' and hasattr(msg, 'program'):
        # Map program number to instrument name (basic GM mapping)
        gm_instruments = {
            0: 'Acoustic Grand Piano', 1: 'Bright Acoustic Piano', 2: 'Electric Grand Piano',
            4: 'Electric Piano 1', 5: 'Electric Piano 2',
            # Add more as needed, or use a full dict
            # Drums on channel 10, but program may vary
        }
        prog = getattr(msg, 'program')
        return gm_instruments.get(prog, f"Program {prog}")
    return track_name or None


def extract_from_file(file_path: str) -> MidiPatternData:
    """
    Extract structured pattern data from a MIDI file.

    Args:
        file_path: Path to the MIDI file.

    Returns:
        MidiPatternData object with extracted information.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"MIDI file not found: {file_path}")

    midi = MidiFile(file_path)
    tempos: List[TempoEvent] = []
    time_signatures: List[TimeSignatureEvent] = []
    all_notes: List[NoteEvent] = []
    track_info = {}

    current_tempo = 500000  # Default 120 BPM in microseconds per beat
    abs_time = 0.0

    for track_idx, track in enumerate(midi.tracks):
        track_name = track.name if hasattr(track, 'name') and track.name else f"Track {track_idx}"
        track_info[track_name] = {'index': track_idx, 'events_count': len(track)}

        active_notes: Dict[int, Tuple[float, int]] = {}  # note -> (start_time, velocity)
        track_time = 0.0
        instrument = None

        for msg in track:
            track_time += msg.time
            abs_time = max(abs_time, track_time)  # For file length

            if getattr(msg, 'type') == 'set_tempo':
                current_tempo = getattr(msg, 'tempo')
                tempos.append(TempoEvent(track_time, 60000000 / current_tempo))
            elif getattr(msg, 'type') == 'time_signature':
                time_signatures.append(TimeSignatureEvent(
                    track_time, getattr(msg, 'numerator'), msg.denominator
                ))
            elif getattr(msg, 'type') == 'program_change':
                instrument = get_instrument_name(msg, track_name)
            elif getattr(msg, 'type') == 'note_on' and msg.velocity > 0:
                active_notes[getattr(msg, 'note')] = (track_time, msg.velocity)
            elif getattr(msg, 'type') == 'note_off' or (getattr(msg, 'type') == 'note_on' and msg.velocity == 0):
                if getattr(msg, 'note') in active_notes:
                    start_time, velocity = active_notes.pop(getattr(msg, 'note'))
                    duration = track_time - start_time
                    all_notes.append(NoteEvent(
                        start_time=start_time,
                        note=getattr(msg, 'note'),
                        velocity=velocity,
                        duration=duration,
                        channel=getattr(msg, 'channel'),
                        instrument=instrument
                    ))

        # Handle any remaining active notes (rare, but close at end)
        end_time = track_time
        for note, (start_time, velocity) in active_notes.items():
            duration = end_time - start_time
            all_notes.append(NoteEvent(
                start_time=start_time,
                note=note,
                velocity=velocity,
                duration=duration,
                channel=0,  # Default if not specified
                instrument=instrument
            ))

    # Sort notes by start time
    all_notes.sort(key=lambda n: n.start_time)

    return MidiPatternData(
        file_path=file_path,
        ticks_per_beat=midi.ticks_per_beat,
        tracks=len(midi.tracks),
        length_ticks=int(abs_time),
        tempos=sorted(tempos, key=lambda t: t.time),
        time_signatures=sorted(time_signatures, key=lambda ts: ts.time),
        notes=all_notes,
        track_info=track_info
    )


def extract_from_directory(dir_path: str) -> List[MidiPatternData]:
    """
    Extract patterns from all MIDI files in a directory.

    Args:
        dir_path: Directory containing MIDI files.

    Returns:
        List of MidiPatternData for each file.
    """
    dir_path_obj = Path(dir_path)
    if not dir_path_obj.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    patterns = []
    for file_path in dir_path_obj.glob("*.mid"):
        try:
            patterns.append(extract_from_file(str(file_path)))
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    return patterns


# ---------------------------
# Rhythm pattern extraction utilities (read-only by default)
# ---------------------------
import math
import statistics

def _first_time_signature_str(mpd: MidiPatternData) -> str:
    if mpd.time_signatures:
        ts = mpd.time_signatures[0]
        return f"{ts.numerator}/{ts.denominator}"
    return "4/4"

def _bar_length_beats(mpd: MidiPatternData) -> float:
    if mpd.time_signatures:
        ts = mpd.time_signatures[0]
        # beats in a bar = numerator * (4/denominator)
        return ts.numerator * (4.0 / float(ts.denominator))
    return 4.0

def _tempo_range_bpm(mpd: MidiPatternData) -> Tuple[Optional[float], Optional[float]]:
    if not mpd.tempos:
        return (None, None)
    vals = [t.bpm for t in mpd.tempos if t.bpm and t.bpm > 0]
    if not vals:
        return (None, None)
    mn = min(vals)
    mx = max(vals)
    # If a single constant tempo, set both min/max; else return range
    if abs(mx - mn) < 1e-6:
        return (mn, mn)
    return (mn, mx)

def _choose_best_subdivision(onsets_beats: List[float]) -> int:
    """
    Choose a pragmatic PPQ subdivision (pulses per quarter note) that minimizes quantization error.
    Candidates biased toward common grids.
    """
    if not onsets_beats:
        return 4
    candidates = [2, 3, 4, 6, 8, 12, 16, 24]
    best = 4
    best_err = float("inf")
    for sub in candidates:
        step = 1.0 / sub
        errs = []
        for b in onsets_beats:
            q = round(b / step) * step
            errs.append(abs(b - q))
        if errs:
            median_err = statistics.median(errs)
            if median_err < best_err:
                best_err = median_err
                best = sub
    return best

def normalize_events_to_pattern_json(
    events: List[NoteEvent],
    *,
    ticks_per_beat: int,
    subdivision: int,
    start_beat: float = 0.0,
    length_beats: Optional[float] = None,
    include_pitch_channel: bool = True,
    quantize: bool = True,
) -> List[Dict[str, Any]]:
    """
    Convert note events to pattern_json events with onset/duration in beats.
    - Quarter note = 1.0
    - For percussive hits, duration_beats = 0.0
    - onset_beats are relative to start_beat
    - If quantize=True, snap to given subdivision grid
    """
    step = 1.0 / max(1, subdivision)
    pattern: List[Dict[str, Any]] = []
    if not events:
        return pattern

    # Convert to beats
    raw = []
    for ev in events:
        onset_b = float(ev.start_time) / float(ticks_per_beat)
        dur_b = float(ev.duration) / float(ticks_per_beat)
        raw.append((onset_b, dur_b, ev.velocity, ev.note, ev.channel))

    # Filter to window and quantize if requested
    end_beat = None if length_beats is None else (start_beat + length_beats + 1e-9)
    for onset_b, dur_b, vel, note, ch in raw:
        if onset_b < start_beat - 1e-9:
            continue
        if end_beat is not None and onset_b > end_beat:
            continue
        qb = onset_b
        if quantize:
            qb = round(onset_b / step) * step
        rel = qb - start_beat
        if rel < -1e-9:
            continue
        item: Dict[str, Any] = {
            "onset_beats": float(rel),
            "duration_beats": 0.0,  # percussive hit default
            "velocity": int(vel),
        }
        if include_pitch_channel:
            item["pitch"] = int(note)
            item["channel"] = int(ch)
        pattern.append(item)

    # Sort and deduplicate near-coincident hits after quantization
    pattern.sort(key=lambda d: (round(d["onset_beats"] / step) * step, d.get("pitch", 0)))
    dedup: List[Dict[str, Any]] = []
    last_key = None
    for d in pattern:
        key = (round(d["onset_beats"] / step) * step, d.get("pitch", -1))
        if key != last_key:
            # clamp onset exactly to grid to avoid drift in JSON
            d["onset_beats"] = round(d["onset_beats"] / step) * step
            dedup.append(d)
            last_key = key
    return dedup

def compute_syncopation_from_events(pattern_events: List[Dict[str, Any]]) -> float:
    """
    Syncopation estimate in [0,1]: fraction of hits not near integer beats.
    Uses onset_beats; expects events normalized to beats.
    """
    if not pattern_events:
        return 0.0
    off = 0
    total = 0
    tol = 1e-3
    for e in pattern_events:
        b = float(e.get("onset_beats", 0.0))
        frac = abs(b - round(b))  # distance to nearest integer beat
        total += 1
        if frac > tol:
            off += 1
    return max(0.0, min(1.0, off / total if total else 0.0))

def compute_density_from_events(pattern_events: List[Dict[str, Any]], *, length_beats: float) -> float:
    """
    Density estimate in [0,1]: hits per beat normalized by a reference max of 4 hits/beat (sixteenth grid).
    """
    if length_beats <= 0:
        return 0.0
    hits_per_beat = len(pattern_events) / float(length_beats)
    density = hits_per_beat / 4.0
    return max(0.0, min(1.0, density))

def estimate_swing_from_events(pattern_events: List[Dict[str, Any]]) -> Optional[float]:
    """
    Optional swing estimate [0,1] or None.
    Heuristic: look at alternating inter-onset intervals; larger alternation implies stronger swing.
    """
    if len(pattern_events) < 6:
        return None
    onsets = sorted(float(e["onset_beats"]) for e in pattern_events)
    iois = [onsets[i+1] - onsets[i] for i in range(len(onsets) - 1)]
    if len(iois) < 5:
        return None
    odds = iois[1::2]
    evens = iois[0::2]
    if not odds or not evens:
        return None
    avg_odd = statistics.mean(odds)
    avg_even = statistics.mean(evens)
    if avg_odd <= 0 or avg_even <= 0:
        return None
    ratio = min(avg_odd, avg_even) / max(avg_odd, avg_even)
    swing_strength = 1.0 - ratio  # 0=no swing, ->1 strong alternation
    # Require some consistency to consider it reliable
    stdev = statistics.pstdev(iois)
    mean = statistics.mean(iois)
    if mean <= 1e-6:
        return None
    variability = stdev / mean
    if variability > 0.6:
        return None
    return max(0.0, min(1.0, swing_strength))

def estimate_humanization_from_events(
    original_onsets_beats: List[float],
    quantized_onsets_beats: List[float],
    *,
    subdivision: int,
) -> Optional[float]:
    """
    Optional humanization estimate [0,1] based on timing jitter stddev relative to grid step.
    """
    if not original_onsets_beats or not quantized_onsets_beats:
        return None
    if len(original_onsets_beats) != len(quantized_onsets_beats):
        n = min(len(original_onsets_beats), len(quantized_onsets_beats))
        original_onsets_beats = original_onsets_beats[:n]
        quantized_onsets_beats = quantized_onsets_beats[:n]
    if len(original_onsets_beats) < 4:
        return None
    residuals = [o - q for o, q in zip(original_onsets_beats, quantized_onsets_beats)]
    try:
        jitter = statistics.pstdev(residuals)
    except statistics.StatisticsError:
        return None
    grid = 1.0 / max(1, subdivision)
    # Normalize: 0.5 of a grid step stddev maps ~1.0; clamp
    norm = jitter / (grid * 0.5)
    return max(0.0, min(1.0, float(norm)))

def _is_drum_like_file(file_path: str) -> bool:
    name = os.path.basename(file_path).lower()
    drum_keywords = ["drum", "kick", "snare", "hihat", "hi-hat", "hat", "clap", "perc", "rim", "shaker", "tom", "cymbal"]
    return any(k in name for k in drum_keywords)

def extract_rhythm_patterns_from_midi(
    midi_path: str,
    *,
    instrument_hint: str = "drums",
    genre: Optional[str] = None,
    mood: Optional[str] = None,
    db_path: Optional[str] = None,
    repository: Optional[Any] = None,
    tags: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    High-level extractor:
    - Parse MIDI, focus on percussive/drum-like content
    - Normalize hits to pattern_json-ready events
    - Compute metrics: syncopation, density, swing (optional), humanization (optional)
    - Optionally persist if repository or db_path provided
    Returns list of pattern dicts ready for repository upsert.
    """
    mpd = extract_from_file(midi_path)
    ts_str = _first_time_signature_str(mpd)
    bar_len = _bar_length_beats(mpd)
    bpm_min, bpm_max = _tempo_range_bpm(mpd)
    inferred_instrument = instrument_hint or "drums"

    # Decide which notes to consider as percussive hits:
    # Prefer GM drum channel 9; if none found, and file appears drum-like, use all notes.
    drum_notes = [n for n in mpd.notes if n.channel == 9]
    if not drum_notes and (_is_drum_like_file(midi_path) or inferred_instrument.lower() == "drums"):
        drum_notes = list(mpd.notes)

    if not drum_notes:
        return []

    # Build onset lists in beats
    onsets_beats_all = [float(n.start_time) / float(mpd.ticks_per_beat) for n in drum_notes]
    if not onsets_beats_all:
        return []

    first = min(onsets_beats_all)
    last = max(onsets_beats_all)

    # Align start to bar boundary, compute length as integer bars
    start_bar = math.floor(first / bar_len)
    start_beat = start_bar * bar_len
    raw_span = max(0.0, last - start_beat)
    bars = max(1, math.ceil((raw_span + 1e-6) / bar_len))
    length_beats = bars * bar_len

    # Choose subdivision and prepare pattern_json
    subdiv = _choose_best_subdivision(onsets_beats_all)
    step = 1.0 / max(1, subdiv)

    # Original and quantized onset lists (within pattern window)
    original_onsets = []
    quantized_onsets = []
    for n in drum_notes:
        ob = float(n.start_time) / float(mpd.ticks_per_beat)
        if ob < start_beat - 1e-9 or ob > start_beat + length_beats + 1e-9:
            continue
        qb = round(ob / step) * step
        original_onsets.append(ob - start_beat)
        quantized_onsets.append(qb - start_beat)

    pattern_events = normalize_events_to_pattern_json(
        drum_notes,
        ticks_per_beat=mpd.ticks_per_beat,
        subdivision=subdiv,
        start_beat=start_beat,
        length_beats=length_beats,
        include_pitch_channel=True,
        quantize=True,
    )

    # Metrics
    sync = compute_syncopation_from_events(pattern_events)
    dens = compute_density_from_events(pattern_events, length_beats=length_beats)
    swing = estimate_swing_from_events(pattern_events)
    human = estimate_humanization_from_events(original_onsets, quantized_onsets, subdivision=subdiv)

    # Tags
    out_tags: Optional[List[str]] = list(tags) if tags else None
    if bars == 1:
        out_tags = (out_tags or []) + ["loop-1bar"]

    pattern_dict = {
        "source_path": midi_path,
        "source_track": None,  # unknown without per-track mapping
        "instrument": "drums" if inferred_instrument is None else inferred_instrument,
        "genre": genre or "unknown",
        "mood": mood,
        "time_signature": ts_str,
        "subdivision": int(subdiv),
        "length_beats": float(length_beats),
        "bpm_min": bpm_min,
        "bpm_max": bpm_max,
        "syncopation": float(sync),
        "density": float(dens),
        "swing": None if swing is None else float(swing),
        "humanization": None if human is None else float(human),
        "quality_score": None,  # leave None unless a separate heuristic is desired
        "pattern_json": pattern_events,
        "accent_profile_json": None,
        "tags_json": out_tags,
    }
    results = [pattern_dict]

    # Optional persistence (kept lightweight to avoid circular deps)
    if repository is not None or db_path is not None:
        try:
            # Lazy import to avoid import cycles at module import time
            from . import store_writer  # type: ignore
        except Exception:
            store_writer = None
        if store_writer is not None:
            repo = repository or store_writer.get_repository(db_path=db_path)
            try:
                src_id = store_writer.upsert_source_for_midi(repo, midi_path, track_name=None)
                _ = store_writer.upsert_patterns(repo, src_id, results)
            finally:
                # Only close if we created it
                if repository is None:
                    try:
                        repo.close()  # type: ignore[attr-defined]
                    except Exception:
                        pass

    return results
if __name__ == "__main__":
    # Example usage
    import json
    data = extract_from_file("reference_midis/midi6/05_808 Attack_Cm_148bpm_Snare.mid")  # Example file
    print(json.dumps(asdict(data), indent=2, default=str))