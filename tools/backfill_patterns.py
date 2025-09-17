#!/usr/bin/env python3
"""
Backfill CLI for ingesting rhythm patterns into SQLite repository.

Sources:
- MIDI files from a directory or single file
- Hardcoded genre libraries (if genres/genre_rules.py is present)

Usage:
  - As module: python -m tools.backfill_patterns [args]
  - As script: python tools/backfill_patterns.py [args]

Only uses standard library and existing project modules.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

# Resolve project-root imports reliably when run as a script
# Ensure repository root is on sys.path (directory containing this tools/ folder)
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# External project modules (existing)
from analyzers.analysis_api import analyze_midi_and_store  # [`python.analyze_midi_and_store()`](analyzers/analysis_api.py:48)
from analyzers import store_writer  # [`python.get_repository()`](analyzers/store_writer.py:25)
from analyzers.midi_pattern_extractor import (  # Metrics and normalization helpers
    NoteEvent,  # [`python.NoteEvent()`](analyzers/midi_pattern_extractor.py:9)
    normalize_events_to_pattern_json,  # [`python.normalize_events_to_pattern_json()`](analyzers/midi_pattern_extractor.py:227)
    compute_syncopation_from_events,  # [`python.compute_syncopation_from_events()`](analyzers/midi_pattern_extractor.py:292)
    compute_density_from_events,  # [`python.compute_density_from_events()`](analyzers/midi_pattern_extractor.py:310)
    estimate_swing_from_events,  # [`python.estimate_swing_from_events()`](analyzers/midi_pattern_extractor.py:320)
)
from data_store.pattern_repository import PatternRepository  # [`python.PatternRepository()`](data_store/pattern_repository.py:11)


# ---------------------------
# Logging setup
# ---------------------------

logger = logging.getLogger("backfill_patterns")


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    # Silence very chatty libraries if any
    logging.getLogger("mido").setLevel(logging.WARNING)


# ---------------------------
# CLI argument parsing
# ---------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Backfill rhythm patterns into SQLite repository from MIDI and/or hardcoded genre libraries."
    )
    p.add_argument("--midi-dir", default="reference_midis/", help="Directory to scan for MIDI files (default: reference_midis/)")
    p.add_argument("--file", dest="file", help="Optional single MIDI file to ingest; if provided, --midi-dir is ignored")
    p.add_argument("--recursive", action="store_true", help="Recurse into subdirectories when scanning --midi-dir")
    p.add_argument("--ingest-midi", action="store_true", default=True, help="Enable MIDI ingestion path (default: True)")
    p.add_argument("--ingest-hardcoded", action="store_true", default=False, help="Enable hardcoded genre library ingestion path (default: False)")
    p.add_argument("--genre", help="Optional genre override applied to ingested patterns")
    p.add_argument("--mood", help="Optional mood label")
    p.add_argument("--instrument", default="drums", help="Instrument label to assign (default: drums)")
    p.add_argument("--db-path", help="Optional DB path; overrides MIDIMASTER_DB_PATH and default")
    p.add_argument("--tags", nargs="+", help="Optional tags (e.g., fill intro loop-1bar)")
    p.add_argument("--dry-run", action="store_true", help="Run extraction and logging only; skip DB upserts")
    p.add_argument("--verbose", action="store_true", help="Increase logging verbosity")
    return p


# ---------------------------
# Helpers: DB path resolution
# ---------------------------

def resolve_db_path(cli_db_path: Optional[str]) -> str:
    """
    Mimic repository path resolution for logging:
    - CLI --db-path if provided
    - else env MIDIMASTER_DB_PATH
    - else default data/pattern_store.sqlite
    """
    if cli_db_path:
        return cli_db_path
    env_path = os.environ.get("MIDIMASTER_DB_PATH")
    if env_path:
        return env_path
    return str(Path("data") / "pattern_store.sqlite")


# ---------------------------
# MIDI ingestion
# ---------------------------

@dataclass
class MidiIngestResult:
    files_processed: int = 0
    sources_seen: int = 0
    patterns_inserted_or_existing: int = 0
    failures: int = 0


def iter_midi_files(root: Path, recursive: bool) -> Iterable[Path]:
    patterns = ("*.mid", "*.midi", "*.MID", "*.MIDI")
    if recursive:
        for pat in patterns:
            for p in root.rglob(pat):
                if p.is_file():
                    yield p
    else:
        for pat in patterns:
            for p in root.glob(pat):
                if p.is_file():
                    yield p


def ingest_midi_files(
    *,
    file: Optional[Path],
    midi_dir: Path,
    recursive: bool,
    instrument_hint: str,
    genre: Optional[str],
    mood: Optional[str],
    db_path: Optional[str],
    tags: Optional[List[str]],
    dry_run: bool,
) -> MidiIngestResult:
    res = MidiIngestResult()
    files: List[Path] = []

    if file:
        if file.exists():
            files = [file]
        else:
            logger.error(f"--file not found: {file}")
            res.failures += 1
            return res
    else:
        if not midi_dir.exists():
            logger.warning(f"--midi-dir not found: {midi_dir} (skipping MIDI ingestion)")
            return res
        files = list(iter_midi_files(midi_dir, recursive))

    logger.info(f"MIDI ingestion: {len(files)} file(s) to process")

    # Use a single repository instance when not dry-run to amortize open/close
    repo: Optional[PatternRepository] = None
    if not dry_run:
        repo = store_writer.get_repository(db_path=db_path)  # [`python.get_repository()`](analyzers/store_writer.py:25)

    try:
        for f in files:
            try:
                if dry_run:
                    # Extract only, skip persistence
                    from analyzers.midi_pattern_extractor import extract_rhythm_patterns_from_midi  # local import to avoid cycles
                    patterns = extract_rhythm_patterns_from_midi(
                        str(f),
                        instrument_hint=instrument_hint,
                        genre=genre,
                        mood=mood,
                        repository=None,
                        db_path=None,
                        tags=tags,
                    )
                    logger.info(f"[dry-run] MIDI {f.name}: extracted {len(patterns)} pattern(s)")
                    res.patterns_inserted_or_existing += len(patterns)
                    res.sources_seen += 1  # would upsert source row even for 0 patterns
                else:
                    result = analyze_midi_and_store(
                        str(f),
                        instrument_hint=instrument_hint,
                        genre=genre,
                        mood=mood,
                        db_path=db_path if repo is None else None,  # avoid passing if repo provided
                        repository=repo,
                        tags=tags,
                    )
                    pid_count = len(result.get("pattern_ids", []))
                    logger.info(f"MIDI {f.name}: source_id={result.get('source_id')} patterns={pid_count}")
                    res.patterns_inserted_or_existing += pid_count
                    res.sources_seen += 1
                res.files_processed += 1
            except Exception as e:
                logger.error(f"Failed to process MIDI file '{f}': {e}")
                logger.debug(traceback.format_exc())
                res.failures += 1
    finally:
        if repo is not None:
            try:
                repo.close()
            except Exception:
                pass

    return res


# ---------------------------
# Hardcoded library ingestion
# ---------------------------

@dataclass
class HardcodedIngestResult:
    classes_seen: int = 0
    sources_seen: int = 0
    patterns_inserted_or_existing: int = 0
    failures: int = 0


def _infer_subdivision_from_onsets(onsets: List[float]) -> int:
    """
    Heuristic subdivision selection based on onsets (beats).
    Chooses among common PPQ subdivisions: 2,3,4,6,8,12,16,24.
    """
    if not onsets:
        return 4
    candidates = [2, 3, 4, 6, 8, 12, 16, 24]
    step_errors: List[Tuple[int, float]] = []
    for sub in candidates:
        step = 1.0 / sub
        err = 0.0
        for b in onsets:
            q = round(b / step) * step
            err += abs(b - q)
        step_errors.append((sub, err))
    step_errors.sort(key=lambda t: t[1])
    return step_errors[0][0]


def _infer_subdivision_from_durations(durations_beats: List[float]) -> int:
    # Detect triplet-friendly values (near multiples of 1/3)
    def near(x: float, base: float, tol: float = 1e-6) -> bool:
        k = round(x / base)
        return abs(x - k * base) <= 1e-3

    if any(near(d, 1.0 / 3.0) or near(d, 2.0 / 3.0) for d in durations_beats if d > 0):
        # Triplet grid
        return 3  # pulses per quarter
    # Default to sixteenth grid
    return 4


def _normalize_tag_list(tags: Optional[Sequence[str]], extra: Optional[Sequence[str]] = None) -> Optional[List[str]]:
    if not tags and not extra:
        return None
    final: List[str] = []
    if tags:
        final.extend([str(t) for t in tags if str(t).strip()])
    if extra:
        final.extend([str(t) for t in extra if str(t).strip()])
    if not final:
        return None
    # de-duplicate while preserving order
    seen = set()
    deduped = []
    for t in final:
        if t not in seen:
            seen.add(t)
            deduped.append(t)
    return deduped


def _pattern_from_durations(
    durations: List[float],
    *,
    instrument: str,
    genre: str,
    mood: Optional[str],
    time_signature: str = "4/4",
    default_pitch: int = 36,
    velocity: int = 100,
    tags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Build a repository-ready dict from a list of beat durations (sum defines length).
    - Creates percussive events at cumulative onsets.
    - Uses normalize_events_to_pattern_json for stable JSON.
    - Computes metrics.
    """
    # Guard
    durs = [float(x) for x in durations if float(x) >= 0.0]
    if not durs:
        raise ValueError("Empty duration pattern")
    length_beats = float(sum(durs))
    # Handle common bar normalization (if durations are fractions of a bar)
    # If the sum is within (0, 1.0001], assume fractions-of-bar, convert to 4/4 beats
    if 0.0 < length_beats <= 1.0001:
        length_beats *= 4.0

    # Onset list
    onsets = []
    t = 0.0
    for d in durs:
        onsets.append(t)
        t += d

    # Subdivision heuristic
    subdivision = _infer_subdivision_from_durations(durs)
    # Simulate ticks; normalize_events_to_pattern_json consumes ticks->beats using tpq
    tpq = 480
    events: List[NoteEvent] = []
    for ob in onsets:
        events.append(
            NoteEvent(
                start_time=ob * tpq,
                note=int(default_pitch),
                velocity=int(velocity),
                duration=0.0,
                channel=9,
                instrument=instrument,
            )
        )
    pattern_events = normalize_events_to_pattern_json(
        events,
        ticks_per_beat=tpq,
        subdivision=subdivision,
        start_beat=0.0,
        length_beats=length_beats,
        include_pitch_channel=True,
        quantize=True,
    )

    sync = compute_syncopation_from_events(pattern_events)
    dens = compute_density_from_events(pattern_events, length_beats=length_beats)
    swing = estimate_swing_from_events(pattern_events)
    # Humanization not applicable for hardcoded quantized patterns
    human = None

    quality_score = 0.6
    out_tags = _normalize_tag_list(tags, extra=["from-hardcoded"])
    if abs(length_beats - 4.0) <= 1e-6:
        out_tags = _normalize_tag_list(out_tags, extra=["loop-1bar"])

    pattern_dict: Dict[str, Any] = {
        "source_path": "genres/genre_rules.py",
        "source_track": None,
        "instrument": instrument,
        "genre": genre,
        "mood": mood,
        "time_signature": time_signature,
        "subdivision": int(subdivision),
        "length_beats": float(length_beats),
        "bpm_min": None,
        "bpm_max": None,
        "syncopation": float(sync),
        "density": float(dens),
        "swing": None if swing is None else float(swing),
        "humanization": None if human is None else float(human),
        "quality_score": float(quality_score),
        "pattern_json": pattern_events,
        "accent_profile_json": None,
        "tags_json": out_tags,
    }
    return pattern_dict


def _pattern_from_steps(
    steps: Sequence[Union[int, bool]],
    *,
    instrument: str,
    genre: str,
    mood: Optional[str],
    time_signature: str = "4/4",
    default_pitch: int = 36,
    velocity: int = 100,
    tags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Build a repository-ready pattern from a boolean/0-1 step array.
    Assumes 16-step per 4/4 bar by default (sixteenth grid).
    """
    n = len(steps)
    if n <= 0:
        raise ValueError("Empty step pattern")
    # Infer one bar length based on common step counts: 16 -> 4/4, 12 -> 4/4 triplets, 8 -> 4/4 eighths
    if n in (16, 8, 4, 32):
        length_beats = 4.0
    elif n in (12, 24):
        length_beats = 4.0
    else:
        # Fallback: assume each step is a sixteenth
        length_beats = float(n) / 4.0

    # Subdivision (pulses per quarter): 4 for 16 steps, 3 for 12-step triplets
    if n % 12 == 0 and n % 16 != 0:
        subdivision = 3
    else:
        subdivision = 4

    tpq = 480
    step_len_beats = length_beats / float(n)
    events: List[NoteEvent] = []
    for i, s in enumerate(steps):
        if bool(s):
            onset = i * step_len_beats
            events.append(
                NoteEvent(
                    start_time=onset * tpq,
                    note=int(default_pitch),
                    velocity=int(velocity),
                    duration=0.0,
                    channel=9,
                    instrument=instrument,
                )
            )

    pattern_events = normalize_events_to_pattern_json(
        events,
        ticks_per_beat=tpq,
        subdivision=subdivision,
        start_beat=0.0,
        length_beats=length_beats,
        include_pitch_channel=True,
        quantize=True,
    )

    sync = compute_syncopation_from_events(pattern_events)
    dens = compute_density_from_events(pattern_events, length_beats=length_beats)
    swing = estimate_swing_from_events(pattern_events)
    human = None

    quality_score = 0.6
    out_tags = _normalize_tag_list(tags, extra=["from-hardcoded"])
    if abs(length_beats - 4.0) <= 1e-6:
        out_tags = _normalize_tag_list(out_tags, extra=["loop-1bar"])

    pattern_dict: Dict[str, Any] = {
        "source_path": "genres/genre_rules.py",
        "source_track": None,
        "instrument": instrument,
        "genre": genre,
        "mood": mood,
        "time_signature": time_signature,
        "subdivision": int(subdivision),
        "length_beats": float(length_beats),
        "bpm_min": None,
        "bpm_max": None,
        "syncopation": float(sync),
        "density": float(dens),
        "swing": None if swing is None else float(swing),
        "humanization": None if human is None else float(human),
        "quality_score": float(quality_score),
        "pattern_json": pattern_events,
        "accent_profile_json": None,
        "tags_json": out_tags,
    }
    return pattern_dict


def _derive_genre_from_classname(cls_name: str) -> str:
    name = cls_name
    if name.lower().endswith("rules"):
        name = name[: -len("rules")]
    return name.replace("_", "-").replace(" ", "-").lower() or "unknown"


def _extract_time_signature_from_rule(rule_obj: Any) -> str:
    # Try common attribute/method names
    for attr in ("time_signature", "meter", "default_time_signature"):
        if hasattr(rule_obj, attr):
            val = getattr(rule_obj, attr)
            if isinstance(val, str) and "/" in val:
                return val
            if isinstance(val, tuple) and len(val) == 2:
                try:
                    return f"{int(val[0])}/{int(val[1])}"
                except Exception:
                    pass
    for meth in ("get_time_signature", "get_meter"):
        if hasattr(rule_obj, meth) and callable(getattr(rule_obj, meth)):
            try:
                val = getattr(rule_obj, meth)()
                if isinstance(val, str) and "/" in val:
                    return val
                if isinstance(val, tuple) and len(val) == 2:
                    return f"{int(val[0])}/{int(val[1])}"
            except Exception:
                logger.debug("Time signature getter failed on %s", type(rule_obj).__name__)
    return "4/4"


def _iter_rule_patterns(rule_obj: Any) -> Iterable[Union[List[float], List[int], List[bool]]]:
    """
    Yield pattern representations from a rule object. Supports:
    - rule.get_rhythm_patterns()
    - rule.get_patterns()
    - rule.patterns
    - rule.rhythm_patterns
    Each item can be:
      * list[float] durations (beats or fractions of a bar)
      * list[int|bool] step array (0/1 or bool)
      * nested sequences; we flatten one level if necessary
    """
    candidates: List[Union[List[Any], Any]] = []

    for attr in ("get_rhythm_patterns", "get_patterns"):
        if hasattr(rule_obj, attr) and callable(getattr(rule_obj, attr)):
            try:
                vals = getattr(rule_obj, attr)()
                candidates.append(vals)
            except Exception:
                logger.debug("Pattern getter %s failed on %s", attr, type(rule_obj).__name__)

    for attr in ("patterns", "rhythm_patterns"):
        if hasattr(rule_obj, attr):
            try:
                candidates.append(getattr(rule_obj, attr))
            except Exception:
                pass

    seen_any = False
    for c in candidates:
        if c is None:
            continue
        # If it's a dict, consider values
        if isinstance(c, dict):
            iterables = c.values()
        else:
            iterables = c
        try:
            for item in iterables:  # type: ignore
                seen_any = True
                # Flatten one level if nested like [[...], [...]]
                if item and isinstance(item, (list, tuple)) and len(item) > 0 and isinstance(item[0], (list, tuple)):
                    for sub in item:
                        yield list(sub)  # type: ignore
                else:
                    yield list(item)  # type: ignore
        except TypeError:
            # Not iterable
            continue

    if not seen_any:
        logger.debug("No discoverable patterns on rule %s", type(rule_obj).__name__)


def ingest_hardcoded_rules(
    *,
    instrument: str,
    genre_override: Optional[str],
    mood: Optional[str],
    db_path: Optional[str],
    tags: Optional[List[str]],
    dry_run: bool,
) -> HardcodedIngestResult:
    res = HardcodedIngestResult()

    # Try importing genres/genre_rules.py dynamically
    try:
        rules_mod = importlib.import_module("genres.genre_rules")
    except Exception as e:
        logger.warning("Hardcoded ingestion requested but genres/genre_rules.py not found or import failed: %s", e)
        return res

    # Discover rule classes: any class ending with 'Rules'
    rule_classes: List[type] = []
    for name in dir(rules_mod):
        obj = getattr(rules_mod, name)
        if isinstance(obj, type) and name.lower().endswith("rules"):
            rule_classes.append(obj)

    if not rule_classes:
        logger.warning("No '*Rules' classes discovered in genres/genre_rules.py; nothing to ingest.")
        return res

    logger.info(f"Discovered {len(rule_classes)} rule class(es): " + ", ".join([c.__name__ for c in rule_classes]))

    repo: Optional[PatternRepository] = None
    if not dry_run:
        repo = store_writer.get_repository(db_path=db_path)

    try:
        for cls in rule_classes:
            try:
                rule_obj = cls()  # type: ignore[call-arg]
            except Exception as e:
                logger.error("Failed to instantiate %s: %s", cls.__name__, e)
                logger.debug(traceback.format_exc())
                res.failures += 1
                continue

            res.classes_seen += 1

            # Determine genre for this rule
            derived_genre = _derive_genre_from_classname(cls.__name__)
            genre = (genre_override or derived_genre) or "unknown"
            time_sig = _extract_time_signature_from_rule(rule_obj)

            # Upsert a source row for this rule
            if not dry_run:
                try:
                    assert repo is not None
                    source_id = repo.upsert_source(  # [`python.upsert_source()`](data_store/pattern_repository.py:55)
                        source_type="hardcoded_library",
                        source_path="genres/genre_rules.py",
                        source_track=cls.__name__,
                        extracted_by_version=None,
                    )
                    res.sources_seen += 1
                except Exception as e:
                    logger.error("Failed to upsert source for rule %s: %s", cls.__name__, e)
                    logger.debug(traceback.format_exc())
                    res.failures += 1
                    # Still try to process next rules
                    continue
            else:
                # Dry-run: pretend a source was recognized
                res.sources_seen += 1
                source_id = -1

            # Iterate patterns from the rule
            patterns_to_upsert: List[Dict[str, Any]] = []
            count_this_rule = 0

            for pat in _iter_rule_patterns(rule_obj):
                try:
                    # Determine if it's durations or steps
                    is_all_numbers = all(isinstance(x, (int, float)) for x in pat)
                    is_all_boolish = all(isinstance(x, (int, bool)) for x in pat) and any(bool(x) for x in pat)
                    if is_all_numbers and not is_all_boolish:
                        # Treat as durations
                        durations = [float(x) for x in pat]
                        pattern_dict = _pattern_from_durations(
                            durations,
                            instrument=instrument,
                            genre=genre,
                            mood=mood,
                            time_signature=time_sig,
                            tags=tags,
                        )
                    else:
                        # Treat as steps (0/1 or booleans)
                        steps = [bool(x) for x in pat]
                        pattern_dict = _pattern_from_steps(
                            steps,
                            instrument=instrument,
                            genre=genre,
                            mood=mood,
                            time_signature=time_sig,
                            tags=tags,
                        )

                    patterns_to_upsert.append(pattern_dict)
                    count_this_rule += 1
                except Exception as e:
                    logger.error("Failed to convert pattern on %s: %s", cls.__name__, e)
                    logger.debug("Pattern data: %s", pat)
                    logger.debug(traceback.format_exc())
                    res.failures += 1
                    continue

            # Persist or log
            if dry_run:
                logger.info(f"[dry-run] Rule {cls.__name__}: would insert {count_this_rule} pattern(s) (genre={genre}, ts={time_sig})")
                res.patterns_inserted_or_existing += count_this_rule
            else:
                try:
                    assert repo is not None
                    ids = store_writer.upsert_patterns(repo, source_id, patterns_to_upsert)  # [`python.upsert_patterns()`](analyzers/store_writer.py:55)
                    res.patterns_inserted_or_existing += len(ids)
                    logger.info(f"Rule {cls.__name__}: inserted/deduped {len(ids)} pattern(s) (genre={genre}, ts={time_sig})")
                except Exception as e:
                    logger.error("Failed to upsert patterns for rule %s: %s", cls.__name__, e)
                    logger.debug(traceback.format_exc())
                    res.failures += 1
    finally:
        if repo is not None:
            try:
                repo.close()
            except Exception:
                pass

    return res


# ---------------------------
# Main CLI
# ---------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    setup_logging(args.verbose)

    resolved_db_path = resolve_db_path(args.db_path)
    logger.info("Resolved DB path: %s", resolved_db_path)
    logger.info("Modes: ingest-midi=%s, ingest-hardcoded=%s, dry-run=%s", bool(args.ingest_midi), bool(args.ingest_hardcoded), bool(args.dry_run))

    total_files_processed = 0
    total_sources = 0
    total_patterns = 0
    total_failures = 0

    # MIDI ingestion
    if args.ingest_midi:
        midi_result = ingest_midi_files(
            file=Path(args.file).resolve() if args.file else None,
            midi_dir=Path(args.midi_dir).resolve(),
            recursive=bool(args.recursive),
            instrument_hint=args.instrument,
            genre=args.genre,
            mood=args.mood,
            db_path=args.db_path,
            tags=args.tags,
            dry_run=bool(args.dry_run),
        )
        total_files_processed += midi_result.files_processed
        total_sources += midi_result.sources_seen
        total_patterns += midi_result.patterns_inserted_or_existing
        total_failures += midi_result.failures

    # Hardcoded ingestion
    if args.ingest_hardcoded:
        hardcoded_result = ingest_hardcoded_rules(
            instrument=args.instrument,
            genre_override=args.genre,
            mood=args.mood,
            db_path=args.db_path,
            tags=args.tags,
            dry_run=bool(args.dry_run),
        )
        total_files_processed += hardcoded_result.classes_seen  # Treat each class as a "source unit" processed
        total_sources += hardcoded_result.sources_seen
        total_patterns += hardcoded_result.patterns_inserted_or_existing
        total_failures += hardcoded_result.failures

    # Summary
    logger.info("Summary:")
    logger.info("  Processed items: %d", total_files_processed)
    logger.info("  Sources created/upserted: %d", total_sources)
    if args.dry_run:
        logger.info("  Patterns (would insert): %d", total_patterns)
    else:
        logger.info("  Patterns inserted/deduped: %d", total_patterns)
    if total_failures:
        logger.warning("  Failures: %d", total_failures)

    # Exit code: 0 on success; non-zero if unexpected failures encountered
    return 0 if total_failures == 0 else 2


if __name__ == "__main__":
    sys.exit(main())