#!/usr/bin/env python3
"""
Lightweight repository helpers for analyzer modules.

Responsibilities:
- Resolve/open PatternRepository with a sensible default
- Upsert source rows for MIDI files
- Upsert extracted rhythm patterns

These helpers intentionally keep logic minimal to avoid circular dependencies.
"""

from typing import Optional, List, Dict, Any

from data_store.pattern_repository import PatternRepository


__all__ = [
    "get_repository",
    "upsert_source_for_midi",
    "upsert_patterns",
]


def get_repository(*, db_path: Optional[str] = None) -> PatternRepository:
    """
    Open or get a repository instance.

    Resolution order:
    - If db_path provided: use it
    - Else PatternRepository uses env MIDIMASTER_DB_PATH or defaults to data/pattern_store.sqlite
    """
    return PatternRepository(db_path=db_path)


def upsert_source_for_midi(
    repository: PatternRepository,
    midi_path: str,
    track_name: Optional[str] = None,
    *,
    extracted_by_version: Optional[str] = None,
) -> int:
    """
    Upsert the source row for a given MIDI file and optional track name.
    Returns source_id.
    """
    return repository.upsert_source(
        source_type="midi_file",
        source_path=midi_path,
        source_track=track_name,
        extracted_by_version=extracted_by_version,
    )


def upsert_patterns(
    repository: PatternRepository,
    source_id: int,
    patterns: List[Dict[str, Any]],
) -> List[int]:
    """
    Upsert a list of extracted pattern dicts into the repository.
    Returns list of (inserted or existing) pattern ids.
    """
    ids: List[int] = []

    for p in patterns:
        instr = (p.get("instrument") or "drums").strip()
        genre = (p.get("genre") or "unknown").strip()
        mood = p.get("mood")
        time_signature = (p.get("time_signature") or "4/4").strip()

        subdivision = int(p.get("subdivision") or 4)
        length_beats = float(p.get("length_beats") or 4.0)

        bpm_min = p.get("bpm_min")
        bpm_max = p.get("bpm_max")

        syncopation = float(p.get("syncopation") or 0.0)
        density = float(p.get("density") or 0.0)
        swing = p.get("swing")
        humanization = p.get("humanization")

        quality_score = p.get("quality_score")

        pattern_json = p.get("pattern_json") or []
        accent_profile_json = p.get("accent_profile_json")
        tags_json = p.get("tags_json")

        pid = repository.upsert_rhythm_pattern(
            source_id=source_id,
            instrument=instr,
            genre=genre,
            mood=mood,
            time_signature=time_signature,
            subdivision=subdivision,
            length_beats=length_beats,
            bpm_min=bpm_min,
            bpm_max=bpm_max,
            syncopation=syncopation,
            density=density,
            swing=swing,
            humanization=humanization,
            quality_score=quality_score,
            pattern_json=pattern_json,
            accent_profile_json=accent_profile_json,
            tags_json=tags_json,
        )
        ids.append(pid)

    return ids