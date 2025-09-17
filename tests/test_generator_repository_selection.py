import os
import logging
import pytest

from data_store.pattern_repository import PatternRepository
from generators.pattern_orchestrator import PatternOrchestrator
from genres.genre_factory import GenreFactory
from structures.song_skeleton import SongSkeleton


WARNING_FALLBACK_POP = "No compatible rhythm pattern found for pop. Using fallback pattern."


def _insert_pop_4_4_eighths_pattern(repo: PatternRepository, source_path: str = "unit://pop/straight_eighths") -> int:
    # Create a source
    source_id = repo.upsert_source(
        source_type="unit_test",
        source_path=source_path,
        source_track=None,
        extracted_by_version="test-1.0",
    )

    # Build simple straight-eighths events within one bar (4/4)
    # Onsets at 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5
    events = []
    for i in range(8):
        onset = i * 0.5
        events.append({
            "onset_beats": onset,
            "duration_beats": 0.0,
            "velocity": 100,
            "pitch": 36,
            "channel": 9
        })

    # Insert rhythm pattern with metadata aligned to repository query defaults
    pid = repo.upsert_rhythm_pattern(
        source_id=source_id,
        instrument="drums",
        genre="pop",
        mood=None,
        time_signature="4/4",
        subdivision=2,             # eighth-note grid per beat
        length_beats=4.0,          # one bar
        bpm_min=None,
        bpm_max=None,
        syncopation=0.5,           # offbeats on eighths => 0.5 fits widened range
        density=0.5,               # 8 hits / 4 beats = 2 hpB => 2/4 = 0.5
        swing=None,
        humanization=None,
        quality_score=0.9,
        pattern_json=events,
        accent_profile_json=None,
        tags_json=["unit", "loop-1bar"],
    )
    return pid


@pytest.mark.parametrize("mood", ["happy"])
def test_generator_uses_repository_first_avoids_fallback(tmp_path, monkeypatch, caplog, capsys, mood):
    # Fresh temp DB and environment isolation
    tmp_db_path = tmp_path / "pattern_store.sqlite"
    monkeypatch.setenv("MIDIMASTER_DB_PATH", str(tmp_db_path))

    # Prepare repository and insert a compatible POP 4/4 one-bar pattern
    repo = PatternRepository(db_path=str(tmp_db_path))
    _insert_pop_4_4_eighths_pattern(repo)

    # Construct minimal generation context for pop
    genre = "pop"
    tempo = 120
    genre_rules = GenreFactory.create_genre_rules(genre)
    song_skeleton = SongSkeleton(genre, tempo, mood)

    # Orchestrator with repository injected
    orch = PatternOrchestrator(
        genre_rules,
        mood,
        pattern_repository=repo
    )

    # Capture warnings and stdout
    with caplog.at_level(level=logging.WARNING):
        _ = orch.generate_beats_only(song_skeleton, num_bars=1, beat_complexity=0.5)
    out = capsys.readouterr().out

    # Assert the specific fallback warning is NOT present
    assert WARNING_FALLBACK_POP not in caplog.text
    assert WARNING_FALLBACK_POP not in out


@pytest.mark.parametrize("mood", ["happy"])
def test_legacy_fallback_when_repository_empty(tmp_path, monkeypatch, caplog, capsys, mood):
    # New empty DB (no patterns)
    tmp_db_path = tmp_path / "pattern_store.sqlite"
    monkeypatch.setenv("MIDIMASTER_DB_PATH", str(tmp_db_path))
    repo = PatternRepository(db_path=str(tmp_db_path))

    # Minimal generation context for pop
    genre = "pop"
    tempo = 120
    genre_rules = GenreFactory.create_genre_rules(genre)
    song_skeleton = SongSkeleton(genre, tempo, mood)

    orch = PatternOrchestrator(
        genre_rules,
        mood,
        pattern_repository=repo
    )

    with caplog.at_level(level=logging.WARNING):
        _ = orch.generate_beats_only(song_skeleton, num_bars=1, beat_complexity=0.5)
    out = capsys.readouterr().out

    # The legacy path prints the warning (via print), not logging; check stdout primarily.
    assert WARNING_FALLBACK_POP in out or WARNING_FALLBACK_POP in caplog.text