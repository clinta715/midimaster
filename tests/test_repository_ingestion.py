import os
import pytest

from analyzers.analysis_api import analyze_midi_and_store
from data_store.pattern_repository import PatternRepository


def test_analyzer_to_store_ingestion(tmp_path, monkeypatch):
    # 1) Use a tmp SQLite DB via env var to isolate state
    db_path = tmp_path / "pattern_store.sqlite"
    monkeypatch.setenv("MIDIMASTER_DB_PATH", str(db_path))

    # 2) Pick an existing MIDI in-repo (prefer small, deterministic ones)
    candidates = [
        "test_dnb.mid",
        "test_gui.mid",
        "output/pop_energetic_balanced_tempo120_bars16_run1.mid",
        "output/classical_energetic.mid",
    ]
    midi_path = next((p for p in candidates if os.path.exists(p)), None)
    assert midi_path is not None, "No MIDI file found in repository for ingestion test"

    # 3) Analyze and ingest patterns into the tmp repository
    result = analyze_midi_and_store(
        midi_path,
        instrument_hint="drums",
        genre="pop",
        mood=None,
        db_path=str(db_path),
    )
    assert isinstance(result.get("source_id"), int)

    # 4) Open repository pointing at the same tmp DB
    repo = PatternRepository(db_path=str(db_path))

    # 5) Verify stats show data present
    stats = repo.get_pattern_stats()
    assert stats["total_sources"] > 0, "Expected at least one source after ingestion"
    assert stats["total_patterns"] > 0, "Expected at least one stored pattern after ingestion"

    # 6) Optional: sanity check a query for pop 4/4 (resilient - not required to be non-empty)
    try:
        _ = repo.find_rhythm_patterns(
            instrument="drums",
            genre="pop",
            time_signature="4/4",
            limit=1,
        )
    except Exception as e:
        pytest.fail(f"Repository query failed unexpectedly: {e}")