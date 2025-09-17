from pathlib import Path
from typing import List, Tuple

from gui.settings_helpers import validate_rhythms_path, persist_rhythms_default
from core.rhythms_db_resolver import RhythmsDbResolver


def test_validate_rhythms_path_valid(tmp_path: Path):
    # Create a valid directory with at least one .mid file
    midi_dir = tmp_path / "ref_midis"
    midi_dir.mkdir(parents=True, exist_ok=True)
    (midi_dir / "example.mid").write_bytes(b"mThd")  # minimal header marker, existence is enough for our validate

    ok, msg = validate_rhythms_path(str(midi_dir))
    assert ok is True
    assert msg == "OK"


def test_validate_rhythms_path_invalid(tmp_path: Path):
    # Non-existent directory
    missing = tmp_path / "missing_dir"
    ok, msg = validate_rhythms_path(str(missing))
    assert ok is False
    assert "does not exist" in msg.lower()

    # File instead of directory
    some_file = tmp_path / "file.txt"
    some_file.write_text("hi")
    ok2, msg2 = validate_rhythms_path(str(some_file))
    assert ok2 is False
    assert "not a directory" in msg2.lower()


def test_persist_rhythms_default_calls_resolver(tmp_path: Path, monkeypatch):
    # Capture calls made to RhythmsDbResolver.set_config_path
    calls: List[Path] = []

    def fake_set_config_path(self, path: Path) -> None:  # noqa: ANN001
        calls.append(Path(path))

    monkeypatch.setattr(RhythmsDbResolver, "set_config_path", fake_set_config_path, raising=True)

    # Use a sample path; the function should call resolver.set_config_path with it
    sample_dir = tmp_path / "db"
    # Do not require this to exist for persistence call; validate is separate
    persist_rhythms_default(str(sample_dir))

    assert len(calls) == 1
    # The helper coerces path via Path(...) but should preserve the value
    assert calls[0] == sample_dir