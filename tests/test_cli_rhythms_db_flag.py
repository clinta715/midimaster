import os
import sys
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.config_loader import resolve_effective_settings
from core.rhythms_db_resolver import RhythmsDbResolver


def _make_dir_with_mid(tmp_path: Path) -> Path:
    d = tmp_path / "rhythms"
    d.mkdir(parents=True, exist_ok=True)
    (d / "dummy.mid").write_bytes(b"\x00")
    return d


def test_rhythms_db_resolves_from_cli_namespace(tmp_path: Path):
    # Create a valid rhythms DB directory with a .mid
    db_dir = _make_dir_with_mid(tmp_path)

    # Minimal CLI args namespace (match main.py defaults for unrelated args)
    args = SimpleNamespace(
        # Core
        genre="pop",
        mood="happy",
        tempo=120,
        bars=16,
        # New flags
        filename_template=None,
        rhythms_db=str(db_dir),
        load_preset=None,
        save_preset=None,
        list_presets=False,
    )

    effective, filename_template, rhythms_db_path = resolve_effective_settings(args)
    assert rhythms_db_path is not None
    assert Path(rhythms_db_path).resolve() == db_dir.resolve()

    # Validate via resolver
    resolver = RhythmsDbResolver(settings_dir="configs")
    ok, msg = resolver.validate_path(Path(rhythms_db_path))
    assert ok, msg


def test_invalid_rhythms_db_cli_exits_nonzero(tmp_path: Path):
    # Non-existent path (no directory created)
    bad_dir = tmp_path / "does_not_exist"

    project_root = Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        "main.py",
        "--genre", "pop",
        "--tempo", "120",
        "--mood", "happy",
        "--bars", "1",
        "--rhythms-db", str(bad_dir),
        "--output", str(tmp_path / "out.mid"),
    ]
    result = subprocess.run(cmd, cwd=str(project_root), capture_output=True, text=True)

    assert result.returncode != 0
    assert "Invalid --rhythms-db" in (result.stderr or "")