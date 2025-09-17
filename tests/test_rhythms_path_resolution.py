import os
import json
from pathlib import Path
import pytest

from core.rhythms_db_resolver import RhythmsDbResolver, coerce_to_path, find_first_midi


def make_dir_with_mid(tmp_path: Path, name: str = "db") -> Path:
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    # create a tiny .mid file (content not parsed by resolver)
    f = d / "dummy.mid"
    f.write_bytes(b"\x00")
    return d


def test_coerce_to_path_and_find_first_midi(tmp_path: Path):
    d = make_dir_with_mid(tmp_path)
    # coerce
    p = coerce_to_path(str(d))
    assert isinstance(p, Path)
    assert p.exists()
    # find
    found = find_first_midi(d)
    assert found is not None
    assert found.suffix == ".mid"


def test_validate_path_success(tmp_path: Path):
    d = make_dir_with_mid(tmp_path)
    resolver = RhythmsDbResolver(settings_dir=tmp_path / "configs")
    ok, msg = resolver.validate_path(d)
    assert ok, msg
    assert msg == "OK"


def test_validate_path_nonexistent(tmp_path: Path):
    d = tmp_path / "does_not_exist"
    resolver = RhythmsDbResolver(settings_dir=tmp_path / "configs")
    ok, msg = resolver.validate_path(d)
    assert not ok
    assert "does not exist" in msg


def test_validate_path_empty_dir(tmp_path: Path):
    d = tmp_path / "empty_dir"
    d.mkdir(parents=True, exist_ok=True)
    resolver = RhythmsDbResolver(settings_dir=tmp_path / "configs")
    ok, msg = resolver.validate_path(d)
    assert not ok
    assert "No .mid files found" in msg


def test_resolution_priority_override_arg_over_env_and_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Prepare three different dirs: config D1, env D2, override D3
    d1 = make_dir_with_mid(tmp_path, "config_db")
    d2 = make_dir_with_mid(tmp_path, "env_db")
    d3 = make_dir_with_mid(tmp_path, "override_db")

    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    settings_file = cfg_dir / "settings.json"
    settings_file.write_text(json.dumps({"rhythms_db_path": str(d1)}), encoding="utf-8")

    monkeypatch.setenv("MIDIMASTER_RHYTHMS_DB", str(d2))

    resolver = RhythmsDbResolver(settings_dir=cfg_dir)
    chosen = resolver.get_rhythms_db_path(override=d3)
    assert chosen == d3.resolve()


def test_resolution_priority_instance_override_over_env_and_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    d1 = make_dir_with_mid(tmp_path, "config_db")
    d2 = make_dir_with_mid(tmp_path, "env_db")
    d3 = make_dir_with_mid(tmp_path, "instance_override_db")

    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "settings.json").write_text(json.dumps({"rhythms_db_path": str(d1)}), encoding="utf-8")
    monkeypatch.setenv("MIDIMASTER_RHYTHMS_DB", str(d2))

    resolver = RhythmsDbResolver(override=d3, settings_dir=cfg_dir)
    chosen = resolver.get_rhythms_db_path()
    assert chosen == d3.resolve()


def test_resolution_priority_env_over_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    d1 = make_dir_with_mid(tmp_path, "config_db")
    d2 = make_dir_with_mid(tmp_path, "env_db")

    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "settings.json").write_text(json.dumps({"rhythms_db_path": str(d1)}), encoding="utf-8")
    monkeypatch.setenv("MIDIMASTER_RHYTHMS_DB", str(d2))

    resolver = RhythmsDbResolver(settings_dir=cfg_dir)
    chosen = resolver.get_rhythms_db_path()
    assert chosen == d2.resolve()


def test_resolution_priority_config_settings_json(tmp_path: Path):
    d1 = make_dir_with_mid(tmp_path, "config_db")
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "settings.json").write_text(json.dumps({"rhythms_db_path": str(d1)}), encoding="utf-8")

    resolver = RhythmsDbResolver(settings_dir=cfg_dir)
    chosen = resolver.get_rhythms_db_path()
    assert chosen == d1.resolve()


def test_resolution_priority_config_temp_settings_json_fallback(tmp_path: Path):
    d1 = make_dir_with_mid(tmp_path, "config_db")
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    # No settings.json; provide temp_settings.json
    (cfg_dir / "temp_settings.json").write_text(json.dumps({"rhythms_db_path": str(d1)}), encoding="utf-8")

    resolver = RhythmsDbResolver(settings_dir=cfg_dir)
    chosen = resolver.get_rhythms_db_path()
    assert chosen == d1.resolve()


def test_resolution_default_reference_midis_absolute(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    # Ensure no env and no configs influence
    monkeypatch.delenv("MIDIMASTER_RHYTHMS_DB", raising=False)
    resolver = RhythmsDbResolver(settings_dir=None)
    chosen = resolver.get_rhythms_db_path()
    assert isinstance(chosen, Path)
    assert chosen.is_absolute()
    # default points to ./reference_midis resolved from CWD; we just require absolute normalization


def test_set_config_path_persists_and_round_trips(tmp_path: Path):
    target = make_dir_with_mid(tmp_path, "persisted_db")
    cfg_dir = tmp_path / "configs"

    resolver = RhythmsDbResolver(settings_dir=cfg_dir)
    # Write
    resolver.set_config_path(target)

    settings_file = cfg_dir / "settings.json"
    assert settings_file.exists()
    data = json.loads(settings_file.read_text(encoding="utf-8"))
    assert data.get("rhythms_db_path") == str(target)

    # Now a clean resolver with same settings_dir should resolve from config
    resolver2 = RhythmsDbResolver(settings_dir=cfg_dir)
    chosen = resolver2.get_rhythms_db_path()
    assert chosen == target.resolve()


def test_validation_with_hidden_artifacts_is_ignored(tmp_path: Path):
    d = tmp_path / "db_hidden"
    d.mkdir(parents=True, exist_ok=True)
    # Hidden/system file should be ignored
    (d / "._artifact.mid").write_bytes(b"\x00")
    resolver = RhythmsDbResolver(settings_dir=tmp_path / "configs")
    ok, msg = resolver.validate_path(d)
    assert not ok
    assert "No .mid files found" in msg

    # Add a real midi file; should pass
    (d / "real.mid").write_bytes(b"\x00")
    ok2, msg2 = resolver.validate_path(d)
    assert ok2, msg2
    assert msg2 == "OK"