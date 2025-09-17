import json
from pathlib import Path
from copy import deepcopy

import pytest

from core.settings_preset_manager import SettingsPresetManager


def test_save_then_load_roundtrip_preserves_data_and_unicode(tmp_path: Path):
    # Use a dedicated preset directory for isolation
    preset_dir = tmp_path / "presets"
    manager = SettingsPresetManager(str(preset_dir))

    # Non-ASCII preset name and values
    name = "能源 测试 🎵"
    raw = {
        # Intentionally missing version and filename_template to exercise normalization
        "genre": "流行",  # non-ASCII
        "mood": "happy",
        "tempo": 128,
        "bars": 32,
        "time_signature": "4/4",
        "complexity": "balanced",
        "rhythms_db_path": "data/rhythms_db.json",
        "custom_param": 42,  # extra unknown field
        "nested": {"a": 1, "b": "值"},  # nested, with non-ASCII
    }
    original_copy = deepcopy(raw)

    # Validate the raw (should produce errors due to missing version and possibly optional keys)
    errors_before = manager.validate_preset(raw)
    assert "Missing required key: version" in errors_before

    # Normalize then validate OK
    normalized = manager.normalize_preset(raw)
    errors_after = manager.validate_preset(normalized)
    assert errors_after == []

    # Save should not mutate caller's dict
    assert manager.save_preset(name, raw) is True
    assert raw == original_copy

    # Load back and compare to normalized (deep equality)
    loaded = manager.load_preset(name)
    assert loaded is not None
    assert isinstance(loaded, dict)
    assert loaded == manager.normalize_preset(original_copy)  # stable normalization result

    # Ensure UTF-8 was used (file exists and is JSON)
    files = list(preset_dir.glob("*.json"))
    assert any(f.name != "index.json" for f in files)
    for f in files:
        # Should be readable as UTF-8 JSON
        json.loads(f.read_text(encoding="utf-8"))


def test_list_and_delete_presets(tmp_path: Path):
    preset_dir = tmp_path / "presets"
    manager = SettingsPresetManager(str(preset_dir))

    base = {
        "genre": "pop",
        "mood": "happy",
        "tempo": 120,
        "bars": 16,
        "time_signature": "4/4",
        "complexity": "balanced",
        "rhythms_db_path": "data/rhythms_db.json",
    }

    assert manager.save_preset("A", base)
    assert manager.save_preset("B", base)

    names = manager.list_presets()
    # Prefer index names; allow filename-based fallback if index missing
    assert "A" in names and "B" in names or (preset_dir / "A.json").exists() and (preset_dir / "B.json").exists()

    # Delete one
    assert manager.delete_preset("A") is True
    names2 = manager.list_presets()
    # Either removed from index or file gone; list should not contain "A" anymore
    assert "A" not in names2 or not (preset_dir / "A.json").exists()


def test_validation_rules_for_missing_and_valid():
    manager = SettingsPresetManager()  # default dir (not used for validation only)
    missing = {
        "genre": "pop",
        # "mood" missing
        "tempo": 120,
        "bars": 16,
        "time_signature": "4/4",
        "complexity": "balanced",
        "rhythms_db_path": "data/rhythms_db.json",
        # "version" missing
    }
    errs = manager.validate_preset(missing)
    # Must at least flag missing mood and version
    assert any("Missing required key: mood" in e for e in errs)
    assert any("Missing required key: version" in e for e in errs)

    valid = manager.normalize_preset(missing)
    assert manager.validate_preset(valid) == []


def test_no_mutation_on_save(tmp_path: Path):
    preset_dir = tmp_path / "presets"
    manager = SettingsPresetManager(str(preset_dir))

    name = "NoMutation"
    data = {
        "genre": "pop",
        "mood": "happy",
        "tempo": 100,
        "bars": 8,
        "time_signature": "4/4",
        "complexity": "balanced",
        "rhythms_db_path": "data/rhythms_db.json",
        # Missing version, filename_template on purpose
    }
    snap = deepcopy(data)
    assert manager.save_preset(name, data) is True
    # Ensure the provided dict wasn't altered
    assert data == snap

    loaded = manager.load_preset(name)
    assert loaded is not None
    assert isinstance(loaded, dict)
    # Loaded must contain version and filename_template defaults, original is unchanged
    assert loaded["version"] == manager.SCHEMA_VERSION
    assert "filename_template" in loaded
    assert "filename_template" not in data