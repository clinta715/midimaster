from pathlib import Path
from copy import deepcopy

from core.settings_preset_manager import SettingsPresetManager


def test_migration_from_legacy_dict(tmp_path: Path):
    preset_dir = tmp_path / "presets"
    mgr = SettingsPresetManager(str(preset_dir))

    # Legacy-like dict missing version and filename_template, with minimal required fields
    legacy = {
        "genre": "rock",
        "mood": "energetic",
        "tempo": 140,
        "bars": 24,
        "time_signature": "4/4",
        "complexity": "dense",
        "rhythms_db_path": "data/rhythms_db.json",
    }
    legacy_copy = deepcopy(legacy)

    # Normalize should add version and filename_template and preserve fields
    normalized = mgr.normalize_preset(legacy)
    assert normalized["version"] == mgr.SCHEMA_VERSION
    assert "filename_template" in normalized
    # Ensure original was not mutated
    assert legacy == legacy_copy

    # Should validate cleanly after normalization
    assert mgr.validate_preset(normalized) == []

    # Save should persist successfully
    assert mgr.save_preset("legacy_ported", legacy) is True
    loaded = mgr.load_preset("legacy_ported")
    assert isinstance(loaded, dict)
    # Loaded form must equal normalization of original legacy dict
    assert loaded == mgr.normalize_preset(legacy_copy)


def test_saving_legacy_settings_adds_version_without_mutation(tmp_path: Path):
    preset_dir = tmp_path / "presets"
    mgr = SettingsPresetManager(str(preset_dir))

    legacy = {
        "genre": "pop",
        "mood": "happy",
        "tempo": 120,
        "bars": 16,
        "time_signature": "3/4",
        "complexity": "balanced",
        "rhythms_db_path": "data/rhythms_db.json",
        # no version, no filename_template
    }
    before = deepcopy(legacy)
    assert mgr.save_preset("legacy_no_mutate", legacy) is True
    # Caller dict must be unchanged
    assert legacy == before

    loaded = mgr.load_preset("legacy_no_mutate")
    assert isinstance(loaded, dict)
    assert loaded["version"] == mgr.SCHEMA_VERSION
    assert "filename_template" in loaded


def test_validation_allows_extra_unknown_fields(tmp_path: Path):
    preset_dir = tmp_path / "presets"
    mgr = SettingsPresetManager(str(preset_dir))

    data = {
        "genre": "jazz",
        "mood": "calm",
        "tempo": 90,
        "bars": 12,
        "time_signature": "6/8",
        "complexity": "sparse",
        "rhythms_db_path": "data/rhythms_db.json",
        "version": mgr.SCHEMA_VERSION,
        # Unknown/extra fields should be preserved and not rejected
        "extra_knob": 0.75,
        "nested": {"x": 1, "y": "z"},
    }
    assert mgr.validate_preset(data) == []
    assert mgr.save_preset("with_extras", data) is True
    loaded = mgr.load_preset("with_extras")
    assert isinstance(loaded, dict)
    assert loaded["extra_knob"] == 0.75
    assert loaded["nested"] == {"x": 1, "y": "z"}