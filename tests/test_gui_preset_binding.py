from core.settings_preset_manager import SettingsPresetManager


def test_preset_save_list_load_delete(tmp_path):
    mgr = SettingsPresetManager(preset_dir=str(tmp_path / "presets"))

    # Minimal valid settings (manager fills defaults/normalizes)
    settings = {
        "genre": "pop",
        "mood": "happy",
        "tempo": 120,
        "bars": 16,
        "time_signature": "4/4",
        "complexity": "balanced",
        "filename_template": "{genre}_{mood}_{tempo}_{bars}",
        "rhythms_db_path": "reference_midis",
        "version": "1.0",
    }

    # Save
    ok = mgr.save_preset("MyPreset", settings)
    assert ok is True

    # List contains the preset
    names = mgr.list_presets()
    assert "MyPreset" in names

    # Load and validate shape
    data = mgr.load_preset("MyPreset")
    assert isinstance(data, dict)
    assert data.get("genre") == "pop"
    assert data.get("mood") == "happy"

    # Validate normalized preset has no errors
    normalized = mgr.normalize_preset(data)
    errors = mgr.validate_preset(normalized)
    assert errors == []

    # Save same name again (update) should still succeed and keep single entry
    ok2 = mgr.save_preset("MyPreset", {**settings, "tempo": 130})
    assert ok2 is True
    names2 = mgr.list_presets()
    assert "MyPreset" in names2

    # Delete preset
    deleted = mgr.delete_preset("MyPreset")
    assert deleted is True
    names3 = mgr.list_presets()
    # Allow index/scan differences; at least name should not be present
    assert "MyPreset" not in names3