import os
from gui.settings_helpers import build_preview_filename, validate_template_str


def test_validate_template_ok_and_unknown():
    ok, reason = validate_template_str("{genre}_{mood}_{tempo}_{bars}")
    assert ok and reason == ""

    bad_ok, bad_reason = validate_template_str("{unknown}_{genre}")
    assert bad_ok is False
    assert "Unknown placeholders" in bad_reason


def test_build_preview_filename_basic():
    settings = {"genre": "pop", "mood": "happy", "tempo": 120, "bars": 16}
    template = "{genre}/{mood}/{genre}_{mood}_{tempo}_{bars}"
    preview = build_preview_filename(settings, template)
    # Preview is pure string with sanitized components, no I/O
    assert isinstance(preview, str)
    assert preview.endswith(".mid")
    # Should include placeholders resolved
    assert "pop" in preview
    assert "happy" in preview
    assert "120" in preview
    assert "16" in preview
    # Should not contain backslashes regardless of platform
    assert "\\" not in preview


def test_build_preview_filename_with_base_dir_prefix_only_str_join():
    settings = {"genre": "rock", "mood": "energetic", "tempo": 160, "bars": 32}
    template = "{genre}_{mood}_{tempo}_{bars}"
    preview = build_preview_filename(settings, template, base_dir="output")
    # No I/O guarantees, but base_dir should prefix the relative path
    assert isinstance(preview, str)
    assert preview.endswith(".mid")
    # Accept both separators in assertion for cross-platform
    assert preview.replace("\\", "/").startswith("output/")