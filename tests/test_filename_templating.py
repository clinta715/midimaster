import re
from pathlib import Path

import pytest

from core.filename_templater import (
    format_filename,
    validate_template,
    resolve_placeholders,
    ensure_unique,
    sanitize_component,
)


def test_basic_formatting_and_sanitization(tmp_path: Path):
    template = "{genre}_{mood}_{tempo}_{bars}_{timestamp}_{stem}.mid"
    settings = {"genre": "hip-hop", "mood": "energetic!", "tempo": 120, "bars": 16}
    context = {"stem": "melody"}

    path = format_filename(template, settings, context, base_dir=tmp_path)
    assert path.suffix == ".mid"
    assert path.parent == tmp_path

    # energetic! should sanitize to energetic; genre keeps dash
    name = path.name
    assert name.startswith("hip-hop_energetic_120_16_")
    assert name.endswith("_melody.mid")

    # timestamp format: YYYYMMDD_HHMMSS
    m = re.search(r"\d{8}_\d{6}", name)
    assert m, f"timestamp not found in {name}"


def test_validate_template_unknown_placeholder():
    ok, reason = validate_template("{genre}_{unknown}_{tempo}.mid")
    assert not ok
    assert "Unknown placeholders" in reason


def test_collision_appends_counter(tmp_path: Path):
    template = "{genre}_{mood}.mid"
    settings = {"genre": "jazz", "mood": "calm"}
    # Fix timestamp for deterministic collisions
    context = {"timestamp": "20240101_010101"}

    # First path
    p1 = format_filename(template, settings, context, base_dir=tmp_path)
    p1.parent.mkdir(parents=True, exist_ok=True)
    p1.touch()

    # Second call should append _2
    p2 = format_filename(template, settings, context, base_dir=tmp_path)
    assert p2.stem == f"{p1.stem}_2"
    assert p2.exists() is False  # ensure_unique returns a non-existing path


def test_unique_id_fallback_after_99(tmp_path: Path):
    template = "song.mid"  # constant template to force same stem
    settings = {}
    context = {"timestamp": "20240101_010101"}  # deterministic

    base = format_filename(template, settings, context, base_dir=tmp_path)
    base.parent.mkdir(parents=True, exist_ok=True)

    stem = base.stem
    # Create base and _2.._99
    (base).touch()
    for i in range(2, 100):
        (base.parent / f"{stem}_{i}.mid").touch()

    # Now this should fall back to _<8-char uid>
    p = format_filename(template, settings, context, base_dir=tmp_path)
    assert re.fullmatch(rf"{re.escape(stem)}_[a-f0-9]{{8}}", p.stem), f"unexpected stem {p.stem}"
    assert p.parent == base.parent


def test_subdirectory_creation_with_path_components(tmp_path: Path):
    template = "{genre}/songs/{mood}_{tempo}.mid"
    settings = {"genre": "Pop", "mood": "Happy*", "tempo": 120}

    p = format_filename(template, settings, base_dir=tmp_path)
    # Expect sanitized directory structure to exist (created by ensure_unique)
    assert p.parent.exists()
    # Genre sanitized component (case preserved) and nested folder "songs"
    assert p.parent == tmp_path / "Pop" / "songs"
    assert p.name.startswith("Happy_120") and p.suffix == ".mid"