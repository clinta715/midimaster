import os
import sys
import json
import subprocess
from pathlib import Path

import pytest


def run_cli(cmd_args, cwd):
    return subprocess.run(
        [sys.executable, "main.py"] + cmd_args,
        cwd=str(cwd),
        capture_output=True,
        text=True,
    )


def test_cli_preset_workflow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    project_root = Path(__file__).resolve().parents[1]
    presets_dir = tmp_path / "presets"
    presets_dir.mkdir(parents=True, exist_ok=True)

    # Point the app at a temp presets directory
    monkeypatch.setenv("MIDIMASTER_PRESETS_DIR", str(presets_dir))

    # 1) Save a preset via CLI
    save_args = [
        "--genre", "jazz",
        "--tempo", "90",
        "--mood", "calm",
        "--bars", "2",
        "--save-preset", "preset_one",
        # keep generation short/minimal; do not pass --output so default naming is used
    ]
    res_save = run_cli(save_args, cwd=project_root)
    assert res_save.returncode == 0, f"Save preset failed: {res_save.stderr}"
    assert "Preset 'preset_one' saved." in (res_save.stdout or "")

    # Verify file created or indexed
    index_file = presets_dir / "index.json"
    assert index_file.exists(), "index.json not created in presets directory"
    idx = json.loads(index_file.read_text(encoding="utf-8"))
    assert "presets" in idx and "preset_one" in idx["presets"]

    # 2) List presets prints the saved name
    list_args = ["--list-presets"]
    res_list = run_cli(list_args, cwd=project_root)
    assert res_list.returncode == 0, f"List presets failed: {res_list.stderr}"
    listed = (res_list.stdout or "").strip().splitlines()
    assert "preset_one" in listed

    # 3) Load preset and ensure CLI overrides beat loaded preset (e.g., tempo)
    #    We set the preset tempo=90, then override with CLI tempo=150 and run a 1-bar generation
    load_args = [
        "--load-preset", "preset_one",
        "--tempo", "150",
        "--bars", "1",
        "--genre", "jazz",
        "--mood", "calm",
    ]
    res_load = run_cli(load_args, cwd=project_root)
    assert res_load.returncode == 0, f"Load preset with override failed: {res_load.stderr}"

    # Check that at least one output MIDI reflecting new run was created
    # Default naming includes genre_mood_tempo_timeSig_timestamp.mid
    # We look for 'jazz_calm_150_' in file names inside the project output dir
    out_dir = project_root / "output"
    # Allow some time for filesystem? Not necessary; the run is synchronous.
    candidates = [p.name for p in out_dir.glob("*.mid") if "jazz_calm_150_" in p.name]
    assert candidates, "Expected an output file with tempo 150, none found"