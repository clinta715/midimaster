import os
import sys
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


def test_cli_filename_template_creates_expected_path(tmp_path: Path):
    project_root = Path(__file__).resolve().parents[1]

    # Use a minimal generation with 1 bar to be fast, no explicit --output so templater is used
    template = "test_tpl/{genre}_{mood}_{tempo}_{bars}_{stem}"

    args = [
        "--genre", "pop",
        "--mood", "happy",
        "--tempo", "120",
        "--bars", "1",
        "--filename-template", template,
    ]

    res = run_cli(args, cwd=project_root)
    assert res.returncode == 0, f"CLI failed: {res.stderr}"

    # Expect file under output/test_tpl/pop_happy_120_1_combined.mid (templater ensures .mid, uniqueness, dirs)
    expected_dir = project_root / "output" / "test_tpl"
    # Gather candidates for combined file
    candidates = list(expected_dir.glob("pop_happy_120_1_combined*.mid"))
    assert candidates, f"No templated output found in {expected_dir}"


def test_cli_filename_template_invalid_exits(tmp_path: Path):
    project_root = Path(__file__).resolve().parents[1]

    # Unknown placeholder should be rejected
    bad_template = "{genre}_{unknown_key}"

    args = [
        "--genre", "pop",
        "--mood", "happy",
        "--tempo", "120",
        "--bars", "1",
        "--filename-template", bad_template,
    ]

    res = run_cli(args, cwd=project_root)
    assert res.returncode != 0, "CLI should fail for invalid template"
    assert "Invalid filename template" in (res.stderr or "") or "Invalid filename template" in (res.stdout or "")