from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Local imports
from core.settings_preset_manager import SettingsPresetManager
from core.rhythms_db_resolver import RhythmsDbResolver

# Note: Keep this module pure and without global state. All functions return new values.


# Defaults aligned with main.py argparse defaults for backward compatibility detection
_DEFAULTS: Dict[str, Any] = {
    "genre": "pop",
    "mood": "happy",
    "tempo": 120,
    "bars": 16,
}


def load_settings_json(path: str | Path = "configs/settings.json") -> Dict[str, Any]:
    """
    Load optional user settings JSON. Returns {} if not present or invalid.
    Supported keys in this subtask:
      - filename_template: Optional[str]
      - rhythms_db_path: Optional[str] (read only; persistence handled elsewhere)
      - default_preset: Optional[str]
    """
    p = Path(path)
    if not p.exists():
        return {}
    try:
        raw = p.read_text(encoding="utf-8")
        data = json.loads(raw) if raw.strip() else {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _load_temp_settings_json(path: str | Path = "configs/temp_settings.json") -> Dict[str, Any]:
    """
    Load legacy temp settings JSON if present (read-only). Returns {} on failure.
    Only rhythms_db_path is considered in this subtask.
    """
    p = Path(path)
    if not p.exists():
        return {}
    try:
        raw = p.read_text(encoding="utf-8")
        data = json.loads(raw) if raw.strip() else {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def merge_settings(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pure merge: return a new dict with override values applied on top of base.
    Shallow merge is sufficient for this subtask.
    """
    merged = dict(base or {})
    if override:
        for k, v in override.items():
            merged[k] = v
    return merged


def _is_explicit_cli(value: Any, default: Any) -> bool:
    """
    Heuristic: consider CLI flag as explicitly provided when its value differs from the parser default.
    Note: If a user passes exactly the default value, this returns False and preset/config may override.
    This behavior is acceptable for the current subtask and tests.
    """
    try:
        return value is not None and value != default
    except Exception:
        return False


def resolve_effective_settings(cli_args) -> Tuple[Dict[str, Any], Optional[str], Optional[str]]:
    """
    Resolve/merge effective settings according to priority (low -> high):

      1) Built-in defaults and legacy defaults
      2) configs/settings.json (filename_template, rhythms_db_path, default_preset)
      3) configs/temp_settings.json (legacy; rhythms_db_path only)
      4) --load-preset NAME (or default_preset from settings.json if present)
         - Loaded contents merged on top, unknown keys preserved
         - Validate preset; on invalid, raise ValueError with message(s)
      5) CLI flags (--filename-template, --rhythms-db) override all above
         For core fields like tempo/genre/mood/bars, explicit CLI overrides win over preset

    Returns:
      (effective_settings_dict, filename_template_or_None, rhythms_db_path_or_None)
    """
    # 1) Start with built-in defaults
    effective: Dict[str, Any] = {
        "genre": _DEFAULTS["genre"],
        "mood": _DEFAULTS["mood"],
        "tempo": _DEFAULTS["tempo"],
        "bars": _DEFAULTS["bars"],
    }

    # 2) Load settings.json and pick up supported keys for later
    user_cfg = load_settings_json("configs/settings.json")
    cfg_filename_template = user_cfg.get("filename_template") if isinstance(user_cfg, dict) else None
    cfg_rhythms_db_from_settings = user_cfg.get("rhythms_db_path") if isinstance(user_cfg, dict) else None
    cfg_default_preset = user_cfg.get("default_preset") if isinstance(user_cfg, dict) else None

    # 3) Load legacy temp_settings.json for rhythms_db_path fallback
    legacy_cfg = _load_temp_settings_json("configs/temp_settings.json")
    cfg_rhythms_db_from_legacy = legacy_cfg.get("rhythms_db_path") if isinstance(legacy_cfg, dict) else None

    # 4) Preset load (explicit CLI preset first, else default_preset from settings.json)
    preset_to_load: Optional[str] = getattr(cli_args, "load_preset", None) or cfg_default_preset
    if preset_to_load:
        preset_dir = os.environ.get("MIDIMASTER_PRESETS_DIR", "configs/presets")
        spm = SettingsPresetManager(preset_dir=preset_dir)
        preset_data = spm.load_preset(preset_to_load)
        if preset_data is None:
            raise ValueError(f"Preset not found: {preset_to_load}")
        # Validate preset
        errors = spm.validate_preset(spm.normalize_preset(preset_data))
        if errors:
            raise ValueError("Invalid preset:\n" + "\n".join(f"- {e}" for e in errors))
        # Merge normalized preset on top of defaults
        normalized = spm.normalize_preset(preset_data)
        effective = merge_settings(effective, normalized)

    # 5) Apply explicit CLI overrides for core fields (tempo/genre/mood/bars) if provided
    #    Use heuristic comparing to known argparse defaults from main.py.
    cli_genre = getattr(cli_args, "genre", _DEFAULTS["genre"])
    if _is_explicit_cli(cli_genre, _DEFAULTS["genre"]):
        effective["genre"] = cli_genre

    cli_mood = getattr(cli_args, "mood", _DEFAULTS["mood"])
    if _is_explicit_cli(cli_mood, _DEFAULTS["mood"]):
        effective["mood"] = cli_mood

    cli_tempo = getattr(cli_args, "tempo", _DEFAULTS["tempo"])
    if _is_explicit_cli(cli_tempo, _DEFAULTS["tempo"]):
        effective["tempo"] = int(cli_tempo)

    cli_bars = getattr(cli_args, "bars", _DEFAULTS["bars"])
    if _is_explicit_cli(cli_bars, _DEFAULTS["bars"]):
        effective["bars"] = int(cli_bars)

    # Filename template resolution:
    # CLI --filename-template overrides; otherwise settings.json; otherwise preset; else None.
    filename_template: Optional[str] = None
    cli_template = getattr(cli_args, "filename_template", None)
    if isinstance(cli_template, str) and cli_template.strip():
        filename_template = cli_template.strip()
    elif isinstance(cfg_filename_template, str) and cfg_filename_template.strip():
        filename_template = cfg_filename_template.strip()
    else:
        # Try preset-provided template (if any) after normalization
        ft = effective.get("filename_template")
        if isinstance(ft, str) and ft.strip():
            filename_template = ft.strip()

    # Rhythms DB path resolution uses the dedicated resolver in priority order:
    # override (CLI) -> env -> settings.json -> temp_settings.json -> default ./reference_midis
    resolver = RhythmsDbResolver(settings_dir="configs")
    override_path = getattr(cli_args, "rhythms_db", None)
    rhythms_db_path_abs = resolver.get_rhythms_db_path(override=override_path)

    # Return a copy of effective (pure) and resolved auxiliary values
    return dict(effective), filename_template, str(rhythms_db_path_abs)