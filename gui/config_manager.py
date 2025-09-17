"""
Lightweight GUI configuration manager.

- Persists temporary GUI settings such as user_key/mode expected by CLI (main.py)
- Tracks the last generated song path for auto-loading on startup
- Stores settings in config/gui_settings.json
- Provides generic load_config/save_config helpers for arbitrary JSON configs
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime


class ConfigManager:
    """
    Simple JSON-backed settings manager for the GUI.

    Public API (compatible with existing usage in main.py and GUI):
      - load_temp_settings() -> Dict[str, Any]
      - save_temp_settings(settings: Dict[str, Any]) -> None
      - get_last_song_path() -> str
      - set_last_song_path(path: str) -> None
      - load_config(path: str) -> Dict[str, Any]
      - save_config(data: Dict[str, Any], path: str) -> None
    """

    def __init__(self, settings_path: Optional[str] = None) -> None:
        # Default path inside ./config
        self.settings_path = Path(settings_path) if settings_path else Path("config") / "gui_settings.json"
        self.settings_path.parent.mkdir(parents=True, exist_ok=True)

        # In-memory cache
        self._data: Dict[str, Any] = {}
        self._load_file()

    # ===== Public API used by CLI/main.py =====
    def load_temp_settings(self) -> Dict[str, Any]:
        """
        Return a dict of temporary settings, including:
          - user_key: Optional[str]
          - user_mode: Optional[str]
        """
        temp = self._data.get("temp_settings", {})
        if not isinstance(temp, dict):
            temp = {}
        # Ensure keys exist (main.py expects these keys when present)
        temp.setdefault("user_key", None)
        temp.setdefault("user_mode", None)
        return temp

    def save_temp_settings(self, settings: Dict[str, Any]) -> None:
        """
        Update temporary settings. Expected keys include:
          - user_key
          - user_mode
        """
        current = self._data.get("temp_settings", {})
        if not isinstance(current, dict):
            current = {}

        current.update(settings or {})
        self._data["temp_settings"] = current
        self._data["updated_at"] = datetime.now().isoformat(timespec="seconds")
        self._save_file()

    # ===== Extended GUI helpers =====
    def get_last_song_path(self) -> str:
        """
        Return the last generated song file path ('' if not found).
        """
        path = self._data.get("last_song_path", "")
        if isinstance(path, str):
            return path
        return ""

    def set_last_song_path(self, path: str) -> None:
        """
        Persist the last generated song file path.
        """
        self._data["last_song_path"] = str(path or "")
        self._data["updated_at"] = datetime.now().isoformat(timespec="seconds")
        self._save_file()

    # ===== Generic config helpers (for Save/Open menu actions) =====
    def load_config(self, path: str) -> Dict[str, Any]:
        """
        Load a JSON configuration file from an arbitrary path.
        Returns an empty dict on failure.
        """
        try:
            p = Path(path)
            raw = p.read_text(encoding="utf-8")
            data = json.loads(raw) if raw.strip() else {}
            if isinstance(data, dict):
                return data
            return {}
        except Exception:
            return {}

    def save_config(self, data: Dict[str, Any], path: str) -> None:
        """
        Save a JSON configuration file to an arbitrary path.
        Silently ignores failures to avoid crashing the GUI.
        """
        try:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(data or {}, indent=2, ensure_ascii=False)
            p.write_text(payload, encoding="utf-8")
        except Exception:
            pass

    # ===== Internal I/O =====
    def _load_file(self) -> None:
        try:
            if self.settings_path.exists():
                raw = self.settings_path.read_text(encoding="utf-8")
                self._data = json.loads(raw) if raw.strip() else {}
            else:
                self._data = {}
        except Exception:
            # Corrupted or unreadable settings; reset in memory
            self._data = {}

        # Minimal schema defaults
        self._data.setdefault("version", "1.0")
        self._data.setdefault("temp_settings", {"user_key": None, "user_mode": None})
        self._data.setdefault("last_song_path", "")

    def _save_file(self) -> None:
        try:
            payload = json.dumps(self._data, indent=2, ensure_ascii=False)
            self.settings_path.write_text(payload, encoding="utf-8")
        except Exception:
            # Silently ignore persistence failures to avoid crashing GUI/CLI
            pass