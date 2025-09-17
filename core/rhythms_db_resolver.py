from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional, Tuple, Union


def coerce_to_path(value: Union[str, os.PathLike]) -> Path:
    """
    Coerce a string or os.PathLike into a pathlib.Path with user expansion (e.g., "~").
    """
    if isinstance(value, Path):
        return value.expanduser()
    return Path(value).expanduser()


def find_first_midi(path: Path) -> Optional[Path]:
    """
    Find the first .mid file under the given directory (recursive).
    Skips hidden/system artifact files like '._*' commonly created by macOS.
    """
    try:
        for p in path.rglob("*.mid"):
            try:
                name = p.name
            except Exception:
                # In edge cases where name can't be read, skip
                continue
            # Skip AppleDouble files and hidden files
            if name.startswith("._") or name.startswith("."):
                continue
            if p.is_file():
                return p
    except Exception:
        # Any unexpected error while scanning is treated as "not found"
        return None
    return None


class RhythmsDbResolver:
    """
    Resolve the path to the MIDI rhythms database according to a strict order:

    1) Explicit override passed into the resolver instance or get_rhythms_db_path (highest priority).
    2) Environment variable MIDIMASTER_RHYTHMS_DB.
    3) User config value (if accessible without refactors):
       - configs/settings.json's "rhythms_db_path"
       - fallback to configs/temp_settings.json's "rhythms_db_path" if present
    4) Project default: "./reference_midis"

    All returned paths are absolute (normalized via resolve()).

    Validation is provided via validate_path(path) and is not enforced implicitly;
    callers can choose to validate or not based on their own needs to maintain
    backward-compatible behavior.
    """

    ENV_VAR = "MIDIMASTER_RHYTHMS_DB"

    def __init__(
        self,
        override: Optional[Union[str, os.PathLike]] = None,
        settings_dir: Optional[Union[str, os.PathLike]] = "configs",
    ) -> None:
        self._instance_override: Optional[Path] = coerce_to_path(override) if override is not None else None
        self._settings_dir: Optional[Path] = coerce_to_path(settings_dir) if settings_dir is not None else None

    def get_rhythms_db_path(self, override: Optional[Union[str, os.PathLike]] = None) -> Path:
        """
        Determine the rhythms DB directory path using the resolution order.

        Returns:
            Absolute Path to the chosen rhythms DB directory (may or may not be valid).
            Use validate_path() to verify.
        """
        # 1) Explicit override (call-level), else resolver instance override
        if override is not None:
            chosen = coerce_to_path(override)
            return self._to_abs(chosen)

        if self._instance_override is not None:
            return self._to_abs(self._instance_override)

        # 2) Environment variable
        env_val = os.environ.get(self.ENV_VAR)
        if env_val:
            return self._to_abs(coerce_to_path(env_val))

        # 3) User config value
        cfg_path = self._read_config_path()
        if cfg_path is not None:
            return self._to_abs(cfg_path)

        # 4) Default project directory
        return self._to_abs(Path("./reference_midis"))

    def validate_path(self, path: Path) -> Tuple[bool, str]:
        """
        Validate that:
          - Path exists and is a directory
          - Is readable by the process
          - Contains at least one .mid file (recursively)

        Returns:
            (is_valid, message)
        """
        try:
            if not isinstance(path, Path):
                path = coerce_to_path(path)
            # Normalize (don't require existence here to avoid exceptions)
            path = self._to_abs(path)

            if not path.exists():
                return False, f"Path does not exist: {path}"
            if not path.is_dir():
                return False, f"Path is not a directory: {path}"

            # Check readability by attempting to iterate
            try:
                _ = next(path.iterdir(), None)
            except PermissionError:
                return False, f"Directory is not readable by current process: {path}"
            except OSError as e:
                return False, f"Error accessing directory {path}: {e}"

            first_midi = find_first_midi(path)
            if first_midi is None:
                return False, f"No .mid files found under: {path}"

            return True, "OK"
        except Exception as e:
            return False, f"Validation error for {path}: {e}"

    def set_config_path(self, path: Path) -> None:
        """
        Persist the provided path to a user config file (configs/settings.json).
        This is a no-op unless explicitly called.

        Behavior:
          - Creates settings directory if needed (when settings_dir is provided).
          - Merges with existing JSON if present.
          - Writes UTF-8 JSON with an additive 'rhythms_db_path' key.

        Note:
          - This method does not validate the path; callers should use validate_path() first if desired.
        """
        if self._settings_dir is None:
            # No settings dir configured; treat as no-op by design
            return

        settings_dir = self._settings_dir
        settings_dir.mkdir(parents=True, exist_ok=True)
        settings_file = settings_dir / "settings.json"

        data = {}
        if settings_file.exists():
            try:
                data = json.loads(settings_file.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    data = {}
            except Exception:
                # If corrupted, reset to empty dict to be additive and non-breaking
                data = {}

        data["rhythms_db_path"] = str(coerce_to_path(path))

        settings_file.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    # -----------------
    # Internal helpers
    # -----------------

    def _to_abs(self, path: Path) -> Path:
        try:
            # resolve(strict=False) to avoid raising if the path does not yet exist
            return path.resolve()
        except Exception:
            # Fallback to simple absolute conversion
            return Path(os.path.abspath(str(path)))

    def _read_config_path(self) -> Optional[Path]:
        """
        Attempt to read rhythms_db_path from:
          - <settings_dir>/settings.json
          - <settings_dir>/temp_settings.json (fallback)
        """
        if self._settings_dir is None:
            return None

        candidates = [
            self._settings_dir / "settings.json",
            self._settings_dir / "temp_settings.json",
        ]

        for file in candidates:
            if not file.exists():
                continue
            try:
                data = json.loads(file.read_text(encoding="utf-8"))
                if isinstance(data, dict) and "rhythms_db_path" in data:
                    return coerce_to_path(data["rhythms_db_path"])
            except Exception:
                # Ignore unreadable or malformed files
                continue

        return None