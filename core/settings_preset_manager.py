from __future__ import annotations

import json
import re
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class SettingsPresetManager:
    """
    Manage named generation settings presets persisted on disk.

    - Presets are saved as UTF-8 pretty JSON under {preset_dir}/{safe_name}.json
    - Optional index file {preset_dir}/index.json maintains name -> file mapping and metadata
    - Schema versioning supported via `version` field (default "1.0")
    - Validation ensures presence and basic types for required fields
    - Extra/unknown fields are preserved as-is

    Default preset_dir: "configs/presets"
    """

    SCHEMA_VERSION = "1.0"
    REQUIRED_KEYS = {
        "genre": str,
        "mood": str,
        "tempo": int,
        "bars": int,
        "time_signature": str,
        "complexity": str,
        # filename_template is optional string
        "rhythms_db_path": str,
        "version": str,
    }

    INVALID_FILENAME_CHARS = set('<>:"/\\|?*')  # Windows-incompatible chars

    def __init__(self, preset_dir: str = "configs/presets") -> None:
        self.preset_dir = Path(preset_dir)
        self.preset_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self.preset_dir / "index.json"

    # ---------------------------
    # Public API
    # ---------------------------
    def save_preset(self, name: str, settings_dict: Dict[str, Any]) -> bool:
        """
        Persist a named preset to disk. Returns True on success, False otherwise.

        - Does not mutate the caller's dict
        - Performs normalization/migration and validation before save
        - Updates/creates index entry with creation_date if missing

        Storage:
          - File: {preset_dir}/{slug(name)}.json
          - Index: {preset_dir}/index.json
        """
        if not isinstance(name, str) or not name.strip():
            return False

        data = self.normalize_preset(settings_dict)
        errors = self.validate_preset(data)
        if errors:
            # Refuse to save invalid presets
            return False

        idx = self._load_index()
        slug = self._slugify(name)
        file_name = self._unique_file_name_for_slug(slug, idx, prefer_existing_for=name)
        file_path = self.preset_dir / file_name

        # Write preset file (pretty, stable ordering, UTF-8)
        self._safe_write_json(file_path, data)

        # Update index
        presets = idx.setdefault("presets", {})
        meta = presets.get(name, {})
        if "creation_date" not in meta:
            meta["creation_date"] = datetime.now().isoformat(timespec="seconds")
        meta["file"] = file_name
        meta.setdefault("description", "")
        presets[name] = meta
        self._save_index(idx)
        return True

    def load_preset(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Load a named preset from disk. Returns dict or None if not found.
        """
        idx = self._load_index()
        file_path = self._file_for_name(name, idx)
        if file_path and file_path.exists():
            try:
                return json.loads(file_path.read_text(encoding="utf-8"))
            except Exception:
                return None

        # Fallback: derive from slug if index is missing/outdated
        fallback = self.preset_dir / f"{self._slugify(name)}.json"
        if fallback.exists():
            try:
                return json.loads(fallback.read_text(encoding="utf-8"))
            except Exception:
                return None
        return None

    def list_presets(self) -> List[str]:
        """
        Return a list of known preset names. Prefers index.json; falls back to scanning files.
        """
        idx = self._load_index()
        names = sorted(idx.get("presets", {}).keys())
        if names:
            return names

        # No index entries: scan directory for *.json presets
        found: List[str] = []
        for fp in sorted(self.preset_dir.glob("*.json")):
            if fp.name == "index.json":
                continue
            # Derive a display name from filename (reverse of slugify without decoding)
            name = fp.stem
            found.append(name)
        return found

    def delete_preset(self, name: str) -> bool:
        """
        Delete the preset file and remove index entry. Returns True if something was deleted.
        """
        deleted = False
        idx = self._load_index()
        file_path = self._file_for_name(name, idx)
        if file_path and file_path.exists():
            try:
                file_path.unlink()
                deleted = True
            except Exception:
                pass

        # Remove from index
        if "presets" in idx and name in idx["presets"]:
            idx["presets"].pop(name, None)
            self._save_index(idx)
            deleted = True

        # Fallback: attempt slug-derived file if we didn't have an index mapping
        if not deleted:
            fallback = self.preset_dir / f"{self._slugify(name)}.json"
            if fallback.exists():
                try:
                    fallback.unlink()
                    deleted = True
                except Exception:
                    pass
        return deleted

    def get_default_preset(self) -> Dict[str, Any]:
        """
        Return a default, valid preset dict (does not reference system state).
        """
        return {
            "version": self.SCHEMA_VERSION,
            "genre": "pop",
            "mood": "happy",
            "tempo": 120,
            "bars": 16,
            "time_signature": "4/4",
            "complexity": "balanced",
            "filename_template": "{genre}_{mood}_{tempo}_{bars}",
            "rhythms_db_path": "data/rhythms_db.json",
        }

    def validate_preset(self, settings_dict: Dict[str, Any]) -> List[str]:
        """
        Validate the provided settings dict against minimal schema requirements.
        Returns a list of error messages (empty list means valid).
        """
        errors: List[str] = []

        def ensure_type(key: str, expected_type: type) -> None:
            if key not in settings_dict:
                errors.append(f"Missing required key: {key}")
                return
            val = settings_dict[key]
            if expected_type is int:
                if not isinstance(val, int):
                    errors.append(f"Key '{key}' must be int, got {type(val).__name__}")
                else:
                    if key in ("tempo", "bars") and val <= 0:
                        errors.append(f"Key '{key}' must be > 0")
            elif expected_type is str:
                if not isinstance(val, str):
                    errors.append(f"Key '{key}' must be str, got {type(val).__name__}")
                elif not val.strip():
                    errors.append(f"Key '{key}' cannot be empty")

        # Required keys and types
        for k, t in self.REQUIRED_KEYS.items():
            # filename_template is optional; handled below
            if k == "version":
                # version presence/type check
                ensure_type("version", str)
            else:
                ensure_type(k, t)

        # Optional: filename_template must be string if present
        if "filename_template" in settings_dict and settings_dict["filename_template"] is not None:
            if not isinstance(settings_dict["filename_template"], str):
                errors.append("Key 'filename_template' must be str when provided")

        # Light time_signature format check (e.g., "4/4", "7/8")
        ts = settings_dict.get("time_signature")
        if isinstance(ts, str):
            if "/" not in ts or not re.match(r"^\s*\d+\s*/\s*\d+\s*$", ts):
                # Not a hard failure; warn as validation error per spec
                errors.append("Key 'time_signature' should be in 'N/D' form like '4/4'")

        return errors

    def normalize_preset(self, settings_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a normalized copy of settings suitable for persistence:
        - Adds missing required fields using defaults
        - Coerces types where safe (tempo/bars to int)
        - Preserves unknown fields
        - Sets version to SCHEMA_VERSION if missing
        """
        # Deep copy to avoid mutating caller-provided data
        src = deepcopy(settings_dict) if isinstance(settings_dict, dict) else {}
        out = deepcopy(src)

        defaults = self.get_default_preset()

        # Ensure version
        if not isinstance(out.get("version"), str) or not out["version"].strip():
            out["version"] = self.SCHEMA_VERSION

        # Strings with fallback
        for key in ("genre", "mood", "time_signature", "complexity", "rhythms_db_path"):
            val = out.get(key)
            if not isinstance(val, str) or not val.strip():
                out[key] = defaults[key]

        # Optional filename template: fill if missing or invalid
        ft = out.get("filename_template", defaults["filename_template"])
        if not isinstance(ft, str) or not ft:
            ft = defaults["filename_template"]
        out["filename_template"] = ft

        # Int coercions
        out["tempo"] = self._coerce_int(out.get("tempo"), defaults["tempo"], min_value=1)
        out["bars"] = self._coerce_int(out.get("bars"), defaults["bars"], min_value=1)

        # Light normalization of time_signature spacing
        ts = out.get("time_signature", defaults["time_signature"])
        if isinstance(ts, str):
            out["time_signature"] = ts.strip()
        else:
            out["time_signature"] = defaults["time_signature"]

        return out

    # ---------------------------
    # Internal helpers
    # ---------------------------
    def _coerce_int(self, v: Any, default: int, min_value: Optional[int] = None) -> int:
        try:
            iv = int(v)
            if min_value is not None and iv < min_value:
                return default
            return iv
        except Exception:
            return default

    def _slugify(self, name: str) -> str:
        # Replace invalid characters while preserving non-ASCII characters
        cleaned = []
        for ch in name:
            if ch in self.INVALID_FILENAME_CHARS:
                cleaned.append("_")
            else:
                cleaned.append(ch)
        slug = "".join(cleaned).strip()
        # Windows: disallow trailing space/dot
        slug = slug.rstrip(" .")
        if not slug:
            slug = "preset"
        return slug

    def _unique_file_name_for_slug(self, slug: str, idx: Dict[str, Any], prefer_existing_for: Optional[str] = None) -> str:
        """
        Ensure filename uniqueness. If index already maps the given preset name, reuse its file.
        Otherwise, avoid collisions by adding numeric suffix.
        """
        if prefer_existing_for:
            existing = idx.get("presets", {}).get(prefer_existing_for)
            if existing and isinstance(existing, dict):
                f = existing.get("file")
                if f:
                    return f

        base = f"{slug}.json"
        candidate = base
        n = 2
        existing_files = {m.get("file") for m in idx.get("presets", {}).values() if isinstance(m, dict)}
        while candidate in existing_files or (self.preset_dir / candidate).exists():
            candidate = f"{slug}_{n}.json"
            n += 1
        return candidate

    def _file_for_name(self, name: str, idx: Optional[Dict[str, Any]] = None) -> Optional[Path]:
        try:
            idx = idx if idx is not None else self._load_index()
            meta = idx.get("presets", {}).get(name)
            if isinstance(meta, dict) and "file" in meta:
                return self.preset_dir / meta["file"]
            # Fallback to slug
            slug = self._slugify(name)
            return self.preset_dir / f"{slug}.json"
        except Exception:
            return None

    def _load_index(self) -> Dict[str, Any]:
        if not self._index_path.exists():
            return {"presets": {}}
        try:
            raw = self._index_path.read_text(encoding="utf-8")
            data = json.loads(raw) if raw.strip() else {}
            if not isinstance(data, dict):
                return {"presets": {}}
            data.setdefault("presets", {})
            if not isinstance(data["presets"], dict):
                data["presets"] = {}
            return data
        except Exception:
            return {"presets": {}}

    def _save_index(self, idx: Dict[str, Any]) -> None:
        try:
            payload = json.dumps(idx, indent=2, ensure_ascii=False, sort_keys=True)
            self._index_path.write_text(payload, encoding="utf-8")
        except Exception:
            # Best effort; don't raise
            pass

    def _safe_write_json(self, path: Path, data: Dict[str, Any]) -> None:
        try:
            payload = json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True)
            path.write_text(payload, encoding="utf-8")
        except Exception:
            # Best effort; don't raise
            pass