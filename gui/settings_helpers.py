from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import re

# Core helpers
from core.filename_templater import (
    resolve_placeholders,
    sanitize_component,
    validate_template as _validate_template,
)
from core.rhythms_db_resolver import RhythmsDbResolver


def _render_template_str(template: str, mapping: Dict[str, str]) -> str:
    """
    Render a template using {placeholder} substitutions with provided mapping.
    Pure string replacement. No filesystem side effects.
    """
    # Replace any {name} with mapping.get(name, "")
    def repl(match: "re.Match[str]") -> str:
        key = match.group(1)
        return mapping.get(key, "")
    return re.sub(r"\{([A-Za-z0-9_]+)\}", repl, template or "")


def validate_template_str(template: str) -> Tuple[bool, str]:
    """
    Validate filename template string using core validation.

    Returns:
      (True, "") when valid or empty.
      (False, "reason") when invalid.
    """
    return _validate_template(template or "")


def build_preview_filename(settings: Dict[str, Any], template: str, base_dir: Optional[str] = None) -> str:
    """
    Build a pure string preview of the resolved filename based on current settings.

    - Does NOT touch disk
    - Does NOT ensure uniqueness
    - Applies per-component sanitization
    - Appends .mid if template yields no extension

    Args:
      settings: dict with at least {genre, mood, tempo, bars}
      template: filename template string with placeholders
      base_dir: optional base directory to prefix (pure join, no I/O)

    Returns:
      Preview path string (may include base_dir if provided)
    """
    if not template:
        # No template => return empty to signal "use legacy behavior"
        return ""

    ok, reason = validate_template_str(template)
    if not ok:
        raise ValueError(f"Invalid filename template: {reason}")

    mapping = resolve_placeholders(
        {
            "genre": settings.get("genre", ""),
            "mood": settings.get("mood", ""),
            "tempo": settings.get("tempo", ""),
            "bars": settings.get("bars", ""),
        },
        context={"stem": "combined"},
    )
    rendered = _render_template_str(template, mapping)

    # Split on separators, sanitize each path component
    parts = [p for p in re.split(r"[\\/]+", rendered) if p]
    safe_parts = [sanitize_component(p) for p in parts]

    # Reassemble relative path
    preview = "/".join(safe_parts) if safe_parts else "output"

    # Ensure .mid extension
    if not preview.lower().endswith(".mid"):
        preview = preview + ".mid"

    if base_dir:
        # Pure join
        base = str(Path(base_dir))
        return str(Path(base) / preview)
    return preview


def validate_rhythms_path(path: str) -> Tuple[bool, str]:
    """
    Validate rhythms DB path via RhythmsDbResolver.validate_path.

    Returns:
      (True, "OK") when valid
      (False, "reason") when invalid
    """
    resolver = RhythmsDbResolver(settings_dir="configs")
    return resolver.validate_path(Path(path))


def persist_rhythms_default(path: str) -> None:
    """
    Persist the rhythms DB path to configs/settings.json using the resolver API.
    Caller should validate the path first.
    """
    resolver = RhythmsDbResolver(settings_dir="configs")
    resolver.set_config_path(Path(path))