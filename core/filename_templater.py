from __future__ import annotations

import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union


# Allowed placeholders
_ALLOWED_KEYS = {
    "genre",
    "mood",
    "tempo",
    "bars",
    "timestamp",
    "stem",
    "run_index",
    "unique_id",
}


def sanitize_component(value: str) -> str:
    """
    Sanitize a single path component to be safe across OSes.
    - Strip leading/trailing whitespace
    - Replace unsafe characters with underscore
    - Collapse whitespace to single underscore
    - Remove control characters
    - Trim repeated underscores/dashes
    - Avoid reserved names on Windows
    """
    if value is None:
        value = ""
    # Normalize to string and strip
    s = str(value).strip()

    # Replace path separators proactively
    s = s.replace("\\", "_").replace("/", "_")

    # Remove control chars
    s = "".join(ch for ch in s if ch.isprintable())

    # Replace invalid filesystem chars
    invalid_chars = '<>:"/\\|?*'
    for ch in invalid_chars:
        s = s.replace(ch, "_")

    # Collapse whitespace to single underscore
    s = re.sub(r"\s+", "_", s)

    # Allow alnum, underscore, dash, dot
    s = re.sub(r"[^A-Za-z0-9._-]", "_", s)

    # Collapse multiple underscores/dashes
    s = re.sub(r"[_-]{2,}", lambda m: m.group(0)[0], s)

    # Trim leading/trailing dots/underscores/dashes
    s = s.strip("._-")

    # Reserved device names on Windows
    reserved = {
        "con", "prn", "aux", "nul",
        *{f"com{i}" for i in range(1, 10)},
        *{f"lpt{i}" for i in range(1, 10)},
    }
    if s.lower() in reserved or s == "":
        s = (s + "_file") if s else "file"

    # Limit component length
    if len(s) > 100:
        s = s[:100]

    return s


def resolve_placeholders(
    settings: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None,
    timestamp_fmt: str = "%Y%m%d_%H%M%S",
) -> Dict[str, str]:
    """
    Build a placeholder mapping from provided settings/context.
    settings can include: genre, mood, tempo, bars
    context can include: stem, run_index, timestamp (overrides), unique_id (overrides)

    Returns stringified and sanitized values (component-safe).
    """
    context = context or {}

    # Timestamp and UID (allow overrides via context)
    ts = context.get("timestamp")
    if not ts:
        ts = datetime.now().strftime(timestamp_fmt)

    uid = context.get("unique_id")
    if not uid:
        uid = uuid.uuid4().hex[:8].lower()

    # Gather values
    values: Dict[str, Any] = {
        "genre": settings.get("genre", ""),
        "mood": settings.get("mood", ""),
        "tempo": settings.get("tempo", ""),
        "bars": settings.get("bars", ""),
        "timestamp": ts,
        "stem": context.get("stem", ""),
        "run_index": context.get("run_index", ""),
        "unique_id": uid,
    }

    # Stringify and sanitize components (timestamp and uid also sanitized but should be safe already)
    resolved = {k: sanitize_component(str(v)) if v is not None else "" for k, v in values.items()}
    return resolved


def validate_template(template: str) -> Tuple[bool, str]:
    """
    Validate that the template only uses allowed placeholders.
    Returns (True, "") if valid, else (False, reason).
    Empty/None template is considered valid.
    """
    if not template:
        return True, ""

    # Find all {name} occurrences
    names = set(re.findall(r"\{([A-Za-z0-9_]+)\}", template))
    unknown = [n for n in names if n not in _ALLOWED_KEYS]
    if unknown:
        return False, f"Unknown placeholders in template: {', '.join(sorted(unknown))}"
    return True, ""


def ensure_unique(path: Path) -> Path:
    """
    Ensure the path is unique. If it exists, append _2, _3, ... up to _99.
    If still a collision, append an 8-char UID as fallback.
    Creates parent directories if needed (no other side effects).
    """
    # Ensure parent exists
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix or ".mid"

    # Try _2.._99
    for i in range(2, 100):
        candidate = parent / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return candidate

    # Fallback: append uid
    uid = uuid.uuid4().hex[:8].lower()
    candidate = parent / f"{stem}_{uid}{suffix}"
    return candidate


def _render_template(template: str, mapping: Dict[str, str]) -> str:
    def repl(match: re.Match) -> str:
        key = match.group(1)
        return mapping.get(key, "")
    return re.sub(r"\{([A-Za-z0-9_]+)\}", repl, template)


def format_filename(
    template: str,
    settings: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None,
    base_dir: Union[str, Path] = "output",
    timestamp_fmt: str = "%Y%m%d_%H%M%S",
) -> Path:
    """
    Render a filename from a template and ensure safety/uniqueness.

    - base_dir defaults to 'output'
    - Template may include path separators to form subdirectories (treated as relative)
    - Ensures '.mid' extension if not provided
    - Sanitizes every path component
    - Creates directories as needed, and resolves collisions with ensure_unique

    Note: Callers should bypass templating (keep legacy) by passing template as None/empty.
    """
    ok, reason = validate_template(template)
    if not ok:
        raise ValueError(f"Invalid filename template: {reason}")

    mapping = resolve_placeholders(settings, context, timestamp_fmt)
    rendered = _render_template(template, mapping)

    # Split into components on both types of separators
    raw_parts = [p for p in re.split(r"[\\/]+", rendered) if p]

    # Sanitize each component
    parts = [sanitize_component(p) for p in raw_parts if p is not None]

    # Rebuild path under base_dir
    base = Path(base_dir) if base_dir else Path("output")
    out_path = base.joinpath(*parts) if parts else base / "output"

    # Ensure .mid extension
    if out_path.suffix.lower() != ".mid":
        out_path = out_path.with_suffix(".mid")

    # Make unique and ensure directories
    out_path = ensure_unique(out_path)
    return out_path