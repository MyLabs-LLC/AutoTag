"""Load the repository .env without overriding variables already set in the shell.

HUGGINGFACE_API_KEY is copied to the names Hugging Face libraries read.
"""

from __future__ import annotations

import os
from pathlib import Path

_LOADED = False


def load_env() -> None:
    global _LOADED
    if _LOADED:
        return
    _LOADED = True
    for path in _candidates():
        if path.is_file():
            _apply(path)
            break
    _alias_huggingface()


def _candidates() -> list[Path]:
    roots = [Path.cwd(), Path(__file__).resolve().parents[1]]
    seen: set[Path] = set()
    found: list[Path] = []
    for root in roots:
        for directory in [root, *root.parents]:
            if directory in seen:
                continue
            seen.add(directory)
            found.append(directory / ".env")
            if directory == Path.home():
                break
    return found


def _apply(path: Path) -> None:
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def _alias_huggingface() -> None:
    token = os.environ.get("HUGGINGFACE_API_KEY", "").strip()
    if not token:
        return
    os.environ.setdefault("HF_TOKEN", token)
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)
