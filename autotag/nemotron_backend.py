"""Nemotron classification through the local Unsloth Studio API.

The server is OpenAI-compatible at http://127.0.0.1:8888/v1. The bearer token
is the one Unsloth Studio writes for that process. One request scores both
axes. A label under the threshold, or one that is not in the closed list,
abstains.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from autotag.classify import DEFAULT_THRESHOLD, Classification
from autotag.env import dotenv_value, load_env
from autotag.gemini_backend import _classifications
from autotag.grok_backend import _parse_text, _prompt
from autotag.openjev_backend import PREMISE_CHARS

NAME = "nemotron-3.5-bolt"
DEFAULT_MODEL = "unsloth/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-GGUF:UD-Q3_K_XL"
DEFAULT_BASE = "http://127.0.0.1:8888/v1"
_SECRET_FILE = Path.home() / ".unsloth/studio/auth/.desktop_secret"
_ATTEMPTS = 3


class NemotronBackend:
    """Ask the local Nemotron server for one industry and one document type."""

    def __init__(self) -> None:
        self.model_id = DEFAULT_MODEL
        self.revision = ""
        self.device = "api"
        self.device_note = "unsloth"
        self.dtype = None
        self.threads = None
        self._key = ""
        self._base = DEFAULT_BASE
        self._cache: dict[str, tuple[Classification, Classification]] = {}

    @property
    def name(self) -> str:
        return NAME

    def load(self) -> None:
        load_env()
        self._base = (dotenv_value("NEMOTRON_BASE_URL") or os.environ.get("NEMOTRON_BASE_URL") or DEFAULT_BASE).rstrip("/")
        self._key = _bearer(self._base)
        if not self._key:
            raise RuntimeError(
                "Unsloth Studio rejected local sign-in. In Settings > API, create a key and set NEMOTRON_API_KEY."
            )
        chosen = dotenv_value("NEMOTRON_MODEL") or os.environ.get("NEMOTRON_MODEL", "").strip()
        self.model_id = chosen or _loaded_model(self._base, self._key) or DEFAULT_MODEL

    def close(self) -> None:
        self._cache.clear()

    def propose_industry(self, text: str, threshold: float = DEFAULT_THRESHOLD) -> Classification:
        return self._pair(text, threshold)[0]

    def classify_doc_type(self, text: str, threshold: float = DEFAULT_THRESHOLD) -> Classification:
        return self._pair(text, threshold)[1]

    def _pair(self, text: str, threshold: float) -> tuple[Classification, Classification]:
        if not self._key:
            self.load()
        cached = self._cache.get(text)
        if cached is None:
            payload = _request(self._base, self._key, self.model_id, _prompt(text[:PREMISE_CHARS]))
            cached = _classifications(payload, self.name, threshold)
            self._cache[text] = cached
        return cached


def _bearer(base: str) -> str:
    """An sk-unsloth key from the environment, or a short-lived token from the local desktop secret."""
    for name in ("NEMOTRON_API_KEY", "UNSLOTH_API_KEY"):
        value = dotenv_value(name) or os.environ.get(name, "").strip()
        if value:
            return value
    secret = _desktop_secret()
    if not secret:
        return ""
    root = base.removesuffix("/v1")
    body = json.dumps({"secret": secret}).encode("utf-8")
    request = urllib.request.Request(
        f"{root}/api/auth/desktop-login",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError):
        return ""
    if not isinstance(payload, dict):
        return ""
    return str(payload.get("access_token") or "").strip()


def _desktop_secret() -> str:
    if not _SECRET_FILE.is_file():
        return ""
    try:
        secret = _SECRET_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        return ""
    if secret.startswith("desktop-") and len(secret) > len("desktop-"):
        return secret
    return ""


def _loaded_model(base: str, key: str) -> str:
    request = urllib.request.Request(f"{base}/models", headers={"Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError):
        return ""
    items = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        return ""
    ids = [str(item.get("id")) for item in items if isinstance(item, dict) and item.get("id")]
    for model_id in ids:
        if "nemotron" in model_id.lower():
            return model_id
    return ids[0] if ids else ""


def _request(base: str, key: str, model_id: str, prompt: str) -> dict[str, Any]:
    body = {"model": model_id, "temperature": 0, "messages": [{"role": "user", "content": prompt}]}
    encoded = json.dumps(body).encode("utf-8")
    last = "request failed"
    for attempt in range(_ATTEMPTS):
        request = urllib.request.Request(
            f"{base}/chat/completions",
            data=encoded,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                payload = json.loads(response.read().decode("utf-8"))
            return _parse_text(_message_text(payload))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            last = f"HTTP {exc.code}: {detail}"
            if exc.code not in {408, 429, 500, 502, 503, 504} or attempt + 1 == _ATTEMPTS:
                raise RuntimeError(f"Nemotron API {last}") from exc
        except urllib.error.URLError as exc:
            last = str(exc.reason)
            if attempt + 1 == _ATTEMPTS:
                raise RuntimeError(f"Nemotron API request failed: {last}") from exc
        time.sleep(2 ** attempt)
    raise RuntimeError(f"Nemotron API request failed: {last}")


def _message_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        raise RuntimeError(f"Nemotron API returned no choices: {list(payload)[:6]}")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content
    if isinstance(content, list):
        texts = [str(part.get("text", "")) for part in content if isinstance(part, dict) and part.get("text")]
        if texts:
            return "\n".join(texts)
    raise RuntimeError("Nemotron API returned no text.")
