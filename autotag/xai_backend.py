"""Grok 4.7 Medium classification through the xAI API.

The key is GROK_API_KEY in the repository .env. The model is grok-4.7 at
medium reasoning. One request scores both axes. A label under the threshold,
or one that is not in the closed list, abstains.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from autotag.classify import DEFAULT_THRESHOLD, Classification
from autotag.env import dotenv_value, load_env
from autotag.gemini_backend import _classifications
from autotag.grok_backend import _parse_text, _prompt
from autotag.openjev_backend import PREMISE_CHARS

MODEL_ID = "grok-4.7"
NAME = "grok-4.7-medium-api"
_URL = "https://api.x.ai/v1/chat/completions"
_ATTEMPTS = 3


class XaiGrokBackend:
    """Ask Grok 4.7 Medium, via GROK_API_KEY, for one industry and one document type."""

    def __init__(self) -> None:
        self.model_id = MODEL_ID
        self.revision = ""
        self.device = "api"
        self.device_note = "xai"
        self.dtype = None
        self.threads = None
        self._key = ""
        self._cache: dict[str, tuple[Classification, Classification]] = {}

    @property
    def name(self) -> str:
        return NAME

    def load(self) -> None:
        load_env()
        self._key = dotenv_value("GROK_API_KEY") or os.environ.get("GROK_API_KEY", "").strip()
        if not self._key:
            raise RuntimeError("GROK_API_KEY is missing. Put it in .env at the repository root.")

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
            payload = _request(self._key, _prompt(text[:PREMISE_CHARS]))
            cached = _classifications(payload, self.name, threshold)
            self._cache[text] = cached
        return cached


def _request(key: str, prompt: str) -> dict[str, Any]:
    body = {
        "model": MODEL_ID,
        "reasoning_effort": "medium",
        "messages": [{"role": "user", "content": prompt}],
    }
    encoded = json.dumps(body).encode("utf-8")
    last = "request failed"
    for attempt in range(_ATTEMPTS):
        request = urllib.request.Request(
            _URL,
            data=encoded,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                payload = json.loads(response.read().decode("utf-8"))
            return _parse_text(_message_text(payload))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            last = f"HTTP {exc.code}: {detail}"
            if exc.code not in {408, 429, 500, 502, 503, 504} or attempt + 1 == _ATTEMPTS:
                raise RuntimeError(f"xAI API {last}") from exc
        except urllib.error.URLError as exc:
            last = str(exc.reason)
            if attempt + 1 == _ATTEMPTS:
                raise RuntimeError(f"xAI API request failed: {last}") from exc
        time.sleep(2 ** attempt)
    raise RuntimeError(f"xAI API request failed: {last}")


def _message_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        raise RuntimeError(f"xAI API returned no choices: {list(payload)[:6]}")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content
    if isinstance(content, list):
        texts = [str(part.get("text", "")) for part in content if isinstance(part, dict) and part.get("text")]
        if texts:
            return "\n".join(texts)
    raise RuntimeError("xAI API returned no text.")
