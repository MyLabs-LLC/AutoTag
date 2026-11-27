"""Composer 2.5 classification through the Cursor API.

The key is CURSOR_API_KEY in the repository .env. One local agent prompt,
with no tools, scores both axes. A label under the threshold, or one that is
not in the closed list, abstains.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from autotag.classify import DEFAULT_THRESHOLD, Classification
from autotag.env import dotenv_value, load_env
from autotag.gemini_backend import _classifications, _prompt as _gemini_prompt
from autotag.openjev_backend import PREMISE_CHARS

MODEL_ID = "composer-2.5"
NAME = "composer-2.5"
_ROOT = Path(__file__).resolve().parents[1]
_ATTEMPTS = 3


class GrokBackend:
    """Ask Composer 2.5 for one industry and one document type."""

    def __init__(self) -> None:
        self.model_id = MODEL_ID
        self.revision = ""
        self.device = "api"
        self.device_note = "cursor"
        self.dtype = None
        self.threads = None
        self._key = ""
        self._cache: dict[str, tuple[Classification, Classification]] = {}

    @property
    def name(self) -> str:
        return NAME

    def load(self) -> None:
        load_env()
        self._key = dotenv_value("CURSOR_API_KEY") or os.environ.get("CURSOR_API_KEY", "").strip()
        if not self._key:
            raise RuntimeError("CURSOR_API_KEY is missing. Put it in .env at the repository root.")

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


def _prompt(text: str) -> str:
    return (
        _gemini_prompt(text)
        + "\n\nReply with only a JSON object with keys industry_label, "
        "industry_score, doc_type_label, and doc_type_score."
    )


def _request(key: str, prompt: str) -> dict[str, Any]:
    from cursor_sdk import Agent, AgentOptions, LocalAgentOptions, ModelSelection
    from cursor_sdk.errors import APITimeoutError, CursorAgentError, InternalServerError, NetworkError, RateLimitError

    options = AgentOptions(
        api_key=key,
        model=ModelSelection(id=MODEL_ID),
        tools=[],
        local=LocalAgentOptions(cwd=_ROOT),
    )
    last: Exception | None = None
    for attempt in range(_ATTEMPTS):
        try:
            result = Agent.prompt(prompt, options)
        except (APITimeoutError, InternalServerError, NetworkError, RateLimitError) as exc:
            last = exc
            time.sleep(2 ** attempt)
            continue
        except CursorAgentError as exc:
            raise RuntimeError(f"Cursor API request failed: {exc}") from exc
        if result.status != "finished":
            detail = result.error_message or result.status
            raise RuntimeError(f"Cursor API run {result.status}: {detail}")
        return _parse_text(result.result or "")
    raise RuntimeError(f"Cursor API request failed: {last}")


def _parse_text(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        text = text.removesuffix("```").strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise RuntimeError(f"Cursor API returned non-JSON: {text[:240]}")
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Cursor API returned non-JSON: {text[:240]}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError("Cursor API JSON was not an object.")
    return parsed
