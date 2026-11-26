"""Gemini 4 Hydrogen classification through the Gemini API.

The key is GEMINI_API_KEY in the repository .env. One request scores both
axes. A label under the threshold, or one that is not in the closed list,
abstains.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from autotag.classify import DEFAULT_THRESHOLD, Classification
from autotag.env import load_env
from autotag.labels import DOC_TYPES, INDUSTRIES, industry_names, doc_type_names
from autotag.openjev_backend import PREMISE_CHARS

MODEL_ID = "gemini-4-hydrogen"
_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_ID}:generateContent"


class GeminiBackend:
    """Ask Gemini 4 Hydrogen for one industry and one document type."""

    def __init__(self) -> None:
        self.model_id = MODEL_ID
        self.revision = ""
        self.device = "api"
        self.device_note = ""
        self.dtype = None
        self.threads = None
        self._keys: list[str] = []
        self._cursor = 0
        self._cache: dict[str, tuple[Classification, Classification]] = {}

    @property
    def name(self) -> str:
        return self.model_id

    def load(self) -> None:
        load_env()
        raw = os.environ.get("GEMINI_API_KEY", "")
        self._keys = [part.strip() for part in raw.split(",") if part.strip()]
        if not self._keys:
            raise RuntimeError("GEMINI_API_KEY is missing. Put it in .env at the repository root.")

    def close(self) -> None:
        self._cache.clear()

    def propose_industry(self, text: str, threshold: float = DEFAULT_THRESHOLD) -> Classification:
        return self._pair(text, threshold)[0]

    def classify_doc_type(self, text: str, threshold: float = DEFAULT_THRESHOLD) -> Classification:
        return self._pair(text, threshold)[1]

    def _pair(self, text: str, threshold: float) -> tuple[Classification, Classification]:
        if not self._keys:
            self.load()
        cached = self._cache.get(text)
        if cached is None:
            payload = _request(self._next_key(), _prompt(text[:PREMISE_CHARS]))
            cached = _classifications(payload, self.name, threshold)
            self._cache[text] = cached
        return cached

    def _next_key(self) -> str:
        key = self._keys[self._cursor % len(self._keys)]
        self._cursor += 1
        return key


def _prompt(text: str) -> str:
    industries = ", ".join(label.id for label in INDUSTRIES)
    doc_types = ", ".join(label.id for label in DOC_TYPES)
    return (
        "Classify the document. Pick one industry id and one document type id "
        "from the lists. Use other when none fit. score is your confidence from "
        "0 to 1 that the chosen id is right.\n\n"
        f"Industries: {industries}, other\n"
        f"Document types: {doc_types}, other\n\n"
        f"Document:\n{text}"
    )


def _request(key: str, prompt: str) -> dict[str, Any]:
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
            "thinkingConfig": {"thinkingLevel": "LOW"},
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "industry_label": {"type": "STRING"},
                    "industry_score": {"type": "NUMBER"},
                    "doc_type_label": {"type": "STRING"},
                    "doc_type_score": {"type": "NUMBER"},
                },
                "required": ["industry_label", "industry_score", "doc_type_label", "doc_type_score"],
            },
        },
    }
    request = urllib.request.Request(
        _URL,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "x-goog-api-key": key},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"Gemini API HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Gemini API request failed: {exc.reason}") from exc
    return _parse_model_json(payload)


def _parse_model_json(payload: dict[str, Any]) -> dict[str, Any]:
    candidates = payload.get("candidates") or []
    if not candidates:
        raise RuntimeError(f"Gemini API returned no candidates: {list(payload)[:6]}")
    parts = ((candidates[0].get("content") or {}).get("parts") or [])
    texts = [str(part.get("text", "")) for part in parts if part.get("text")]
    if not texts:
        raise RuntimeError("Gemini API returned no text.")
    raw = "\n".join(texts).strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1]
        raw = raw.removesuffix("```").strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Gemini API returned non-JSON: {raw[:240]}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError("Gemini API JSON was not an object.")
    return parsed


def _classifications(
    payload: dict[str, Any],
    backend: str,
    threshold: float,
) -> tuple[Classification, Classification]:
    industry = _one(
        "industry",
        str(payload.get("industry_label", "other")),
        payload.get("industry_score", 0),
        industry_names(),
        backend,
        threshold,
    )
    doc_type = _one(
        "doc_type",
        str(payload.get("doc_type_label", "other")),
        payload.get("doc_type_score", 0),
        doc_type_names(),
        backend,
        threshold,
    )
    return industry, doc_type


def _one(
    axis: str,
    label: str,
    score: object,
    names: dict[str, str],
    backend: str,
    threshold: float,
) -> Classification:
    try:
        value = float(score)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        value = 0.0
    value = min(1.0, max(0.0, value))
    chosen = label.strip().lower().replace(" ", "_")
    known = chosen in names
    abstained = (not known) or value < threshold
    shown = "other" if abstained else chosen
    return Classification(
        axis=axis,
        label=shown,
        label_name="Other" if abstained else names[chosen],
        score=value,
        abstained=abstained,
        scores={shown: round(value, 4)},
        backend=backend,
    )
