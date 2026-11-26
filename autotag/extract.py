"""Turn a document into the short text the model actually reads.

The model sees at most the first 16 KiB. Plain text, HTML, XML, JSON, and CSV
are taken from the first 16 KiB of the file. PDF and Office files are parsed
with Kreuzberg, a native extractor for the same family of formats as Apache
Tika, and then cut to 16 KiB of text.

Kreuzberg runs in a child process. A Rust panic in one file, including the
PDF metadata bug that slices through a multibyte character, is reported on
that file and the run continues.
"""

from __future__ import annotations

import os
import select
import subprocess
import sys
import time
from pathlib import Path

# 16 KiB. The premise cap in the OpenJev backend uses the same limit.
TEXT_CHARS = 16 * 1024
_FILE_TIMEOUT_S = 120.0

PLAIN_SUFFIXES = frozenset({".txt", ".text", ".log", ".md"})
MARKUP_SUFFIXES = frozenset({".htm", ".html", ".xml", ".json", ".csv"})
BINARY_SUFFIXES = frozenset({".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".rtf"})
SUPPORTED_SUFFIXES = PLAIN_SUFFIXES | MARKUP_SUFFIXES | BINARY_SUFFIXES

_MARKUP_MIME = {
    ".htm": "text/html",
    ".html": "text/html",
    ".xml": "application/xml",
    ".json": "application/json",
    ".csv": "text/csv",
}

# Dataset bookkeeping, not a document to classify.
SKIP_NAMES = frozenset({"manifest.json", "manifest.md", "intent.json", ".ds_store"})

_worker: _Extractor | None = None


class ExtractError(RuntimeError):
    """The file is a supported type, but its text could not be read."""


def is_document(path: Path) -> bool:
    return path.is_file() and path.name.lower() not in SKIP_NAMES and path.suffix.lower() in SUPPORTED_SUFFIXES


def extract_text(path: Path) -> str:
    """Text for one document, at most TEXT_CHARS characters."""
    suffix = path.suffix.lower()
    if suffix in PLAIN_SUFFIXES:
        return _plain(path)
    if suffix not in SUPPORTED_SUFFIXES:
        raise ExtractError(f"Unsupported document type: {path.name}")
    return _worker_extract(path)


def _plain(path: Path) -> str:
    with path.open("rb") as handle:
        raw = handle.read(TEXT_CHARS)
    return raw.decode("utf-8", errors="replace")[:TEXT_CHARS]


def _extract_local(path: Path) -> str:
    """Parse one file in this process. Used by the extractor child."""
    suffix = path.suffix.lower()
    if suffix in MARKUP_SUFFIXES:
        return _markup(path)
    if suffix in BINARY_SUFFIXES:
        return _binary(path)
    raise ExtractError(f"Unsupported document type: {path.name}")


def _markup(path: Path) -> str:
    with path.open("rb") as handle:
        raw = handle.read(TEXT_CHARS)
    if not raw:
        return ""
    try:
        text = _kreuzberg_bytes(raw, _MARKUP_MIME[path.suffix.lower()])
    except ExtractError:
        return raw.decode("utf-8", errors="replace")[:TEXT_CHARS]
    return text[:TEXT_CHARS]


def _binary(path: Path) -> str:
    return _kreuzberg_file(path)[:TEXT_CHARS]


def _kreuzberg_file(path: Path) -> str:
    extract_file_sync, config = _kreuzberg()
    try:
        result = extract_file_sync(path, config=config)
    except Exception as exc:
        raise ExtractError(f"{path.name}: {exc}") from exc
    return str(result.content or "")


def _kreuzberg_bytes(raw: bytes, mime: str) -> str:
    _extract_file_sync, config = _kreuzberg()
    from kreuzberg import extract_bytes_sync

    try:
        result = extract_bytes_sync(raw, mime, config)
    except Exception as exc:
        raise ExtractError(str(exc)) from exc
    return str(result.content or "")


def _kreuzberg():
    try:
        from kreuzberg import ExtractionConfig, PdfConfig, extract_file_sync
    except ImportError as exc:
        raise ExtractError(
            "Kreuzberg is required to read this document. "
            "Install autotag/requirements-openjev.txt into this environment."
        ) from exc
    # extract_metadata walks PDF info strings and panics on a bad character boundary.
    config = ExtractionConfig(
        enable_quality_processing=False,
        force_ocr=False,
        pdf_options=PdfConfig(extract_metadata=False, extract_images=False),
    )
    return extract_file_sync, config


def _serve_stdio() -> None:
    stdin = sys.stdin.buffer
    while True:
        line = stdin.readline()
        if not line:
            return
        path = Path(line.decode("utf-8", errors="replace").rstrip("\r\n"))
        try:
            text = _extract_local(path)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as exc:
            _respond(b"E", f"{path.name}: {exc}")
            continue
        _respond(b"K", text)


def _respond(kind: bytes, text: str) -> None:
    data = text.encode("utf-8", errors="replace")
    out = sys.stdout.buffer
    out.write(kind + b"\n")
    out.write(str(len(data)).encode("ascii") + b"\n")
    out.write(data)
    out.flush()


class _Extractor:
    """One Kreuzberg process. Restarted after a panic."""

    def __init__(self) -> None:
        self.proc: subprocess.Popen[bytes] | None = None
        self._start()

    def _start(self) -> None:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "autotag.extract"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
            env=env,
        )

    def _stop(self) -> None:
        proc = self.proc
        self.proc = None
        if proc is None:
            return
        if proc.poll() is None:
            proc.kill()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass

    def extract(self, path: Path) -> str:
        if self.proc is None or self.proc.poll() is not None:
            self._stop()
            self._start()
        proc = self.proc
        assert proc is not None and proc.stdin is not None and proc.stdout is not None
        try:
            proc.stdin.write(str(path).encode("utf-8") + b"\n")
            proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            self._stop()
            raise ExtractError(f"{path.name}: extractor panicked") from exc
        kind = _read_line(proc, _FILE_TIMEOUT_S)
        if kind is None:
            self._stop()
            raise ExtractError(f"{path.name}: extractor panicked")
        length_line = _read_line(proc, _FILE_TIMEOUT_S)
        if length_line is None:
            self._stop()
            raise ExtractError(f"{path.name}: extractor panicked")
        try:
            length = int(length_line)
        except ValueError as exc:
            self._stop()
            raise ExtractError(f"{path.name}: extractor panicked") from exc
        payload = _read_exact(proc, length, _FILE_TIMEOUT_S)
        if payload is None:
            self._stop()
            raise ExtractError(f"{path.name}: extractor panicked")
        text = payload.decode("utf-8", errors="replace")
        if kind == b"K":
            return text
        raise ExtractError(text)

    def close(self) -> None:
        proc = self.proc
        if proc is not None and proc.poll() is None and proc.stdin is not None:
            try:
                proc.stdin.close()
            except OSError:
                pass
        self._stop()


def _read_line(proc: subprocess.Popen[bytes], timeout: float) -> bytes | None:
    assert proc.stdout is not None
    fd = proc.stdout.fileno()
    deadline = time.monotonic() + timeout
    buf = bytearray()
    while time.monotonic() < deadline:
        if proc.poll() is not None and not buf:
            ready, _, _ = select.select([fd], [], [], 0)
            if not ready:
                return None
        remaining = deadline - time.monotonic()
        ready, _, _ = select.select([fd], [], [], max(0.0, remaining))
        if not ready:
            return None
        chunk = os.read(fd, 1)
        if not chunk:
            return bytes(buf) if buf else None
        if chunk == b"\n":
            return bytes(buf)
        buf.extend(chunk)
        if len(buf) > 100_000:
            return None
    return None


def _read_exact(proc: subprocess.Popen[bytes], length: int, timeout: float) -> bytes | None:
    if length < 0:
        return None
    assert proc.stdout is not None
    fd = proc.stdout.fileno()
    deadline = time.monotonic() + timeout
    buf = bytearray()
    while len(buf) < length:
        if time.monotonic() >= deadline:
            return None
        if proc.poll() is not None and len(buf) < length:
            ready, _, _ = select.select([fd], [], [], 0)
            if not ready:
                return None
        remaining = deadline - time.monotonic()
        ready, _, _ = select.select([fd], [], [], max(0.0, remaining))
        if not ready:
            return None
        chunk = os.read(fd, length - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def _worker_extract(path: Path) -> str:
    global _worker
    if _worker is None:
        _worker = _Extractor()
    return _worker.extract(path)


if __name__ == "__main__":
    _serve_stdio()
