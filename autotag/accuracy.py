"""Run every selected model on every selected dataset and record what happened.

Document-type labels are compared to intent.json when that file covers the
documents. That file was written with the samples. It is not a human study
and not an LLM judgment.
"""

from __future__ import annotations

import gc
import json
import random
import resource
import sys
import time
from pathlib import Path

from autotag.catalog import DEFAULT_OUTPUT, DatasetSpec, ModelSpec, document_id, load_documents
from autotag.extract import ExtractError, extract_text
from autotag.classify import DEFAULT_THRESHOLD, StubBackend, Tagger
from autotag.labels import industry_questions

_SCORE = (
    "P(entailment | entailment or contradiction) for each question on its own. "
    "Neutral is ignored. Labels are not renormalized against each other. "
    "The premise is the first 16 KiB of extracted text."
)


def _score_meaning(kind: str) -> str:
    if kind == "openjev":
        return _SCORE
    if kind == "gemini":
        return "Gemini 4 Hydrogen confidence for the chosen label. Not an entailment score."
    if kind == "grok":
        return "Composer 2.5 confidence for the chosen label, through the Cursor API. Not an entailment score."
    if kind == "xai":
        return "Grok 4.7 Medium confidence for the chosen label, through the xAI API. Not an entailment score."
    if kind == "nemotron":
        return "Local Nemotron confidence for the chosen label, through the Unsloth Studio API. Not an entailment score."
    return "Stub phrase hits on the closed label ids. Not an entailment score."


_NOT_MEASURED = [
    "Agreement with a human labeler. No labeled set of about 50 documents, and no second annotator.",
    "Agreement with a large language model. No comparison model was named or run.",
    "A threshold fit on held-out labels. 0.5 was fixed before reading these scores.",
]


def _rss_mb() -> float:
    # ru_maxrss is bytes on macOS and kilobytes on Linux.
    per_mb = 1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / per_mb


def _load_model(spec: ModelSpec, cpu: bool) -> tuple[Tagger, float, float]:
    if spec.kind == "stub":
        return StubBackend(), 0.0, 0.0
    if spec.kind == "gemini":
        from autotag.gemini_backend import GeminiBackend

        backend = GeminiBackend()
        started = time.perf_counter()
        backend.load()
        return backend, time.perf_counter() - started, 0.0
    if spec.kind == "grok":
        from autotag.grok_backend import GrokBackend

        backend = GrokBackend()
        started = time.perf_counter()
        backend.load()
        return backend, time.perf_counter() - started, 0.0
    if spec.kind == "xai":
        from autotag.xai_backend import XaiGrokBackend

        backend = XaiGrokBackend()
        started = time.perf_counter()
        backend.load()
        return backend, time.perf_counter() - started, 0.0
    if spec.kind == "nemotron":
        from autotag.nemotron_backend import NemotronBackend

        backend = NemotronBackend()
        started = time.perf_counter()
        backend.load()
        return backend, time.perf_counter() - started, 0.0
    from autotag.openjev_backend import OpenJevBackend

    if not spec.model_id:
        raise RuntimeError(f"Model {spec.name} has no checkpoint id.")
    backend = OpenJevBackend(spec.model_id, cpu=cpu)
    started = time.perf_counter()
    backend.load()
    warmup_started = time.perf_counter()
    backend.entail("warmup", ["This is a warmup hypothesis."])
    warmup_s = time.perf_counter() - warmup_started
    load_s = time.perf_counter() - started - warmup_s
    return backend, load_s, warmup_s


def _release(backend: Tagger) -> None:
    close = getattr(backend, "close", None)
    if close is not None:
        close()
    gc.collect()


_log: object | None = None


def log_path_for(models: list[ModelSpec]) -> Path:
    """output/YYMMDD_HHMMSS_<model>_run.txt. A long model list uses a short name.

    The full list is the first line of the log. macOS rejects a filename past 255 bytes.
    """
    stamp = time.strftime("%y%m%d_%H%M%S")
    slug = "-".join(_safe_name(spec.name) for spec in models) or "run"
    filename = f"{stamp}_{slug}_run.txt"
    if len(filename.encode()) > 200:
        filename = f"{stamp}_{len(models)}models_run.txt"
    return Path("output") / filename


def open_log(path: Path) -> None:
    global _log
    path.parent.mkdir(parents=True, exist_ok=True)
    _log = path.open("w", encoding="utf-8")


def close_log() -> None:
    global _log
    handle = _log
    _log = None
    if handle is not None:
        handle.close()


def _safe_name(name: str) -> str:
    return "".join(char if char.isalnum() or char in "._+-" else "_" for char in name)


def _say(message: str) -> None:
    print(message, flush=True)
    handle = _log
    if handle is not None:
        handle.write(message + "\n")
        handle.flush()


def _sample_paths(paths: list[Path], count: int | None, seed: int | None) -> tuple[list[Path], int | None]:
    """Return the documents to score, and the seed that chose them."""
    if count is None:
        return paths, None
    if count < 1:
        raise ValueError("--rand must be at least 1.")
    used = seed if seed is not None else random.SystemRandom().randrange(1, 2**31)
    if count >= len(paths):
        return list(paths), used
    return random.Random(used).sample(paths, count), used


def run_selection(
    models: list[ModelSpec],
    datasets: list[DatasetSpec],
    output: Path = DEFAULT_OUTPUT,
    threshold: float = DEFAULT_THRESHOLD,
    rand: int | None = None,
    seed: int | None = None,
    cpu: bool = False,
) -> dict[str, object]:
    runs: list[dict[str, object]] = []
    for spec in models:
        _say(f"loading {spec.name}" + (f" ({spec.model_id})" if spec.model_id else ""))
        try:
            backend, load_s, warmup_s = _load_model(spec, cpu)
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            _say(f"{spec.name} failed to load: {message}")
            for dataset in datasets:
                runs.append(_failed_run(spec, dataset, threshold, message))
            continue
        device = getattr(backend, "device", "cpu")
        note = getattr(backend, "device_note", "")
        where = f"{device} ({note})" if note else str(device)
        _say(f"loaded {spec.name} on {where} in {load_s:.2f}s, warmup {warmup_s:.2f}s")
        try:
            for dataset in datasets:
                try:
                    runs.append(_score_dataset(spec, backend, dataset, threshold, load_s, warmup_s, rand, seed))
                except Exception as exc:
                    message = f"{type(exc).__name__}: {exc}"
                    _say(f"{spec.name} on {dataset.name} failed: {message}")
                    runs.append(_failed_run(spec, dataset, threshold, message))
        finally:
            _release(backend)

    payload: dict[str, object] = {
        "models": [spec.name for spec in models],
        "datasets": [spec.name for spec in datasets],
        "threshold": threshold,
        "rand": rand,
        "seed": seed,
        "runs": runs,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def _failed_run(spec: ModelSpec, dataset: DatasetSpec, threshold: float, error: str) -> dict[str, object]:
    return {
        "model": spec.name,
        "kind": spec.kind,
        "dataset": dataset.name,
        "dataset_path": str(dataset.path),
        "model_id": spec.model_id or spec.name,
        "error": error,
        "threshold": threshold,
        "documents": [],
        "fixture_intent_doc_type": None,
    }


def _score_dataset(
    spec: ModelSpec,
    backend: Tagger,
    dataset: DatasetSpec,
    threshold: float,
    load_s: float,
    warmup_s: float,
    rand: int | None,
    seed: int | None,
) -> dict[str, object]:
    found, intent = load_documents(dataset.path, dataset.recursive)
    paths, used_seed = _sample_paths(found, rand, seed)
    scope = "recursive" if dataset.recursive else "this directory only"
    sample = ""
    if rand is not None:
        if rand >= len(found):
            sample = f", random {rand} requested but only {len(found)} documents, using all, seed {used_seed}"
        else:
            sample = f", random {len(paths)} of {len(found)}, seed {used_seed}"
    _say(f"{spec.name} on {dataset.name}: {len(paths)} documents, {scope}, threshold {threshold}{sample}")
    documents: list[dict[str, object]] = []
    matches = 0
    scored = 0
    extract_errors = 0
    total = len(paths)
    for index, path in enumerate(paths, start=1):
        doc_id = document_id(dataset.path, path)
        expected = None
        if intent is not None:
            expected = intent.get(doc_id)
            if expected is None:
                expected = intent.get(path.name)
        prefix = f"[{index}/{total}] {doc_id}"
        extract_started = time.perf_counter()
        try:
            text = extract_text(path)
        except ExtractError as exc:
            extract_s = time.perf_counter() - extract_started
            extract_errors += 1
            _say(f"{prefix} [Null::Null] extract {extract_s:.4f}s model 0.0000s error: {exc}")
            documents.append(
                {
                    "doc_id": doc_id,
                    "chars": 0,
                    "extract_s": round(extract_s, 4),
                    "model_s": None,
                    "extract_error": str(exc),
                    "industry": None,
                    "doc_type": None,
                    "fixture_intent_doc_type": expected,
                    "doc_type_matches_fixture_intent": None,
                }
            )
            continue
        extract_s = time.perf_counter() - extract_started
        model_started = time.perf_counter()
        try:
            industry = backend.propose_industry(text, threshold)
            doc_type = backend.classify_doc_type(text, threshold)
        except Exception as exc:
            model_s = time.perf_counter() - model_started
            _say(f"{prefix} [Null::Null] extract {extract_s:.4f}s model {model_s:.4f}s error: {exc}")
            documents.append(
                {
                    "doc_id": doc_id,
                    "chars": len(text),
                    "extract_s": round(extract_s, 4),
                    "model_s": round(model_s, 4),
                    "extract_error": None,
                    "model_error": str(exc),
                    "industry": None,
                    "doc_type": None,
                    "fixture_intent_doc_type": expected,
                    "doc_type_matches_fixture_intent": None,
                }
            )
            continue
        model_s = time.perf_counter() - model_started
        matched = None if expected is None else doc_type.label == expected
        if matched is not None:
            scored += 1
            matches += int(matched)
        documents.append(
            {
                "doc_id": doc_id,
                "chars": len(text),
                "extract_s": round(extract_s, 4),
                "model_s": round(model_s, 4),
                "extract_error": None,
                "model_error": None,
                "industry": industry.to_dict(),
                "doc_type": doc_type.to_dict(),
                "fixture_intent_doc_type": expected,
                "doc_type_matches_fixture_intent": matched,
            }
        )
        _say(f"{prefix} {_document_line(documents[-1])}")
    intent_note = f", fixture intent {matches}/{scored}" if scored else ""
    device = getattr(backend, "device", "cpu")
    dtype = getattr(backend, "dtype", None)
    threads = getattr(backend, "threads", None)
    _say(
        f"{spec.name} on {dataset.name} done: {len(documents)} documents, "
        f"{extract_errors} extract errors{intent_note}; "
        f"device {device} dtype {dtype} threads {threads} peak_rss_mb {round(_rss_mb(), 1)}"
    )

    intent_summary: dict[str, object] | None
    if scored == 0:
        intent_summary = None
    else:
        intent_summary = {
            "n": scored,
            "matches": matches,
            "meaning": (
                "Count of document-type labels equal to intent.json in the dataset. "
                "The intent file was written with the samples. This is not agreement "
                "with a person or with a large language model."
            ),
        }
    return {
        "model": spec.name,
        "kind": spec.kind,
        "dataset": dataset.name,
        "dataset_path": str(dataset.path),
        "model_id": getattr(backend, "model_id", spec.model_id or spec.name),
        "revision": getattr(backend, "revision", None) or None,
        "device": getattr(backend, "device", "cpu"),
        "dtype": getattr(backend, "dtype", None),
        "threads": getattr(backend, "threads", None),
        "threshold": threshold,
        "industry_questions": industry_questions(),
        "score": _score_meaning(spec.kind),
        "load_s": round(load_s, 4),
        "warmup_s": round(warmup_s, 4),
        "peak_rss_mb": round(_rss_mb(), 1),
        "error": None,
        "documents": documents,
        "fixture_intent_doc_type": intent_summary,
        "not_measured": _NOT_MEASURED,
    }


def _tag(industry: dict[str, object], doc_type: dict[str, object]) -> str:
    """[industry::doc_type]. A side that abstains is Null."""
    industry_name = "Null" if industry["abstained"] else str(industry["label"])
    doc_name = "Null" if doc_type["abstained"] else str(doc_type["label"])
    return f"[{industry_name}::{doc_name}]"


def _document_line(row: dict[str, object]) -> str:
    industry = row["industry"]
    doc_type = row["doc_type"]
    assert isinstance(industry, dict) and isinstance(doc_type, dict)
    if row["fixture_intent_doc_type"] is None:
        mark = ""
    else:
        outcome = "match" if row["doc_type_matches_fixture_intent"] else "differ"
        mark = f" fixture {row['fixture_intent_doc_type']} {outcome}"
    return (
        f"{_tag(industry, doc_type)} extract {row['extract_s']:.4f}s model {row['model_s']:.4f}s "
        f"({row['chars']} chars){mark}"
    )
