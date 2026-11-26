"""Models and datasets the one AutoTag command can run.

`all` is every registered name. A Hugging Face id is an extra OpenJev model.
A filesystem path is an extra dataset: a directory of documents, or one document.
`-r` includes files in subdirectories.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from autotag.extract import SUPPORTED_SUFFIXES, is_document
from autotag.gemini_backend import MODEL_ID as GEMINI_MODEL
from autotag.openjev_backend import FALLBACK_MODEL, PREFERRED_MODEL

PACKAGE = Path(__file__).resolve().parent
FIXTURES = PACKAGE / "fixtures"
DATASETS_ROOT = PACKAGE / "datasets"
DEFAULT_OUTPUT = PACKAGE / "measurements" / "run.json"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: str
    model_id: str | None


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    path: Path
    recursive: bool = False


MODELS: tuple[ModelSpec, ...] = (
    ModelSpec("stub", "stub", None),
    ModelSpec("openjev-flash", "openjev", PREFERRED_MODEL),
    ModelSpec("openjev-71m", "openjev", FALLBACK_MODEL),
    ModelSpec("gemini-4-hydrogen", "gemini", GEMINI_MODEL),
)


def model_names() -> list[str]:
    return [spec.name for spec in MODELS]


def dataset_names() -> list[str]:
    return [spec.name for spec in registered_datasets()]


def registered_datasets() -> tuple[DatasetSpec, ...]:
    specs = [DatasetSpec("fixtures", FIXTURES)]
    seen = {spec.name for spec in specs}
    if DATASETS_ROOT.is_dir():
        for child in sorted(DATASETS_ROOT.iterdir()):
            if not child.is_dir() or child.name in seen:
                continue
            if any(is_document(path) for path in child.rglob("*")):
                specs.append(DatasetSpec(child.name, child))
                seen.add(child.name)
    return tuple(specs)


def resolve_models(tokens: list[str] | None) -> list[ModelSpec]:
    chosen = _flatten(tokens)
    if not chosen or "all" in chosen:
        if chosen and "all" in chosen and any(token != "all" for token in chosen):
            raise ValueError("Use --model all on its own, or name the models to run.")
        return list(MODELS)
    by_name = {spec.name: spec for spec in MODELS}
    resolved: list[ModelSpec] = []
    seen: set[str] = set()
    for token in chosen:
        if token in seen:
            continue
        seen.add(token)
        if token in by_name:
            resolved.append(by_name[token])
            continue
        if "/" in token:
            resolved.append(ModelSpec(token, "openjev", token))
            continue
        known = ", ".join(model_names())
        raise ValueError(f"Unknown model {token!r}. Choose from: {known}, all, or a Hugging Face model id.")
    return resolved


def resolve_datasets(tokens: list[str] | None, recursive: bool = False) -> list[DatasetSpec]:
    chosen = _flatten(tokens)
    if not chosen or "all" in chosen:
        if chosen and "all" in chosen and any(token != "all" for token in chosen):
            raise ValueError("Use --dataset all on its own, or name the datasets to run.")
        specs = list(registered_datasets())
    else:
        by_name = {spec.name: spec for spec in registered_datasets()}
        specs = []
        seen: set[str] = set()
        for token in chosen:
            if token in seen:
                continue
            seen.add(token)
            if token in by_name:
                specs.append(by_name[token])
                continue
            path = Path(token)
            if path.exists():
                specs.append(DatasetSpec(path.name, path))
                continue
            known = ", ".join(dataset_names())
            raise ValueError(f"Unknown dataset {token!r}. Choose from: {known}, all, or a path.")
    resolved = [DatasetSpec(spec.name, spec.path, recursive) for spec in specs]
    for spec in resolved:
        load_documents(spec.path, spec.recursive)
    return resolved


def document_id(dataset: Path, path: Path) -> str:
    """Path relative to the dataset, so files in different folders stay distinct."""
    if dataset.is_file():
        return path.name
    return path.relative_to(dataset).as_posix()


def load_documents(path: Path, recursive: bool = False) -> tuple[list[Path], dict[str, str] | None]:
    """Documents in a dataset, plus fixture intent when intent.json covers them."""
    if path.is_file():
        if not is_document(path):
            suffixes = ", ".join(sorted(SUPPORTED_SUFFIXES))
            raise ValueError(f"Unsupported document type: {path}. Supported suffixes: {suffixes}")
        intent = _intent_file(path.parent / "intent.json")
        if intent is not None and path.name not in intent:
            intent = None
        return [path], intent
    if not path.is_dir():
        raise ValueError(f"Not a dataset directory: {path}")
    documents = sorted(_iter_documents(path, recursive))
    if not documents:
        hint = ""
        if not recursive and _has_nested_document(path):
            hint = " Use -r to read subdirectories."
        suffixes = ", ".join(sorted(SUPPORTED_SUFFIXES))
        raise ValueError(f"No documents in {path}. Supported suffixes: {suffixes}.{hint}")
    intent = _intent_file(path / "intent.json")
    if intent is not None:
        missing = [
            document_id(path, item)
            for item in documents
            if document_id(path, item) not in intent and item.name not in intent
        ]
        if missing:
            shown = ", ".join(missing[:8])
            extra = f" (+{len(missing) - 8} more)" if len(missing) > 8 else ""
            raise ValueError(f"{path / 'intent.json'} is missing intent for: {shown}{extra}")
    return documents, intent


def _iter_documents(path: Path, recursive: bool):
    walker = path.rglob("*") if recursive else path.glob("*")
    for item in walker:
        if is_document(item):
            yield item


def nested_document_count(path: Path) -> int:
    """Documents below the top directory. Zero when recursion is already on, or the path is a file."""
    if not path.is_dir():
        return 0
    return sum(1 for item in path.rglob("*") if is_document(item) and item.parent != path)


def _has_nested_document(path: Path) -> bool:
    return nested_document_count(path) > 0


def _intent_file(path: Path) -> dict[str, str] | None:
    if not path.is_file():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must be a JSON object of filename to document type.")
    intent: dict[str, str] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError(f"{path} must map filenames to document-type strings.")
        intent[key] = value
    return intent


def _flatten(tokens: list[str] | None) -> list[str]:
    if not tokens:
        return []
    flat: list[str] = []
    for token in tokens:
        for piece in token.split(","):
            name = piece.strip()
            if name:
                flat.append(name)
    return flat
