"""Entailment with a small OpenJev checkpoint.

The default device is CUDA on Linux and MPS on Mac. `--cpu` forces the CPU.
torch and transformers are imported inside `load` so the stub path stays
on the standard library. Install them from autotag/requirements-openjev.txt
into a separate environment. They are not Axion training dependencies.
"""

from __future__ import annotations

import os
import sys
from typing import Any

from autotag.classify import DEFAULT_THRESHOLD, _classification
from autotag.labels import doc_type_hypotheses, doc_type_names, industry_hypotheses, industry_names

PREFERRED_MODEL = "xxparthparekhxx/openjev-flash"
FALLBACK_MODEL = "RazvanManolache/raz-systemone-nli-xsmall-openjev"
MAX_LENGTH = 512
BATCH = 8
# jevos document-classification guide: the type is announced on the first page.
# The model reads the first 16 KiB and ignores the rest of the file.
PREMISE_CHARS = 16 * 1024


class OpenJevBackend:
    """Score hypotheses with P(entailment). Scores are not renormalized across labels."""

    def __init__(
        self,
        model_id: str = PREFERRED_MODEL,
        revision: str | None = None,
        cpu: bool = False,
    ) -> None:
        self.model_id = model_id
        self.revision = revision or ""
        self.cpu = cpu
        self.device = "cpu"
        self.device_note = ""
        self.dtype = "float32"
        self._model: Any = None
        self._tokenizer: Any = None
        self._entailment_index = 1

    @property
    def name(self) -> str:
        return self.model_id

    def load(self) -> None:
        if self._model is not None:
            return
        from autotag.env import load_env

        load_env()
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        _require_runtime(torch.__version__)

        threads = os.cpu_count() or 1
        torch.set_num_threads(threads)
        self.threads = threads
        self.device, self.device_note = _select_device(torch, self.cpu)
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        model = AutoModelForSequenceClassification.from_pretrained(self.model_id)
        model.eval()
        model.to(self.device)
        self._entailment_index = _entailment_index(model.config)
        self._contradiction_index = _negative_index(model.config, self._entailment_index)
        self._tokenizer = tokenizer
        self._model = model
        self.revision = _cached_revision(self.model_id)

    def close(self) -> None:
        self._model = None
        self._tokenizer = None

    def entail(self, premise: str, hypotheses: list[str]) -> list[float]:
        """Independent yes-score for each hypothesis.

        Softmax over contradiction and entailment only, the same reduction the
        Hugging Face zero-shot pipeline uses when labels are not forced to sum
        to 1. Neutral is left out, and labels are not renormalized against
        each other, so a weak best label can still fall under the threshold.
        """
        if not hypotheses:
            return []
        self.load()
        import torch

        premise = premise[:PREMISE_CHARS]
        scores: list[float] = []
        assert self._model is not None and self._tokenizer is not None
        for start in range(0, len(hypotheses), BATCH):
            chunk = hypotheses[start : start + BATCH]
            encoded = self._tokenizer(
                [premise] * len(chunk),
                chunk,
                truncation="only_first",
                max_length=MAX_LENGTH,
                padding=True,
                return_tensors="pt",
            )
            encoded = encoded.to(self.device)
            with torch.inference_mode():
                logits = self._model(**encoded).logits.float()
            entailment = logits[:, self._entailment_index]
            contradiction = logits[:, self._contradiction_index]
            paired = torch.stack((contradiction, entailment), dim=-1)
            yes = torch.softmax(paired, dim=-1)[:, 1]
            scores.extend(float(value) for value in yes.tolist())
        return scores

    def propose_industry(self, text: str, threshold: float = DEFAULT_THRESHOLD):
        hypotheses = industry_hypotheses()
        ordered = list(hypotheses)
        probabilities = self.entail(text, [hypotheses[label_id] for label_id in ordered])
        scores = dict(zip(ordered, probabilities))
        return _classification("industry", scores, threshold, self.name, industry_names())

    def classify_doc_type(self, text: str, threshold: float = DEFAULT_THRESHOLD):
        hypotheses = doc_type_hypotheses()
        ordered = list(hypotheses)
        probabilities = self.entail(text, [hypotheses[label_id] for label_id in ordered])
        scores = dict(zip(ordered, probabilities))
        return _classification("doc_type", scores, threshold, self.name, doc_type_names())


def _select_device(torch: Any, force_cpu: bool) -> tuple[str, str]:
    """CUDA on Linux, MPS on Mac, CPU when --cpu is set or no GPU is present."""
    if force_cpu:
        return "cpu", ""
    if torch.cuda.is_available():
        return "cuda", ""
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps", ""
    return "cpu", "no CUDA or MPS GPU, using cpu"


def _require_runtime(torch_version: str) -> None:
    """OpenJev runs on Python 3.14 and PyTorch 2.14.1.

    The Linux and Windows CPU wheel reports 2.14.1+cpu. The macOS wheel reports 2.14.1.
    """
    running = f"{sys.version_info.major}.{sys.version_info.minor}"
    if sys.version_info[:2] != (3, 14):
        raise RuntimeError(f"AutoTag requires Python 3.14. This interpreter is {running}.")
    base = torch_version.split("+", 1)[0]
    if base != "2.14.1":
        raise RuntimeError(f"AutoTag requires PyTorch 2.14.1. Installed torch is {torch_version}.")


def _id2label(config: Any) -> dict[int, str]:
    raw = getattr(config, "id2label", {}) or {}
    return {int(key): str(value).lower() for key, value in raw.items()}


def _entailment_index(config: Any) -> int:
    """The positive class. `not_entailment` is the negative class on binary checkpoints."""
    labels = _id2label(config)
    for index, name in labels.items():
        if name in {"entailment", "entail"}:
            return index
    for index, name in labels.items():
        if "entail" in name and not name.startswith("not"):
            return index
    raise RuntimeError(f"Checkpoint labels have no entailment class: {labels}")


def _negative_index(config: Any, entailment_index: int) -> int:
    """Contradiction when the checkpoint has one, otherwise the other class of a binary model."""
    labels = _id2label(config)
    for index, name in labels.items():
        if "contradict" in name:
            return index
    others = [index for index in labels if index != entailment_index]
    if len(labels) == 2 and len(others) == 1:
        return others[0]
    raise RuntimeError(f"Checkpoint labels have no contradiction class: {labels}")


def _cached_revision(model_id: str) -> str:
    """Commit hash recorded by the Hub cache, when the snapshot has one."""
    try:
        from huggingface_hub import try_to_load_from_cache

        resolved = try_to_load_from_cache(model_id, "config.json")
    except (ImportError, OSError):
        return ""
    if not isinstance(resolved, str):
        return ""
    parts = resolved.split(os.sep)
    if "snapshots" in parts:
        return parts[parts.index("snapshots") + 1]
    return ""
