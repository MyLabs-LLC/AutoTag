"""One command for every AutoTag model and dataset.

From the repository root:

    python -m autotag
    python -m autotag --model stub --dataset fixtures
    python -m autotag --model openjev-flash --dataset fixtures
    python -m autotag --model all --dataset all
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from autotag.accuracy import close_log, log_path_for, open_log, run_selection, _say
from autotag.catalog import (
    DEFAULT_OUTPUT,
    compare_model_names,
    dataset_names,
    model_names,
    nested_document_count,
    resolve_datasets,
    resolve_models,
)
from autotag.classify import DEFAULT_THRESHOLD
from autotag.env import load_env
from autotag.report import render_summary, write_report


def build_parser() -> argparse.ArgumentParser:
    models = ", ".join(model_names())
    compare = ", ".join(compare_model_names())
    datasets = ", ".join(dataset_names())
    parser = argparse.ArgumentParser(
        prog="autotag",
        description="Run AutoTag across models and datasets. One command controls the whole run.",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=None,
        metavar="MODEL",
        help=(
            f"Model to run. Names: {models}. "
            f"`compare` is {compare}. "
            "Also `all`, a comma-separated list, or a Hugging Face model id. "
            "Repeat the flag to run several. Default: all."
        ),
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        metavar="DATASET",
        help=(
            f"Dataset to score. Names: {datasets}. "
            "Also `all`, a comma-separated list, a directory of documents, or one document. "
            "A directory named under autotag/datasets/ is included in `all`. "
            "Repeat the flag to run several. Default: all."
        ),
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Read documents in subdirectories. The model sees the first 16 KiB of each file.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Run OpenJev on the CPU. The default is CUDA on Linux and MPS on Mac.",
    )
    parser.add_argument("--rand", type=int, default=None, metavar="N", help="Score N documents chosen at random.")
    parser.add_argument("--seed", type=int, default=None, metavar="N", help="Seed for --rand. The same seed picks the same documents.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Abstain below this score. Default: {DEFAULT_THRESHOLD}.",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help=f"JSON record. Default: {DEFAULT_OUTPUT}.")
    return parser


def main(argv: list[str] | None = None) -> int:
    load_env()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        models = resolve_models(args.model)
        datasets = resolve_datasets(args.dataset, recursive=args.recursive)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if args.rand is not None and args.rand < 1:
        print("--rand must be at least 1.", file=sys.stderr)
        return 2
    if args.rand is not None and args.seed is None:
        args.seed = random.SystemRandom().randrange(1, 2**31)
    log_path = log_path_for(models)
    open_log(log_path)
    try:
        _say("models: " + ", ".join(spec.name for spec in models))
        _say("datasets: " + ", ".join(spec.name for spec in datasets))
        if any(spec.kind == "openjev" for spec in models):
            _say("device: cpu" if args.cpu else "device: gpu (CUDA on Linux, MPS on Mac)")
        if args.rand is not None:
            _say(f"rand: {args.rand} seed: {args.seed}")
        for spec in datasets:
            if spec.recursive:
                continue
            hidden = nested_document_count(spec.path)
            if hidden:
                _say(f"{spec.path}: {hidden} documents are in subdirectories. Pass -r to read them.")
        try:
            payload = run_selection(
                models,
                datasets,
                Path(args.output),
                args.threshold,
                rand=args.rand,
                seed=args.seed,
                cpu=args.cpu,
            )
        except (OSError, ValueError, RuntimeError) as exc:
            _say(str(exc))
            print(str(exc), file=sys.stderr)
            return 3
        report_path = log_path.with_name(log_path.name.replace("_run.txt", "_report.html"))
        write_report(payload, report_path)
        _say(render_summary(payload).rstrip())
        _say(f"wrote {args.output}")
        _say(f"log {log_path}")
        _say(f"report {report_path}")
    finally:
        close_log()
    runs = payload["runs"]
    assert isinstance(runs, list)
    if any(isinstance(run, dict) and run.get("error") for run in runs):
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
