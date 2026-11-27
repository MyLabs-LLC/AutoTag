# AutoTag

Score each document on two closed lists: one industry and one document type. Abstention prints as `Null`. The printed tag is `[industry::doc_type]`.

One command runs everything, from the repository root:

```bash
autotag/.venv/bin/python -m autotag --model compare \
  --dataset ../data/v0.6.3_cluster_files -r --rand 1000 --seed 77
```

`--model all` is only `stub`, `openjev-flash`, `openjev-71m`, and `gemini-4-hydrogen`. `--model compare` is the bake-off, in this order:

1. `gemini-4-hydrogen` — Gemini API, `GEMINI_API_KEY`
2. `composer-2.5` — Composer 2.5 through the Cursor API, `CURSOR_API_KEY`
3. `grok-4.7-medium-api` — Grok 4.7 Medium through the xAI API, `GROK_API_KEY`, `reasoning_effort=medium`
4. `nemotron-3.5-bolt` — local Nemotron through Unsloth Studio at `http://127.0.0.1:8888/v1`
5. The local entailment models in `COMPARE_MODELS`

A Hugging Face id with a `/` is accepted as another entailment model. `-r` walks subdirectories. `--rand` and `--seed` pick the same files for every model. `--cpu` forces the CPU. The default device is CUDA on Linux and MPS on Mac.

## What a run writes

- Transcript: `output/YYMMDD_HHMMSS_<name>_run.txt`. Many models use a short name such as `12models_run.txt`. Keep the filename under 200 bytes.
- Review page: `output/..._report.html`. Rows are documents, columns are models, cells are the guessed tag. Open this file to review a run.
- JSON: `autotag/measurements/run.json`, written only when the run returns.

Keys live in `.env`. Do not commit `.env`, `output/`, or `run.json`. `HUGGINGFACE_API_KEY` is copied to `HF_TOKEN`. For Cursor, prefer the `.env` value of `CURSOR_API_KEY` over a key already injected into the shell.

## Layout

| Path | Role |
|---|---|
| `autotag/__main__.py` | The one CLI |
| `autotag/catalog.py` | Model and dataset names. `all` and `compare` are separate lists |
| `autotag/labels.py` | Industry and document-type questions |
| `autotag/extract.py` | First 16 KiB of text. Kreuzberg runs in a child process |
| `autotag/openjev_backend.py` | Local entailment. Premise is cut to 512 tokens |
| `autotag/gemini_backend.py` | Gemini 4 Hydrogen |
| `autotag/grok_backend.py` | Composer 2.5 via Cursor, no tools |
| `autotag/xai_backend.py` | Grok 4.7 Medium via `GROK_API_KEY` |
| `autotag/nemotron_backend.py` | Local Nemotron via the Unsloth Studio API |
| `autotag/accuracy.py` | Scores models and writes the transcript |
| `autotag/report.py` | The HTML review page |

Entailment models need Python 3.14 and PyTorch 2.14.1 from `autotag/requirements-openjev.txt`. The stub path stays on the standard library. A binary checkpoint labeled `entailment` / `not_entailment` is valid. A three-way checkpoint uses contradiction versus entailment, and neutral is left out of the score.

Labels are not renormalized across the list. Below 0.2, that side is `Null`.
