# AutoTag package

Two labels on a document:

- **Industry**, one name from the sixteen industries in `labels.py`, or
  `other` when the score is below the threshold.
- **Document type**, one of ten: invoice, receipt, contract, policy, report,
  letter, email, memo, specification, financial_statement. One best label, or
  `other` when the score is below the threshold.

The project README at the repository root is the full description.

The CPU model is `xxparthparekhxx/openjev-flash` (about 150M). If that
checkpoint does not load, the accuracy command tries
`RazvanManolache/raz-systemone-nli-xsmall-openjev` (71M). Both are entailment
models: they score a hypothesis, they do not generate a new sentence. Each
industry and each document type is one yes/no question. The model scores the
statement form of that question (the answer yes). The score is P(entailment).
It is not renormalized across labels, so a weak best label can abstain.

## Run

From the repository root, one command controls every model and dataset:

```bash
python -m autotag
python -m autotag --model stub --dataset fixtures
python -m autotag --model stub --dataset fixtures --rand 100 --seed 77
```

`--model` is `stub`, `openjev-flash`, `openjev-71m`, `gemini-4-hydrogen`, `all`, or a Hugging Face
id. `--dataset` is `fixtures`, `all`, a directory of documents, or one
document. `-r` reads subdirectories. The model sees the first 16 KiB of each
file. Directories added under `autotag/datasets/` join `all`.
The JSON record is `autotag/measurements/run.json`. The transcript is
`output/YYMMDD_HHMMSS_<model>_run.txt`. `--rand N` scores N random
documents. `--seed` repeats that choice.

`stub` needs only the standard library. OpenJev needs the environment in
`requirements-openjev.txt`:

```bash
uv venv autotag/.venv --python 3.14
uv pip install --python autotag/.venv/bin/python -r autotag/requirements-openjev.txt
autotag/.venv/bin/python -m autotag
```

`autotag/fixtures/intent.json` is the document type each sample was written
as. Matching it is a fixture check. It is not agreement with a person or
with a large language model. `python -m autotag.selfcheck` checks the stub
and does not load a checkpoint.

## Layout

| Path | Role |
|---|---|
| `labels.py` | Industries and document types, one question each |
| `classify.py` | Stub backend and the one-label-or-abstain rule |
| `openjev_backend.py` | CPU OpenJev entailment |
| `catalog.py` | Model and dataset names the command accepts |
| `accuracy.py` | Scores each selected model on each selected dataset |
| `cluster.py` | Lexical clustering from the first exploration |
| `fixtures/` | Plain-text samples and `intent.json` |
