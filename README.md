# AutoTag

AutoTag reads a plain-text document and assigns two labels:

- **Industry.** One of twenty-eight everyday industry names, or `other`.
- **Document type.** One name from the document-type list, or no type when the score is below 0.5. The list is the original twenty plus the types that appear in the cluster corpus.

Each label is a separate yes/no question. The score is how strongly the text entails the “yes” form of that question. The best score wins. If it is below **0.5**, the label is `other`. Scores are not forced to add up to 1, so abstaining is possible on both axes. Industry is chosen from the list below. The model does not invent a phrase from the page.

The CPU checkpoint used for the recorded run is [`xxparthparekhxx/openjev-flash`](https://huggingface.co/xxparthparekhxx/openjev-flash) (about 150M parameters, Ettin encoder). It scores a premise against a hypothesis. It does not generate text. If that checkpoint fails to load, the accuracy command tries `RazvanManolache/raz-systemone-nli-xsmall-openjev` (71M).

There is **no human/LLM agreement rate yet**. The fixture file `autotag/fixtures/intent.json` only records the document type each sample was written as. Matching it is a check against those samples, not agreement with a person or with a large language model. No industry gold set exists. The design notes and the latest CPU table are in [`docs/autotag-plan.md`](docs/autotag-plan.md). The raw scores are in `autotag/measurements/cpu-accuracy.json`.

## Industries

One question each: “Is this document mainly about the {name} industry ({gloss})?”

| Id | Gloss |
|---|---|
| healthcare | hospitals, clinics, patients, or medical care |
| software | software products, SaaS, or hosted software platforms |
| finance | banking, lending, or investment |
| insurance | insurance policies, premiums, or claims |
| legal | law firms, attorneys, or legal services |
| education | schools, universities, or teaching |
| energy | electricity, oil, gas, or utilities |
| manufacturing | factories or making physical goods |
| retail | selling goods to consumers in stores |
| transportation | freight, shipping, or passenger transport |
| real_estate | property sales, leasing, or real estate |
| construction | building construction or contractors |
| government | government or public administration |
| hospitality | hotels, restaurants, or food service |
| telecommunications | telephone, internet access, or broadcasting |
| agriculture | farming, crops, or livestock |
| media | news, publishing, film, or television |
| marketing | advertising, campaigns, or promoting a brand |
| nonprofit | charities, donations, or nonprofit organizations |
| pharmaceuticals | drug makers, pharmacies, or medicines |
| automotive | cars, vehicle makers, or car dealers |
| food | food makers, groceries, or packaged drinks |
| consulting | advising a client or a consulting engagement |
| staffing | recruiting, temp agencies, or placing workers |
| mining | mines, ore, or quarries |
| aerospace | aircraft, spacecraft, or satellites |
| logistics | warehousing, fulfillment, or moving goods for others |
| sports | teams, leagues, stadiums, or sporting events |

## Document types

| Id | Question |
|---|---|
| invoice | Is this document an invoice asking for payment? |
| receipt | Is this document a receipt confirming a payment already made? |
| contract | Is this document a contract or agreement between parties? |
| policy | Is this document a policy that states rules someone must follow? |
| report | Is this document a narrative report of findings, an investigation, or a clinical course? |
| letter | Is this document a letter addressed to a person or organisation? |
| email | Is this document an email message with a sender and a subject? |
| memo | Is this document an internal memorandum to colleagues? |
| specification | Is this document a specification of requirements for a product or system? |
| financial_statement | Is this document a financial statement, such as an income statement, a balance sheet, or a cash flow statement? |
| proposal | Is this document a proposal offering work or a plan for someone to accept? |
| purchase_order | Is this document a purchase order telling a seller what to supply? |
| resume | Is this document a resume or CV describing a person's jobs and education? |
| presentation | Is this document a presentation or a slide deck? |
| manual | Is this document a manual or user guide explaining how to do something? |
| press_release | Is this document a press release announcing news to the public? |
| meeting_notes | Is this document meeting notes or minutes of a discussion? |
| form | Is this document a form with blanks for someone to fill in? |
| quote | Is this document a price quote or an estimate for work or goods? |
| article | Is this document a news article or a magazine article written for readers? |

## Run

One command scores every selected model on every selected dataset. With no flags it runs all of them: `stub`, `openjev-flash`, `openjev-71m`, and `gemini-4-hydrogen`, on every dataset (`fixtures`, plus any directory of `.txt` files under `autotag/datasets/`). Gemini 4 Hydrogen and Hugging Face downloads use the keys in `.env`.

```bash
python -m autotag
```

The same command is the control surface. Name a model, a dataset, or both:

```bash
python -m autotag --model stub --dataset fixtures
python -m autotag --model openjev-flash --dataset fixtures
python -m autotag --model stub,openjev-flash --dataset fixtures
python -m autotag --model all --dataset autotag/fixtures/hospital_invoice.txt
python -m autotag --model stub --dataset ../data/v0.6.3_cluster_files -r
```

`--model` accepts `stub`, `openjev-flash`, `openjev-71m`, `all`, or a Hugging Face model id. `--dataset` accepts `fixtures`, `all`, a directory of documents, or one document. `-r` reads subdirectories. The model sees the first 16 KiB of each file. PDF, Word, Excel, PowerPoint, RTF, HTML, XML, JSON, and CSV are read with Kreuzberg. `--threshold` defaults to 0.5. The JSON record is `autotag/measurements/run.json` unless `--output` says otherwise.

`stub` is a deterministic lexicon and needs no download. Its scores are not the OpenJev numbers in the plan. OpenJev models need the environment below. A model that fails to load is recorded in the JSON, and the command continues with the other models.

## Run OpenJev on CPU

The model path requires **Python 3.14** and **PyTorch 2.14.1**. `.python-version` pins the interpreter. `autotag/requirements-openjev.txt` pins the CPU wheel on Linux and Windows (`2.14.1+cpu`). macOS has no `+cpu` wheel, so Apple Silicon installs PyPI's `2.14.1` build. Loading the checkpoint on any other Python or PyTorch version stops with an error.

```bash
uv venv autotag/.venv --python 3.14
uv pip install --python autotag/.venv/bin/python -r autotag/requirements-openjev.txt
autotag/.venv/bin/python -m autotag
```

That command classifies every selected dataset and writes `autotag/measurements/run.json`. Weights download into the Hugging Face cache. Do not commit the virtualenv or the weights. `python -m autotag.selfcheck` still checks the stub lexicon and does not load a checkpoint.

On the recorded CPU run (Python 3.14, PyTorch 2.14.1+cpu, float32, threshold 0.5), eight documents receive an industry (`healthcare`, or `software` for the SaaS agreement). The grocery note, the lab-results email, and the nursing memo abstain. Document type matches the fixture file on 10 of 11. The income statement is labeled `specification` (0.903) over `financial_statement` (0.731). That count is not agreement with a person or a large language model. The full table is in the plan.
