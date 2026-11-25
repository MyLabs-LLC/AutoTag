# AutoTag

AutoTag reads a plain-text document and assigns two labels:

- **Industry.** One of sixteen named industries, or `other`.
- **Document type.** One of ten document types, or `other`.

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

## Document types

| Id | Question |
|---|---|
| invoice | Is this document an invoice asking for payment? |
| receipt | Is this document a receipt confirming a payment already made? |
| contract | Is this document a contract or agreement between parties? |
| policy | Is this document a policy that states rules someone must follow? |
| report | Is this document a report that presents findings or results? |
| letter | Is this document a letter addressed to a person or organisation? |
| email | Is this document an email message with a sender and a subject? |
| memo | Is this document an internal memorandum to colleagues? |
| specification | Is this document a specification of requirements for a product or system? |
| financial_statement | Is this document a financial statement such as a balance sheet, income statement, or cash flow statement? |

## Run without the model

From this repository root, with the standard library only:

```bash
python -m autotag classify autotag/fixtures/hospital_invoice.txt
python -m autotag cluster autotag/fixtures
python -m autotag.selfcheck
```

The default backend is a deterministic lexicon (`stub`) so the commands run with no download. Stub scores are not the OpenJev numbers in the plan.

## Run OpenJev-flash on CPU

`torch` and `transformers` are not installed by a project lockfile. Create a virtualenv and install the CPU wheel first. See `autotag/requirements-openjev.txt`.

```bash
uv venv autotag/.venv --python 3.12
uv pip install --python autotag/.venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python autotag/.venv/bin/python -r autotag/requirements-openjev.txt
autotag/.venv/bin/python -m autotag accuracy
```

That command classifies every file in `autotag/fixtures/` and rewrites `autotag/measurements/cpu-accuracy.json`. Weights download into the Hugging Face cache. Do not commit the virtualenv or the weights.

On the recorded CPU run (float32, 4 threads, threshold 0.5), two documents received an industry, both `healthcare` (the hospital invoice at 0.813 and the hospital income statement at 0.535). The other nine abstained. Document type matched the fixture file on 9 of 11. The full table is in the plan.
