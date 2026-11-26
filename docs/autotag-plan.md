# AutoTag plan

AutoTag assigns two labels to a document. Industry is one name from a list of sixteen, or `other` when the score is below 0.5. Document type is one of ten fixed labels, or `other` on the same rule. The code is on branch `AutoTag` under `autotag/`. The trainer, `CLAUDE.md`, and `pyproject.toml` stay unchanged. OpenJev's Python packages are listed in `autotag/requirements-openjev.txt` and installed into a separate virtualenv.

## Decisions

1. The CPU model is `xxparthparekhxx/openjev-flash` (~150M). The 71M checkpoint remains the fallback if flash does not load. jevos-v5 is not the target. A generic DeBERTa MNLI model is not the target.
2. Industry is a short named list, written in `autotag/labels.py`. The model picks one industry from that list, or abstains. It does not propose a phrase copied from the page. The earlier open-phrase run is retired: winners such as `quarter ending` and `closes six` were page bigrams, not industries.
3. Document type stays the fixed ten: invoice, receipt, contract, policy, report, letter, email, memo, specification, financial_statement. One best label per axis. If the winning score is below 0.5, the label is `other`.
4. The measurement is a CPU accuracy run on the fixtures, not a tagging UI. There is still no human-labeled set and no comparison LLM, so fixture-intent counts are not an agreement rate.

## Industry list

Sixteen names. Each question is asked once. OpenJev-flash scores the statement form of the question (the answer "yes"), because the checkpoint is an entailment model and does not answer a question by generating text. P(entailment) is not renormalized across the sixteen, so the best industry can still fall under 0.5.

| Id | Yes/no question |
|---|---|
| healthcare | Is this document mainly about the healthcare industry (hospitals, clinics, patients, or medical care)? |
| software | Is this document mainly about the software industry (software products, SaaS, or hosted software platforms)? |
| finance | Is this document mainly about the finance industry (banking, lending, or investment)? |
| insurance | Is this document mainly about the insurance industry (insurance policies, premiums, or claims)? |
| legal | Is this document mainly about the legal industry (law firms, attorneys, or legal services)? |
| education | Is this document mainly about the education industry (schools, universities, or teaching)? |
| energy | Is this document mainly about the energy industry (electricity, oil, gas, or utilities)? |
| manufacturing | Is this document mainly about the manufacturing industry (factories or making physical goods)? |
| retail | Is this document mainly about the retail industry (selling goods to consumers in stores)? |
| transportation | Is this document mainly about the transportation industry (freight, shipping, or passenger transport)? |
| real_estate | Is this document mainly about the real estate industry (property sales, leasing, or real estate)? |
| construction | Is this document mainly about the construction industry (building construction or contractors)? |
| government | Is this document mainly about the government industry (government or public administration)? |
| hospitality | Is this document mainly about the hospitality industry (hotels, restaurants, or food service)? |
| telecommunications | Is this document mainly about the telecommunications industry (telephone, internet access, or broadcasting)? |
| agriculture | Is this document mainly about the agriculture industry (farming, crops, or livestock)? |

This is a compact common-industry list. It is not NAICS, and nothing in the list forces a SaaS document to a numeric sector code.

## Document type

| Id | Yes/no question |
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

Industry is scored first, then document type, on the same text. Document type does not depend on the industry label.

## Model

| Field | Value |
|---|---|
| Model id | `xxparthparekhxx/openjev-flash` |
| Revision | `9f754e6582c1c5cb7f03bf24832cd64388fc004f` |
| Device | CPU, 4 threads, float32 |
| Score | P(entailment) from the 3-way softmax |
| Threshold | 0.5, unchanged from the previous run, not refit on these scores |
| Record | `autotag/measurements/cpu-accuracy.json` |

## CPU results on the fixtures

Every industry label below is one of the sixteen names, or `other`. None is a span copied from the page. The parenthetical score on an abstention is the best industry in the list, which stayed under 0.5.

| Document | Industry | Document type | Fixture intent (type) |
|---|---|---|---|
| hospital_invoice.txt | healthcare (0.813) | invoice (0.850) | invoice |
| discharge_summary.txt | other (healthcare 0.437) | report (0.642) | report |
| saas_agreement.txt | other (software 0.457) | contract (0.965) | contract |
| payment_receipt.txt | other (healthcare 0.456) | receipt (0.814) | receipt |
| privacy_policy.txt | other (healthcare 0.208) | policy (0.918) | policy |
| customer_letter.txt | other (healthcare 0.249) | letter (0.596) | letter |
| status_email.txt | other (healthcare 0.289) | email (0.963) | email |
| team_memo.txt | other (healthcare 0.332) | memo (0.909) | memo |
| api_specification.txt | other (healthcare 0.254) | other (specification 0.391) | specification |
| income_statement.txt | healthcare (0.535) | other (report 0.326) | financial_statement |
| kitchen_note.txt | other (retail 0.103) | other (0.055) | other |

Two documents receive an industry: the hospital invoice and the hospital income statement, both healthcare. The other nine abstain. On the hospital discharge summary, the receipt, and the SaaS agreement, the leading industry is the sensible one (healthcare or software) and sits just under 0.5 (0.437, 0.456, 0.457). The grocery note's best industry is retail at 0.103.

Document type matches the fixture file on 9 of 11. The specification sample abstains with specification at 0.391. The income statement abstains with report at 0.326. That 9/11 count is a check against text written with the samples. It is not agreement with a person or with a large language model. No industry gold file exists, so this run does not score industry against an intent.

On this process, with the weights already cached: load 2.9 s, then about 0.17–0.31 s per document. Peak resident set 701 MB.

## Still missing for a real agreement number

1. Documents nobody on this branch wrote, on the order of 50.
2. At least two people labeling both axes from these questions, plus an adjudicated label where they disagree.
3. One named large language model, temperature 0, asked the same questions.
4. A threshold chosen on one slice and scored on the rest. 0.5 is still the predeclared cutoff. Several industry scores land between 0.43 and 0.46, so the cutoff moves those rows between a label and `other`.
5. `autotag/agreement.py` computes accuracy and Cohen's kappa. It has no human or LLM labels to score.

## How to run it again

The model path is Python 3.14 and PyTorch 2.14.1. `.python-version` selects 3.14. The requirements file installs the CPU wheel `torch==2.14.1+cpu` on Linux and Windows. Apple Silicon installs PyPI's `torch==2.14.1` wheel, because the CPU index publishes no macOS build. `OpenJevBackend.load` refuses any other interpreter or torch version. The check compares the version before the `+` suffix, so both `2.14.1` and `2.14.1+cpu` pass.

```bash
uv venv autotag/.venv --python 3.14
uv pip install --python autotag/.venv/bin/python -r autotag/requirements-openjev.txt
autotag/.venv/bin/python -m autotag accuracy
```

`python -m autotag.selfcheck` uses the lexicon stub and does not load the checkpoint.
