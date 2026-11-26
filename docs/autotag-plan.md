# AutoTag plan

AutoTag assigns two labels to a document. Industry is one everyday name from a list of twenty-eight, or `other` when the score is below 0.5. Document type is one of twenty plain names, or `other` on the same rule. The project is the `autotag` package in this repository. OpenJev's Python packages are listed in `autotag/requirements-openjev.txt`. The runtime is Python 3.14 and PyTorch 2.14.1 (`torch==2.14.1+cpu`).

## Decisions

1. The CPU model is `xxparthparekhxx/openjev-flash` (~150M). The 71M checkpoint remains the fallback if flash does not load. jevos-v5 is not the target. A generic DeBERTa MNLI model is not the target.
2. Industry is a short named list, written in `autotag/labels.py`. The model picks one industry from that list, or abstains. It does not propose a phrase copied from the page. The earlier open-phrase run is retired: winners such as `quarter ending` and `closes six` were page bigrams, not industries.
3. Document type is a plain-language list: invoice, receipt, contract, policy, report, letter, email, memo, specification, financial statement, proposal, purchase order, resume, presentation, manual, press release, meeting notes, form, quote, and article. One best label per axis. If the winning score is below 0.5, the label is `other`. The words are what a person would say, not codes.
4. The measurement is a CPU accuracy run on the fixtures, not a tagging UI. There is still no human-labeled set and no comparison LLM, so fixture-intent counts are not an agreement rate.

## Industry list

Twenty-eight names a person would say at a glance: healthcare, not a sector code. Each question is asked once. OpenJev-flash scores the statement form of the question (the answer "yes"). The yes-score is P(entailment | entailment or contradiction) for that question alone, so the best industry can still fall under 0.5.

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
| media | Is this document mainly about the media industry (news, publishing, film, or television)? |
| marketing | Is this document mainly about the marketing industry (advertising, campaigns, or promoting a brand)? |
| nonprofit | Is this document mainly about the nonprofit industry (charities, donations, or nonprofit organizations)? |
| pharmaceuticals | Is this document mainly about the pharmaceuticals industry (drug makers, pharmacies, or medicines)? |
| automotive | Is this document mainly about the automotive industry (cars, vehicle makers, or car dealers)? |
| food | Is this document mainly about the food and beverage industry (food makers, groceries, or packaged drinks)? |
| consulting | Is this document mainly about the consulting industry (advising a client or a consulting engagement)? |
| staffing | Is this document mainly about the staffing industry (recruiting, temp agencies, or placing workers)? |
| mining | Is this document mainly about the mining industry (mines, ore, or quarries)? |
| aerospace | Is this document mainly about the aerospace industry (aircraft, spacecraft, or satellites)? |
| logistics | Is this document mainly about the logistics industry (warehousing, fulfillment, or moving goods for others)? |
| sports | Is this document mainly about the sports industry (teams, leagues, stadiums, or sporting events)? |

The names are everyday words. The list is not NAICS, and a SaaS document is not forced to a numeric sector code.

## Document type

| Id | Yes/no question |
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

Industry is scored first, then document type, on the same text. Document type does not depend on the industry label. The premise sent to the model is the first 4,000 characters. The fixtures are shorter than that, so the cap does not change them. It is there for longer files, where the jevos guide says the type is visible at the start and the rest of the file buries it.

## What the 2024–2026 sources say, and what this repo uses

| Source | Finding | Adopted |
|---|---|---|
| [jevos document-classification guide](https://github.com/feder-cr/jev/wiki/document-classification-with-a-local-llm) and [yes/no zero-shot guide](https://github.com/feder-cr/jev/wiki/zero-shot-text-classification-yes-no-questions) | One yes/no question per label. Put the distinguishing feature in the question. Use "mainly", keep questions parallel, and abstain below a threshold chosen on labeled documents. Do not add an "other" question. Probabilities are independent and do not sum to 1. Prefer the first page over the whole file. | Already the label design. Added the 4,000-character premise cap. Left the 0.5 cutoff where it was: these fixtures are not a labeled tuning set. |
| [Hugging Face zero-shot pipeline](https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/zero_shot_classification.py) and Climate-NLI (PACLIC 2024, [anthology](https://aclanthology.org/2024.paclic-1.57.pdf)) | When labels are independent, the yes-score is a softmax of the entailment logit against the contradiction logit. A softmax across labels is a different mode, and it cannot abstain. | Yes. Each question is now scored that way. The previous 3-way P(entailment) left neutral mass in the score, so healthcare and software sat at 0.43–0.46 and abstained. |
| [openjev-flash model card](https://huggingface.co/xxparthparekhxx/openjev-flash) | A 150M entailment model checks the text in front of it. It is weak when the answer is not in the premise, and its scores are not calibrated across tasks (published JevBench ECE about 0.14–0.18). | Stayed on this checkpoint. Did not ask it to invent an industry code from world knowledge. |
| ANLP 2025, [From NLI to Classification](https://www.anlp.jp/proceedings/annual_meeting/2025/pdf_dir/P6-8.pdf) | On their biomedical set, a bare label hypothesis beat a "this text is about" gloss, using BART-large-MNLI. | Not adopted. The jevos guide and the company-classification result below both say the gloss is what a small entailment model can check. No held-out set to re-pick the wording. |
| MDPI Information, 31 Jan 2024, [company classification](https://www.mdpi.com/2078-2489/15/2/77) | Zero-shot GICS with sector names scored F1 0.56; TF-IDF-expanded names scored 0.64; a trained RoBERTa scored 0.81. | Kept the short glosses already in the questions. Did not add a TF-IDF label rewriter or a supervised head. |
| GLiClass, arXiv [2508.07662](https://arxiv.org/abs/2508.07662) (Aug 2025) and [gliclass-v4-edge](https://huggingface.co/knowledgator/gliclass-v4-edge) | One encoder pass scores every label. The edge model is built for CPUs. Few-shot (8 examples per label) lifts the small models by a large margin. DeBERTa backbones beat ModernBERT in their table. | Not adopted. It is a new package and a different checkpoint. It is the CPU candidate to try if OpenJev-flash is replaced, after a labeled set exists. |
| SetFit zero-shot ([Hugging Face docs](https://huggingface.co/docs/setfit/how_to/zero_shot)) | Label-name embeddings from a small bi-encoder (they use `BAAI/bge-small-en-v1.5`) are a fast baseline. The published latency comparison is on GPU. Few-shot SetFit needs examples. | Not adopted. No labeled examples, and it is a second model stack. |
| BEACON (Journal of Data Science, 2025, [doi 10.6339/25-jds1180](https://doi.org/10.6339/25-jds1180)) and CLIPS (EPICOH 2025, [doi 10.1136/oemed-2025-epicohabstracts.59](https://doi.org/10.1136/oemed-2025-epicohabstracts.59)) | Census and job-text systems that rank NAICS codes. CLIPS top-1 agreement with an expert was 47.4% at 5 digits; the expert code was in the top 10 for 80%. | Not adopted. The industry list stays the twenty-eight everyday names. NAICS is a different task and both systems need training text. |
| [OpenJev-Flash-9B](https://huggingface.co/openjev/OpenJev-Flash-9B) and the [OpenRouter Jev cookbook](https://openrouter.ai/docs/cookbook/evaluate-and-optimize/jev-classification) | Same noul / choice API. A `choice` question is for labels that cannot overlap. A 9B model is the small member of that family. | Not adopted. 9B is outside this CPU budget. A single `choice` softmax cannot abstain the way an independent threshold can, so the runtime stays one noul-style question per label. |

## Model

| Field | Value |
|---|---|
| Model id | `xxparthparekhxx/openjev-flash` |
| Revision | `9f754e6582c1c5cb7f03bf24832cd64388fc004f` |
| Runtime | Python 3.14, PyTorch 2.14.1 CPU (`torch==2.14.1+cpu`), 4 threads, float32 |
| Score | P(entailment given entailment or contradiction), per question. Not renormalized across labels. |
| Threshold | 0.5, not refit on these scores |
| Record | `autotag/measurements/cpu-accuracy.json` |

## CPU results on the fixtures

Every industry label below is one of the twenty-eight names, or `other`. None is a code or a span copied from the page. The parenthetical score on an abstention is the best industry in the list, which stayed under 0.5.

| Document | Industry | Document type | Fixture intent (type) |
|---|---|---|---|
| hospital_invoice.txt | healthcare (0.919) | invoice (0.963) | invoice |
| discharge_summary.txt | healthcare (0.726) | report (0.884) | report |
| saas_agreement.txt | software (0.777) | contract (0.987) | contract |
| payment_receipt.txt | healthcare (0.782) | receipt (0.987) | receipt |
| privacy_policy.txt | healthcare (0.839) | policy (0.988) | policy |
| customer_letter.txt | healthcare (0.775) | letter (0.957) | letter |
| status_email.txt | other (healthcare 0.363) | email (0.976) | email |
| team_memo.txt | other (healthcare 0.457) | memo (0.982) | memo |
| api_specification.txt | healthcare (0.611) | specification (0.977) | specification |
| income_statement.txt | healthcare (0.861) | specification (0.903) | financial_statement |
| kitchen_note.txt | other (logistics 0.492) | other (0.462) | other |

Eight documents receive an industry. Hospital documents are healthcare. The SaaS agreement is software. The grocery note abstains (logistics 0.492). The lab-results email and the nursing memo abstain, with healthcare at 0.363 and 0.457.

Document type matches the fixture file on 10 of 11. The income statement is the miss: specification is 0.903 and financial statement is 0.731, so the one-label rule keeps specification. A person looking at the title "INCOME STATEMENT" would say financial statement. The questions were not tuned further on this one file.

That 10/11 figure is a check against text written with the samples. It is not agreement with a person or with a large language model. No industry gold file exists.

This run used Python 3.14.10 and `torch==2.14.1+cpu`. Load 2.3 s with weights cached, then about 0.33–0.55 s per document. Peak resident set 757 MB.

## Still missing for a real agreement number

1. Documents nobody on this branch wrote, on the order of 50.
2. At least two people labeling both axes from these questions, plus an adjudicated label where they disagree.
3. One named large language model, temperature 0, asked the same questions.
4. A threshold chosen on one slice and scored on the rest. 0.5 is still the predeclared cutoff. The nursing memo's healthcare score is 0.457, so a small move in the cutoff changes that row.
5. `autotag/agreement.py` computes accuracy and Cohen's kappa. It has no human or LLM labels to score.

## How to run it again

The model path is Python 3.14 and PyTorch 2.14.1. `.python-version` selects 3.14. The requirements file installs the CPU wheel `torch==2.14.1+cpu` on Linux and Windows. Apple Silicon installs PyPI's `torch==2.14.1` wheel, because the CPU index publishes no macOS build. `OpenJevBackend.load` refuses any other interpreter or torch version. The check compares the version before the `+` suffix, so both `2.14.1` and `2.14.1+cpu` pass.

```bash
uv venv autotag/.venv --python 3.14
uv pip install --python autotag/.venv/bin/python -r autotag/requirements-openjev.txt
autotag/.venv/bin/python -m autotag accuracy
```

`python -m autotag.selfcheck` uses the lexicon stub and does not load the checkpoint.
