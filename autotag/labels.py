"""Closed label spaces for industry and document type.

Industry is one name from INDUSTRIES. Document type is one name from DOC_TYPES.
Each label has one yes/no question. OpenJev scores the declarative hypothesis,
which is that question with the answer yes. Abstention is a threshold in the
caller, not an extra question.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Label:
    id: str
    name: str
    question: str
    hypothesis: str


def _industry(id_: str, name: str, gloss: str) -> Label:
    question = f"Is this document mainly about the {name} industry ({gloss})?"
    hypothesis = f"This document is mainly about the {name} industry ({gloss})."
    return Label(id_, name, question, hypothesis)


def _doc_type(id_: str, name: str, question: str, hypothesis: str) -> Label:
    return Label(id_, name, question, hypothesis)


def _article(name: str) -> str:
    return "an" if name[:1].lower() in "aeiou" else "a"


def _from_id(id_: str) -> Label:
    """A document type named by its id, asked as one yes/no question."""
    name = id_.replace("_", " ")
    article = _article(name)
    return _doc_type(
        id_,
        name,
        f"Is this document {article} {name}?",
        f"This document is {article} {name}.",
    )


# Sixteen common industries. Names are the labels. This is not a NAICS encoding.
INDUSTRIES: tuple[Label, ...] = (
    _industry("healthcare", "healthcare", "hospitals, clinics, patients, or medical care"),
    _industry("software", "software", "software products, SaaS, or hosted software platforms"),
    _industry("finance", "finance", "banking, lending, or investment"),
    _industry("insurance", "insurance", "insurance policies, premiums, or claims"),
    _industry("legal", "legal", "law firms, attorneys, or legal services"),
    _industry("education", "education", "schools, universities, or teaching"),
    _industry("energy", "energy", "electricity, oil, gas, or utilities"),
    _industry("manufacturing", "manufacturing", "factories or making physical goods"),
    _industry("retail", "retail", "selling goods to consumers in stores"),
    _industry("transportation", "transportation", "freight, shipping, or passenger transport"),
    _industry("real_estate", "real estate", "property sales, leasing, or real estate"),
    _industry("construction", "construction", "building construction or contractors"),
    _industry("government", "government", "government or public administration"),
    _industry("hospitality", "hospitality", "hotels, restaurants, or food service"),
    _industry("telecommunications", "telecommunications", "telephone, internet access, or broadcasting"),
    _industry("agriculture", "agriculture", "farming, crops, or livestock"),
    _industry("media", "media", "news, publishing, film, or television"),
    _industry("marketing", "marketing", "advertising, campaigns, or promoting a brand"),
    _industry("nonprofit", "nonprofit", "charities, donations, or nonprofit organizations"),
    _industry("pharmaceuticals", "pharmaceuticals", "drug makers, pharmacies, or medicines"),
    _industry("automotive", "automotive", "cars, vehicle makers, or car dealers"),
    _industry("food", "food and beverage", "food makers, groceries, or packaged drinks"),
    _industry("consulting", "consulting", "advising a client or a consulting engagement"),
    _industry("staffing", "staffing", "recruiting, temp agencies, or placing workers"),
    _industry("mining", "mining", "mines, ore, or quarries"),
    _industry("aerospace", "aerospace", "aircraft, spacecraft, or satellites"),
    _industry("logistics", "logistics", "warehousing, fulfillment, or moving goods for others"),
    _industry("sports", "sports", "teams, leagues, stadiums, or sporting events"),
)

DOC_TYPES: tuple[Label, ...] = (
    _doc_type(
        "invoice",
        "Invoice",
        "Is this document an invoice asking for payment?",
        "This document is an invoice asking for payment.",
    ),
    _doc_type(
        "receipt",
        "Receipt",
        "Is this document a receipt confirming a payment already made?",
        "This document is a receipt confirming a payment already made.",
    ),
    _doc_type(
        "contract",
        "Contract",
        "Is this document a contract or agreement between parties?",
        "This document is a contract or agreement between parties.",
    ),
    _doc_type(
        "policy",
        "Policy",
        "Is this document a policy that states rules someone must follow?",
        "This document is a policy that states rules someone must follow.",
    ),
    _doc_type(
        "report",
        "Report",
        "Is this document a narrative report of findings, an investigation, or a clinical course?",
        "This document is a narrative report of findings, an investigation, or a clinical course.",
    ),
    _doc_type(
        "letter",
        "Letter",
        "Is this document a letter addressed to a person or organisation?",
        "This document is a letter addressed to a person or organisation.",
    ),
    _doc_type(
        "email",
        "Email",
        "Is this document an email message with a sender and a subject?",
        "This document is an email message with a sender and a subject.",
    ),
    _doc_type(
        "memo",
        "Memo",
        "Is this document an internal memorandum to colleagues?",
        "This document is an internal memorandum to colleagues.",
    ),
    _doc_type(
        "specification",
        "Specification",
        "Is this document a specification of requirements for a product or system?",
        "This document is a specification of requirements for a product or system.",
    ),
    _doc_type(
        "financial_statement",
        "Financial statement",
        "Is this document a financial statement, such as an income statement, a balance sheet, or a cash flow statement?",
        "This document is a financial statement, such as an income statement, a balance sheet, or a cash flow statement.",
    ),
    _doc_type(
        "proposal",
        "Proposal",
        "Is this document a proposal offering work or a plan for someone to accept?",
        "This document is a proposal offering work or a plan for someone to accept.",
    ),
    _doc_type(
        "purchase_order",
        "Purchase order",
        "Is this document a purchase order telling a seller what to supply?",
        "This document is a purchase order telling a seller what to supply.",
    ),
    _doc_type(
        "resume",
        "Resume",
        "Is this document a resume or CV describing a person's jobs and education?",
        "This document is a resume or CV describing a person's jobs and education.",
    ),
    _doc_type(
        "presentation",
        "Presentation",
        "Is this document a presentation or a slide deck?",
        "This document is a presentation or a slide deck.",
    ),
    _doc_type(
        "manual",
        "Manual",
        "Is this document a manual or user guide explaining how to do something?",
        "This document is a manual or user guide explaining how to do something.",
    ),
    _doc_type(
        "press_release",
        "Press release",
        "Is this document a press release announcing news to the public?",
        "This document is a press release announcing news to the public.",
    ),
    _doc_type(
        "meeting_notes",
        "Meeting notes",
        "Is this document meeting notes or minutes of a discussion?",
        "This document is meeting notes or minutes of a discussion.",
    ),
    _doc_type(
        "form",
        "Form",
        "Is this document a form with blanks for someone to fill in?",
        "This document is a form with blanks for someone to fill in.",
    ),
    _doc_type(
        "quote",
        "Quote",
        "Is this document a price quote or an estimate for work or goods?",
        "This document is a price quote or an estimate for work or goods.",
    ),
    _doc_type(
        "article",
        "Article",
        "Is this document a news article or a magazine article written for readers?",
        "This document is a news article or a magazine article written for readers.",
    ),
)

# Types that show up in real corpora and are not already one of the questions above.
# "other" is not a question. Abstaining prints Null.
_EXTRA_DOC_TYPE_IDS: tuple[str, ...] = (
    "procurement_record",
    "audit_and_compliance_report",
    "audit_report",
    "annual_report",
    "environmental_impact_assessment",
    "imaging_report",
    "treatment_plan",
    "lab_results",
    "medical_history_form",
    "immigration_form",
    "patient_consent_form",
    "insurance_policy",
    "merger_agreement",
    "tax_return",
    "market_analysis",
    "loan_application",
    "forecast",
    "bank_statement",
    "petition",
    "insurance_claim_form",
    "expense_report",
    "land_use_permit",
    "financial_report",
    "statement_of_work",
    "marriage_license_form",
    "birth_certificate_form",
    "driver_license_form",
    "budget_proposal",
    "compliance_report",
    "procedure_guide",
    "meeting_minutes",
    "physical_therapy_plan",
    "acquisition_agreement",
    "occupational_therapy_plan",
    "profit_and_loss_statement",
    "pathology_report",
    "home_health_care_plan",
    "medical_imaging_report",
    "pathology_summary",
    "rehabilitation_plan",
    "blood_test_report",
    "quality_assurance_report",
    "patient_education_materials",
    "balance_sheet",
    "forecast_variance_report",
    "training_materials",
    "customer_blood_type_profile",
    "policy_manual",
    "cash_flow_statement",
    "solvency_report",
    "revenue_report",
    "supplier_evaluation_form",
    "petition_for_asylum",
    "health_history",
    "allergy_information",
    "zoning_compliance_certificate",
    "green_card_application",
    "passport",
    "health_insurance_claim_form",
    "risk_assessment",
    "medical_expense_reimbursement",
    "immunization_schedule",
    "grant_application",
    "emergency_medical_information",
    "health_risk_assessment_form",
    "admission_checklist",
    "capital_allocation_report",
    "inpatient_admission_form",
    "billing_statement",
    "medical_record",
    "worker_s_compensation_claim",
    "patient_assessment",
    "discharge_checklist",
    "blood_bank_inventory_record",
    "budget_variance_report",
    "corporate_bylaws",
    "diagnosis_report",
    "valuation_report",
    "employment_contract",
    "business_plan",
    "emergency_response_plan",
    "purchase_agreement",
    "clinical_trial_screening_form",
    "lab_specimen_label",
    "job_application",
    "blood_donor_registration_form",
    "futures_contract",
    "investment_proposal",
    "confidentiality_agreement",
    "financial_plan",
    "freedom_of_information_act_request",
    "stock_certificate",
    "derivative_contract",
    "risk_management_plan",
    "medication_list",
    "portfolio_report",
    "prospectus",
    "dividend_payment",
    "forward_contract",
    "time_off_request",
    "interest_payment",
    "equity_offering",
    "shareholder_agreement",
    "health_insurance_form",
    "consultation_summary",
    "debt_offering",
    "emergency_contact_information",
    "border_entry_biometric_scan_receipt",
    "disability_insurance_claim",
    "consultation_notes",
    "patient_demographics",
    "code_of_conduct",
    "security_policy",
    "disaster_recovery_plan",
    "fund_performance_report",
    "unemployment_claim",
    "property_tax_payment",
    "swap_agreement",
    "retirement_plan_enrollment",
    "credit_report",
    "debt_allocation_report",
    "security_offering_memorandum",
    "lease_agreement",
    "prescription",
    "employment_verification",
    "option_contract",
    "medical_discharge_form",
    "recipient_transfusion_log",
    "divorce_decree",
    "vaccination_record",
    "bond_certificate",
    "biometric_enrollment_form",
    "hospital_admission_record",
    "employee_benefits_enrollment",
    "international_health_certificate",
    "underwriting_agreement",
    "subscription_agreement",
    "patient_biometric_id_record",
    "investment_performance_report",
    "medical_certificate",
    "outpatient_visit_notes",
    "401k_contribution_form",
    "ethics_policy",
    "discharge_summary",
    "clinical_trial_participant_biometric_file",
    "equity_allocation_report",
    "customer_order_confirmation",
    "health_insurance_enrollment",
    "repurchase_agreement",
    "return_merchandise_authorization",
    "incident_postmortem_report",
    "system_architecture_design_doc",
)

_EXISTING_DOC_TYPE_IDS = frozenset(label.id for label in DOC_TYPES)
DOC_TYPES = DOC_TYPES + tuple(
    _from_id(label_id) for label_id in _EXTRA_DOC_TYPE_IDS if label_id not in _EXISTING_DOC_TYPE_IDS
)

ABSTAIN = "other"

INDUSTRY_IDS: frozenset[str] = frozenset(label.id for label in INDUSTRIES)


def industry_hypotheses() -> dict[str, str]:
    return {label.id: label.hypothesis for label in INDUSTRIES}


def industry_names() -> dict[str, str]:
    return {label.id: label.name for label in INDUSTRIES}


def industry_questions() -> dict[str, str]:
    return {label.id: label.question for label in INDUSTRIES}


def doc_type_hypotheses() -> dict[str, str]:
    return {label.id: label.hypothesis for label in DOC_TYPES}


def doc_type_names() -> dict[str, str]:
    return {label.id: label.name for label in DOC_TYPES}
