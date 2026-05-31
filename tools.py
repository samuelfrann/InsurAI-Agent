import io
import os
import joblib
import pandas as pd
import pdfplumber

from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from catboost import CatBoostClassifier, Pool
from langchain_core.tools import tool, Tool
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional

# ── Temp file store ───────────────────────────────────────────────────
# app.py stores uploaded files here; tools read from here
file_store: dict = {}


@tool
def pdf_reader_tool(file_key: str) -> str:
    """Read and extract text and tables from an uploaded PDF document.
    Use this whenever a staff member uploads a PDF file and asks about its contents."""
    entry = file_store.get(file_key)
    if not entry:
        return "File not found or already processed. Ask the staff member to re-upload the document."

    try:
        with pdfplumber.open(io.BytesIO(entry["bytes"])) as pdf:
            pages = []
            for i, page in enumerate(pdf.pages, 1):
                text = (page.extract_text() or "").strip()
                tables = page.extract_tables()

                section = f"── Page {i} ──\n{text}" if text else f"── Page {i} ── (no text found)"

                for table in tables:
                    rows = [" | ".join(str(cell or "").strip() for cell in row) for row in table]
                    section += "\n\nTable:\n" + "\n".join(rows)

                pages.append(section)

        del file_store[file_key]  # clean up after reading

        result = f"Document: {entry['filename']}\n\n" + "\n\n".join(pages)
        return result[:8000]

    except Exception as e:
        return f"Failed to read PDF: {str(e)}"


search = DuckDuckGoSearchRun()
search_tool = Tool(
    name = 'search',
    func = search.run,
    description = 'Search the web for information'
)

api_wrapper = WikipediaAPIWrapper(top_k_results=5, doc_content_chars_max=100)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'fraud_catboost.cbm')

fraud_model = CatBoostClassifier()
fraud_model.load_model(r'/home/frank/MACHINE LEARNING WSL/InsurAI-Agent/models/catboost_fraud_model.cbm')

FRAUD_THRESHOLD = 0.52

print(f"✅ Fraud model loaded.")

CAT_FEATURES = [
    'Month', 'DayOfWeek', 'Make', 'AccidentArea',
    'DayOfWeekClaimed', 'MonthClaimed', 'Sex', 'MaritalStatus',
    'Fault', 'VehicleCategory', 'VehiclePrice',
    'Days_Policy_Accident', 'Days_Policy_Claim',
    'PastNumberOfClaims', 'AgeOfVehicle', 'AgeOfPolicyHolder',
    'PoliceReportFiled', 'WitnessPresent', 'AgentType',
    'NumberOfSuppliments', 'AddressChange_Claim',
    'NumberOfCars', 'BasePolicy'
]


REQUIRED_FIELDS = [
    'Fault', 'BasePolicy', 'VehicleCategory', 'Month', 'Age',
    'DayOfWeek', 'Year', 'DayOfWeekClaimed', 'Make',
    'AgeOfPolicyHolder', 'NumberOfSuppliments', 'MonthClaimed',
    'AgeOfVehicle', 'PastNumberOfClaims', 'VehiclePrice', 'Sex', 'PoliceReportFiled'
]

DEFAULTS = {
    'WeekOfMonth':          3,
    'WeekOfMonthClaimed':   3,
    'MaritalStatus':        'Single',
    'AccidentArea':         'Urban',
    'Deductible':           400,
    'DriverRating':         1,
    'Days_Policy_Accident': 'more than 30',
    'Days_Policy_Claim':    'more than 30',
    'WitnessPresent':       'No',
    'AgentType':            'External',
    'AddressChange_Claim':  'no change',
    'NumberOfCars':         '1 vehicle',
}


class ClaimInput(BaseModel):
    # ── Top 17 required fields ────────────────────────────────────────
    Fault: str                          = Field(description="'Policy Holder' or 'Third Party'")
    BasePolicy: str                     = Field(description="'Liability', 'Collision', 'All Perils'")
    VehicleCategory: str                = Field(description="'Sport', 'Sedan', 'Utility'")
    Month: str                          = Field(description="Month of accident e.g. 'Dec'")
    Age: int                            = Field(description="Age of claimant")
    DayOfWeek: str                      = Field(description="Day of accident e.g. 'Monday'")
    Year: int                           = Field(description="Year of claim e.g. 1994")
    DayOfWeekClaimed: str               = Field(description="Day claim was filed e.g. 'Tuesday'")
    Make: str                           = Field(description="Vehicle make e.g. 'Honda'")
    AgeOfPolicyHolder: str              = Field(description="e.g. '26 to 30'")
    NumberOfSuppliments: str            = Field(description="e.g. 'none', '1 to 2', '3 to 5'")
    MonthClaimed: str                   = Field(description="Month claim was filed e.g. 'Jan'")
    AgeOfVehicle: str                   = Field(description="e.g. '3 years', 'new', 'more than 7'")
    PastNumberOfClaims: str             = Field(description="e.g. 'none', '1', '2 to 4'")
    VehiclePrice: str                   = Field(description="e.g. 'more than 69000', '20000 to 29000'")
    Sex: str                            = Field(None, description="'Male' or 'Female'")
    PoliceReportFiled: str              = Field(None, description="'Yes' or 'No'")

    # ── 12 auto-filled fields — staff can provide, not required ───────
    WeekOfMonth: Optional[int]          = Field(None, description="Week of month 1-5")
    WeekOfMonthClaimed: Optional[int]   = Field(None, description="Week of month claim was filed")
    MaritalStatus: Optional[str]        = Field(None, description="'Single', 'Married', 'Divorced', 'Widow'")
    AccidentArea: Optional[str]         = Field(None, description="'Urban' or 'Rural'")
    Deductible: Optional[int]           = Field(None, description="Deductible amount")
    DriverRating: Optional[int]         = Field(None, description="Driver rating 1-4")
    Days_Policy_Accident: Optional[str] = Field(None, description="e.g. 'more than 30', '1 to 7'")
    Days_Policy_Claim: Optional[str]    = Field(None, description="e.g. 'more than 30', '1 to 7'")
    WitnessPresent: Optional[str]       = Field(None, description="'Yes' or 'No'")
    AgentType: Optional[str]            = Field(None, description="'Internal' or 'External'")
    AddressChange_Claim: Optional[str]  = Field(None, description="e.g. 'no change', '1 year'")
    NumberOfCars: Optional[str]         = Field(None, description="e.g. '1 vehicle', '2 vehicles'")


@tool(args_schema=ClaimInput)
def fraud_detection_tool(**kwargs) -> str:
    """
    Assess an insurance claim for fraud risk using a trained CatBoost model.
    Top 15 fields required. Remaining 14 fields auto-filled with defaults.
    """
    try:
        none_fields = [k for k, v in kwargs.items() if v is None]
        print(f"🔍 Fraud Detection Tool called with input:")

        auto_filled = []
        for field, default in DEFAULTS.items():
            if kwargs.get(field) is None:
                kwargs[field] = default
                auto_filled.append((field, default))

        none_fields_after = [k for k, v in kwargs.items() if v is None]

        if none_fields_after:
            return (
                f"❌ Cannot run model — these fields are still missing after "
                f"auto-fill: {none_fields_after}. "
                f"Please provide them and try again."
            )

        TRAINING_ORDER = [
            'Month', 'WeekOfMonth', 'DayOfWeek', 'Make', 'AccidentArea',
            'DayOfWeekClaimed', 'MonthClaimed', 'WeekOfMonthClaimed',
            'Sex', 'MaritalStatus', 'Age', 'Fault', 'VehicleCategory',
            'VehiclePrice', 'Deductible', 'DriverRating',
            'Days_Policy_Accident', 'Days_Policy_Claim',
            'PastNumberOfClaims', 'AgeOfVehicle', 'AgeOfPolicyHolder',
            'PoliceReportFiled', 'WitnessPresent', 'AgentType',
            'NumberOfSuppliments', 'AddressChange_Claim',
            'NumberOfCars', 'Year', 'BasePolicy'
        ]

        df = pd.DataFrame([kwargs])[TRAINING_ORDER]

        # ── Predict via Pool ───────────────────────────────────────────
        pool = Pool(df, cat_features=CAT_FEATURES)
        probability = float(fraud_model.predict_proba(pool)[:, 1][0])
        is_fraud    = probability >= FRAUD_THRESHOLD

        if probability >= 0.70:
            risk_level     = "HIGH"
            recommendation = "Immediate investigation required before processing."
        elif probability >= FRAUD_THRESHOLD:
            risk_level     = "MEDIUM"
            recommendation = "Flag for review by claims investigator."
        elif probability >= 0.30:
            risk_level     = "LOW-MEDIUM"
            recommendation = "Process normally but monitor closely."
        else:
            risk_level     = "LOW"
            recommendation = "Process normally."

        auto_fill_section = ""
        if auto_filled:
            auto_fill_section = (
                f"\n\n⚠️ **Auto-Filled Fields ({len(auto_filled)}):**\n"
                + "\n".join(f"  - {f}: '{v}'" for f, v in auto_filled)
                + "\n\n**Defaults used. Confirm or correct for higher accuracy.**"
            )

        return (
            f"**Fraud Assessment Result**\n"
            f"{'='*40}\n"
            f"- Fraud Probability:  {probability:.4f} ({probability*100:.1f}%)\n"
            f"- Classification:     {'⚠️  SUSPICIOUS' if is_fraud else '✅ NORMAL'}\n"
            f"- Risk Level:         {risk_level}\n"
            f"- Threshold Used:     {FRAUD_THRESHOLD}\n"
            f"- Recommendation:     {recommendation}"
            + auto_fill_section
            + "\n\n*Triage signal only — "
            "final decision must be made by a human investigator.*"
        )

    except Exception as e:
        import traceback
        return f"❌ Error during fraud assessment: {str(e)}\nTraceback: {traceback.format_exc()}"