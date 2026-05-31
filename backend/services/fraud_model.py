import pandas as pd
from catboost import CatBoostClassifier, Pool
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Optional
from functools import lru_cache

from backend.config import settings


@lru_cache(maxsize=1)
def get_fraud_model() -> CatBoostClassifier:
    """Load model once from the path in config (no hardcoded paths)."""
    model = CatBoostClassifier()
    model.load_model(settings.model_path)
    print("✅ Fraud model loaded.")
    return model


FRAUD_THRESHOLD = 0.52

CAT_FEATURES = [
    "Month", "DayOfWeek", "Make", "AccidentArea", "DayOfWeekClaimed",
    "MonthClaimed", "Sex", "MaritalStatus", "Fault", "VehicleCategory",
    "VehiclePrice", "Days_Policy_Accident", "Days_Policy_Claim",
    "PastNumberOfClaims", "AgeOfVehicle", "AgeOfPolicyHolder",
    "PoliceReportFiled", "WitnessPresent", "AgentType",
    "NumberOfSuppliments", "AddressChange_Claim", "NumberOfCars", "BasePolicy",
]

TRAINING_ORDER = [
    "Month", "WeekOfMonth", "DayOfWeek", "Make", "AccidentArea",
    "DayOfWeekClaimed", "MonthClaimed", "WeekOfMonthClaimed",
    "Sex", "MaritalStatus", "Age", "Fault", "VehicleCategory",
    "VehiclePrice", "Deductible", "DriverRating",
    "Days_Policy_Accident", "Days_Policy_Claim", "PastNumberOfClaims",
    "AgeOfVehicle", "AgeOfPolicyHolder", "PoliceReportFiled",
    "WitnessPresent", "AgentType", "NumberOfSuppliments",
    "AddressChange_Claim", "NumberOfCars", "Year", "BasePolicy",
]

DEFAULTS = {
    "WeekOfMonth": 3, "WeekOfMonthClaimed": 3, "MaritalStatus": "Single",
    "AccidentArea": "Urban", "Deductible": 400, "DriverRating": 1,
    "Days_Policy_Accident": "more than 30", "Days_Policy_Claim": "more than 30",
    "WitnessPresent": "No", "AgentType": "External",
    "AddressChange_Claim": "no change", "NumberOfCars": "1 vehicle",
}


class ClaimInput(BaseModel):
    Fault: str                          = Field(description="'Policy Holder' or 'Third Party'")
    BasePolicy: str                     = Field(description="'Liability', 'Collision', 'All Perils'")
    VehicleCategory: str                = Field(description="'Sport', 'Sedan', 'Utility'")
    Month: str                          = Field(description="Month of accident e.g. 'Dec'")
    Age: int                            = Field(description="Age of claimant")
    DayOfWeek: str                      = Field(description="Day of accident e.g. 'Monday'")
    Year: int                           = Field(description="Year of claim")
    DayOfWeekClaimed: str               = Field(description="Day claim was filed")
    Make: str                           = Field(description="Vehicle make e.g. 'Honda'")
    AgeOfPolicyHolder: str              = Field(description="e.g. '26 to 30'")
    NumberOfSuppliments: str            = Field(description="'none', '1 to 2', '3 to 5', 'more than 5'")
    MonthClaimed: str                   = Field(description="Month claim was filed")
    AgeOfVehicle: str                   = Field(description="'new', '3 years', 'more than 7'")
    PastNumberOfClaims: str             = Field(description="'none', '1', '2 to 4', 'more than 4'")
    VehiclePrice: str                   = Field(description="e.g. 'more than 69000'")
    Sex: str                            = Field(description="'Male' or 'Female'")
    PoliceReportFiled: str              = Field(description="'Yes' or 'No'")
    WeekOfMonth: Optional[int]          = Field(None)
    WeekOfMonthClaimed: Optional[int]   = Field(None)
    MaritalStatus: Optional[str]        = Field(None)
    AccidentArea: Optional[str]         = Field(None)
    Deductible: Optional[int]           = Field(None)
    DriverRating: Optional[int]         = Field(None)
    Days_Policy_Accident: Optional[str] = Field(None)
    Days_Policy_Claim: Optional[str]    = Field(None)
    WitnessPresent: Optional[str]       = Field(None)
    AgentType: Optional[str]            = Field(None)
    AddressChange_Claim: Optional[str]  = Field(None)
    NumberOfCars: Optional[str]         = Field(None)

def explain_prediction(df: pd.DataFrame, pool: Pool, top_n: int = 5) -> list[dict]:
    """Return top N features driving this prediction using CatBoost's native SHAP.
    No extra library needed — CatBoost computes these internally already."""
    model = get_fraud_model()
    shap_values = model.get_feature_importance(pool, type="ShapValues")
    contributions = shap_values[0][:-1]  # last column is the base value — drop it

    values = df.iloc[0].to_dict()
    paired = [
        (name, values[name], float(contrib))
        for name, contrib in zip(df.columns, contributions)
    ]
    paired.sort(key=lambda x: abs(x[2]), reverse=True)

    return [
        {
            "feature": name,
            "value": value,
            "impact": round(contrib, 4),
            "direction": "increases risk" if contrib > 0 else "lowers risk",
        }
        for name, value, contrib in paired[:top_n]
    ]

@tool(args_schema=ClaimInput)
def fraud_detection_tool(**kwargs) -> str:
    """Assess an insurance claim for fraud risk using a trained CatBoost model."""
    try:
        fraud_model = get_fraud_model()

        auto_filled = []
        for field, default in DEFAULTS.items():
            if kwargs.get(field) is None:
                kwargs[field] = default
                auto_filled.append((field, default))

        still_missing = [k for k, v in kwargs.items() if v is None]
        if still_missing:
            return f"❌ Cannot run model — fields still missing after auto-fill: {still_missing}"

        df = pd.DataFrame([kwargs])[TRAINING_ORDER]
        pool = Pool(df, cat_features=CAT_FEATURES)
        probability = float(fraud_model.predict_proba(pool)[:, 1][0])
        is_fraud = probability >= FRAUD_THRESHOLD

        if probability >= 0.70:
            risk_level, recommendation = "HIGH", "Immediate investigation required before processing."
        elif probability >= FRAUD_THRESHOLD:
            risk_level, recommendation = "MEDIUM", "Flag for review by claims investigator."
        elif probability >= 0.30:
            risk_level, recommendation = "LOW-MEDIUM", "Process normally but monitor closely."
        else:
            risk_level, recommendation = "LOW", "Process normally."

        auto_fill_section = ""
        if auto_filled:
            auto_fill_section = (
                f"\n\n⚠️ **Auto-Filled Fields ({len(auto_filled)}):**\n"
                + "\n".join(f"  - {f}: '{v}'" for f, v in auto_filled)
                + "\n\n**Defaults used. Confirm or correct for higher accuracy.**"
            )

            # SHAP explanation
        factors = explain_prediction(df, pool, top_n=5)
        risk_drivers = [f for f in factors if f["direction"] == "increases risk"]
        explanation_section = ""
        if risk_drivers:
            explanation_section = "\n\n**Key risk drivers:**\n" + "\n".join(
                f"  - {f['feature']} = '{f['value']}'" for f in risk_drivers
            )

        return (
            f"**Fraud Assessment Result**\n{'='*40}\n"
            f"- Fraud Probability:  {probability:.4f} ({probability*100:.1f}%)\n"
            f"- Classification:     {'⚠️  SUSPICIOUS' if is_fraud else '✅ NORMAL'}\n"
            f"- Risk Level:         {risk_level}\n"
            f"- Threshold Used:     {FRAUD_THRESHOLD}\n"
            f"- Recommendation:     {recommendation}"
            + explanation_section
            + auto_fill_section
            + "\n\n*Triage signal only — final decision must be made by a human investigator.*"
        )
    except Exception as e:
        import traceback
        return f"❌ Error during fraud assessment: {str(e)}\nTraceback: {traceback.format_exc()}"