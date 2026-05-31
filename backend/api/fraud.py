from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional
import json
import time
import random
from datetime import datetime, timezone
import pandas as pd
from catboost import Pool

from backend.db.database import conn as _sdb
from backend.core.security import get_current_user
from backend.services.fraud_model import get_fraud_model, DEFAULTS, CAT_FEATURES, explain_prediction
from backend.services.tools import analyze_photo_exif

fraud_model = get_fraud_model()

router = APIRouter()

FRAUD_THRESHOLD = 0.20

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

class FraudAssessRequest(BaseModel):
    Fault: str
    BasePolicy: str
    VehicleCategory: str
    Month: str
    Age: int
    DayOfWeek: str
    Year: int
    DayOfWeekClaimed: str
    Make: str
    AgeOfPolicyHolder: str
    NumberOfSuppliments: str
    MonthClaimed: str
    AgeOfVehicle: str
    PastNumberOfClaims: str
    VehiclePrice: str
    Sex: str
    PoliceReportFiled: str

    WeekOfMonth: Optional[int] = None
    WeekOfMonthClaimed: Optional[int] = None
    MaritalStatus: Optional[str] = None
    AccidentArea: Optional[str] = None
    Deductible: Optional[int] = None
    DriverRating: Optional[int] = None
    Days_Policy_Accident: Optional[str] = None
    Days_Policy_Claim: Optional[str] = None
    WitnessPresent: Optional[str] = None
    AgentType: Optional[str] = None
    AddressChange_Claim: Optional[str] = None
    NumberOfCars: Optional[str] = None
    vehicle_ngn: Optional[float] = None
    thread_id: Optional[str] = None

class PhotoMetadataRequest(BaseModel):
    image_data: str
    file_name: str = ""

@router.post("/fraud/check-photo-metadata")
async def check_photo_metadata(
    body: PhotoMetadataRequest,
    current_user: str = Depends(get_current_user)
):
    result = analyze_photo_exif(body.image_data, body.file_name)
    return result

@router.post("/fraud/sessions")
async def create_fraud_session_endpoint(current_user: str = Depends(get_current_user)):
    tid = f"fraud_{int(time.time())}_{random.randint(1000,9999)}"
    now = datetime.now(timezone.utc).isoformat()
    _sdb.execute(
        "INSERT INTO sessions (thread_id, username, title, session_type, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
        (tid, current_user, 'New Assessment', 'fraud', now, now)
    )
    _sdb.commit()
    return {"thread_id": tid}

@router.post("/fraud/assess")
async def assess_claim(claim: FraudAssessRequest, current_user: str = Depends(get_current_user)):
    claim_dict = claim.model_dump(exclude={'vehicle_ngn', 'thread_id'})

    auto_filled = []
    for field, default in DEFAULTS.items():
        if claim_dict.get(field) is None:
            claim_dict[field] = default
            auto_filled.append({"field": field, "value": default})

    df = pd.DataFrame([claim_dict])[TRAINING_ORDER]
    pool = Pool(df, cat_features=CAT_FEATURES)
    probability = float(fraud_model.predict_proba(pool)[:, 1][0])
    factors = explain_prediction(df, pool, top_n=5)
    is_fraud = probability >= FRAUD_THRESHOLD

    if probability >= 0.70:
        risk_level = "HIGH"
        recommendation = "Immediate investigation required before processing."
    elif probability >= FRAUD_THRESHOLD:
        risk_level = "MEDIUM"
        recommendation = "Flag for review by claims investigator."
    elif probability >= 0.30:
        risk_level = "LOW-MEDIUM"
        recommendation = "Process normally but monitor closely."
    else:
        risk_level = "LOW"
        recommendation = "Process normally."

    thread_id = claim.thread_id or f"fraud_{int(time.time())}_{random.randint(1000,9999)}"
    result_dict = {
        "thread_id": thread_id,
        "probability_pct": round(probability * 100, 1),
        "is_fraud": is_fraud,
        "risk_level": risk_level,
        "recommendation": recommendation,
        "auto_filled": auto_filled,
        "factors": factors,  
    }

    now = datetime.now(timezone.utc).isoformat()
    vehicle_make = claim_dict.get("Make", "Vehicle")
    title = f"{vehicle_make} — {risk_level} RISK"

    try:
        existing_session = _sdb.execute("SELECT 1 FROM sessions WHERE thread_id = ?", (thread_id,)).fetchone()
        if existing_session:
            _sdb.execute("UPDATE sessions SET title=?, updated_at=? WHERE thread_id=?", (title, now, thread_id))
        else:
            _sdb.execute(
                "INSERT INTO sessions (thread_id, username, title, session_type, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                (thread_id, current_user, title, 'fraud', now, now)
            )

        _sdb.execute("DELETE FROM fraud_assessments WHERE thread_id = ?", (thread_id,))
        _sdb.execute(
            """INSERT INTO fraud_assessments
               (thread_id, username, created_at, form_data, result, risk_level, probability, vehicle_ngn)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (thread_id, current_user, now, json.dumps(claim_dict), json.dumps(result_dict), risk_level, probability, claim.vehicle_ngn or 0.0)
        )
        _sdb.commit()
    except Exception as e:
        print(f"Error saving assessment: {e}")
        _sdb.rollback()

    return result_dict

@router.get("/fraud/sessions/{thread_id}")
async def get_fraud_session(thread_id: str, current_user: str = Depends(get_current_user)):
    row = _sdb.execute("""
        SELECT form_data, result, vehicle_ngn, created_at
        FROM fraud_assessments
        WHERE thread_id = ? AND username = ?
    """, (thread_id, current_user)).fetchone()

    if not row:
        return {"form_data": None, "result": None}

    return {
        "form_data":   json.loads(row[0]) if row[0] else None,
        "result":      json.loads(row[1]) if row[1] else None,
        "vehicle_ngn": row[2],
        "created_at":  row[3],
    }

@router.post("/fraud/sessions/{thread_id}/result")
async def save_fraud_result(thread_id: str, payload: dict, current_user: str = Depends(get_current_user)):
    now = datetime.now(timezone.utc).isoformat()
    form_data = payload.get("form_data")
    result    = payload.get("result")
    vehicle_ngn = payload.get("vehicle_ngn", 0.0)

    existing = _sdb.execute(
        "SELECT form_data FROM fraud_assessments WHERE thread_id = ? AND username = ?",
        (thread_id, current_user)
    ).fetchone()

    if existing:
        _sdb.execute(
            "UPDATE fraud_assessments SET form_data=?, result=?, vehicle_ngn=? WHERE thread_id=? AND username=?",
            (json.dumps(form_data), json.dumps(result), vehicle_ngn, thread_id, current_user)
        )
    else:
        risk_level  = result.get("risk_level", "LOW") if result else "LOW"
        probability = result.get("probability_pct", 0) / 100 if result else 0.0
        _sdb.execute(
            """INSERT INTO fraud_assessments
               (thread_id, username, created_at, form_data, result, risk_level, probability, vehicle_ngn)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (thread_id, current_user, now, json.dumps(form_data), json.dumps(result), risk_level, probability, vehicle_ngn)
        )

    _sdb.execute("UPDATE sessions SET updated_at=? WHERE thread_id=? AND username=?", (now, thread_id, current_user))
    _sdb.commit()
    return {"ok": True}