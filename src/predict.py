import mlflow.sklearn
import pandas as pd
import numpy as np
import os
import sys
import uuid
import json
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator, Field
import uvicorn

sys.path.insert(0, os.path.dirname(__file__))
from features import engineer_features, get_feature_columns


MODEL_NAME   = "credit-risk-model"
LOG_PATH     = os.path.join(os.path.dirname(__file__), "..", "monitoring", "prediction_log.csv")
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

app = FastAPI(
    title="Credit Risk Scoring API",
    description="Predicts probability of credit card default for individual customers.",
    version="1.0.0"
)

# load model at startup
model = None

@app.on_event("startup")
def load_model():
    global model
    try:
        model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/Production")
        print(f"Production model loaded: models:/{MODEL_NAME}/Production")
    except Exception as e:
        print(f"WARNING: Could not load Production model: {e}")
        print("Run train.py and evaluate.py first to register a Production model.")


# ─────────────────────────────────────────────
#  Request / Response schemas
# ─────────────────────────────────────────────

class CustomerInput(BaseModel):
    LIMIT_BAL:  float = Field(..., description="Credit limit (NT dollar)")
    SEX:        int   = Field(..., ge=1, le=2, description="1=male, 2=female")
    EDUCATION:  int   = Field(..., ge=1, le=4, description="1=grad, 2=university, 3=high school, 4=other")
    MARRIAGE:   int   = Field(..., ge=1, le=3, description="1=married, 2=single, 3=other")
    AGE:        int   = Field(..., ge=18, le=100)
    PAY_0:      int   = Field(..., description="Repayment status Sep (-1=on time, 1=1mo late, ...)")
    PAY_2:      int   = Field(..., description="Repayment status Aug")
    PAY_3:      int   = Field(..., description="Repayment status Jul")
    PAY_4:      int   = Field(..., description="Repayment status Jun")
    PAY_5:      int   = Field(..., description="Repayment status May")
    PAY_6:      int   = Field(..., description="Repayment status Apr")
    BILL_AMT1:  float = Field(..., description="Bill amount Sep")
    BILL_AMT2:  float = Field(..., description="Bill amount Aug")
    BILL_AMT3:  float = Field(..., description="Bill amount Jul")
    BILL_AMT4:  float = Field(..., description="Bill amount Jun")
    BILL_AMT5:  float = Field(..., description="Bill amount May")
    BILL_AMT6:  float = Field(..., description="Bill amount Apr")
    PAY_AMT1:   float = Field(..., description="Payment amount Sep")
    PAY_AMT2:   float = Field(..., description="Payment amount Aug")
    PAY_AMT3:   float = Field(..., description="Payment amount Jul")
    PAY_AMT4:   float = Field(..., description="Payment amount Jun")
    PAY_AMT5:   float = Field(..., description="Payment amount May")
    PAY_AMT6:   float = Field(..., description="Payment amount Apr")

    @validator("LIMIT_BAL")
    def limit_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError("LIMIT_BAL must be positive")
        return v


class PredictionResponse(BaseModel):
    request_id:          str
    default_probability: float
    risk_score:          int    # 0-100 scale
    decision:            str    # APPROVE / REVIEW / REJECT
    decision_reason:     str
    timestamp:           str


# ─────────────────────────────────────────────
#  Helper functions
# ─────────────────────────────────────────────

def make_decision(probability: float) -> tuple:
    """Convert probability to tiered decision with reason."""
    if probability >= 0.70:
        return "REJECT", "High default risk detected based on payment history patterns."
    elif probability >= 0.45:
        return "REVIEW", "Moderate risk — manual underwriter review recommended."
    else:
        return "APPROVE", "Low default risk. Customer demonstrates consistent repayment behaviour."


def log_prediction(request_id: str, customer: CustomerInput, probability: float, decision: str):
    """Append prediction to CSV log for monitoring."""
    record = {
        "request_id":  request_id,
        "timestamp":   datetime.utcnow().isoformat(),
        "probability": probability,
        "decision":    decision,
        **customer.dict()
    }
    row = pd.DataFrame([record])
    header = not os.path.exists(LOG_PATH)
    row.to_csv(LOG_PATH, mode="a", header=header, index=False)


# ─────────────────────────────────────────────
#  Endpoints
# ─────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerInput):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run train.py and evaluate.py first."
        )

    request_id = str(uuid.uuid4())

    # build dataframe
    # build dataframe
    df = pd.DataFrame([customer.dict()])

    # run feature engineering — model expects engineered features
    df = engineer_features(df)

    try:
        probability = float(model.predict_proba(df)[0][1])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    risk_score           = int(round(probability * 100))
    decision, reason     = make_decision(probability)

    # log for monitoring
    try:
        log_prediction(request_id, customer, probability, decision)
    except Exception:
        pass    # logging failure should never block a prediction

    return PredictionResponse(
        request_id=request_id,
        default_probability=round(probability, 4),
        risk_score=risk_score,
        decision=decision,
        decision_reason=reason,
        timestamp=datetime.utcnow().isoformat()
    )


@app.get("/model/info")
def model_info():
    """Return info about the currently loaded model."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "model_name":  MODEL_NAME,
        "stage":       "Production",
        "log_path":    LOG_PATH,
        "predictions_logged": _count_logged_predictions()
    }


def _count_logged_predictions() -> int:
    try:
        if os.path.exists(LOG_PATH):
            return len(pd.read_csv(LOG_PATH))
        return 0
    except Exception:
        return 0


if __name__ == "__main__":
    uvicorn.run("predict:app", host="0.0.0.0", port=8000, reload=True)