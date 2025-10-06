# src/api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
import os
from typing import List, Dict
from src.model import load_model, predict_proba
from src.features import build_features
from src.data import load_data, prepare_labels

app = FastAPI(title="Ethereum Fraud Detection API")

MODEL_PATH = "models/xgb_model.joblib"
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
else:
    model = None

class PredictRequest(BaseModel):
    transactions: List[Dict]  # list of dicts, each dict is a transaction row

class PredictResponse(BaseModel):
    fraud_probs: List[float]

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train model first.")
    try:
        import pandas as pd
        df = pd.DataFrame(req.transactions)
        df = prepare_labels(df)
        X, _, _ = build_features(df)
        probs = predict_proba(model, X)
        return PredictResponse(fraud_probs=probs.tolist())
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
def root():
    return {"status": "ok", "model_loaded": model is not None}
