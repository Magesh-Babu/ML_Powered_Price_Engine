# quote_api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Literal, Tuple
import pandas as pd
import numpy as np
import joblib
import os
from schemas import QuoteSchemaV1  # Ensure this is imported from your existing schema
from model_development import load_model, get_latest_version  # Reuse from earlier

app = FastAPI(title="Quote Prediction API")

# Input schema for the API
class QuoteRequest(BaseModel):
    user_id: str
    quote_data: dict

class QuoteResponse(BaseModel):
    predicted_price: float
    confidence_interval: Tuple[float, float]
    model_version: str

@app.post("/predict", response_model=QuoteResponse)
def predict_quote(request: QuoteRequest):
    user_id = request.user_id
    input_dict = request.quote_data

    try:
        # Validate input
        validated = QuoteSchemaV1(**input_dict)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Input validation error: {str(e)}")

    # Convert to dataframe
    input_df = pd.DataFrame([validated.model_dump()])
    input_df = input_df.drop(columns=["schema_version", "Quote_Date"], errors="ignore")

    # Load user-specific model
    version = get_latest_version(user_id)
    try:
        model = load_model(user_id, version)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Model not found for user")

    # Predict using ensemble for uncertainty
    preds = [est.predict(input_df)[0] for est in model.estimators_]
    mean_pred = np.mean(preds)
    lower, upper = np.percentile(preds, [5, 95])

    return QuoteResponse(
        predicted_price=round(mean_pred, 4),
        confidence_interval=(round(lower, 4), round(upper, 4)),
        model_version=version
    )
