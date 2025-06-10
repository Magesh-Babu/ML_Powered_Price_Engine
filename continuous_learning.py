# continuous_learning.py
import pandas as pd
import joblib
import os
from datetime import datetime
from schemas import QuoteSchemaV1
from model_registry import save_model, get_latest_version
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Define a secure user-specific data/model directory
data_dir = "user_data"
model_dir = "models"
os.makedirs(data_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# 1. Feedback loop: new quote data ingestion and validation
def append_new_quote(user_id: str, new_quote: dict):
    try:
        validated = QuoteSchemaV1(**new_quote)
    except Exception as e:
        raise ValueError(f"Validation failed: {e}")

    user_file = os.path.join(data_dir, f"{user_id}_quotes.csv")
    new_df = pd.DataFrame([validated.model_dump()])
    new_df = new_df.drop(columns=["schema_version"], errors="ignore")

    if os.path.exists(user_file):
        df = pd.read_csv(user_file)
        df = pd.concat([df, new_df], ignore_index=True)
    else:
        df = new_df

    df.to_csv(user_file, index=False)
    print("✅ New quote appended successfully.")
    return df


# 2. Retraining strategy: simple retrain from updated user data
def retrain_user_model(user_id: str):
    user_file = os.path.join(data_dir, f"{user_id}_quotes.csv")
    if not os.path.exists(user_file):
        raise FileNotFoundError("User data not found.")

    df = pd.read_csv(user_file)
    if "Quote_Date" in df.columns:
        df = df.sort_values("Quote_Date")

    X = df.drop(columns=["Quote_Price_SEK", "Quote_Date"], errors="ignore")
    y = df["Quote_Price_SEK"]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    mae = mean_absolute_error(y_val, preds)

    new_version = save_model(model, user_id)
    print(f"✅ Model retrained. New version: {new_version}, MAE: {mae:.4f}")
    return new_version, mae



# Optional: wire into an API endpoint
def api_trigger_retraining(user_id: str, new_quote: dict):
    print(f"\n🟡 API triggered retraining for user: {user_id}")
    append_new_quote(user_id, new_quote)
    version, mae = retrain_user_model(user_id)
    print(f"\n✅ Retraining complete. Model v{version} | MAE: {mae:.4f}")
    return {"model_version": version, "mae": mae}
