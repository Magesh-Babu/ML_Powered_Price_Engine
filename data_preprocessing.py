import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import IsolationForest
from category_encoders import CountEncoder

def apply_imputation(df: pd.DataFrame) -> pd.DataFrame:

    # Split dataset: rows with and without missing Lead Time
    df_lead_train = df[df["Lead_Time_weeks"].notnull()]
    df_lead_missing = df[df["Lead_Time_weeks"].isnull()]

    # Features to use for prediction (drop leakages and target)
    features = ["Alloy", "Finish", "Length_m", "Weight_kg_m", "Profile_Name", "Tolerances", "GD_T",
                "Order_Quantity", "LME_Price_EUR", "Customer_Category", "Quote_Price_SEK"]

    # Encode categorical variables
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_train = df_lead_train[features].copy()
    X_missing = df_lead_missing[features].copy()

    X_train[["Alloy", "Finish", "Profile_Name", "GD_T", "Customer_Category"]] = encoder.fit_transform(
        X_train[["Alloy", "Finish", "Profile_Name", "GD_T", "Customer_Category"]]
    )
    X_missing[["Alloy", "Finish", "Profile_Name", "GD_T", "Customer_Category"]] = encoder.transform(
        X_missing[["Alloy", "Finish", "Profile_Name", "GD_T", "Customer_Category"]]
    )

    y_train = df_lead_train["Lead_Time_weeks"]

    # Train a random forest regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict and impute
    predicted_lead_time = model.predict(X_missing)
    df.loc[df["Lead_Time_weeks"].isnull(), "Lead_Time_weeks"] = predicted_lead_time

    return df

def treat_outlier(df: pd.DataFrame) -> pd.DataFrame:

    # Isolation Forest for outlier detection on 'Weight'
    iso_forest = IsolationForest(contamination=0.02, random_state=42)
    df['Weight_outlier'] = iso_forest.fit_predict(df[['Weight_kg_m']])

    # -1 means outlier, 1 means normal
    # We'll handle outliers by replacing them using interpolation from neighbors

    # Mark outliers as NaN
    df.loc[df['Weight_outlier'] == -1, 'Weight_kg_m'] = np.nan

    # Interpolate missing (outlier) values using linear interpolation
    df['Weight_kg_m'] = df['Weight_kg_m'].interpolate(method='linear', limit_direction='both')

    # Drop the helper column
    df.drop(columns=['Weight_outlier'], inplace=True)

    # Check if any NaNs remain
    #df['Weight'].isnull().sum()
    return df