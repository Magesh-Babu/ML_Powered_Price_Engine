import sys
import pandas as pd
import numpy as np
import model_storage
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import IsolationForest
from category_encoders import CountEncoder
from utils import get_user_logger

def load_csv(user_id: str) -> pd.DataFrame:
    """
    Loads historical quote data for a given user from a CSV file.

    Args:
        user_id (str): The user ID whose data should be loaded.

    Returns:
        pd.DataFrame: DataFrame containing the user's historical quotes.
    """
    base_dir = config['storage']['base_dir']
    file_path = f"{base_dir}/{user_id}/simulated_dataset.csv"
    # This will raise FileNotFoundError if the file does not exist
    df = pd.read_csv(file_path)
    return df

def apply_imputation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputes missing 'Lead_Time_weeks' values using a trained Random Forest regressor.

    Args:
        df (pd.DataFrame): DataFrame containing product features and lead time.

    Returns:
        pd.DataFrame: DataFrame with missing lead times imputed.
    """
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
    """
    Detects and replaces outliers in the 'Weight_kg_m' column using interpolation.

    Args:
        df (pd.DataFrame): DataFrame containing the 'Weight_kg_m' column.

    Returns:
        pd.DataFrame: DataFrame with outliers treated and interpolated.
    """
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

def encode_cat_features(df: pd.DataFrame, user_id: str):
    """
    Applies frequency and ordinal encoding to categorical features and saves the encoders.

    Args:
        df (pd.DataFrame): Training DataFrame containing categorical features.
        user_id (str): Identifier for the user, used to save encoders.

    Returns:
        pd.DataFrame: DataFrame with encoded categorical features.
    """
    # Frequency encoding
    freq_encoder = CountEncoder(cols=['Profile_Name'], normalize=True)
    df['Profile_Name_encoded'] = freq_encoder.fit_transform(df['Profile_Name'])
    model_storage.save_model(user_id, freq_encoder, component="freq_encoder")  # ✅ Save using your versioning logic

    # Ordinal encoding
    ordinal_features = ['Alloy', 'Finish', 'GD_T', 'Customer_Category']
    ordinal_encoder = OrdinalEncoder()
    df[ordinal_features] = ordinal_encoder.fit_transform(df[ordinal_features])
    model_storage.save_model(user_id, ordinal_encoder, component="ordinal_encoder")  # ✅ Save this too

    # Clean column
    df.drop(columns=['Profile_Name'], inplace=True)
    df.rename(columns={'Profile_Name_encoded': 'Profile_Name'}, inplace=True)

    return df

def feature_engineering(df: pd.DataFrame, user_id: str):
    """
    Applies domain-specific feature engineering and saves derived components for future use.

    Args:
        df (pd.DataFrame): DataFrame with raw or encoded training features.
        user_id (str): Identifier for the user, used to save derived components.

    Returns:
        pd.DataFrame: DataFrame with additional engineered features.
    """
    df['Quote_Date'] = pd.to_datetime(df['Quote_Date'])
    df = df.sort_values('Quote_Date')

    # Profile complexity
    df['Profile_Complexity'] = (df['Weight_kg_m'] / df['Length_m'] *df['Tolerances'] * (df['GD_T'] + 1))

    # Manufacturing difficulty
    df['Manufacturing_Difficulty'] = df['Tolerances'] * (df['GD_T'] + 1)

    # LME moving average and lag
    df['LME_MA_7'] = df['LME_Price_EUR'].rolling(window=7, min_periods=1).mean()
    df['LME_Lag_1'] = df['LME_Price_EUR'].shift(1).bfill()

    lme_features = {
        "LME_MA_7": df['LME_MA_7'].iloc[-1],
        "LME_Lag_1": df['LME_Lag_1'].iloc[-1]
    }
    model_storage.save_model(user_id, lme_features, component="lme")  # ✅ Save this as well

    return df

def load_and_preprocess(user_id: str) -> pd.DataFrame:
    """
    Loads a user's dataset and applies preprocessing and feature engineering steps.

    Args:
        user_id (str): Identifier for the user whose data is being processed.

    Returns:
        pd.DataFrame: Cleaned and feature-engineered DataFrame ready for model training.
    """    
    logger = get_user_logger(user_id)
    df = load_csv(user_id)
    df = apply_imputation(df)
    df = treat_outlier(df)
    df = encode_cat_features(df, user_id)
    df = feature_engineering(df, user_id)
    logger.info(f"Data Preprocessing completed.")
    #print("Data preprocessing complete. Final dataset shape:", df.shape)
    return df