import pandas as pd
import model_storage
from utils import get_user_logger

def transform_user_input(user_id: str, user_input: dict) -> pd.DataFrame:
    """
    Transforms raw user input into a model-ready format using saved encoders and feature mappings.

    Args:
        user_id (str): Identifier for the user whose encoders and mappings are loaded.
        user_input (dict): Raw input features for prediction.

    Returns:
        pd.DataFrame: Transformed DataFrame ready for model inference.
    """       
    df = pd.DataFrame([user_input])

    # Load components via versioned loading
    freq_encoder, _ = model_storage.load_model(user_id, component="freq_encoder")
    ordinal_encoder, _ = model_storage.load_model(user_id, component="ordinal_encoder")
    lme_statics, _ = model_storage.load_model(user_id, component="lme")

    # Apply frequency encoding
    df['Profile_Name'] = freq_encoder.transform(df['Profile_Name'])

    # Apply ordinal encoding
    ordinal_features = ['Alloy', 'Finish', 'GD_T', 'Customer_Category']
    df[ordinal_features] = ordinal_encoder.transform(df[ordinal_features])

    # Profile complexity
    df['Profile_Complexity'] = (df['Weight_kg_m'] / df['Length_m'] *df['Tolerances'] * (df['GD_T'] + 1))

    # Manufacturing difficulty
    df['Manufacturing_Difficulty'] = df['Tolerances'] * (df['GD_T'] + 1)

    # LME features (static from training)
    df['LME_MA_7'] = lme_statics['LME_MA_7']
    df['LME_Lag_1'] = lme_statics['LME_Lag_1']

    return df