import pandas as pd
import numpy as np
import model_storage
import model_training
import data_preprocessing
from utils import get_user_logger

def predict_price(user_id: str, features: dict):
    """
    Predicts a price quote for a user based on input product features.

    Args:
        user_id (str): Identifier for the user.
        features (dict): Raw input features for the prediction.

    Returns:
        tuple: (float, int) - Predicted price and the model version used.
    
    Raises:
        ValueError: If no model or training data is available, or if input transformation fails.
    """
    logger = get_user_logger(user_id)
    logger.info(f"Received prediction request: {features}")
    try:
        # Try to load the latest model for the user
        model, version = model_storage.load_model(user_id, component="model")
    except FileNotFoundError:
        # No model found, attempt to train one if data exists
        try:
            model, version = model_training.train_model(user_id)
        except FileNotFoundError as e:
            # No data available to train a model
            raise ValueError(f"No model available for user '{user_id}' (no training data).") from e
    
    # Step 1: Transform the user input to match training data format
    try:
        X_new = data_preprocessing.transform_user_input(user_id, features)
    except Exception as e:
        logger.error(f"Input transformation failed: {e}")
        raise ValueError(f"Error in transforming user input: {str(e)}") from e

    # Step 2: Perform prediction
    all_preds = [tree.predict(X_new) for tree in model.estimators_]
    mean_pred = np.mean(all_preds)
    std_dev = np.std(all_preds)
    predicted_price = float(mean_pred)

    #Step 3: Calculate Prediction interval
    lower = mean_pred - 1.96 * std_dev
    upper = mean_pred + 1.96 * std_dev

    logger.info(f"Generated quote: {predicted_price:.2f} Prediction Interval: ({lower:.2f},{upper:.2f})(using model v{version})")
    return predicted_price, lower, upper, version
