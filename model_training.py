import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from skopt import BayesSearchCV
from skopt.space import Real, Integer

from pricing_engine import data_preprocessing
from pricing_engine import model_storage
from pricing_engine.utils import get_user_logger
from pricing_engine.config import config

def train_model(user_id: str):
    """
    Trains a Random Forest model with hyperparameter tuning using the user's historical data.

    Args:
        user_id (str): Identifier for the user whose model is being trained.

    Returns:
        tuple: A tuple containing the trained model and its saved version number.
    """
    logger = get_user_logger(user_id)
    # Load and preprocess the data
    df = data_preprocessing.load_and_preprocess(user_id)
    if df.empty:
        raise ValueError("No training data available after preprocessing.")
    df_sorted = df.sort_values("Quote_Date").reset_index(drop=True)

    # Split features and target
    X = df_sorted.drop(columns=["Quote_Price_SEK", "Quote_Date"])
    y = df_sorted["Quote_Price_SEK"]
    logger.info(f"Training data columns: {X.columns}")
    # Define Bayesian search space
    search_space = {
        'n_estimators': Integer(100, 300),
        'max_depth': Integer(5, 30),
        'min_samples_split': Integer(2, 10),
        'min_samples_leaf': Integer(1, 10),
        'max_features': Real(0.1, 1.0)
    }

    # Time Series CV
    tscv = TimeSeriesSplit(n_splits=5)

    # BayesSearchCV
    bayes_cv = BayesSearchCV(
        estimator=RandomForestRegressor(random_state=42),
        search_spaces=search_space,
        cv=tscv,
        n_iter=20,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
        random_state=42
    )
    bayes_cv.fit(X, y)

    # Train/Test split after sorting
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Train final model with best parameters
    final_model = RandomForestRegressor(**bayes_cv.best_params_, random_state=42)
    final_model.fit(X_train, y_train)

    # Predict and evaluate
    preds = final_model.predict(X_test)
    metrics = {
        "Best Hyperparameters": bayes_cv.best_params_,
        "MAE": round(mean_absolute_error(y_test, preds), 4),
        "RMSE": round(mean_squared_error(y_test, preds), 4),
        "R2 Score": round(r2_score(y_test, preds), 4),
        "MAPE (%)": round(mean_absolute_percentage_error(y_test, preds) * 100, 2)
    }
    #print(metrics)
    logger.info(f"Model training completed with metrics: {metrics}")

    # Save the trained model to disk with a new version
    version = model_storage.save_model(user_id, final_model, component="model")
    logger.info(f"Model training completed. Model version {version} saved.")
    return final_model, version
