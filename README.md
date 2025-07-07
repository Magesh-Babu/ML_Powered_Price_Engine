# AI-Powered Price Engine

## Project Overview

The AI-Powered Price Engine is a machine learning project that predicts manufacturing component prices based on various features. By leveraging historical data and commodity price trends, this engine provides quick price predictions in a manufacturing context. The project demonstrates an end-to-end ML pipeline from data preprocessing and feature engineering to model training and deployment for price prediction.

## System Architecture and Data Flow

The system is designed as a modular pipeline where data flows through sequential stages from raw input to price predictions. The major components and data flow are as follows:

1. **Data Loading**: Reads historical component data from a CSV file.
2. **Data Preprocessing**: Cleans missing values, treats outliers, and encodes categorical variables.
3. **Feature Engineering**: Creates new domain-specific features to enhance prediction quality.
4. **Model Training**: Trains a Random Forest Regressor.
5. **Model Evaluation**: Assesses performance using MAE, RMSE, R², and MAPE.
6. **Model Storage**: Saves the trained model for reuse.
7. **Inference**: Exposes a FastAPI app for serving predictions from the trained model.
8. **Continous Learning**: Re-training the model from scratch each time with the new data included.

## Technology and Model Choices

The engine uses a **Random Forest Regressor** for price quoting. This ensemble method was chosen for its strong balance of accuracy, robustness, and interpretability, particularly well-suited for structured tabular data. Random Forests can model non-linear relationships and feature interactions effectively. They also naturally handle outliers and noisy observations — which is crucial in pricing scenarios where atypical quotes or edge cases may arise.

To maximize performance, the model's hyperparameters are tuned using **Bayesian Optimization with BayesSearchCV**. This method intelligently explores the search space to find optimal configurations (e.g., `n_estimators`, `max_depth`, `min_samples_leaf`, etc.) in fewer iterations than grid or random search. This approach ensures that the model generalizes well even with a relatively modest dataset (~1000 rows). For reproducibility and consistent evaluation, the `random_state` is fixed at 42.

## Data Preprocessing

Advanced techniques are applied to prepare the data:

🔧 **Imputation**
- `Lead_Time_weeks` variable had missing values.
- Implemented advanced imputation technique using a Random Forest Regressor, making use of relationships in the data.

🚨 **Outlier Detection**
- `Weight_kg_m` variable contained outliers.
- Treated using Isolation Forest, a tree-based anomaly detection method.

🧠 **Encoding**
- `Profile_Name` variable has high cardinality, applied frequency encoding to convert categories into meaningful numeric frequencies.
- Other categorical varaibles, applied one-hot encoding.

🧬 **Feature Engineering**
- Creates a feature `Profile_Complexity` which is geometry-driven complexity using weight, length, tolerances and GD_T. It captures physical + process complexity and includes geometric properties.
- Creates a feature `Manufacturing_Difficulty` and it focuses only on process precision requirements. Higher tolerance precision and GD&T complexity leads to more difficulty in manufacturing processes.
- Creates time series feature `LME_MA_7` which is calculation of average of the past 7 LME prices (sliding window), it smooths out short-term noise in raw LME prices and captures trend direction
- Creates another time series feature  `LME_Lag_1` Shifts the LME price column by 1 time step, it captures the last known LME price available before the current quote date — useful in time-aware prediction models.

In the current setup, scaling is left as passthrough for numeric features because tree-based models like random forest do not require feature scaling (they split on raw values). If we were using a linear model or neural network, we would likely enable scaling (e.g., using StandardScaler) 

## Model Evaluation

The following metrics are used:

| Metric   | Value | Description |
|-------------|---------|---------|
| **MAE**  | 0.0021 | Average absolute error |
| **RMSE** | 0.0 | 	Penalizes larger errors more |
| **R² Score**  | 1.0 | Perfect variance explanation |
| **MAPE (%)** | 0.08 | Only 0.08% average error in percentage terms |

The model demonstrates exceptionally high performance across all evaluation metrics:
- MAE (0.0021) indicates that, on average, the absolute difference between predicted and actual prices is extremely small — making the model very precise for practical use.
- RMSE (0.0) is nearly zero, which shows that even large prediction errors (which RMSE penalizes more heavily) are virtually nonexistent in the current evaluation.
- R² Score (1.0) suggests the model captures 100% of the variance in the target variable, meaning the model predictions align almost perfectly with actual prices.
- MAPE (0.08%) shows that the model's predictions are off by less than one-tenth of one percent, making it extremely reliable even in percentage terms — especially useful in business reporting contexts.

These metrics are consistent with a model that is both well-fitted and generalizable to the test set. However, the real-world data drift or feedback may introduce noise over time, so the pipeline is designed to support continuous learning and feedback integration to maintain this level of accuracy.

## Price Prediction and Uncertainty

The predicted price is computed as the mean prediction from all individual trees in the Random Forest ensemble. To estimate prediction uncertainty, calculating the standard deviation of predictions from each tree, assuming a normal distribution. The 95% prediction interval is then derived as:
```
[mean - 1.96 × std, mean + 1.96 × std]
```
This interval reflects the range within which the true quote is expected to fall with 95% confidence, helping users assess pricing reliability and risk.

## Isolation and Multi-tenancy

Each user's data, models and components are isolated in separate folders. There is **no overlap or sharing of data** between users. In the FastAPI routes, the `{user_id}` path parameter differentiates users, and the backend uses this to read from the correct file and load the correct model and components. This design achieves multi-tenancy with simple file-based storage. This isolation is important for both data privacy and correctness (no risk of one user's data contaminating another's model). Additionally, by isolating at the filesystem level, we can easily apply different encryption keys or retention policies per user if needed in the future.

## FastAPI Interface

The application provides a RESTful API built with **FastAPI**. FastAPI was chosen for its performance and ease of use (especially its integration with Pydantic for validation). There are two main endpoints:

- `POST /users/{user_id}/quote`: Generate a new price quote for user `{user_id}` given a product specification in the request body.
- `POST /users/{user_id}/feedback`: Submit a new data point (product spec + actual price) for user `{user_id}`, triggering a model update.

## Setup and Installation
### 🔨 Step-by-Step
1. **Clone the Repo**
```
git clone https://github.com/Magesh-Babu/ML_Powered_Price_Engine.git
cd ML_Powered_Price_Engine
```
2. **Create Virtual Environment (Optional but Recommended)**
```
# 1. Create a virtual environment
python -m venv venv

# 2. Activate the virtual environment
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```
3. **Install Dependencies**
```
pip install -r requirements.txt
```

## Running Inference via FastAPI
The trained model can be served using a REST API with FastAPI.
### 🚀 Start the API Server
```
uvicorn main:app --reload
```
Visit the API at: http://127.0.0.1:8000
### Try the API
Go to: http://127.0.0.1:8000/docs
Use the interactive Swagger UI to:
- Submit input feature data via `/quote` 
- Receive predicted price in response

### 📤 Sample Request
**user_id**: `user_1` or `user_2`

**Request body**:
```
{
    "Alloy": "Aluminium",
    "Finish": "Powder coated",
    "Length_m": 23.8,
    "Weight_kg_m": 1.342,
    "Tolerances": 0.1,
    "GD_T": "medium",
    "Order_Quantity": 85000,
    "LME_Price_EUR": 3.4,
    "Customer_Category": "medium",
    "Lead_Time_weeks": 6,
    "Profile_Name": "Karmlist"
}
```


