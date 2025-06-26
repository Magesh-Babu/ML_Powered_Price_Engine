import os
import csv
from config import config
import model_training
from utils import get_user_logger

def add_feedback(user_id: str, feedback_data: dict) -> int:
    """
    Appends user feedback to the dataset and retrains the model.

    Args:
        user_id (str): Identifier for the user.
        feedback_data (dict): Dictionary containing input features, actual price, and date.

    Returns:
        int: The new model version after retraining.

    Raises:
        Exception: If writing to the dataset file fails.
    """    
    logger = get_user_logger(user_id)
    logger.info(f"Received feedback data: {feedback_data}")
    base_dir = config['storage']['base_dir']
    user_dir = f"{base_dir}/{user_id}"
    os.makedirs(user_dir, exist_ok=True)
    file_path = f"{user_dir}/simulated_dataset.csv"
    file_exists = os.path.isfile(file_path)
    # Determine order of columns from config (features + target)
    columns = config['features']['indep_var'] + [config['features']['target']] + [config['features']['date']]
    # Append the new data to CSV
    try:
        with open(file_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                # Write header if file didn't exist
                writer.writerow(columns)
            # Ensure values are in the correct order
            row = [feedback_data.get(col) for col in columns]
            writer.writerow(row)
    except Exception as e:
        logger.error(f"Failed to write feedback data to CSV: {e}")
        raise
    # Retrain the model with the updated data
    model, new_version = model_training.train_model(user_id)
    logger.info(f"Feedback processed: model retrained to version {new_version}")
    return new_version