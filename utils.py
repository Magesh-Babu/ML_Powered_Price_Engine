import os
import logging
from pricing_engine.config import config

def get_user_logger(user_id: str):
    """
    Get a logger for a specific user. Creates a separate log file per user.
    """
    logger_name = f"user_{user_id}"
    logger = logging.getLogger(logger_name)
    if not logger.handlers:
        # Ensure log directory exists
        base_dir = config['storage']['base_dir']
        log_dir = f"{base_dir}/{user_id}/logs"
        os.makedirs(log_dir, exist_ok=True)
        # Configure file handler for user-specific log
        file_path = f"{log_dir}/app.log"
        fh = logging.FileHandler(file_path)
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s\n")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        logger.setLevel(logging.INFO)
    return logger
