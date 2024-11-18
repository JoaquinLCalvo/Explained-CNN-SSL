import os
from utils.logger import setup_logger
from training.train_classifier import train_classifier
from configs.config import Config

if __name__ == "__main__":
    # Setup logger
    logger = setup_logger(Config.saved_models_path, log_file="classifier_training.log")
    logger.info("Starting MLP Classifier training...")

    try:
        train_classifier()
        logger.info("MLP Classifier training completed successfully.")
    except Exception as e:
        logger.error(f"An error occurred during MLP Classifier training: {e}")
        raise
