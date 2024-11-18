import os
from utils.logger import setup_logger
from training.train_simclr import train_simclr
from configs.config import Config

if __name__ == "__main__":
    # Setup logger
    logger = setup_logger(Config.saved_models_path, log_file="simclr_training.log")
    logger.info("Starting SimCLR training...")

    try:
        train_simclr()
        logger.info("SimCLR training completed successfully.")
    except Exception as e:
        logger.error(f"An error occurred during SimCLR training: {e}")
        raise