from utils.logger import setup_logger
from training.train_simclr import train_simclr
from scripts.run_explainability import run_explainability

def run_pipeline():
    logger = setup_logger(log_dir='logs/', log_file='pipeline.log')
    # train_simclr()
    run_explainability()


if __name__ == "__main__":
    logger = setup_logger(log_dir='logs/', log_file='pipeline.log')

    try:
        # run_pipeline()
        run_explainability()
        logger.info("Explainability pipeline completed successfully.")
    except Exception as e:
        logger.error(f"An error occurred during explainablity pipeline: {e}")
        raise