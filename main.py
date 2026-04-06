import sys
from us_visa.logger import logging
from us_visa.exception import USvisaException
from us_visa.pipeline.training_pipeline import TrainingPipeline

if __name__ == "__main__":
    try:
        logging.info("Starting Training Pipeline...")
        pipeline = TrainingPipeline()
        pipeline.run_pipeline()
        logging.info("Training Pipeline Finished!")
    except Exception as e:
        raise USvisaException(e, sys)