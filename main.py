import sys

from src.exception import CustomException
from src.logger import logger
from src.pipeline.training_pipeline import TrainingPipeline


def main():
    try:
        logger.info("=" * 70)
        logger.info("Starting Global Mobility Analyzer Training Pipeline")
        logger.info("=" * 70)

        training_pipeline = TrainingPipeline()
        training_pipeline.run_pipeline()

        logger.info("=" * 70)
        logger.info("Training Pipeline Completed Successfully")
        logger.info("=" * 70)

    except Exception as e:
        logger.exception("Training Pipeline Failed")
        raise CustomException(e, sys) from e


if __name__ == "__main__":
    main()