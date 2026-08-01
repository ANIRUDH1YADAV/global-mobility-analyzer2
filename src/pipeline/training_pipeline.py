import sys

from src.components.data_ingestion import DataIngestion
from src.components.data_eda import DataEDA
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation
from src.components.model_pusher import ModelPusher

from src.exception import CustomException
from src.logger import logger


class TrainingPipeline:

    def __init__(self):

        self.ingestion = DataIngestion()
        self.eda = DataEDA()
        self.validation = DataValidation()
        self.transformation = DataTransformation()
        self.trainer = ModelTrainer()
        self.evaluator = ModelEvaluation()
        self.pusher = ModelPusher()

    def run_pipeline(self):

        try:

            logger.info("=" * 60)
            logger.info("Training Pipeline Started")
            logger.info("=" * 60)

            logger.info("Step 1/6 : Data Ingestion")
            ingestion_artifact = (
                self.ingestion.initiate_data_ingestion()
            )

            logger.info("Step 2/7 : Data EDA")
            eda_artifact = self.eda.initiate_eda(ingestion_artifact)

            logger.info("Step 3/7 : Data Validation")
            validation_artifact = (
                self.validation.initiate_data_validation(
                    ingestion_artifact
                )
            )

            if not validation_artifact.validation_status:
                raise ValueError(
                    "Data Validation Failed."
                )

            logger.info("Step 4/7 : Data Transformation")
            transformation_artifact = (
                self.transformation.initiate_data_transformation(
                    ingestion_artifact
                )
            )

            logger.info("Step 5/7 : Model Training")
            trainer_artifact = (
                self.trainer.initiate_model_training(
                    transformation_artifact
                )
            )

            logger.info("Step 6/7 : Model Evaluation")
            evaluation_artifact = (
                self.evaluator.initiate_model_evaluation(
                    trainer_artifact
                )
            )

            if not evaluation_artifact.is_model_accepted:
                raise ValueError(
                    "Model Rejected During Evaluation."
                )

            logger.info("Step 7/7 : Model Pusher")

            model_pusher_artifact = (
                self.pusher.initiate_model_pusher(
                    trainer_artifact,
                    transformation_artifact,
                )
            )

            logger.info("=" * 60)
            logger.info("Training Pipeline Completed Successfully")
            logger.info("=" * 60)

            return {

                "ingestion_artifact":
                    ingestion_artifact,

                "eda_artifact":
                    eda_artifact,

                "validation_artifact":
                    validation_artifact,

                "transformation_artifact":
                    transformation_artifact,

                "trainer_artifact":
                    trainer_artifact,

                "evaluation_artifact":
                    evaluation_artifact,

                "model_pusher_artifact":
                    model_pusher_artifact,
            }

        except Exception as e:
            raise CustomException(e, sys) from e