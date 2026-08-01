import sys

from src.configuration.configuration import ConfigurationManager
from src.entity.config_entity import (
    ModelTrainerArtifact,
    ModelEvaluationArtifact,
)
from src.exception import CustomException
from src.logger import logger
from src.utils.main_utils import write_yaml_file


class ModelEvaluation:

    def __init__(self):
        self.config = ConfigurationManager.get_model_evaluation_config(
            min_expected_auc=0.60
        )

    def initiate_model_evaluation(
        self,
        trainer_artifact: ModelTrainerArtifact,
    ) -> ModelEvaluationArtifact:

        try:
            logger.info("Starting Model Evaluation")

            accepted = (
                trainer_artifact.test_auc
                >= self.config.min_expected_auc
            )

            report = {

                "model_name":
                    trainer_artifact.model_name,

                "test_auc":
                    trainer_artifact.test_auc,

                "minimum_expected_auc":
                    self.config.min_expected_auc,

                "accepted":
                    accepted,

            }

            write_yaml_file(
                self.config.evaluation_report_path,
                report,
            )

            logger.info(
                f"Model Accepted : {accepted}"
            )

            return ModelEvaluationArtifact(

                is_model_accepted=accepted,

                evaluation_report_path=self.config.evaluation_report_path,

            )

        except Exception as e:

            raise CustomException(e, sys)