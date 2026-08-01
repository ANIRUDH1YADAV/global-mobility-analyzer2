import os
import sys
import shutil

from src.configuration.configuration import ConfigurationManager
from src.entity.config_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
    ModelPusherArtifact,
)
from src.exception import CustomException
from src.logger import logger
from src.utils.main_utils import create_directories


class ModelPusher:

    def __init__(self):
        self.config = ConfigurationManager.get_model_pusher_config()

    def initiate_model_pusher(
        self,
        trainer_artifact: ModelTrainerArtifact,
        transformation_artifact: DataTransformationArtifact,
    ) -> ModelPusherArtifact:

        try:
            logger.info("Starting Model Pusher")

            create_directories([
                os.path.dirname(self.config.pushed_model_path),
                os.path.dirname(self.config.pushed_preprocessor_path),
                os.path.dirname(self.config.pushed_label_encoder_path),
            ])

            logger.info("Copying trained model")

            shutil.copy2(
                trainer_artifact.trained_model_path,
                self.config.pushed_model_path,
            )

            logger.info("Copying preprocessor")

            shutil.copy2(
                transformation_artifact.preprocessor_path,
                self.config.pushed_preprocessor_path,
            )

            logger.info("Copying label encoder")

            shutil.copy2(
                transformation_artifact.label_encoder_path,
                self.config.pushed_label_encoder_path,
            )

            logger.info("Model Pusher Completed Successfully")

            return ModelPusherArtifact(
                pushed_model_path=self.config.pushed_model_path,
                pushed_preprocessor_path=self.config.pushed_preprocessor_path,
                pushed_label_encoder_path=self.config.pushed_label_encoder_path,
            )

        except Exception as e:
            raise CustomException(e, sys)