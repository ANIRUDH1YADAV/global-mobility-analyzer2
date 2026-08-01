import os
import sys
import shutil

from dotenv import load_dotenv
from huggingface_hub import HfApi

from src.configuration.configuration import ConfigurationManager
from src.entity.config_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
    ModelPusherArtifact,
)
from src.exception import CustomException
from src.logger import logger
from src.utils.main_utils import create_directories

load_dotenv()


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

            logger.info("Uploading model artifacts to Hugging Face")

            api = HfApi()

            repo_id = os.getenv("HUGGINGFACE_REPO")
            token = os.getenv("HUGGINGFACE_TOKEN")

            if repo_id is None:
                raise ValueError(
                    "HUGGINGFACE_REPO not found in .env file."
                )

            if token is None:
                raise ValueError(
                    "HUGGINGFACE_TOKEN not found in .env file."
                )

            # Upload model
            api.upload_file(
                path_or_fileobj=self.config.pushed_model_path,
                path_in_repo="model.pkl",
                repo_id=repo_id,
                repo_type="model",
                token=token,
            )

            logger.info("Uploaded model.pkl")

            # Upload preprocessor
            api.upload_file(
                path_or_fileobj=self.config.pushed_preprocessor_path,
                path_in_repo="preprocessor.pkl",
                repo_id=repo_id,
                repo_type="model",
                token=token,
            )

            logger.info("Uploaded preprocessor.pkl")

            # Upload label encoder
            api.upload_file(
                path_or_fileobj=self.config.pushed_label_encoder_path,
                path_in_repo="label_encoder.pkl",
                repo_id=repo_id,
                repo_type="model",
                token=token,
            )

            logger.info("Uploaded label_encoder.pkl")

            logger.info(
                "All model artifacts uploaded successfully to Hugging Face"
            )

            logger.info("Model Pusher Completed Successfully")

            return ModelPusherArtifact(
                pushed_model_path=self.config.pushed_model_path,
                pushed_preprocessor_path=self.config.pushed_preprocessor_path,
                pushed_label_encoder_path=self.config.pushed_label_encoder_path,
            )

        except Exception as e:
            logger.exception("Model Pusher Failed")
            raise CustomException(e, sys) from e