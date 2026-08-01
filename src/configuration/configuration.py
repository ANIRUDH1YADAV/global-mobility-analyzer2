from dataclasses import dataclass
import os

from dotenv import load_dotenv

from src.constants import (
    COLLECTION_NAME,
    DATABASE_NAME,
    EDA_PLOTS_DIR,
    EDA_REPORT_PATH,
    RAW_FILE_PATH,
    FEATURE_SELECTION_REPORT_PATH,
    TRAIN_FILE_PATH,
    TEST_FILE_PATH,
    VALIDATION_REPORT_PATH,
    PREPROCESSOR_PATH,
    LABEL_ENCODER_PATH,
    TRANSFORMED_TRAIN_PATH,
    TRANSFORMED_TEST_PATH,
    TRAINED_MODEL_PATH,
    TRAINING_REPORT_PATH,
    EVALUATION_REPORT_PATH,
    PUSHED_MODEL_PATH,
    PUSHED_PREPROCESSOR_PATH,
    PUSHED_LABEL_ENCODER_PATH,
)

load_dotenv()


# ==========================
# Mongo Configuration
# ==========================

@dataclass
class MongoConfig:
    uri: str
    db_name: str
    collection: str


# ==========================
# Data Ingestion
# ==========================

@dataclass
class DataIngestionConfig:
    raw_file_path: str
    train_file_path: str
    test_file_path: str
    train_ratio: float


# ==========================
# Data Validation
# ==========================

@dataclass
class DataValidationConfig:
    validation_report_path: str


@dataclass
class DataEDAConfig:
    eda_report_path: str
    plots_dir: str


# ==========================
# Data Transformation
# ==========================

@dataclass
class DataTransformationConfig:
    preprocessor_path: str
    label_encoder_path: str
    transformed_train_path: str
    transformed_test_path: str
    feature_selection_report_path: str


# ==========================
# Model Trainer
# ==========================

@dataclass
class ModelTrainerConfig:
    trained_model_path: str
    training_report_path: str
    target_column: str
    random_state: int


# ==========================
# Model Evaluation
# ==========================

@dataclass
class ModelEvaluationConfig:
    evaluation_report_path: str
    min_expected_auc: float


# ==========================
# Model Pusher
# ==========================

@dataclass
class ModelPusherConfig:
    pushed_model_path: str
    pushed_preprocessor_path: str
    pushed_label_encoder_path: str


class ConfigurationManager:

    @staticmethod
    def get_mongo_config() -> MongoConfig:
        return MongoConfig(
            uri=os.getenv("MONGODB_URI"),
            db_name=os.getenv("DATABASE_NAME", DATABASE_NAME),
            collection=os.getenv("COLLECTION_NAME", COLLECTION_NAME),
        )

    @staticmethod
    def get_data_ingestion_config(
        train_ratio: float,
    ) -> DataIngestionConfig:

        return DataIngestionConfig(
            raw_file_path=RAW_FILE_PATH,
            train_file_path=TRAIN_FILE_PATH,
            test_file_path=TEST_FILE_PATH,
            train_ratio=train_ratio,
        )

    @staticmethod
    def get_data_validation_config() -> DataValidationConfig:

        return DataValidationConfig(
            validation_report_path=VALIDATION_REPORT_PATH,
        )

    @staticmethod
    def get_data_eda_config() -> DataEDAConfig:

        return DataEDAConfig(
            eda_report_path=EDA_REPORT_PATH,
            plots_dir=EDA_PLOTS_DIR,
        )

    @staticmethod
    def get_data_transformation_config() -> DataTransformationConfig:

        return DataTransformationConfig(
            preprocessor_path=PREPROCESSOR_PATH,
            label_encoder_path=LABEL_ENCODER_PATH,
            transformed_train_path=TRANSFORMED_TRAIN_PATH,
            transformed_test_path=TRANSFORMED_TEST_PATH,
            feature_selection_report_path=FEATURE_SELECTION_REPORT_PATH,
        )

    @staticmethod
    def get_model_trainer_config(
        target_column: str,
        random_state: int,
    ) -> ModelTrainerConfig:

        return ModelTrainerConfig(
            trained_model_path=TRAINED_MODEL_PATH,
            training_report_path=TRAINING_REPORT_PATH,
            target_column=target_column,
            random_state=random_state,
        )

    @staticmethod
    def get_model_evaluation_config(
        min_expected_auc: float = 0.60,
    ) -> ModelEvaluationConfig:

        return ModelEvaluationConfig(
            evaluation_report_path=EVALUATION_REPORT_PATH,
            min_expected_auc=min_expected_auc,
        )

    @staticmethod
    def get_model_pusher_config() -> ModelPusherConfig:

        return ModelPusherConfig(
            pushed_model_path=PUSHED_MODEL_PATH,
            pushed_preprocessor_path=PUSHED_PREPROCESSOR_PATH,
            pushed_label_encoder_path=PUSHED_LABEL_ENCODER_PATH,
        )