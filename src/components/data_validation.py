import sys
import pandas as pd

from src.configuration.configuration import ConfigurationManager
from src.constants import TARGET_COLUMN
from src.entity.config_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
)
from src.exception import CustomException
from src.logger import logger
from src.utils.main_utils import write_yaml_file


class DataValidation:

    def __init__(self):
        self.config = ConfigurationManager.get_data_validation_config()

    def initiate_data_validation(
        self,
        ingestion_artifact: DataIngestionArtifact,
    ) -> DataValidationArtifact:

        try:
            logger.info("Starting Data Validation")

            train_df = pd.read_csv(
                ingestion_artifact.train_file_path
            )

            test_df = pd.read_csv(
                ingestion_artifact.test_file_path
            )

            report = {

                "train_rows": len(train_df),

                "test_rows": len(test_df),

                "target_present_train":
                    TARGET_COLUMN in train_df.columns,

                "target_present_test":
                    TARGET_COLUMN in test_df.columns,

                "same_columns_train_test":
                    sorted(train_df.columns.tolist())
                    ==
                    sorted(test_df.columns.tolist()),

                "missing_values_train":
                    int(train_df.isnull().sum().sum()),

                "missing_values_test":
                    int(test_df.isnull().sum().sum()),

                "duplicate_rows_train":
                    int(train_df.duplicated().sum()),

                "duplicate_rows_test":
                    int(test_df.duplicated().sum()),

            }

            validation_status = all(

                [

                    report["target_present_train"],

                    report["target_present_test"],

                    report["same_columns_train_test"],

                    report["train_rows"] > 0,

                    report["test_rows"] > 0,

                ]

            )

            report["validation_status"] = validation_status

            write_yaml_file(
                self.config.validation_report_path,
                report,
            )

            logger.info(
                "Validation Report Saved Successfully"
            )

            logger.info(
                f"Validation Status : {validation_status}"
            )

            return DataValidationArtifact(

                validation_status=validation_status,

                validation_report_path=self.config.validation_report_path,

            )

        except Exception as e:

            raise CustomException(e, sys)