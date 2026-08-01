import sys
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

from src.configuration.configuration import ConfigurationManager
from src.constants import TARGET_COLUMN
from src.entity.config_entity import (
    DataIngestionArtifact,
    DataTransformationArtifact,
)
from src.exception import CustomException
from src.logger import logger
from src.utils.main_utils import save_object


class DataTransformation:
    def __init__(self):
        self.config = ConfigurationManager.get_data_transformation_config()

    @staticmethod
    def _feature_engineering(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        working_df = df.copy()
        engineered_columns = []

        text_columns = working_df.select_dtypes(include=["object"]).columns.tolist()
        for column in text_columns:
            working_df[column] = working_df[column].astype(str).str.strip()

        for column in ["has_job_experience", "requires_job_training", "full_time_position"]:
            if column in working_df.columns:
                working_df[column] = working_df[column].str.upper()
                engineered_columns.append(column)

        for column in ["continent", "education_of_employee", "region_of_employment", "unit_of_wage"]:
            if column in working_df.columns:
                working_df[column] = working_df[column].str.replace("  ", " ", regex=False)
                engineered_columns.append(column)

        report = {
            "engineered_columns": sorted(set(engineered_columns)),
            "text_columns_cleaned": text_columns,
            "rows_processed": int(len(working_df)),
        }

        return working_df, report

    @staticmethod
    def _select_features(x_train: pd.DataFrame, x_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        selected_columns = [column for column in x_train.columns if column != "case_id"]
        dropped_columns = [column for column in x_train.columns if column not in selected_columns]

        report = {
            "selected_columns": selected_columns,
            "dropped_columns": dropped_columns,
            "selected_feature_count": len(selected_columns),
        }

        return x_train[selected_columns].copy(), x_test[selected_columns].copy(), report

    def _build_preprocessor(self, x: pd.DataFrame) -> ColumnTransformer:
        numeric_columns = x.select_dtypes(include=np.number).columns.tolist()
        categorical_columns = [col for col in x.columns if col not in numeric_columns]

        logger.info(f"Numerical Columns: {numeric_columns}")
        logger.info(f"Categorical Columns: {categorical_columns}")

        numerical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore"),
                ),
            ]
        )

        return ColumnTransformer(
            transformers=[
                ("numerical", numerical_pipeline, numeric_columns),
                ("categorical", categorical_pipeline, categorical_columns),
            ]
        )

    def initiate_data_transformation(
        self,
        ingestion_artifact: DataIngestionArtifact,
    ) -> DataTransformationArtifact:
        try:
            logger.info("Starting Data Transformation")

            logger.info("Loading train dataset")
            train_df = pd.read_csv(ingestion_artifact.train_file_path)

            logger.info("Loading test dataset")
            test_df = pd.read_csv(ingestion_artifact.test_file_path)

            logger.info("Separating input and target features")

            x_train = train_df.drop(columns=[TARGET_COLUMN])
            y_train = train_df[TARGET_COLUMN]

            x_test = test_df.drop(columns=[TARGET_COLUMN])
            y_test = test_df[TARGET_COLUMN]

            logger.info("Applying feature engineering")
            x_train, feature_engineering_report = self._feature_engineering(x_train)
            x_test, _ = self._feature_engineering(x_test)

            logger.info("Applying feature selection")
            x_train, x_test, feature_selection_report = self._select_features(x_train, x_test)

            logger.info("Creating preprocessing pipeline")
            preprocessor = self._build_preprocessor(x_train)

            logger.info("Fitting preprocessor on training data")
            x_train = preprocessor.fit_transform(x_train)
            x_test = preprocessor.transform(x_test)

            logger.info("Encoding target variable")
            label_encoder = LabelEncoder()
            y_train = label_encoder.fit_transform(y_train)
            y_test = label_encoder.transform(y_test)

            logger.info("Saving preprocessing object")
            save_object(self.config.preprocessor_path, preprocessor)

            logger.info("Saving label encoder")
            save_object(self.config.label_encoder_path, label_encoder)

            logger.info("Saving feature selection report")
            from src.utils.main_utils import write_yaml_file

            write_yaml_file(
                self.config.feature_selection_report_path,
                {
                    "feature_engineering": feature_engineering_report,
                    "feature_selection": feature_selection_report,
                },
            )

            if hasattr(x_train, "toarray"):
                x_train = x_train.toarray()

            if hasattr(x_test, "toarray"):
                x_test = x_test.toarray()

            train_arr = np.c_[x_train, y_train]
            test_arr = np.c_[x_test, y_test]

            logger.info("Saving transformed train array")
            np.save(self.config.transformed_train_path, train_arr)

            logger.info("Saving transformed test array")
            np.save(self.config.transformed_test_path, test_arr)

            logger.info("Data Transformation Completed Successfully")

            return DataTransformationArtifact(
                preprocessor_path=self.config.preprocessor_path,
                label_encoder_path=self.config.label_encoder_path,
                transformed_train_path=self.config.transformed_train_path,
                transformed_test_path=self.config.transformed_test_path,
                feature_selection_report_path=self.config.feature_selection_report_path,
            )

        except Exception as e:
            raise CustomException(e, sys)