import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

from src.configuration.configuration import ConfigurationManager
from src.configuration.mongo_db_connection import MongoDBClient
from src.constants import TARGET_COLUMN, TRAIN_RATIO, RANDOM_STATE
from src.entity.config_entity import DataIngestionArtifact
from src.exception import CustomException
from src.logger import logger
from src.utils.main_utils import create_directories


class DataIngestion:

    def __init__(self):
        self.config = ConfigurationManager.get_data_ingestion_config(
            train_ratio=TRAIN_RATIO
        )

        create_directories([
            os.path.dirname(self.config.raw_file_path),
            os.path.dirname(self.config.train_file_path),
            os.path.dirname(self.config.test_file_path),
        ])

    def initiate_data_ingestion(self) -> DataIngestionArtifact:

        try:
            logger.info("Starting Data Ingestion")

            logger.info("Connecting to MongoDB")
            collection = MongoDBClient().get_collection()

            logger.info("Fetching records")
            records = list(collection.find())

            df = pd.DataFrame(records)

            if df.empty:
                raise ValueError("MongoDB collection is empty.")

            if "_id" in df.columns:
                df.drop(columns=["_id"], inplace=True)

            if TARGET_COLUMN not in df.columns:
                raise ValueError(
                    f"Target column '{TARGET_COLUMN}' not found."
                )

            logger.info(f"Dataset Shape : {df.shape}")

            df.to_csv(
                self.config.raw_file_path,
                index=False,
            )

            logger.info("Performing Train/Test Split")

            train_df, test_df = train_test_split(
                df,
                train_size=self.config.train_ratio,
                stratify=df[TARGET_COLUMN],
                random_state=RANDOM_STATE,
            )

            train_df.to_csv(
                self.config.train_file_path,
                index=False,
            )

            test_df.to_csv(
                self.config.test_file_path,
                index=False,
            )

            logger.info(
                "Data Ingestion Completed Successfully"
            )

            return DataIngestionArtifact(
                raw_file_path=self.config.raw_file_path,
                train_file_path=self.config.train_file_path,
                test_file_path=self.config.test_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)