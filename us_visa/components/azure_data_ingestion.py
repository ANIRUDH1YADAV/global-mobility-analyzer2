import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

from us_visa.logger import logging
from us_visa.exception import USvisaException
from us_visa.constants import *
from us_visa.configuration.azure_config import get_engine as build_azure_engine


class AzureSQLDataIngestion:

    def __init__(self):
        # Pull credentials from constants (which loads from .env)
        self.server   = AZURE_SQL_SERVER
        self.database = AZURE_SQL_DATABASE
        self.username = AZURE_SQL_USERNAME
        self.password = AZURE_SQL_PASSWORD

    def get_engine(self):
        return build_azure_engine(
            server=self.server,
            database=self.database,
            username=self.username,
            password=self.password,
        )

    def fetch_data(self, table_name=AZURE_SQL_TABLE) -> pd.DataFrame:
        try:
            if not str(table_name).replace("_", "").isalnum():
                raise ValueError(
                    "Invalid table name. Use alphanumeric and underscores only."
                )
            logging.info(f"Fetching data from Azure SQL table: {table_name}")
            engine = self.get_engine()
            df = pd.read_sql(f"SELECT * FROM [{table_name}]", engine)
            logging.info(f"Data fetched successfully. Shape: {df.shape}")
            return df
        except Exception as e:
            raise USvisaException(e, sys)

    def split_data_as_train_test(self, df: pd.DataFrame):
        try:
            logging.info("Splitting data into train and test sets...")
            train_set, test_set = train_test_split(
                df,
                test_size=DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO,
                random_state=42
            )
            save_dir = os.path.join(
                ARTIFACT_DIR,
                DATA_INGESTION_DIR_NAME,
                DATA_INGESTION_INGESTED_DIR
            )
            os.makedirs(save_dir, exist_ok=True)

            train_path = os.path.join(save_dir, "train.csv")
            test_path  = os.path.join(save_dir, "test.csv")

            train_set.to_csv(train_path, index=False)
            test_set.to_csv(test_path,   index=False)

            logging.info(f"Train saved: {train_path}")
            logging.info(f"Test saved:  {test_path}")
            return train_path, test_path
        except Exception as e:
            raise USvisaException(e, sys)

    def initiate_data_ingestion(self):
        try:
            logging.info(">>> Data Ingestion Started")
            df = self.fetch_data()
            train_path, test_path = self.split_data_as_train_test(df)
            logging.info(">>> Data Ingestion Completed")
            return train_path, test_path
        except Exception as e:
            raise USvisaException(e, sys)