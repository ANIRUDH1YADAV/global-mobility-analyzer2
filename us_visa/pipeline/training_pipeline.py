import sys
from us_visa.logger import logging
from us_visa.exception import USvisaException
from us_visa.components.azure_data_ingestion import AzureSQLDataIngestion

class TrainingPipeline:
    def run_pipeline(self):
        try:
            logging.info("===== Training Pipeline Started =====")
            data_ingestion = AzureSQLDataIngestion()
            train_path, test_path = data_ingestion.initiate_data_ingestion()
            logging.info("===== Training Pipeline Completed =====")
            return train_path, test_path
        except Exception as e:
            raise USvisaException(e, sys)