import sys
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.configuration.configuration import ConfigurationManager
from src.constants import TARGET_COLUMN
from src.entity.config_entity import DataEDAArtifact, DataIngestionArtifact
from src.exception import CustomException
from src.logger import logger
from src.utils.main_utils import create_directories, write_yaml_file


class DataEDA:
    def __init__(self):
        self.config = ConfigurationManager.get_data_eda_config()
        create_directories([
            self.config.plots_dir,
            os.path.dirname(self.config.eda_report_path),
        ])

    @staticmethod
    def _safe_savefig(file_path: str) -> None:
        plt.tight_layout()
        plt.savefig(file_path, dpi=150, bbox_inches="tight")
        plt.close()

    def initiate_eda(self, ingestion_artifact: DataIngestionArtifact) -> DataEDAArtifact:
        try:
            logger.info("Starting EDA")
            df = pd.read_csv(ingestion_artifact.raw_file_path)

            report = {
                "shape": {
                    "rows": int(df.shape[0]),
                    "columns": int(df.shape[1]),
                },
                "columns": df.columns.tolist(),
                "dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()},
                "missing_values": {column: int(value) for column, value in df.isna().sum().items()},
                "duplicate_rows": int(df.duplicated().sum()),
            }

            if TARGET_COLUMN in df.columns:
                report["target_distribution"] = {
                    str(index): int(value)
                    for index, value in df[TARGET_COLUMN].value_counts().items()
                }

            numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
            categorical_columns = df.select_dtypes(include=["object"]).columns.tolist()

            report["numeric_columns"] = numeric_columns
            report["categorical_columns"] = categorical_columns

            write_yaml_file(self.config.eda_report_path, report)

            if TARGET_COLUMN in df.columns:
                plt.figure(figsize=(6, 4))
                sns.countplot(x=df[TARGET_COLUMN])
                plt.title("Target Distribution")
                plt.xlabel(TARGET_COLUMN)
                plt.ylabel("Count")
                self._safe_savefig(os.path.join(self.config.plots_dir, "target_distribution.png"))

            if numeric_columns:
                df[numeric_columns].hist(figsize=(14, 10), bins=20)
                plt.suptitle("Numeric Feature Distributions")
                self._safe_savefig(os.path.join(self.config.plots_dir, "numeric_distributions.png"))

            if categorical_columns:
                top_columns = categorical_columns[:4]
            else:
                top_columns = []

            if top_columns:
                fig, axes = plt.subplots(len(top_columns), 1, figsize=(10, 4 * len(top_columns)))
                if len(top_columns) == 1:
                    axes = [axes]
                for axis, column in zip(axes, top_columns):
                    value_counts = df[column].astype(str).value_counts().head(10)
                    sns.barplot(x=value_counts.values, y=value_counts.index, ax=axis)
                    axis.set_title(f"Top values for {column}")
                self._safe_savefig(os.path.join(self.config.plots_dir, "categorical_top_values.png"))

            logger.info("EDA Completed Successfully")

            return DataEDAArtifact(
                eda_report_path=self.config.eda_report_path,
                plots_dir=self.config.plots_dir,
            )

        except Exception as e:
            raise CustomException(e, sys) from e