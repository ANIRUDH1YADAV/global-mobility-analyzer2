import sys
from dataclasses import dataclass

import pandas as pd

from src.constants import (
    PUSHED_MODEL_PATH,
    PUSHED_PREPROCESSOR_PATH,
    PUSHED_LABEL_ENCODER_PATH,
)
from src.exception import CustomException
from src.logger import logger
from src.utils.main_utils import load_object


@dataclass
class VisaData:
    continent: str
    education_of_employee: str
    has_job_experience: str
    requires_job_training: str
    no_of_employees: int
    yr_of_estab: int
    region_of_employment: str
    prevailing_wage: float
    unit_of_wage: str
    full_time_position: str

    def get_data_as_dataframe(self) -> pd.DataFrame:

        return pd.DataFrame(
            [{
                "continent": self.continent,
                "education_of_employee": self.education_of_employee,
                "has_job_experience": self.has_job_experience,
                "requires_job_training": self.requires_job_training,
                "no_of_employees": self.no_of_employees,
                "yr_of_estab": self.yr_of_estab,
                "region_of_employment": self.region_of_employment,
                "prevailing_wage": self.prevailing_wage,
                "unit_of_wage": self.unit_of_wage,
                "full_time_position": self.full_time_position,
            }]
        )


class PredictionPipeline:

    def __init__(self):

        try:

            logger.info("Loading preprocessor")
            self.preprocessor = load_object(
                PUSHED_PREPROCESSOR_PATH
            )

            logger.info("Loading model")
            self.model = load_object(
                PUSHED_MODEL_PATH
            )

            logger.info("Loading label encoder")
            self.label_encoder = load_object(
                PUSHED_LABEL_ENCODER_PATH
            )

        except Exception as e:
            raise CustomException(e, sys) from e

    def predict(
        self,
        features_df: pd.DataFrame,
    ) -> tuple[str, float]:

        try:

            logger.info("Running prediction")

            transformed = self.preprocessor.transform(
                features_df
            )

            if hasattr(transformed, "toarray"):
                transformed = transformed.toarray()

            prediction = self.model.predict(
                transformed
            )[0]

            prediction_label = (
                self.label_encoder.inverse_transform(
                    [prediction]
                )[0]
            )

            probability = self.model.predict_proba(
                transformed
            )[0]

            confidence = round(
                float(max(probability) * 100),
                2,
            )

            return prediction_label, confidence

        except Exception as e:
            raise CustomException(e, sys) from e