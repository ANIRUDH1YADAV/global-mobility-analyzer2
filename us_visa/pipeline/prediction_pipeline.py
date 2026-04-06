import os
import sys
import pandas as pd
from us_visa.logger import logging
from us_visa.exception import USvisaException
from us_visa.constants import PREPROCESSOR_OBJ_FILE_PATH, TRAINED_MODEL_FILE_PATH
from us_visa.utils.main_utils import load_object


class VisaData:
    def __init__(
        self,
        continent: str,
        education_of_employee: str,
        has_job_experience: str,
        requires_job_training: str,
        no_of_employees: int,
        yr_of_estab: int,
        region_of_employment: str,
        prevailing_wage: float,
        unit_of_wage: str,
        full_time_position: str,
    ):
        self.continent             = continent
        self.education_of_employee = education_of_employee
        self.has_job_experience    = has_job_experience
        self.requires_job_training = requires_job_training
        self.no_of_employees       = no_of_employees
        self.yr_of_estab           = yr_of_estab
        self.region_of_employment  = region_of_employment
        self.prevailing_wage       = prevailing_wage
        self.unit_of_wage          = unit_of_wage
        self.full_time_position    = full_time_position

    def get_data_as_dataframe(self) -> pd.DataFrame:
        try:
            return pd.DataFrame({
                "continent":             [self.continent],
                "education_of_employee": [self.education_of_employee],
                "has_job_experience":    [self.has_job_experience],
                "requires_job_training": [self.requires_job_training],
                "no_of_employees":       [self.no_of_employees],
                "yr_of_estab":           [self.yr_of_estab],
                "region_of_employment":  [self.region_of_employment],
                "prevailing_wage":       [self.prevailing_wage],
                "unit_of_wage":          [self.unit_of_wage],
                "full_time_position":    [self.full_time_position],
            })
        except Exception as e:
            raise USvisaException(e, sys)


class PredictionPipeline:

    def __init__(self):
        try:
            logging.info(f"Loading model: {TRAINED_MODEL_FILE_PATH}")
            self.model = load_object(TRAINED_MODEL_FILE_PATH)

            self.preprocessor = None
            if os.path.exists(PREPROCESSOR_OBJ_FILE_PATH):
                logging.info(f"Loading preprocessor: {PREPROCESSOR_OBJ_FILE_PATH}")
                self.preprocessor = load_object(PREPROCESSOR_OBJ_FILE_PATH)
                logging.info("✅ Model and Preprocessor loaded successfully!")
            else:
                logging.info(
                    f"Preprocessor not found at {PREPROCESSOR_OBJ_FILE_PATH}. "
                    "Using model feature alignment without preprocessor."
                )
                logging.info("✅ Model loaded successfully!")

        except Exception as e:
            raise USvisaException(e, sys)

    def _align_to_model_features(self, transformed_df: pd.DataFrame) -> pd.DataFrame:
        """Align prediction-time features to the exact schema used during model fit."""
        if not hasattr(self.model, "feature_names_in_"):
            return transformed_df

        expected_cols = list(self.model.feature_names_in_)

        # Handle preprocessor-generated prefixes like cat__/num__.
        rename_map = {}
        for col in transformed_df.columns:
            if "__" in col:
                stripped = col.split("__", 1)[1]
                if stripped in expected_cols and col not in expected_cols:
                    rename_map[col] = stripped

        if rename_map:
            transformed_df = transformed_df.rename(columns=rename_map)

        missing_cols = [c for c in expected_cols if c not in transformed_df.columns]
        if missing_cols:
            logging.info(f"Adding {len(missing_cols)} missing feature columns with 0.")
            for col in missing_cols:
                transformed_df[col] = 0

        extra_cols = [c for c in transformed_df.columns if c not in expected_cols]
        if extra_cols:
            logging.info(f"Dropping {len(extra_cols)} unexpected feature columns.")

        return transformed_df.reindex(columns=expected_cols, fill_value=0)

    def _prepare_features(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        if self.preprocessor is not None:
            transformed = self.preprocessor.transform(dataframe)

            # Some preprocessors return scipy sparse matrices.
            if hasattr(transformed, "toarray"):
                transformed = transformed.toarray()

            if hasattr(self.preprocessor, "get_feature_names_out"):
                transformed_df = pd.DataFrame(
                    transformed,
                    columns=self.preprocessor.get_feature_names_out(),
                )
            else:
                transformed_df = pd.DataFrame(transformed)

            return self._align_to_model_features(transformed_df)

        # Fallback path when only model.pkl exists.
        transformed_df = pd.get_dummies(dataframe, drop_first=True)
        return self._align_to_model_features(transformed_df)

    def predict(self, dataframe: pd.DataFrame) -> str:
        try:
            logging.info(f"Input Data:\n{dataframe}")

            # Step 1: Prepare features using available artifacts.
            transformed_df = self._prepare_features(dataframe)
            logging.info(f"Transformed feature shape: {transformed_df.shape}")

            # Step 2: Predict
            prediction = self.model.predict(transformed_df)

            # Step 3: Probability
            proba = self.model.predict_proba(transformed_df)

            # ✅ FINAL DEBUG (VISIBLE IN TERMINAL)
            logging.info("\n========== DEBUG ==========")
            logging.info(f"Input: {dataframe.to_dict()}")
            logging.info(f"Prediction: {prediction[0]}")
            logging.info(f"Probabilities: {proba}")
            logging.info(f"Denied: {proba[0][0]}")
            logging.info(f"Certified: {proba[0][1]}")
            logging.info("========== END ==========\n")

            # Step 4: Threshold fix
            if proba[0][1] > 0.4:
                result = "Certified"
            else:
                result = "Denied"

            logging.info(f"Final Result: {result}")
            return result

        except Exception as e:
            raise USvisaException(e, sys)