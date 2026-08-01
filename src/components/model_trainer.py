import sys
import numpy as np
from typing import Dict

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from src.configuration.configuration import ConfigurationManager
from src.constants import RANDOM_STATE, TARGET_COLUMN
from src.entity.config_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
)
from src.exception import CustomException
from src.logger import logger
from src.utils.main_utils import save_object, write_yaml_file


class ModelTrainer:

    def __init__(self):
        self.config = ConfigurationManager.get_model_trainer_config(
            target_column=TARGET_COLUMN,
            random_state=RANDOM_STATE,
        )

    @staticmethod
    def _prepare_xy(arr: np.ndarray):
        x = arr[:, :-1]
        y = arr[:, -1].astype(int)
        return x, y

    @staticmethod
    def _evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, y_prob)),
        }

    def initiate_model_training(
        self,
        transformation_artifact: DataTransformationArtifact,
    ) -> ModelTrainerArtifact:

        try:
            logger.info("Starting Model Training")

            logger.info("Loading transformed train data")
            train_arr = np.load(
                transformation_artifact.transformed_train_path
            )

            logger.info("Loading transformed test data")
            test_arr = np.load(
                transformation_artifact.transformed_test_path
            )

            x_train, y_train = self._prepare_xy(train_arr)
            x_test, y_test = self._prepare_xy(test_arr)

            logger.info("Initializing candidate models")

            candidate_models = {

                "RandomForest": RandomForestClassifier(
                    n_estimators=300,
                    random_state=self.config.random_state,
                    class_weight="balanced",
                    n_jobs=-1,
                ),

                "XGBoost": XGBClassifier(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=6,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    random_state=self.config.random_state,
                    eval_metric="logloss",
                ),
            }

            best_model = None
            best_model_name = None
            best_test_auc = -1
            best_train_auc = -1
            best_metrics = {}
            model_comparison = {}

            for name, model in candidate_models.items():

                logger.info(f"Training {name}")

                model.fit(x_train, y_train)

                train_pred = model.predict(x_train)
                test_pred = model.predict(x_test)

                train_prob = model.predict_proba(x_train)[:, 1]
                test_prob = model.predict_proba(x_test)[:, 1]

                train_auc = roc_auc_score(y_train, train_prob)
                test_auc = roc_auc_score(y_test, test_prob)

                model_comparison[name] = {
                    "train_auc": float(train_auc),
                    "test_auc": float(test_auc),
                    **self._evaluate_predictions(y_test, test_pred, test_prob),
                }

                if test_auc > best_test_auc:

                    best_test_auc = test_auc
                    best_train_auc = train_auc
                    best_model = model
                    best_model_name = name

                    best_metrics = {
                        **self._evaluate_predictions(y_test, test_pred, test_prob),
                    }

            logger.info(
                f"Best Model Selected : {best_model_name}"
            )

            save_object(
                self.config.trained_model_path,
                best_model,
            )

            write_yaml_file(
                self.config.training_report_path,
                {
                    "selected_model": best_model_name,
                    "selected_model_metrics": {
                        "train_auc": float(best_train_auc),
                        "test_auc": float(best_test_auc),
                        "accuracy": float(best_metrics["accuracy"]),
                        "precision": float(best_metrics["precision"]),
                        "recall": float(best_metrics["recall"]),
                        "f1_score": float(best_metrics["f1_score"]),
                        "roc_auc": float(best_metrics["roc_auc"]),
                    },
                    "comparison": model_comparison,
                },
            )

            logger.info("Training Report Saved")
            logger.info("Model Training Completed Successfully")

            return ModelTrainerArtifact(
                trained_model_path=self.config.trained_model_path,
                train_auc=float(best_train_auc),
                test_auc=float(best_test_auc),
                model_name=best_model_name,
            )

        except Exception as e:
            raise CustomException(e, sys)