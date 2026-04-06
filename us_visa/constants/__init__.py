import os
from datetime import date
from dotenv import load_dotenv

load_dotenv()

# ─── Azure SQL Database ──────────────────────────────────
# ✅ These MUST match azure_config.py os.getenv() keys exactly
AZURE_SQL_SERVER   = os.getenv("AZURE_SERVER")      # ← was AZURE_SQL_SERVER
AZURE_SQL_DATABASE = os.getenv("AZURE_DB")          # ← was AZURE_SQL_DATABASE
AZURE_SQL_USERNAME = os.getenv("AZURE_USER")        # ← was AZURE_SQL_USER
AZURE_SQL_PASSWORD = os.getenv("AZURE_PASSWORD")    # ← was AZURE_SQL_PASSWORD
AZURE_SQL_TABLE    = "EasyVisa"

# ─── Pipeline ────────────────────────────────────────────
PIPELINE_NAME = "usvisa"
ARTIFACT_DIR  = "artifact"

# ─── Data Ingestion ──────────────────────────────────────
DATA_INGESTION_DIR_NAME               = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR      = "feature_store"
DATA_INGESTION_INGESTED_DIR           = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO = 0.2

# ─── Data Validation ─────────────────────────────────────
DATA_VALIDATION_DIR_NAME               = "data_validation"
DATA_VALIDATION_DRIFT_REPORT_DIR       = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME = "report.yaml"

# ─── Data Transformation ─────────────────────────────────
DATA_TRANSFORMATION_DIR_NAME               = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR   = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR = "transformed_object"
PREPROCESSING_OBJECT_FILE_NAME             = "preprocessing.pkl"

TARGET_COLUMN = "case_status"
CURRENT_YEAR  = date.today().year

# ─── Model Trainer ───────────────────────────────────────
MODEL_TRAINER_DIR_NAME               = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR      = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME     = "model.pkl"
MODEL_TRAINER_EXPECTED_SCORE         = 0.7
MODEL_TRAINER_MODEL_CONFIG_FILE_PATH = os.path.join("config", "model.yaml")

# ─── Model Evaluation ────────────────────────────────────
MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE = 0.02
MODEL_BUCKET_NAME   = os.getenv("MODEL_BUCKET_NAME", "your-s3-bucket-name")
MODEL_PUSHER_S3_KEY = "model-registry"

# ─── Prediction Paths ────────────────────────────────────
TRAINED_MODEL_FILE_PATH = os.path.join(
    ARTIFACT_DIR,
    MODEL_TRAINER_DIR_NAME,
    MODEL_TRAINER_TRAINED_MODEL_DIR,
    MODEL_TRAINER_TRAINED_MODEL_NAME
)

PREPROCESSOR_OBJ_FILE_PATH = os.path.join(
    ARTIFACT_DIR,
    DATA_TRANSFORMATION_DIR_NAME,
    DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
    PREPROCESSING_OBJECT_FILE_NAME
)

MODEL_FILE_NAME = "model.pkl"