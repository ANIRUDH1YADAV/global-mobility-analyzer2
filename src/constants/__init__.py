import os

# =====================================================
# Project Root
# =====================================================

ROOT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
    )
)

# =====================================================
# Common Directories
# =====================================================

ARTIFACT_DIR = os.path.join(ROOT_DIR, "artifacts")
LOG_DIR = os.path.join(ROOT_DIR, "logs")

# =====================================================
# Dataset Constants
# =====================================================

TARGET_COLUMN = "case_status"

TRAIN_RATIO = 0.80

RANDOM_STATE = 42

# =====================================================
# MongoDB
# =====================================================

DATABASE_NAME = "global_mobility"

COLLECTION_NAME = "visa_data"

# =====================================================
# Data Ingestion
# =====================================================

DATA_INGESTION_DIR = os.path.join(
    ARTIFACT_DIR,
    "data_ingestion",
)

RAW_FILE_PATH = os.path.join(
    DATA_INGESTION_DIR,
    "raw.csv",
)

TRAIN_FILE_PATH = os.path.join(
    DATA_INGESTION_DIR,
    "train.csv",
)

TEST_FILE_PATH = os.path.join(
    DATA_INGESTION_DIR,
    "test.csv",
)

# =====================================================
# Data Validation
# =====================================================

DATA_VALIDATION_DIR = os.path.join(
    ARTIFACT_DIR,
    "data_validation",
)

VALIDATION_REPORT_PATH = os.path.join(
    DATA_VALIDATION_DIR,
    "validation_report.yaml",
)

# =====================================================
# Data EDA
# =====================================================

DATA_EDA_DIR = os.path.join(
    ARTIFACT_DIR,
    "data_eda",
)

EDA_REPORT_PATH = os.path.join(
    DATA_EDA_DIR,
    "eda_report.yaml",
)

EDA_PLOTS_DIR = os.path.join(
    DATA_EDA_DIR,
    "plots",
)

# =====================================================
# Data Transformation
# =====================================================

DATA_TRANSFORMATION_DIR = os.path.join(
    ARTIFACT_DIR,
    "data_transformation",
)

PREPROCESSOR_PATH = os.path.join(
    DATA_TRANSFORMATION_DIR,
    "preprocessor.pkl",
)

LABEL_ENCODER_PATH = os.path.join(
    DATA_TRANSFORMATION_DIR,
    "label_encoder.pkl",
)

TRANSFORMED_TRAIN_PATH = os.path.join(
    DATA_TRANSFORMATION_DIR,
    "train.npy",
)

TRANSFORMED_TEST_PATH = os.path.join(
    DATA_TRANSFORMATION_DIR,
    "test.npy",
)

FEATURE_SELECTION_REPORT_PATH = os.path.join(
    DATA_TRANSFORMATION_DIR,
    "feature_selection_report.yaml",
)

# =====================================================
# Model Trainer
# =====================================================

MODEL_TRAINER_DIR = os.path.join(
    ARTIFACT_DIR,
    "model_trainer",
)

TRAINED_MODEL_PATH = os.path.join(
    MODEL_TRAINER_DIR,
    "model.pkl",
)

TRAINING_REPORT_PATH = os.path.join(
    MODEL_TRAINER_DIR,
    "training_report.yaml",
)

# =====================================================
# Model Evaluation
# =====================================================

MODEL_EVALUATION_DIR = os.path.join(
    ARTIFACT_DIR,
    "model_evaluation",
)

EVALUATION_REPORT_PATH = os.path.join(
    MODEL_EVALUATION_DIR,
    "evaluation_report.yaml",
)

# =====================================================
# Model Pusher
# =====================================================

MODEL_PUSHER_DIR = os.path.join(
    ARTIFACT_DIR,
    "final_model",
)

PUSHED_MODEL_PATH = os.path.join(
    MODEL_PUSHER_DIR,
    "model.pkl",
)

PUSHED_PREPROCESSOR_PATH = os.path.join(
    MODEL_PUSHER_DIR,
    "preprocessor.pkl",
)

PUSHED_LABEL_ENCODER_PATH = os.path.join(
    MODEL_PUSHER_DIR,
    "label_encoder.pkl",
)