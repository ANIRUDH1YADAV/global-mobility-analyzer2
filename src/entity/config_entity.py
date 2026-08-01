from dataclasses import dataclass


# =====================================================
# Data EDA
# =====================================================

@dataclass
class DataEDAArtifact:
    eda_report_path: str
    plots_dir: str


# =====================================================
# Data Ingestion
# =====================================================

@dataclass
class DataIngestionArtifact:
    raw_file_path: str
    train_file_path: str
    test_file_path: str


# =====================================================
# Data Validation
# =====================================================

@dataclass
class DataValidationArtifact:
    validation_status: bool
    validation_report_path: str


# =====================================================
# Data Transformation
# =====================================================

@dataclass
class DataTransformationArtifact:
    preprocessor_path: str
    label_encoder_path: str
    transformed_train_path: str
    transformed_test_path: str
    feature_selection_report_path: str


# =====================================================
# Model Trainer
# =====================================================

@dataclass
class ModelTrainerArtifact:
    trained_model_path: str
    train_auc: float
    test_auc: float
    model_name: str


# =====================================================
# Model Evaluation
# =====================================================

@dataclass
class ModelEvaluationArtifact:
    is_model_accepted: bool
    evaluation_report_path: str


# =====================================================
# Model Pusher
# =====================================================

@dataclass
class ModelPusherArtifact:
    pushed_model_path: str
    pushed_preprocessor_path: str
    pushed_label_encoder_path: str