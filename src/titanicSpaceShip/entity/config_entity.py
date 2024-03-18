from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    local_data_file: Path
    cloud_config_zipfile: Path
    authentication_token: Path

@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path
    params_learning_rate: float
    params_features: int

@dataclass(frozen=True)
class PrepareCallbacksConfig:
    root_dir: Path
    tensorboard_root_log_dir: Path
    checkpoint_model_filepath: Path

@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    trained_model_path: Path
    base_model_path: Path
    training_data: Path
    encoder_traindata: Path
    params_epochs: int
    params_batch_size: int

@dataclass(frozen=True)
class EvaluationConfig:
    base_model_path: Path
    path_of_model: Path
    training_data: Path
    encoder_traindata: Path
    onnx_model_path: Path
    params_features: int
    all_params: dict
    params_batch_size: int