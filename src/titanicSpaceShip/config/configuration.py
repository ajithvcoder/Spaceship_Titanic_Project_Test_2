import os
from titanicSpaceShip.constants import *
from titanicSpaceShip.utils.common import read_yaml, create_directories
from titanicSpaceShip.utils.common import get_size
from titanicSpaceShip.entity.config_entity import DataIngestionConfig, PrepareBaseModelConfig, \
                                                      PrepareCallbacksConfig, TrainingConfig, \
                                                      EvaluationConfig

class ConfigurationManager:
    def __init__(
            self,
            config_filepath = CONFIG_FILE_PATH,
            params_filepath = PARAMS_FILE_PATH):

            self.config = read_yaml(config_filepath)
            self.params = read_yaml(params_filepath)

            create_directories([self.config.artifacts_root])
    def get_data_ingestion_config(self) -> DataIngestionConfig:
          config = self.config.data_ingestion

          create_directories([config.root_dir])

          data_ingestion_config = DataIngestionConfig(
                root_dir=config.root_dir,
                local_data_file=config.local_data_file,
                cloud_config_zipfile=config.cloud_config_zipfile,
                authentication_token=config.authentication_token
          )
          return data_ingestion_config

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        
        create_directories([config.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(
                root_dir=Path(config.root_dir),
                base_model_path=Path(config.base_model_path),
                params_learning_rate=self.params.LEARNING_RATE,
                params_features=self.params.FEATURES
        )

        return prepare_base_model_config
    
    def get_prepare_callback_config(self) -> PrepareCallbacksConfig:
        config = self.config.prepare_callbacks
        model_ckpt_dir = os.path.dirname(config.checkpoint_model_filepath)
        create_directories([
            Path(model_ckpt_dir),
            Path(config.tensorboard_root_log_dir)
        ])

        prepare_callback_config = PrepareCallbacksConfig(
            root_dir=Path(config.root_dir),
            tensorboard_root_log_dir=Path(config.tensorboard_root_log_dir),
            checkpoint_model_filepath=Path(config.checkpoint_model_filepath)
        )

        return prepare_callback_config

    def get_training_config(self) -> TrainingConfig:
        training = self.config.training
        prepare_base_model = self.config.prepare_base_model
        params = self.params
        training_data = os.path.join(self.config.data_ingestion.local_data_file)
        create_directories([
            Path(training.root_dir)
        ])

        training_config = TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            base_model_path=Path(prepare_base_model.base_model_path),
            training_data=Path(training_data),
            encoder_traindata=Path(training.encoder_traindata),
            params_epochs=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE
        )

        return training_config
    
    def get_validation_config(self) -> EvaluationConfig:
        training = self.config.training
        eval_config = EvaluationConfig(
            base_model_path="artifacts/prepare_base_model/base_model.pt",
            path_of_model="artifacts/training/model.pt",
            training_data="artifacts/data_ingestion/data_spaceShip.csv",
            encoder_traindata=Path(training.encoder_traindata),
            onnx_model_path=Path(training.onnx_model_path),
            params_features=self.params.FEATURES,
            all_params=self.params,
            params_batch_size=self.params.BATCH_SIZE
        )
        return eval_config