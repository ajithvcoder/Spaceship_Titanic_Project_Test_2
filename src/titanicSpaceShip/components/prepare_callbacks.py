import os
import urllib.request as request
from zipfile import ZipFile
import time
from torch.utils.tensorboard import SummaryWriter
import torch
from titanicSpaceShip.entity.config_entity import PrepareCallbacksConfig
from titanicSpaceShip import logger

class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's 
    validation acc is greater than the previous least less, then save the
    model state.
    """
    def __init__(
        self, filepath, best_valid_acc=float(0)
    ):
        self.best_valid_acc = best_valid_acc
        self.filepath = filepath
        
    def __call__(
        self, current_valid_acc, 
        epoch, model, optimizer, criterion
    ):
        if current_valid_acc > self.best_valid_acc:
            self.best_valid_acc = current_valid_acc
            logger.info(f"\nBest validation acc: {self.best_valid_acc}")
            logger.info(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, self.filepath)

class PrepareCallback:
    def __init__(self, config: PrepareCallbacksConfig):
        self.config = config
        # self.save_best_model = SaveBestModel()


    
    @property
    def _create_tb_callbacks(self):
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
        tb_running_log_dir = os.path.join(
            self.config.tensorboard_root_log_dir,
            f"tb_logs_at_{timestamp}",
        )
        return SummaryWriter(log_dir=tb_running_log_dir)
    

    @property
    def _create_ckpt_callbacks(self):
        return SaveBestModel(
            filepath=self.config.checkpoint_model_filepath
        )


    def get_tb_ckpt_callbacks(self):
        return [
            self._create_tb_callbacks,
            self._create_ckpt_callbacks
        ]