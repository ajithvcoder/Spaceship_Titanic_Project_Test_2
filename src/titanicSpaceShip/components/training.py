import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from titanicSpaceShip.entity.config_entity import TrainingConfig
from sklearn.preprocessing import OneHotEncoder
from pathlib import Path
import torch.nn as nn
import torch
import shutil
from titanicSpaceShip import logger
import pickle

class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    def __len__(self):
        return len(self.features)
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
        
    def get_base_model(self):
        print(self.config.base_model_path)
        self.model = torch.load(
            self.config.base_model_path
        )
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
    
    
    def train_valid_generator(self):

        data = pd.read_csv(self.config.training_data)
        y = data["Transported"].astype(int)
        X = data.drop(["SNo","Transported"], axis=1)
        encoder = OneHotEncoder(handle_unknown="ignore")
        X["CryoSleep"] = X["CryoSleep"].astype(float)
        X["VIP"] = X["VIP"].astype(float)
        multicol_encoded = encoder.fit(X[["HomePlanet","Destination"]])
        with open(self.config.encoder_traindata, 'wb') as f:
            pickle.dump(multicol_encoded, f)
        multicol_encoded = encoder.transform(X[["HomePlanet","Destination"]])
        multicol_encoded = multicol_encoded.toarray()
        multicol_encoded = pd.DataFrame(multicol_encoded, columns=encoder.get_feature_names_out())
        X.drop(["HomePlanet","Destination"], axis=1, inplace=True)
        X = pd.concat([X, multicol_encoded], axis=1)
        X_tensor = torch.Tensor(X.values)
        y_tensor = torch.Tensor(y.values)
        X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.20, random_state=21)
        train_dataset = CustomDataset(X_train, y_train)
        test_dataset = CustomDataset(X_test, y_test)
        self.train_loader = DataLoader(train_dataset, batch_size= self.config.params_batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size= self.config.params_batch_size, shuffle=True)

    @staticmethod
    def save_model(path: Path, destpath: Path):
        shutil.copyfile(path, destpath)


    def train(self, callback_list: list):
        for epoch in range(self.config.params_epochs):
            overall_loss = 0 
            for inputs, labels in self.train_loader:
                outputs = self.model(inputs)
                loss = self.criterion(outputs[:,0], labels)
                overall_loss += loss.item()
                loss.backward()
                self.optimizer.step()
            logger.info(f"Epoch {epoch+1} completed loss {overall_loss}")
            correct_predictions = 0
            total_samples = 0
            with torch.no_grad():
                for inputs, labels in self.test_loader:
                    outputs_test = self.model(inputs)
                    outputs_test = (outputs_test >= 0.3).float()
                    correct_predictions += torch.sum(outputs_test[:,0]==labels)
                    total_samples += labels.size(0)
            test_accuracy = correct_predictions/total_samples
            logger.info(f"Test Accuracy : {test_accuracy*100:.2f}%")
            tensorboard_writer, save_checkpoint = callback_list
            tensorboard_writer.add_scalar("Loss", overall_loss, epoch+1)
            tensorboard_writer.add_scalar("Test Accuracy", test_accuracy, epoch+1)
            save_checkpoint(test_accuracy*100, epoch+1, self.model, self.optimizer, self.criterion)

        self.save_model(
            path="artifacts/prepare_callbacks/checkpoint_dir/model.pt",
            destpath=self.config.trained_model_path
        )