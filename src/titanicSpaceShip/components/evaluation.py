import torch.nn as nn
import torch
from titanicSpaceShip import logger
import pickle
import shutil
import onnxruntime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from titanicSpaceShip.entity.config_entity import EvaluationConfig
from titanicSpaceShip.utils.common import save_json
from pathlib import Path

class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    def __len__(self):
        return len(self.features)
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config


    def get_base_model(self):
        # print(self.config.base_model_path)
        self.model = torch.load(
            self.config.base_model_path
        )
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def _valid_generator(self):

        data = pd.read_csv(self.config.training_data)
        y = data["Transported"]
        y = data["Transported"].astype(int)
        X = data.drop(["SNo","Transported"], axis=1)
        X["CryoSleep"] = X["CryoSleep"].astype(float)
        X["VIP"] = X["VIP"].astype(float)
        multicol_encoded = []
        print(self.config.encoder_traindata)
        with open(self.config.encoder_traindata, 'rb') as f:
            encoder = pickle.load(f)
            multicol_encoded = encoder.transform(X[["HomePlanet","Destination"]])
            multicol_encoded = multicol_encoded.toarray()
            multicol_encoded = pd.DataFrame(multicol_encoded, columns=encoder.get_feature_names_out())
        X.drop(["HomePlanet","Destination"], axis=1, inplace=True)
        X = pd.concat([X, multicol_encoded], axis=1)
        X_tensor = torch.Tensor(X.values)
        y_tensor = torch.Tensor(y)
        X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=21)
        test_dataset = CustomDataset(X_test, y_test)
        self.test_loader = DataLoader(test_dataset, batch_size= self.config.params_batch_size, shuffle=True)

    def load_model(self, path: Path) -> nn.Module:
        checkpoint = torch.load(path, map_location=torch.device('cpu'))  # You can change 'cpu' to your desired device
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Model loaded from {path}")
        return self.model

    def copymodel_to_torchserve(self, dest_model_path, dest_encoder_path):
        shutil.copyfile(self.config.onnx_model_path, dest_model_path)
        shutil.copyfile(self.config.encoder_traindata, dest_encoder_path)

    def convert_to_onnx(self):
        dummy = torch.randn(self.config.params_features, requires_grad=True)
        torch.onnx.export(self.model,
                  dummy,
                  self.config.onnx_model_path,
                  export_params=True,
                  opset_version=10,
                  do_constant_folding=True,
                  input_names = ['input'],
                  output_names = ['output'],
                  )
        torch_out = self.model(dummy)
        ort_session = onnxruntime.InferenceSession(self.config.onnx_model_path, providers=["CPUExecutionProvider"])

        def to_numpy(tensor):
            return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

        # compute ONNX Runtime output prediction
        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy)}
        ort_outs = ort_session.run(None, ort_inputs)

        # compare ONNX Runtime and PyTorch results
        np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

        logger.info("Exported model has been tested with ONNXRuntime, and the result looks good!")

    def evaluation(self):
        self.model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        correct_predictions = 0
        total_samples = 0
        self.val_loss = 0
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                outputs_test = self.model(inputs)
                outputs_test = (outputs_test >= 0.3).float()
                correct_predictions += torch.sum(outputs_test[:,0]==labels)
                total_samples += labels.size(0)
                loss = self.criterion(outputs_test[:,0], labels)
                self.val_loss += loss.item()
        self.test_accuracy = f"{(correct_predictions/total_samples)*100:.2f}"
        self.save_score()
        print(f"Test Accuracy : {self.test_accuracy}%")
        self.convert_to_onnx()
        self.copymodel_to_torchserve("torchserve/models/spaceship.onnx", "torchserve/utils/encoder_traindata.pickle")

    
    def save_score(self):
        scores = {"loss": self.val_loss, "accuracy": self.test_accuracy}
        save_json(path=Path("scores.json"), data=scores)
