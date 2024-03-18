import os
import pandas as pd
import pickle
from ts.torch_handler.base_handler import BaseHandler
import numpy as np

class ModelHandler(BaseHandler):

    def handle(self, data, context):
        """
        Invoke by TorchServe for prediction request.
        Do pre-processing of data, prediction using model and postprocessing of prediction output
        :param data: Input data for prediction
        :param context: Initial context contains model server system properties.
        :return: prediction output
        """
        model_input = self.preprocess(data)
        model_output = self.inference(model_input)
        return self.postprocess(model_output)

    def initialize(self, context):
        """
        Initialize model. This will be called during model loading time
        :param context: Initial context contains model server system properties.
        :return:
        """
        super().initialize(context)

    def preprocess(self, data):
        """
        Preprocess function to convert the request input to a tensor(Torchserve supported format).
        Args :
            data (list): List of the data from the request input.
        Returns:
            tensor: Returns the tensor data of the input
        """
        converted_data = [{key: [value.decode()] if isinstance(value, bytearray) else value for key, value in entry.items()} for entry in data]
        data = converted_data[0]
        bool_mapping = {'True': True, 'False': False}

        features = pd.DataFrame.from_dict(data)
        bool_columns = ['CryoSleep', 'VIP']
        for col in bool_columns:
            features[col] = features[col].map(bool_mapping)

        features["CryoSleep"] = features["CryoSleep"].astype(float)
        features["VIP"] = features["VIP"].astype(float)

        bool_columns = ['Age','RoomService', 'FoodCourt', 'ShoppingMall','Spa', 'VRDeck']
        for col in bool_columns:
            features[col] = float(features[col])

        with open(os.path.join("encoder_traindata.pickle"), 'rb') as f:
            enc = pickle.load(f)
            transenc = enc.transform(features[["HomePlanet","Destination"]])
            transenc = transenc.toarray()
            transformed_data = pd.DataFrame(transenc, columns=enc.get_feature_names_out())
            features.drop(["HomePlanet","Destination"], axis=1, inplace=True)
            features = pd.concat([features, transformed_data], axis=1)

        return features.to_numpy().astype(np.float32)
    def inference(self, data):
        ort_inputs = {self.model.get_inputs()[0].name: data[0]}

        ort_outs = self.model.run(None, ort_inputs)
        return ort_outs

    def postprocess(self, data):
        data = data[0] > 0.3
        return [{'transported': str(data[0])}]