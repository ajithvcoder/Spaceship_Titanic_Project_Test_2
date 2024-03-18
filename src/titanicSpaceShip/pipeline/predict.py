import requests

class PredictionPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        prediction = requests.post("http://localhost:8080/predictions/spaceship", data=features)
        prediction = prediction.json()
        return prediction