from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS, cross_origin
from titanicSpaceShip.components import decodeData
from titanicSpaceShip.pipeline.predict import PredictionPipeline
# from titanicSpaceShip.components import data_validation
from titanicSpaceShip import logger
import signal
import logging

flask_logger = logging.getLogger()

# create a formatter object
logFormatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")

# add file handler to the root logger
fileHandler = logging.FileHandler("flasklogs.log")
fileHandler.setFormatter(logFormatter)
flask_logger.addHandler(fileHandler)

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)

class ClientApp:
    def __init__(self):
        self.classifier = PredictionPipeline()

@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')

@app.route('/stopServer', methods=['GET'])
@cross_origin()
def stopServer():
    os.kill(os.getpid(), signal.SIGINT)
    return jsonify({ "success": True, "message": "Server is shutting down..." })

@app.route("/train", methods=['GET','POST'])
@cross_origin()
def trainRoute():
    os.system("dvc repro")
    os.system("torchserve --stop")
    os.system("torch-model-archiver -f --model-name spaceship --version 1.0 --serialized-file torchserve/models/spaceship.onnx --export-path torchserve/model-store --handler torchserve/handler.py --extra-files torchserve/utils/encoder_traindata.pickle -f")
    os.system("torchserve --start --ncs --model-store torchserve/model-store --models spaceship=spaceship.mar")
    logger.info("Training done successfully!")
    return "Training done successfully!"

@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    try:
        data = request.json['data']
        decodedFeatures = decodeData(data)
        result = clApp.classifier.predict(decodedFeatures)
        logger.info("Successfully result is responded to UI")
        return jsonify(result)
    except Exception as e:
        logger.info(f"Exception is raised {e}")
        return jsonify([{'transported': "Error processing"}])

if __name__ == "__main__":
    clApp = ClientApp()

    # app.run(host='127.0.0.1', port=5000, debug=True) #local host
    app.run(host='0.0.0.0', port=8085) #local host
    # # app.run(host='0.0.0.0', port=8080) #for AWS