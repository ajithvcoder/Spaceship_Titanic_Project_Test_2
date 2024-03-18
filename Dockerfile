FROM pytorch/torchserve:latest

USER root
RUN apt-get install --reinstall apt
RUN apt update -y && apt install awscli -y
WORKDIR /app

COPY . /app
RUN pip3 install -r requirements.txt
RUN dvc init -f
RUN dvc repro

EXPOSE 8080
EXPOSE 8081
EXPOSE 8085

# RUN torchserve --stop
RUN torch-model-archiver -f --model-name spaceship --version 1.0 --serialized-file torchserve/models/spaceship.onnx --export-path torchserve/model-store --handler torchserve/handler.py --extra-files torchserve/utils/encoder_traindata.pickle -f

CMD ["/bin/bash", "./start_script.sh"]