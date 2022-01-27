FROM python:3.7
WORKDIR /service
VOLUME /data

# Install dependencies
COPY requirements.txt ./
RUN DEBIAN_FRONTEND=noninteractive pip install -r requirements.txt

EXPOSE $COGMENT_VERSE_MLFLOW_PORT

CMD mlflow server --host 0.0.0.0 --port $COGMENT_VERSE_MLFLOW_PORT --backend-store-uri sqlite:////data/mlflow.db --default-artifact-root /data
