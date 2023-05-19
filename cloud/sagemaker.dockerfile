FROM python:3.9

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y swig python3-opencv python3-pip nodejs npm

RUN mkdir -p /opt/ml/code/cloud/
COPY requirements.txt /opt/ml/code/
COPY cloud/sagemaker_utils.py /opt/ml/code/cloud/
COPY cloud/sagemaker_main.py /opt/ml/code/
RUN chmod +x /opt/ml/code

WORKDIR /opt/ml/code
RUN pip install --no-cache-dir -r requirements.txt --timeout 5000
RUN pip install SuperSuit==3.7.0
RUN chmod +x sagemaker_main.py

# Activate license for atari game
RUN AutoROM --accept-license

ENTRYPOINT ["python", "sagemaker_main.py"]

