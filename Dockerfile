#FROM nvidia/cuda:11.8.0-base-ubuntu22.04
FROM python:3.9

ENV DEBIAN_FRONTEND=noninteractive 
RUN apt-get update && apt-get install -y swig python3-opencv python3-pip
RUN apt install -y nodejs
RUN apt-get install -y npm

WORKDIR /cogment-verse

COPY requirements.txt .
RUN pip install -r requirements.txt --timeout 1000

COPY . .

ENV COGMENT_VERSE_WORK_DIR /cogment_verse_work_dir
VOLUME ${COGMENT_VERSE_WORK_DIR}

ENTRYPOINT python3 -m main
