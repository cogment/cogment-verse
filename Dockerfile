ARG BASE_IMAGE=python:3.9
FROM $BASE_IMAGE

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y swig python3-opencv python3-pip nodejs npm

WORKDIR /cogment-verse

COPY requirements.txt .
RUN pip install -r requirements.txt --timeout 5000
RUN pip install SuperSuit==3.7.0

COPY . .

ENV COGMENT_VERSE_WORK_DIR /cogment_verse_work_dir
VOLUME ${COGMENT_VERSE_WORK_DIR}
