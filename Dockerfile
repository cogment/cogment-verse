FROM python:3.9

RUN apt-get update && apt-get install -y swig python3-opencv

WORKDIR /cogment-verse

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8080

ENV COGMENT_VERSE_WORK_DIR /cogment_verse_work_dir
VOLUME ${COGMENT_VERSE_WORK_DIR}

ENTRYPOINT python -m main
