FROM cogment/cogment:v2.9.2 AS cogdock
#FROM local/cogment:latest AS cogdock

FROM ubuntu:20.04

WORKDIR /workspace

ENV TZ=America/New_York
ENV LANG=C.UTF-8
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-venv \
    parallel \
    curl \
    vim \
    sed \
    swig \
    python3-opencv \
    build-essential \
    python3.9-dev
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1

COPY *.txt ./
COPY *.toml ./
COPY *.sh ./

RUN ./run.sh build_python

ENV COGMENT_LOG_LEVEL=warning
ENV orchestrator=9010
ENV orchestrator_web=9011


# Because Git is not installed in container and some python is doing "import git"
ENV GIT_PYTHON_REFRESH=quiet

COPY *.py ./
COPY actors ./actors/
COPY cogment_verse ./cogment_verse/
COPY config ./config/
COPY environments ./environments/
COPY runs ./runs/
COPY tests ./tests/
COPY tender_merkle_0_model ./.cogment_verse/model_registry/tender_merkle_0_model/ 

COPY --from=cogdock /usr/local/bin/cogment /usr/local/bin/cogment

ENV COGVERSE_LOG_LEVEL=DEBUG

ENTRYPOINT ["./run.sh"]

