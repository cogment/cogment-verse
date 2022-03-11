ARG BASE_PYTHON_IMAGE
ARG DEVICE=cpu
FROM ${BASE_PYTHON_IMAGE} as base

#########
# "cpu" is the base image for cpu-based services
FROM python:3.7-slim as cpu

# Install torch as early as possible to help with cache
RUN pip install torch==1.11.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html

# Install system dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y curl xvfb
#########

#########
# "gpu" is the base image for gpu-based services
# Check the support matrix to ensure the nvcr image has the correct pytorch version
# https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html
FROM nvcr.io/nvidia/pytorch:22.02-py3 as gpu

# Install the rest of the dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y python3-opengl xvfb git tk swig wget unrar libglib2.0-0 g++ cmake
RUN pip uninstall -y torchvision torchtext torchaudio
RUN pip install torch==1.11.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
#########

#########
# "dev" is the development image
FROM ${DEVICE} as dev

# Install poetry
ENV POETRY_VERSION=1.1.11
ENV POETRY_HOME="/usr/local/"
ENV POETRY_NO_INTERACTION=1
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV POETRY_VIRTUALENVS_CREATE=false

# Copy a mock of the 'environment' library (checked by poetry even if it's not installed)
COPY mock_environment/setup.py /cogment-verse/environment/setup.py

# Copy the `base_python` library
COPY --from=base /cogment-verse/base_python /cogment-verse/base_python

WORKDIR /cogment-verse/torch_agents

# Install dependencies (w/o torch)
COPY pyproject.toml poetry.lock ./
RUN poetry install --no-root

# start the server in development mode
ENTRYPOINT [ "poetry", "run" ]
CMD ["task", "dev"]
#########

#########
# "prod" is the production image
FROM dev as prod

# Copy the sources and build the package
COPY . ./
RUN poetry run task build

CMD [ "task", "start" ]
#########
