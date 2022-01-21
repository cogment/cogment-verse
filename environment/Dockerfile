ARG BASE_PYTHON_IMAGE
FROM ${BASE_PYTHON_IMAGE} as base_python
FROM python:3.7-slim

# Install system dependencies
RUN apt-get update && apt-get install -y software-properties-common && apt-add-repository non-free
RUN apt-get update && apt-get install -y curl xvfb python3-opengl xvfb git tk swig wget unrar libglib2.0-0 g++ cmake

# Install poetry
ENV POETRY_VERSION=1.1.11
ENV POETRY_HOME="/usr/local/"
ENV POETRY_NO_INTERACTION=1
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV POETRY_VIRTUALENVS_CREATE=false

# Copy the `base_python` library
COPY --from=base_python /base_python /base_python

WORKDIR /environment

# Install dependencies
COPY pyproject.toml poetry.lock ./
RUN poetry install --no-root

# Download the atari roms
RUN poetry run task retrieve_roms

# Build the package
COPY . ./
RUN poetry run task build

ENTRYPOINT []
CMD ["xvfb-run", "-e", "/dev/stdout", "-a", "python", "-m", "main"]
