FROM python:3.7-slim

WORKDIR /cogment-verse/base_python

# Install system dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y curl

# Install poetry
ENV POETRY_VERSION=1.1.11
ENV POETRY_HOME="/usr/local/"
ENV POETRY_NO_INTERACTION=1
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV POETRY_VIRTUALENVS_CREATE=false

# Install dependencies
COPY pyproject.toml poetry.lock ./
RUN poetry install --no-root

# Install the cogment-verse package
COPY . ./
RUN poetry run task build

# Install the package
RUN poetry install
