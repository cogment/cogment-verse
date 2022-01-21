ARG BASE_PYTHON_IMAGE
FROM ${BASE_PYTHON_IMAGE} as base_python
FROM python:3.7-slim

# Install tensorflow as early as possible to help with cache
RUN pip install tensorflow==2.6.2

# Install system dependencies
RUN apt-get update && apt-get install -y curl

# Install poetry
ENV POETRY_VERSION=1.1.11
ENV POETRY_HOME="/usr/local/"
ENV POETRY_NO_INTERACTION=1
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV POETRY_VIRTUALENVS_CREATE=false

# Copy the `base_python` library
COPY --from=base_python /base_python /base_python

WORKDIR /tf_agents

# Install dependencies (w/o tf)
COPY pyproject.toml poetry.lock ./
RUN poetry install --no-root

# Copy the sources and build the package
COPY . ./
RUN poetry run task build

ENTRYPOINT [ "poetry", "run" ]
CMD [ "task", "start" ]
