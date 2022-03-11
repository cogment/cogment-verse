ARG BASE_PYTHON_IMAGE
FROM ${BASE_PYTHON_IMAGE} as base_python
FROM python:3.7-slim

# Install torch as early as possible to help with cache
# In this case we install the generic version because of transitive dependencies from the environment lib
RUN pip install torch==1.11.0 -f https://download.pytorch.org/whl/cpu/torch_stable.html

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

WORKDIR /torch_agents

# Install dependencies (w/o torch, with what's needed for tests )
COPY pyproject.toml poetry.lock ./
RUN poetry install --no-root -E test

# Build the package
COPY . ./
RUN poetry run task build

ENTRYPOINT [ "poetry", "run" ]
CMD [ "task", "test" ]
