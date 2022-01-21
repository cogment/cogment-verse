ARG BASE_PYTHON_IMAGE
FROM ${BASE_PYTHON_IMAGE} as base
FROM nvcr.io/nvidia/pytorch:21.08-py3

# Install the rest of the dependencies
RUN apt-get update && apt-get install -y python3-opengl xvfb git tk swig wget unrar libglib2.0-0
RUN apt-get install -y g++ cmake

# Install poetry
ENV POETRY_VERSION=1.1.11
ENV POETRY_HOME="/usr/local/"
ENV POETRY_NO_INTERACTION=1
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV POETRY_VIRTUALENVS_CREATE=false

COPY --from=base /base_python /base_python

WORKDIR /torch_agents

# Install dependencies (w/o torch it's already in the image)
COPY pyproject.toml ./
# '/environment' is the location of an optional dependency we don't install here,
# still poetry is checking for its existance so let's fake it
COPY null_environment_setup.py /environment/setup.py
RUN poetry install --no-root

# Build the package
COPY . ./
RUN poetry run task build

ENTRYPOINT [ "poetry", "run" ]
CMD [ "task", "start" ]
