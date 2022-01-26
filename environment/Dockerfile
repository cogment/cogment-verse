ARG BASE_PYTHON_IMAGE
FROM ${BASE_PYTHON_IMAGE} as base

#########
# "dev" is the developpment image
FROM python:3.7-slim as dev

# Install system dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y software-properties-common && apt-add-repository non-free
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y curl xvfb python3-opengl xvfb git tk swig wget unrar libglib2.0-0 g++ cmake

# Install poetry
ENV POETRY_VERSION=1.1.11
ENV POETRY_HOME="/usr/local/"
ENV POETRY_NO_INTERACTION=1
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV POETRY_VIRTUALENVS_CREATE=false

# Copy the `base_python` library
COPY --from=base /cogment-verse/base_python /cogment-verse/base_python

WORKDIR /cogment-verse/environment

# Install dependencies
COPY pyproject.toml poetry.lock ./
RUN poetry install --no-root

# Download & import the atari roms
RUN poetry run task import_atari_roms

ENTRYPOINT ["xvfb-run", "-e", "/dev/stdout", "-a"]
# start the server in development mode
CMD ["poetry", "run", "task", "dev"]
#########

#########
# "prod" is the production image
FROM dev as prod

# Build the package
COPY . ./
RUN poetry run task build

CMD ["poetry", "run", "task", "start"]
#########
