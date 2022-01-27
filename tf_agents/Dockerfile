ARG BASE_PYTHON_IMAGE
FROM ${BASE_PYTHON_IMAGE} as base

#########
# "dev" is the developpment image
FROM python:3.7-slim as dev

# Install tensorflow as early as possible to help with cache
RUN pip install tensorflow==2.6.2

# Install system dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y curl

# Install poetry
ENV POETRY_VERSION=1.1.11
ENV POETRY_HOME="/usr/local/"
ENV POETRY_NO_INTERACTION=1
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV POETRY_VIRTUALENVS_CREATE=false

# Copy the `base_python` library
COPY --from=base /cogment-verse/base_python /cogment-verse/base_python

WORKDIR /cogment-verse/tf_agents

# Install dependencies (w/o tf)
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
