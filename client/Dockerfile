FROM python:3.7

# Install system dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y curl

# Install poetry
ENV POETRY_VERSION=1.1.11
ENV POETRY_HOME="/usr/local/"
ENV POETRY_NO_INTERACTION=1
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV POETRY_VIRTUALENVS_CREATE=false

WORKDIR /cogment-verse/client

# Install dependencies
COPY pyproject.toml poetry.lock ./
RUN poetry install --no-root

# Copy the rest of the service sources
COPY . ./

# Generate the proto files
RUN poetry run task build

ENV PYTHONPATH=.
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "-m", "main"]
