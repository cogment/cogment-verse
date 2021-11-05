ARG BASE_PYTHON_IMAGE
FROM ${BASE_PYTHON_IMAGE} as base_python
FROM python:3.7-slim

# Install torch as early as possible to help with cache
RUN pip install torch==1.8.1+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html

# Install system dependencies
RUN apt-get update && apt-get install -y curl xvfb

# Install poetry
ENV POETRY_VERSION=1.1.11
ENV POETRY_HOME="/opt/poetry"
ENV POETRY_VIRTUALENVS_CREATE=false
ENV POETRY_NO_INTERACTION=1
ENV PATH="$POETRY_HOME/bin:$PATH"
RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py | python -

# Copy a mock of the 'environment' library (checked by poetry even if it's not installed)
COPY mock_environment/setup.py /environment/setup.py

# Copy the `base_python` library
COPY --from=base_python /base_python /base_python

WORKDIR /torch_agents

# Install dependencies (w/o torch)
COPY pyproject.toml poetry.lock ./
RUN poetry install --no-root

# Copy the sources and build the package
COPY . ./
RUN poetry run task build

ENTRYPOINT [ "poetry", "run" ]
CMD [ "task", "start" ]
