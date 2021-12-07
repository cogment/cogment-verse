ARG BASE_PYTHON_IMAGE
FROM ${BASE_PYTHON_IMAGE} as base_python
FROM nvcr.io/nvidia/pytorch:21.08-py3

# Install the rest of the dependencies
# RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y python3-opengl xvfb git tk swig wget unrar libglib2.0-0
RUN apt-get install -y g++ cmake

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
