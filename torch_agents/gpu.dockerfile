ARG BASE_PYTHON_IMAGE
FROM ${BASE_PYTHON_IMAGE} as base
FROM nvcr.io/nvidia/pytorch:21.08-py3

# Install the rest of the dependencies
# RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y python3-opengl xvfb git tk swig wget unrar libglib2.0-0
RUN apt-get install -y g++ cmake

# Install poetry
RUN apt-get update && apt-get install -y curl
RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py | python -
ENV PATH="/root/.local/bin:${PATH}"
RUN poetry config virtualenvs.create false

COPY --from=base /base_python /base_python

WORKDIR /torch_agents

# Install dependencies (w/o torch it's already in the image)
COPY pyproject.toml ./
# '/environment' is the location of an optional dependency we don't install here,
# still poetry is checking for its existance so let's fake it
COPY mock_environment/setup.py /environment/setup.py
RUN poetry install --no-root

# Build the package
COPY . ./
RUN poetry run task build

ENTRYPOINT [ "poetry", "run" ]
CMD [ "task", "start" ]
