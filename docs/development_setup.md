# Development setup

This is a practical guide for developers wanting to develop within cogment verse.

## Prerequisites

> 🚧 _in construction_ 🚧

## Local development

Docker is a great tool because it provides a way to isolate environments and make sure they are fully contained: which system packages are used, which runtime is used, which dependencies are used... However during day-to-day development Docker can be hard to use because every change requires to stop the container / rebuild the image / start a new container. The Cogment Verse repository is designed to let developer both execute their code as a docker image (the default way) or locally.

### Full docker development

> 🚧 _in construction_ 🚧

### Local development

In this section we will explain how to develop on a given service, e.g. `torch_agents`, locally while relying on the other services running in Docker.

### Step 1 - Configure how the _other_ services access the _developed_ service

The first step is to make sure the other services knows how to access the developed service. This is achieved by changing their configuration through environment variables.

When everything runs in Docker, the different services can access each other based on their **hostname** as defined in the root [`docker-compose.yaml`](/docker-compose.yaml) inside a virtual network managed by the Docker runtime. For example, the orchestrator is accessible at the hostname `orchestrator`.

When developing and running a service on the host machine, this service is no longer part of the virtual network. It needs to be accessed at a different hostname. Because of the different ways Docker operates on different operating systems, this host name is based on the host machine OS.

Open [`.env`](/.env) and update the endpoit of the service you are developing. For example to work on `torch_agents`, change

```
COGMENT_VERSE_TORCH_AGENTS_ENDPOINT=torch_agents:${COGMENT_VERSE_TORCH_AGENTS_PORT}
```

to (if you are developing on **Linux**)

```
COGMENT_VERSE_TORCH_AGENTS_ENDPOINT=localhost:${COGMENT_VERSE_TORCH_AGENTS_PORT}
```

or (if you are developing on **macOS** or **Windows**)

```
COGMENT_VERSE_TORCH_AGENTS_ENDPOINT=host.docker.internal:${COGMENT_VERSE_TORCH_AGENTS_PORT}
```

### Step 2 - Configure how the _developed_ service accesses the _other_ services

The second step is the reverse, making sure the developed service, that is runing directly on the host machine (and not in Docker), can access the other services it needs.

#### Exposing the required ports

For something running on the host machine to be able to access a service running in the docker virtual network, its ports need to be exposed. Thanks to `docker-compose` configuration override capabilities this can be achieved by creating a new `.yaml` file, we can call it `docker-compose.local.yaml`. In this file we will simply expose the port for the services the developed service requires. For example to work on `torch_agents`, it would look like

```yaml
version: "3.7"

services:
  orchestrator:
    ports:
      - ${COGMENT_VERSE_ORCHESTRATOR_PORT}:${COGMENT_VERSE_ORCHESTRATOR_PORT}
  trial_datastore:
    ports:
      - ${COGMENT_VERSE_TRIAL_DATASTORE_PORT}:${COGMENT_VERSE_TRIAL_DATASTORE_PORT}
  model_registry:
    ports:
      - ${COGMENT_VERSE_MODEL_REGISTRY_PORT}:${COGMENT_VERSE_MODEL_REGISTRY_PORT}
```

Because Cogment Verse already defines unique ports for all the services this should work as-is. If there is a conflict with something else that might run on your machine, you'll need to update the ports in the root [`.env`](/.env) file.

#### Configuring the _developed_ service

To configure the _developed_ service, we will use a dedicated `.env` file. Let's start by copying the root [`.env`](/.env) to the folder of the developed service. Once this is done we need to edit the file to replace the hostname of the accessed service by `localhost`. For example to work on `torch_agents`, we would edit the following:

```
COGMENT_VERSE_TRIAL_DATASTORE_ENDPOINT=trial_datastore:${COGMENT_VERSE_TRIAL_DATASTORE_PORT}
COGMENT_VERSE_MODEL_REGISTRY_ENDPOINT=model_registry:${COGMENT_VERSE_MODEL_REGISTRY_PORT}
COGMENT_VERSE_ORCHESTRATOR_ENDPOINT=orchestrator:${COGMENT_VERSE_ORCHESTRATOR_PORT}
```

to

```
COGMENT_VERSE_TRIAL_DATASTORE_ENDPOINT=localhost:${COGMENT_VERSE_TRIAL_DATASTORE_PORT}
COGMENT_VERSE_MODEL_REGISTRY_ENDPOINT=localhost:${COGMENT_VERSE_MODEL_REGISTRY_PORT}
COGMENT_VERSE_ORCHESTRATOR_ENDPOINT=localhost:${COGMENT_VERSE_ORCHESTRATOR_PORT}
```

### Running the _other_ services

Now that everything is configured, we can build and run the service we rely on in Docker. To do that we will rely directly on `docker-compose up --build`. For example to work on `torch_agents`, we would run:

```
cogment run copy && docker-compose -f docker-compose.yaml -f docker-compose.local.yaml up --build trial_datastore model_registry orchestrator environment tf_agents mlflow grafana prometheus
```

### Developing!

With everything setup and running, you can simply work on the service and trigger run as usual using `cogment run start_run`. For example to work on `torch_agents`, you would do the following to install the dependencies and start the service locally

```
cd torch_agents
poetry install
poetry run task build
poetry run task start
```

And then you could start a run that uses this service, e.g.

```
RUN_PARAMS=cartpole_dqn cogment run start_run
```

## Dependencies

### Python dependencies

Dependencies of the various python libraries (including [`base_python`](./base_python), [`environment`](./environment), [`torch_agents`](./torch_agents), [`tf_agents`](./tf_agents), [`client`](./client)) are managed by [poetry](https://python-poetry.org) which, among other nice properties have much more robust version resolution and can install packages in parallel.

Whenever a change is made to the dependencies in a `pyproject.toml` file, make sure to run `poetry update` on the host machine to run the version resolution and update the `poetry.lock` file. This file will be used to speed up the install process within the docker images.

For larger dependencies, in particular **pytorch** and **tensorflow** we try to use specialized docker images already having the dependency installed. It means those dependencies should be optional and handled with _extras_ (cf. [`torch_agents/pyproject.toml`](/torch_agents/pyproject.toml)) and that versions provided by the docker image should match.
