# Docker

Cogment Verse support development using [Docker](https://www.docker.com).

1. Install Docker 23.0.1
2. Install docker compose plugin v2.16.0. Docker Compose commands may not work with earlier versions.

To run Cogment Verse from docker containers:

1. Build the `cogment_verse` service's image. From the project's root, run the command:

   ```console
   $ docker compose build
   ```

   By default, the base image will be `python:3.9`. Pass the BASE_IMAGE build argument to modify it.

   ```console
   $ docker compose build --build-arg BASE_IMAGE=python:3.9
   ```

2. Launch the container services using docker compose using the command:

   ```console
   $ docker compose run --service-ports cogment_verse [ARGS...]
   ```

   This is equivalent to running `python -m main` locally.

   The same way it is done locally, you can add specific config arguments for hydra. Example:

   ```console
   $ docker compose run --service-ports cogment_verse +experiment=simple_bc/mountain_car
   ```

3. In case the default ports to access the web app, the orchestrator's web endpoint as well as mlflow are not available, you can specify alternates using the following environment variables. The default port values are the following:

   - `WEB_PORT=8080`
   - `ORCHESTRATOR_WEB_PORT=9000`
   - `MLFLOW_PORT=3000`

   For example, to use a different port for the web app and mlflow, use the command:

   ```console
   $ WEB_PORT=8081 MLFLOW_PORT=5000 docker compose run --service-ports cogment_verse
   ```

4. Open Chrome (other web browser might work but haven't tested) and navigate to http://localhost:8080

## Docker on Remote Instance

To run the docker version on a remote instance, make sure that the default ports are available. See the section above to pass a different combination of ports. Additionally, in order to connect to the remotly hosted web app from your browser, you must pass the public address of the host instance through the `WEB_HOST` environment variable.

For example, from the remote instance, run the command:

```console
$ WEB_HOST=[public ip address] docker compose run --service-ports cogment_verse
```

Open Chrome (other web browser might work but haven't tested) and navigate to http://[public ip address]:8080

## Troubleshooting

On M1/M2 Macs you'll need to force Docker to use the `linux/amd64` platform as a few dependencies are not availabe for `linux/arm64`. The environment variable `DOCKER_DEFAULT_PLATFORM` needs to be set to `linux/amd64`, e.g:

```console
$ DOCKER_DEFAULT_PLATFORM=linux/amd64 docker compose build
```
