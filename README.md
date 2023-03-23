# Cogment Verse

[![Apache 2 License](https://img.shields.io/badge/license-Apache%202-green?style=flat-square)](./LICENSE) [![Changelog](https://img.shields.io/badge/-Changelog%20-blueviolet?style=flat-square)](./CHANGELOG.md)

> 🚧 Cogment-Verse is still under heavy development but has already been successfully used by several research teams, join the [Cogment community](https://cogment.ai/docs/community-channels) to get support and be notified of updates.

[Cogment](https://cogment.ai) is an innovative open source AI platform designed to leverage the advent of AI to benefit humankind through human-AI collaboration developed by [AI Redefined](https://ai-r.com). Cogment enables AI researchers and engineers to build, train and operate AI agents in simulated or real environments shared with humans. For the full user documentation visit <https://docs.cogment.ai>

This repository contains a library of environments and agent implementations
to get started with Human In the Loop Learning (HILL) and Reinforcement
Learning (RL) with Cogment in minutes. Cogment Verse is designed both
for practitioners discovering the field as well as for experienced
researchers or engineers as a framework to develop and benchmark new
approaches.

Cogment verse includes environments from:

- [OpenAI Gym](https://gym.openai.com),
- [Petting Zoo](https://www.pettingzoo.ml).
- [MinAtar](https://github.com/kenjyoung/MinAtar).
- [Procgen](https://github.com/openai/procgen).

## Documentation table of contents

- [Getting started](#getting-started)
- Tutorials 🚧
  - [Simple Behavior Cloning](/docs/tutorials/simple_bc.md)
- Experimental results 🚧
  - [A2C](/docs/results/a2c.md)
  - [REINFORCE](/docs/results/REINFORCE.md)
- Develop 🚧
  - [Development Setup](/docs/development_setup.md)
  - [Debug](#debug)
  - [Environment development](/docs/environment.md)
- Deploy 🚧
  - [Tunnel unsing ngrok](/docs/deployment/tunnel_using_ngrok.md)
- [Changelog](/CHANGELOG.md)
- [Contributors guide](/CONTRIBUTING.md)
- [Community code of conduct](/CODE_OF_CONDUCT.md)
- [Acknowledgments](#acknowledgements)

## Getting started

1. Clone this repository
2. Install [Python 3.9](https://www.python.org/)
3. Depending on your specific machine, you might also need to following dependencies:

   - `swig`, which is required for the Box2d gym environments, it can be installed using `apt-get install swig` on ubuntu or `brew install swig` on macOS
   - `python3-opencv`, which is required on ubuntu systems, it can be installed using `apt-get install python3-opencv`

4. Create and activate a virtual environment by runnning

   ```console
   $ python -m venv .venv
   $ source .venv/bin/activate
   ```

5. Install the python dependencies by running
   ```console
   $ pip install -r requirements.txt
   ```
6. In another terminal, launch a mlflow server on port 3000 by running
   ```console
   $ source .venv/bin/activate
   $ python -m simple_mlflow
   ```
7. Start the default Cogment Verse run using `python -m main`
8. Open Chrome (other web browser might work but haven't tested) and navigate to http://localhost:8080/
9. Play the game!

That's the basic setup for Cogment Verse, you are now ready to train AI agents.

### Configuration

Cogment Verse relies on [hydra](https://hydra.cc) for configuration. This enables easy configuration and composition of configuration directly from yaml files and the command line.

The configuration files are located in the `config` directory, with defaults defined in `config/config.yaml`.

Here are a few examples:

- Launch a Simple Behavior Cloning run with the [Mountain Car Gym environment](https://www.gymlibrary.ml/environments/classic_control/mountain_car/) (which is the default environment)
  ```console
  $ python -m main +experiment=simple_bc/mountain_car
  ```
- Launch a Simple Behavior Cloning run with the [Lunar Lander Gym environment](https://www.gymlibrary.ml/environments/box2d/lunar_lander/)
  ```console
  $ python -m main +experiment=simple_bc/mountain_car services/environment=lunar_lander
  ```
- Launch and play a single trial of the Lunar Lander Gym environment with continuous controls
  ```console
  $ python -m main services/environment=lunar_lander_continuous
  ```
- Launch an A2C training run with the [Cartpole Gym environment](https://www.gymlibrary.ml/environments/classic_control/cartpole/)

  ```console
  $ python -m main +experiment=simple_a2c/cartpole
  ```

  This one is completely _headless_ (training doens't involve interaction with a human player). It will take a little while to run, you can monitor the progress using mlflow at <http://localhost:3000>

- Launch an DQN self training run with the [Connect Four Petting Zoo environment](https://www.pettingzoo.ml/classic/connect_four)

  ```console
  $ python -m main +experiment=simple_dqn/connect_four
  ```

  The same experiment can be launched with a ratio of human-in-the-loop training trials (that are playable on in the web client)

  ```console
  $ python -m main +experiment=simple_dqn/connect_four +run.hill_training_trials_ratio=0.05
  ```

## Isaac gym

If you want to use Isaac Gym, use python3.8 (not python3.9)

1. download the zip file from [NVIDIA webpage](https://developer.nvidia.com/isaac-gym)
   , unzip and copy the `isaacgym` folder to this repo.
2. clone [IsaacGymEnvs](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs) and copy the
   folder inside the `isaacgym` folder
3. comment out line-32 in `isaacgym/IsaacGymEnvs/isaacgymenvs/__init__.py`
4. (Assuming you already installed requirements.txt), run `pip install -r isaac_requirements.txt`.
5. nvidia-smi` to check that you have NVIDIA drivers and proper cuda installations.
6. (Assuming you already have mlflow running in a different terminal), Run `python -m main services/environment=ant`

## List of publications and submissions using Cogment and/or Cogment Verse

- Analyzing and Overcoming Degradation in Warm-Start Off-Policy Reinforcement Learning [code](https://github.com/benwex93/cogment-verse)
- Multi-Teacher Curriculum Design for Sparse Reward Environments [code](https://github.com/kharyal/cogment-verse/)

(please open a pull request to add missing entries)

## Docker

To run cogment verse from docker containers:

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

   To specify the orchestrator's web endpoint port, set the environment variables `ORCHESTRATOR_WEB_PORT`

   ```console
   $ ORCHESTRATOR_WEB_PORT=9000 docker compose run --service-ports cogment_verse
   ```

3. Open Chrome (other web browser might work but haven't tested) and navigate to http://localhost:8080

### Troubleshooting

On M1/M2 Macs you'll need to force Docker to use the `linux/amd64` platform as a few dependencies are not availabe for `linux/arm64`. The environment variable `DOCKER_DEFAULT_PLATFORM` needs to be set to `linux/amd64`, e.g:

```console
$ DOCKER_DEFAULT_PLATFORM=linux/amd64 docker compose build
```
