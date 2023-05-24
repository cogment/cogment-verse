# Cogment Verse

[![Apache 2 License](https://img.shields.io/badge/license-Apache%202-green?style=flat-square)](./LICENSE) [![Changelog](https://img.shields.io/badge/-Changelog%20-blueviolet?style=flat-square)](./CHANGELOG.md)

Cogment Verse is a SDK helping researchers and developers in the fields of human-in-the-loop learning (HILL) and multi-agent reinforcement learning (MARL) train and validate their agents at scale. Cogment Verse instantiates the open-source [Cogment](https://cogment.ai) platform for environments following the OpenAI Gym mold, making it easy to get started.

Simply clone the repo and start training.

## Documentation table of contents

- [Getting started](#getting-started)
- Tutorials
  - [Simple Behavioral Cloning](/docs/tutorials/simple_bc.md)
- Develop
  - [Development Setup](/docs/develop/development_setup.md)
  - [Docker](/docs/develop/docker.md)
  - [Isaac Gym](/docs/develop/isaac_gym.md)
- Deploy
  - [Tunnel unsing ngrok](/docs/deploy/tunnel_using_ngrok.md)
- Experimental results 🚧
  - [A2C](/docs/results/a2c.md)
  - [REINFORCE](/docs/results/REINFORCE.md)
- [Changelog](/CHANGELOG.md)
- [Contributors guide](/CONTRIBUTING.md)
- [Community code of conduct](/CODE_OF_CONDUCT.md)

## Getting started

> The following will show you how to setup Cogment Verse locally, it is possible to use a Docker based setup instead. Instructions for this can be found [here](/docs/develop/docker.md)

1. Clone this repository
2. Install [Python 3.9](https://www.python.org/)
3. Depending on your specific machine, you might also need to following dependencies:

   - `swig`, which is required for the Box2d gym environments, it can be installed using `apt-get install swig` on ubuntu or `brew install swig` on macOS
   - `python3-opencv`, which is required on ubuntu systems, it can be installed using `apt-get install python3-opencv`
   - `libosmesa6-dev` and `patchelf` are required to run the environment libraries using `mujoco`. They can be installed using `apt-get install libosmesa6-dev patchelf`.

4. Create and activate a virtual environment

   ```console
   $ python -m venv .venv
   $ source .venv/bin/activate
   ```

5. Install the python dependencies. For petting zoo's Atari games, [additional installation](/docs/develop/development_setup.md#petting-zoo-atari-games) is required after this step
   ```console
   $ pip install -r requirements.txt
   $ pip install SuperSuit==3.7.0
   ```
6. In another terminal, launch a mlflow server on port 3000
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

- Petting Zoo's [Atari Pong Environment](https://pettingzoo.farama.org/environments/atari/pong/)

  Example #1: Self-training

  ```console
  $ python -m main +experiment=ppo_atari_pz/pong_pz
  ```

  Example #2: Training with human's demonstrations

  ```console
  $ python -m main +experiment=ppo_atari_pz/hill_pong_pz
  ```

  Example #3: Training with human's feedback

  ```console
  $ python -m main +experiment=ppo_atari_pz/hfb_pong_pz
  ```

  NOTE: Example 2&3 require users to open Chrome and navigate to http://localhost:8080 in order to provide either demonstrations or feedback.

## List of publications and submissions using Cogment and/or Cogment Verse

- Analyzing and Overcoming Degradation in Warm-Start Off-Policy Reinforcement Learning [code](https://github.com/benwex93/cogment-verse)
- Multi-Teacher Curriculum Design for Sparse Reward Environments [code](https://github.com/kharyal/cogment-verse/)

(please open a pull request to add missing entries)
