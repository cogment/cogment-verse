# Cogment Verse

[![Apache 2 License](https://img.shields.io/badge/license-Apache%202-green?style=flat-square)](./LICENSE) [![Changelog](https://img.shields.io/badge/-Changelog%20-blueviolet?style=flat-square)](./CHANGELOG.md)

> 🚧 A new major version of Cogment Verse is under develelopment in the [`next`](https://github.com/cogment/cogment-verse/tree/next). Not all the algorithms and environments are available yet but it is fully operational. Do not hesitate to test it! 
>
> Follow and discuss the development in this [Pull Request](https://github.com/cogment/cogment-verse/pull/71). 

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
- [Changelog](/CHANGELOG.md)
- [Contributors guide](/CONTRIBUTING.md)
- [Community code of conduct](/CODE_OF_CONDUCT.md)
- [Acknowledgments](#acknowledgements)

## Getting started

### Setup, Build and Run

1. Clone this repository
2. Install the following dependencies:
   - [Python 3.9](https://www.python.org/),
   - [Node.JS v14](https://nodejs.org/) or above and npm,
   - `parallel`, on ubuntu it is installable using `apt-get install parallel`, on mac it is available through `brew install parallel`,
   - `unrar`, on ubuntu it is installable using `apt-get install unrar`, on mac it is available through `brew install unrar`,
   - `swig`, (required to install box2d-py), on ubuntu it is installable using `apt-get install swig`, on mac it is available through `brew install swig`,
   - `virtualenv`, installable using `pip install virtualenv`.
3. `./run.sh build`
4. `./run.sh services_start`
5. In a different terminal, start the trials with `./run.sh client start <run-name>`.
   Different run names can be found in `run_params.yaml`
6. (Optional) To launch webclient, run `./run.sh web_client_start` in a different
   terminal. Open http://localhost:8000 to join or visualize trials

#### Run monitoring

You can monitor ongoing run using [mlflow](https://mlflow.org). By default a local instance of mlflow is started by cogment-verse and is accessible at <http://localhost:3000>.

#### Human player

Some of the availabe run involve a human player,
for example `benchmark_lander_hill` enables a human player
to momentarily take control of the lunar lander to help the
AI agents during the training process.

Then start the run

```console
./run.sh client start benchmark_lander_hill
```

Access the playing interface by launching a webclient with
`./run.sh web_client_start` and navigating to <http://localhost:8000>

#### **Play**

The `play` run implementation can be used to have any actor play in any environment. 3 example run parameters are provided:

**`headless_play`** instanciates any agents and start a number of trials.

```console
./run.sh client start headless_play
```

**`observe`** instanciates any agents and start a number of trials with a human observer through the webclient.

```console
./run.sh client start observe
```

**`play`** instanciates let a human player play in a supported environment.

```console
./run.sh client start play
```

They can be inspected and adapted to your needs in `run_params.yaml`:

## List of publications and submissions using Cogment and/or Cogment Verse

- Analyzing and Overcoming Degradation in Warm-Start Off-Policy Reinforcement Learning [code](https://github.com/benwex93/cogment-verse)
- Multi-Teacher Curriculum Design for Sparse Reward Environments [code](https://github.com/kharyal/cogment-verse/)

(please open a pull request to add missing entries)

## Acknowledgements

The subdirectories `/tf_agents/cogment_verse_tf_agents/third_party` and `/torch_agents/cogment_verse_torch_agents/third_party` contains code from third party sources

- `hive`: Taken from the [Hive library](https://github.com/chandar-lab/RLHive)
- `td3`: Taken form the [authors' implementation](https://github.com/sfujim/TD3)
