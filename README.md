# Cogment Verse

[![Apache 2 License](https://img.shields.io/badge/license-Apache%202-green?style=flat-square)](./LICENSE) [![Changelog](https://img.shields.io/badge/-Changelog%20-blueviolet?style=flat-square)](./CHANGELOG.md)

[Cogment](https://cogment.ai) is an innovative open source AI platform designed to leverage the advent of AI to benefit humankind through human-AI collaboration developed by [AI Redefined](https://ai-r.com). Cogment enables AI researchers and engineers to build, train and operate AI agents in simulated or real environments shared with humans. For the full user documentation visit <https://docs.cogment.ai>

🚧 This repository is under construction, it propose a library of environments and agent implementations to get started with Human In the Loop Learning (HILL) and Reinforcement Learning (RL) with Cogment in minutes. Cogment Verse is designed both for practionners discovering the field as well as for experienced researchers or engineers as an framework to develop and benchmark new approaches.

Cogment verse includes environments from:

- [OpenAI Gym](https://gym.openai.com),
- [Petting Zoo](https://www.pettingzoo.ml).
- [MinAtar](https://github.com/kenjyoung/MinAtar).

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

1. Install [cogment](https://docs.cogment.ai/introduction/installation/) (⚠️ version >= 2.2.0 is required)
2. Clone this repository
3. Install parallel
4. `./run.sh build`
5. `./run.sh`
6. In a different terminal, start the trials with `./run.sh client start <run-name>`.
Different run names can be found in `run_params.yaml`

### Copy the shared definitions

After a fresh close or whenever either the `cogment.yaml` or any protobuf file in the root directory is changed, you need to copy those changes to the different services source directories. This is achieved with the following.

```console
cogment run copy
```

#### Run monitoring

You can monitor ongoing run using [mlflow](https://mlflow.org). By default a local instance of mlflow is started by cogment-verse and is accessible at <http://localhost:3000>.

#### Human player

Some of the availabe run involve a human player, for example `benchmark_lander_hill` enables a human player to momentarily take control of the lunar lander to help the AI agents during the training process.

Then start the run

```console
RUN_PARAMS=benchmark_lander_hill cogment run start_run
```

Access the playing interface by navigating to <http://localhost:8080>

#### The **Play** run

The **`play`** is a run that is used to test any agent in an environment. The run is started by

```console
RUN_PARAMS=play cogment run start_run
```

It can be configured with the following parameters (to change in `run_params.yaml`):

```yaml
play:
  implementation: play
  config:
    class_name: data_pb2.PlayRunConfig
    # Set to true to have the ability to observe the run in the web client
    observer: true
    # Number of trials to run
    trial_count: 10
    environment:
      # Reference one of the environment specs defined at the top of `run_params.yaml`
      specs: *cartpole_specs
    actors:
    # Configure the players (only the first ones are used, up to the number of required players)
    ## follows the `cogment_verse.ActorParams` datastructure
    - name: agent_1
      actor_class: agent
      # Select the implementation to use
      implementation: random
      agent_config:
        # Define the agent config here
        ## follows the `cogment_verse.AgentConfig` datastructure
        ## Make sure that the selected model is compatible with the selected implementation
        model_id: compassionate_aryabhata_model
        model_version: -1
    - name: agent_2
      actor_class: agent
      implementation: random
```

## Debug

### Resources monitoring

Cogment verse comes with [prometheus](https://prometheus.io), in [`/prometheus`](/prometheus), and [grafana](https://grafana.com), in [`/grafana`](/grafana), services to facilitate the monitoring of the cluster.

When running with the default `cogment run start`, the grafana dashboard can be accessed at <http://localhost:3001>.

### Profiling

Steps

- Add viztracer to python requirements.txt
- Modify docker-compose override
  - Add a mount for the profile results json/html file
  - Change the entrypoint of the service `viztracer --output_file /output/results.html script.py`
- Rebuild and run jobs

## Acknowledgements

The subdirectories `/tf_agents/cogment_verse_tf_agents/third_party` and `/torch_agents/cogment_verse_torch_agents/third_party` contains code from third party sources

- `hive`: Taken from the Hive library by MILA/CRL
- `td3`: Taken form the [authors' implementation](https://github.com/sfujim/TD3)
