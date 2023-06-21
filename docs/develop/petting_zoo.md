# Petting Zoo

___

# Atari Games

### Additional installation steps

Because of some issues with peting zoo versions and the pip version resolver the following packages need be installed separately after `pip install -r requirements.txt`:

```console
pip install SuperSuit==3.7.0
```

### macOS

This step is only required for Apple silicon-based computers (e.g., M1&2 chips). Clone [Multi-Agent-ALE](https://github.com/Farama-Foundation/Multi-Agent-ALE) repository

```console
git clone https://github.com/Farama-Foundation/Multi-Agent-ALE.git
cd Multi-Agent-ALE
pip install .
```

### License Activation

Activate [AutoROM](https://github.com/Farama-Foundation/AutoROM) license relating to Atari games.

```sh
AutoROM --accept-license
```
___

# MPE Environments


## Simple Tag

In the run configurations, it is necessary to specify all actors included in the experiment. They are listed under `run.players`. The actor naming follows the names given to agents and adversaries by the Simple Tag environment. Actors can be added/modified/removed from the example configuration files listed in the examples below. The `implementation` can be changed to any other implementation. The spec_type parameter is necessary for all non-human actors. Its value is either `mpe_agent` for "good" actors or `mpe_adversary` for adversaries.

When adding or removing actors, it is important to update the `services.environment` config parameters `num_good` and `num_adversaries` for the correct actor counts.

### Human Actor
An example is provided in the configuration file `config/experiment/pz_simple_tag/play`. The first step is to add its configuration under `run.players`. The spec_type parameter should be ommitted, because it is necessary for cogment to use the default spec_type value for human actors. For example:
  ```console
  players:
    - name: agent_0
      implementation: client
  ```
  However, the second step is to tell the environment configuration which observation space to use for the human actor. Add the parameter `web_actor_spec` under `services.environment`. its value should be `mpe_agent` if it is an agent and `mpe_adversaries` if it is an adversary.

Here are a few examples:
- Observe episodes with the [Simple Tag Petting Zoo environment](https://pettingzoo.farama.org/environments/mpe/simple_tag/)
  ```console
  python -m main +experiment=pz_simple_tag/observe
  ```
- Play episodes as an agent with the [Simple Tag Petting Zoo environment](https://pettingzoo.farama.org/environments/mpe/simple_tag/)
  ```console
  python -m main +experiment=pz_simple_tag/play
  ```

