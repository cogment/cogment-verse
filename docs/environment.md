# Environment development

This is a practical guide for developers wanting to develop new environments, or integrate existing third-party environments, within cogment verse.

At the moment, all environment implementations are provided by a single service
with multiple endpoints. This is likely to change in the future and the documentation
below will need to be updated to reflect any changes in the architecture.

## Method 1: Add a new endpoint to the existing environment service

### Step 1 - Update dependencies

If your environment has any additional dependencies, you will need to update the [`Dockerfile`](/environment/Dockerfile) (for system dependencies) and [`pyproject.toml`](/environment/pyproject.toml) (for python dependencies) files to reflect this.
Additional details can be found [here](/docs/development_setup.md#dependencies).

### Step 2 - Write an adapter for your environment

The environment service expects a gym-like API provided by the [BaseEnv](/environment/cogment_verse_environment/base.py) class. Existing third-party environments can be
integrated using a small amount of glue code, see for example

- [Gym](https://github.com/openai/gym) -> [GymEnv](/environment/cogment_verse_environment/gym_env.py)
- [Minatar](https://github.com/kenjyoung/MinAtar) -> [MinAtarEnv](/environment/cogment_verse_environment/minatarenv.py)
- [PettingZoo](https://github.com/Farama-Foundation/PettingZoo) -> [PettingZooEnv](/environment/cogment_verse_environment/zoo_env.py)
- [Procgen](https://github.com/openai/procgen) -> [ProcGenEnv](/environment/cogment_verse_environment/procgen_env.py)

### Step 3 - Register your environment

Edit the [EnvironmentAdapter](/environment/cogment_verse_environment/environment_adapter.py) in the following locations

- Add your environment constructor to the list of `ENVIRONMENT_CONSTRUCTORS`
- Add the names of the provided environments to `self._environments` in the `EnvironmentAdapter` constructor
- Add the names of the environment endpoints to `COGMENT_VERSE_ENVIRONMENT_ENDPOINTS` in (/.env).

### Step 4 - Add unit tests for your environment

The [MockEnvironmentSession](/environment/tests/mock_environment_session.py) class
allows you to create and interact with your environment using the same API that
is exposed to other services. For examples of how to test environments, see
[test_atari](/environment/tests/test_atari.py) and [test_procgen](/environment/tests/test_procgen.py). The tests can be run via the command

```
cogment run test_environment
```

### Step 5 - Add a run configuration to train using your environment

Edit [`run_params`](/run_params.yaml) to create a run configuration using one of the
new environment endpoints that you have registered. The environment endpoint is
determined by the `environment_implementation` field of the run configuration.

## Method 2: Develop your own environment service

Todo!
