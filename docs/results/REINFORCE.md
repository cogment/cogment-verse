# REINFORCE

## About

REINFORCE is an on-policy Monte Carlo variant of a [policy gradient algorithm](https://proceedings.neurips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf). The agent uses its current policy to collect samples over an episode and then consumes the entire trajectory to update its paramters using the [policy gradient theorem](https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html#policy-gradient-theorem).

## Implementations

### Reinforce

Reinforce is `cogment-verse` minimal implementation of REINFORCE written using tensorflow. It is mainly designed as a entry point for people discovering the `cogment-verse` framework, as much it is lacking a lot of bells and whistles: e.g. it only supports a simple multilayer perceptron architecture (MLP) making it only suited for low dimensionality environments.

The full implementation can be found in [`tf_agents/cogment_verse_tf_agents/reinforce/`](/tf_agents/cogment_verse_tf_agents/reinforce/)

#### Cartpole

_Experiment ran on 2021-11-22 on the current `HEAD` version of the code_

The run params were the following:

```yaml
cartpole_REINFORCE:
  implementation: "reinforce_training"
  config:
    <<: *default_config
    agent_implementation: reinforce
    aggregate_by_actor: True
    min_replay_buffer_size: 100000
    max_parallel_trials: 1
```

This is a plot of the total trial reward against the number of trials with an exponential moving average over 50 trials.

![Training total reward for the Reinforce implementation](./REINFORCE.png)

