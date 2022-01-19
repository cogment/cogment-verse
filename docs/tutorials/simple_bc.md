# Simple interactive behavioral cloning (BC)

## About

This is a interactive implementation of behavioral cloning (BC) that can be thought of as a simplified version of the DAGGER algorithm, intended to demonstrate the human-in-the-loop learning within Cogment Verse.

The overall architecture involves two actors

1. An AI agent uing a simple a policy network whose action space is determined by the environment.
2. A human _teacher_ having the same action space extended by a single "take-no-action" `NO-OP` action.

The two actors cooperate during training by the following rule: if the teacher takes the `NO-OP` action, then we sample an action from the AI agent's policy, otherwise we use the action specified by the human. In this way, the human provides feedback to the agent in the form of implicit approval (by taking `NO-OP`) or by demonstration by overriding the agent's actions. Running trials in this way we generate a sequence of pairs `(state, perfomed_action)`. We train the AI agent by using the performed actions as target labels
with a categorical cross-entropy loss.

We train the agent online as trials are running and each trial uses the latest snapshot of the agent's policy network. As the agent learns to mimic the human demonstrations, the human's task becomes easier as fewer interventions are required to correct the agent's behavior.

## Step-by-step implementation guide

- Add `simple_bc` to the list of actor endpoints in `COGMENT_VERSE_ACTOR_ENDPOINTS` in `.env`
- Add `simple_bc_training` to the list of run endpoints in `COGMENT_VERSE_RUN_ENDPOINTS` in `.env`
- Add `SimpleBCTrainingConfig` and `SimpleBCTrainingRunConfig` to `data.proto`
- Add run configurations to `run_params.yaml`
- Implement agent adapter to `simple_bc_agent.py` (details below)
- Register actor and run implementations in `torch_agents/main.py`

## Implement the agent adapter

The `AgentAdapter` base class provides a simple way to implement new agents together with
their corresponding training algorithms. There are only a few methods that must be implemented:

- `_create`: This instantiates the PyTorch model to be used for the agent and training
- `_save` and `_load`: Serialization and deserialization to be used with the model registry
- `_create_actor_implementations`: Returns the list of actor implementations to be registered
- `_create_run_implementations`: Return the list of run implementations (e.g. training/evaluation regimens) to be registered

### Actor implementation

In Cogment/Cogverse terminology, an actor implementation is a function that takes an actor session from a running trial and performs actions for each event in the actor's event loop for the trial, i.e. a funcion that performs an action for each observation received from the environment. In the case of a policy-based actor, we simply sample from the policy corresponding to the given observation.

### Run implementation

A "run implementation" is a function that launches trials and consumes the resulting data that is generated. Examples include training and evaluation.

As part of the run implementation we must define a "sample producer implementation". This is a function that takes the raw event stream from a trial (e.g. `(state, action, reward)` tuples) and emits samples that can be directly consumed by the training/evaluation algorithm (e.g. `(state, action, reward, next_state)` transitions). In the case of simple BC, the logic for dealing with human demonstrations is contained in the sample producer, and we directly emit the training targets in the form `(state, target_action)`.

The run implementation itself contains the following main sections

- Create and register model
- Create trial configurations and launch trials
- Main training loop over events produced by the trials

For our simple BC, the event loop body is very simple. The produced samples are of the form `(state, target_action)` which can be added directly to the replay buffer. After adding the sample to the replay buffer, we sample a batch and traing using a standard categorical cross entropy.

Periodically (e.g. every 100 training steps) we publish the model snapshot and log metrics. It is important that the publication interval is not too long in order to ensure that the models used when launch trials are not too stale. We can optionally archive model snapshots for retrieval after the training run completes.

## Running the demos

First build and start up the services:

```
cogment run copy && cogment run build && cogment run start
```

Once the services are running, start the web client:

```
cogment run start_web_client
```

Once the web client builds and is running, open `localhost:8080`.

Now we can launch a training run. There are three configurations provided:

- simple_bc_cartpole
- simple_bc_mountaincar
- simple_bc_lander

Start a training run as follows

```
RUN_PARAMS=simple_bc_lander cogment run start_run
```

and open the web client to start interacting with the agent while it trains.

You can view the logged metrics (e.g. training loss) by opening `localhost:3000`.
