# Simple interactive behavioral cloning (BC)

## About

This is an interactive implementation of behavioral cloning (BC) that can be thought of as a simplified version of the DAGGER algorithm, intended to demonstrate the human-in-the-loop learning within Cogment Verse.

The overall architecture involves two actors

1. An AI agent using a simple policy network whose action space is determined by the environment.
2. A human _teacher_ having the same action space extended by a single "take-no-action" `NO-OP` action.

The two actors cooperate during training by the following rule: if the teacher takes the `NO-OP` action, then we sample an action from the AI agent's policy, otherwise we use the action specified by the human. In this way, the human provides feedback to the agent in the form of implicit approval (by taking `NO-OP`) or by demonstration by overriding the agent's actions. By running trials in this way, we generate a sequence of pairs `(observation, perfomed_action)`. We train the AI agent by using the performed actions as target labels with a categorical cross-entropy loss.

We train the agent online as trials are running and each trial uses the latest snapshot of the agent's policy network. As the agent learns to mimic the human demonstrations, the human's task becomes easier as fewer interventions are required to correct the agent's behavior.

## Step-by-step implementation guide

> We don't provide (yet) a true step by step implementation guide but more a description of existing four implementation steps.

### Step 1 - Create a basic actor implementation

> Implementation for this step is available at [`/actors/tutorial/tutorial_1.py`](/actors/tutorial/tutorial_1.py)

In this step we create a dedicated **Actor implementation** to implement behavior cloning. **Actors** are a lightweight formalism to help implement agents and training algorithms together.

In this case, it defines two Cogment implementations: an actor called `simple_bc` that is using the trained model, in this case a simple policy network, to take actions and a **runner** called `simple_bc_training` that is orchestrating the training of the model.

#### Actor Implementation

The `SimpleBCActor` class provides a minimal example of an actor implementation together with their corresponding training algorithms. A few methods must be implemented.

- `get_actor_classes`: Returns the list of actor classes supported to this actor. By default, the `PLAYER_ACTOR_CLASS` is returned.
- `impl`: the actor implementation. See next section.

#### First actor implementation, doing random action

In Cogment terminology, an **actor implementation** is a function that takes an `actor_session` from a running trial and performs actions for each event in the agent's event loop for the trial, i.e. a function that performs an action for each observation received from the environment. In this first step, the actor implementation does random actions.

#### First run implementation, starting trials with a human teacher

A **run implementation** consists of a function that launches trials and consumes the resulting data that is generated. Examples include training and evaluation.

The run implementation is paired with a **sample producer implementation** that is tasked with taking the raw event stream from a trial (e.g. `(state, action, reward)` tuples) and emits samples that can be directly consumed by the training/evaluation algorithm (e.g. `(state, action, reward, next_state)` transitions). In this first step the sample producer implementation does little, only logging a message each time a sample is received. [Step 2](#step-2-producing-samples) will go further.

In the first step, the run implementation is pretty minimal. It sets up an experiment metrics tracker using MLFlow, then defines a function to create trial configuration and then starts a bunch of trials. No samples are retrieved yet because sample producer implementation doesn't produce any.

#### Running everything

Make sure you are using step 1 version of the actor implementation by ensuring the following:

1. In `config/experiment/simple_bc/mountain_car.yaml` file, `class_name` is set to `actors.tutorial.tutorial_1.SimpleBCTraining`.
2. In `config/services/actor/simple_bc.yaml` file, `class_name` is set to `actors.tutorial.tutorial_1.SimpleBCActor`.
3. Port 3000 is not in use.

First, launch a mlflow server on port 3000 using:

```
python -m simple_mlflow
```

In a new terminal, launch a simple behavior cloning run with the mountain car gym environment using:

```
python -m main +experiment=simple_bc/mountain_car
```

Open Chrome (other web browser might work but haven't tested) and navigate to http://localhost:8080/. At this step the agent will take random actions, as a human you can take over to play the game. In the console, the run implementation will log every time a sample is retrieved.

### Step 2 - Producing samples
C
> Implementation for this step is available at [`/actors/tutorial/tutorial_2.py`](/actors/tutorial/tutorial_2.py)
>
> Changes from the previous step are surrounded by `############ TUTORIAL STEP 2 ############` comments

In this second step, we properly define the **sample producer implementation**.

At the top of the file, a few helpers are imported in order to convert actions and observations between the Cogment Verse format and PyTorch tensors.

In the event loop of the **sample producer implementation** those helpers are used to convert the observation, the agent action and the teacher action. If the teacher performed an action, this is the one used in the produced sample.

Samples are produced as tuples consisting: a flag identifying if the sample is a demonstration (coming from the teacher), the observation as a tensor, the action as a tensor.

Make sure you are using step 2 version of the actor implementation by editing the "default" export in `config/experiment/simple_bc/mountain_car.yaml` and then launch a run as described in the previous step.

Nothing should change in the web browser but received samples should be logged. Notice that when the human takes over samples are logged with the demonstration flag to `True`.

### Step 3 - Defining a policy model

> Implementation for this step is available at [`/actors/tutorial/tutorial_3.py`](/actors/tutorial/tutorial_3.py)
>
> Changes from the previous step are surrounded by `############ TUTORIAL STEP 3 ############` comments

In this third step, we introduce an actual model: it is initialized in the **run implementation** and used in the **actor implementation**. To achieve that, the `SimpleBCModel` class is created. It extends the base Model class and the methods `get_model_user_data`, `serialize_model`, and `deserialize_model` are fully implemented.

In `SimpleBCModel`, we include the configuration structure for multi-layer perceptrons (MLPs).

`serialize_model` and `deserialize_model` are implemented using PyTorch's load and save function. They are used to convert the pytorch model to a stream of bytes and vice versa. These methods will be used before publishing or storing a model to the Model Registry and afer retrieving a serialized model.

The actor implementation uses `SimpleBCModel.retrieve_model` to retrieve the model having the configured `model_id` and `iteration`. These two model parameters are now specified as a part of the actor params in the runner's implementation. The iteration number is defined as `-1`, which means the latest available iteration. Also in the actor implementation, we fetch the agent's action space from the actor_session using the method `get_action_space` and use the helper function `create` to build an `Action` object from the output of the policy network.

Make sure you are using step 3 version of the actor implementation by editing the "default" export in `config/experiment/simple_bc/mountain_car.yaml` and then launch a run as described in the previous step.

Nothing should change in the web browser - the agent is still performing random actions, but it's now random actions computed by a neural network.

### Step 4 - Training

> Implementation for this step is available at [`/actors/tutorial/tutorial_4.py`](/actors/tutorial/tutorial_4.py)
>
> Changes from the previous step are surrounded by `############ TUTORIAL STEP 4 ############` comments

This fourth step is about actually training the policy, aside from some import at the top, the changes are located in the runner's implementation and should be pretty straightforward for someone familiar with supervised learning with PyTorch.

One thing to notice is the way we deal with publishing new iteration of the model. This part of the code is only executed every 100 training steps, this is a tradeoff between the reactivity of the training and limiting the amount of data exchanged over the network and the time spent serializing and deserializing models.

Make sure you are using step 4 version of the actor implementation by editing the "default" export in `config/experiment/simple_bc/mountain_car.yaml` and then launch a run as described in the previous step.

The agent is now learning, with a few demonstrations it should start to clone the behavior of the human player.

You can view the logged metrics (e.g. training loss) by opening `localhost:3000`.
