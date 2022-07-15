# Copyright 2022 AI Redefined Inc. <dev+cogment@ai-r.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import cogment
import torch

from cogment_verse.specs import (
    AgentConfig,
    cog_settings,
    EnvironmentConfig,
    flatten,
    flattened_dimensions,
    PLAYER_ACTOR_CLASS,
    PlayerAction,
    SpaceValue,
)
from cogment_verse import Model


log = logging.getLogger(__name__)


class SimpleA2CModel(Model):
    def __init__(
        self,
        model_id,
        environment_implementation,
        num_input,
        num_output,
        actor_network_num_hidden_nodes=64,
        critic_network_num_hidden_nodes=64,
        dtype=torch.float,
        version_number=0,
    ):
        super().__init__(model_id, version_number)
        self._dtype = dtype
        self._environment_implementation = environment_implementation
        self._num_input = num_input
        self._num_output = num_output
        self._actor_network_num_hidden_nodes = actor_network_num_hidden_nodes
        self._critic_network_num_hidden_nodes = critic_network_num_hidden_nodes

        self.actor_network = torch.nn.Sequential(
            torch.nn.Linear(self._num_input, self._actor_network_num_hidden_nodes, dtype=self._dtype),
            torch.nn.Tanh(),
            torch.nn.Linear(
                self._actor_network_num_hidden_nodes, self._actor_network_num_hidden_nodes, dtype=self._dtype
            ),
            torch.nn.Tanh(),
            torch.nn.Linear(self._actor_network_num_hidden_nodes, self._num_output, dtype=self._dtype),
        )

        self.critic_network = torch.nn.Sequential(
            torch.nn.Linear(self._num_input, self._critic_network_num_hidden_nodes, dtype=self._dtype),
            torch.nn.Tanh(),
            torch.nn.Linear(
                self._critic_network_num_hidden_nodes, self._critic_network_num_hidden_nodes, dtype=self._dtype
            ),
            torch.nn.Tanh(),
            torch.nn.Linear(self._critic_network_num_hidden_nodes, 1, dtype=self._dtype),
        )

        # version user data
        self.epoch_idx = 0
        self.total_samples = 0

    def get_model_user_data(self):
        return {
            "environment_implementation": self._environment_implementation,
            "num_input": self._num_input,
            "num_output": self._num_output,
            "actor_network_num_hidden_nodes": self._actor_network_num_hidden_nodes,
            "critic_network_num_hidden_nodes": self._critic_network_num_hidden_nodes,
        }

    def save(self, model_data_f):
        torch.save((self.actor_network.state_dict(), self.critic_network.state_dict()), model_data_f)

        return {"epoch_idx": self.epoch_idx, "total_samples": self.total_samples}

    @classmethod
    def load(cls, model_id, version_number, model_user_data, version_user_data, model_data_f):
        # Create the model instance
        model = SimpleA2CModel(
            model_id=model_id,
            version_number=version_number,
            environment_implementation=model_user_data["environment_implementation"],
            num_input=int(model_user_data["num_input"]),
            num_output=int(model_user_data["num_output"]),
            actor_network_num_hidden_nodes=int(model_user_data["actor_network_num_hidden_nodes"]),
            critic_network_num_hidden_nodes=int(model_user_data["critic_network_num_hidden_nodes"]),
        )

        # Load the saved states
        (actor_network_state_dict, critic_network_state_dict) = torch.load(model_data_f)
        model.actor_network.load_state_dict(actor_network_state_dict)
        model.critic_network.load_state_dict(critic_network_state_dict)

        # Load version data
        model.epoch_idx = version_user_data["epoch_idx"]
        model.total_samples = version_user_data["total_samples"]
        return model


class SimpleA2CActor:
    def __init__(self, _cfg):
        self._dtype = torch.float

    def get_actor_classes(self):
        return [PLAYER_ACTOR_CLASS]

    async def impl(self, actor_session):
        actor_session.start()

        config = actor_session.config

        assert config.environment_specs.num_players == 1
        assert len(config.environment_specs.action_space.properties) == 1
        assert config.environment_specs.action_space.properties[0].WhichOneof("type") == "discrete"

        observation_space = config.environment_specs.observation_space

        model, _, _ = await actor_session.model_registry.retrieve_version(
            SimpleA2CModel, config.model_id, config.model_version
        )
        model.actor_network.eval()
        model.critic_network.eval()

        async for event in actor_session.all_events():
            if event.observation and event.type == cogment.EventType.ACTIVE:
                obs_tensor = torch.tensor(
                    flatten(observation_space, event.observation.observation.value), dtype=self._dtype
                )
                probs = torch.softmax(model.actor_network(obs_tensor), dim=-1)
                discrete_action_tensor = torch.distributions.Categorical(probs).sample()
                action_value = SpaceValue(properties=[SpaceValue.PropertyValue(discrete=discrete_action_tensor.item())])
                actor_session.do_action(PlayerAction(value=action_value))


class SimpleA2CTraining:
    default_cfg = {
        "seed": 10,
        "num_epochs": 500,
        "epoch_num_trials": 10,
        "num_parallel_trials": 8,
        "discount_factor": 0.99,
        "entropy_loss_coef": 0.05,
        "value_loss_coef": 0.5,
        "action_loss_coef": 1.0,
        "learning_rate": 0.01,
        "actor_network": {"num_hidden_nodes": 64},
        "critic_network": {"num_hidden_nodes": 64},
    }

    def __init__(self, environment_specs, cfg):
        super().__init__()
        self._dtype = torch.float
        self._environment_specs = environment_specs
        self._cfg = cfg

    async def trial_sample_sequences_producer_impl(self, sample_producer_session):
        observation = []
        action = []
        reward = []
        done = []

        player_actor_params = sample_producer_session.trial_info.parameters.actors[0]

        player_actor_name = player_actor_params.name
        player_observation_space = player_actor_params.config.environment_specs.observation_space

        async for sample in sample_producer_session.all_trial_samples():
            if sample.trial_state == cogment.TrialState.ENDED:
                # This sample includes the last observation and no action
                # The last sample was the last useful one
                done[-1] = torch.ones(1, dtype=self._dtype)
                break

            actor_sample = sample.actors_data[player_actor_name]
            observation.append(
                torch.tensor(flatten(player_observation_space, actor_sample.observation.value), dtype=self._dtype)
            )
            action_value = actor_sample.action.value
            action.append(
                torch.tensor(
                    action_value.properties[0].discrete if len(action_value.properties) > 0 else 0, dtype=self._dtype
                )
            )
            reward.append(
                torch.tensor(actor_sample.reward if actor_sample.reward is not None else 0, dtype=self._dtype)
            )
            done.append(torch.zeros(1, dtype=self._dtype))

        # Keeping the samples grouped by trial by emitting only one grouped sample at the end of the trial
        sample_producer_session.produce_sample((observation, action, reward, done))

    async def impl(self, run_session):
        # Initializing a model
        model_id = f"{run_session.run_id}_model"

        assert self._environment_specs.num_players == 1
        assert len(self._environment_specs.action_space.properties) == 1
        assert self._environment_specs.action_space.properties[0].WhichOneof("type") == "discrete"

        model = SimpleA2CModel(
            model_id,
            environment_implementation=self._environment_specs.implementation,
            num_input=flattened_dimensions(self._environment_specs.observation_space),
            num_output=flattened_dimensions(self._environment_specs.action_space),
            actor_network_num_hidden_nodes=self._cfg.actor_network.num_hidden_nodes,
            critic_network_num_hidden_nodes=self._cfg.critic_network.num_hidden_nodes,
            dtype=self._dtype,
        )
        _model_info, version_info = await run_session.model_registry.publish_initial_version(model)

        run_session.log_params(
            self._cfg,
            model_id=model_id,
            environment_implementation=self._environment_specs.implementation,
            actor_network_num_hidden_nodes=self._cfg.actor_network.num_hidden_nodes,
            critic_network_num_hidden_nodes=self._cfg.critic_network.num_hidden_nodes,
        )

        # Configure the optimizer over the two models
        optimizer = torch.optim.Adam(
            torch.nn.Sequential(model.actor_network, model.critic_network).parameters(),
            lr=self._cfg.learning_rate,
        )

        total_samples = 0
        for epoch_idx in range(self._cfg.num_epochs):
            # Rollout a bunch of trials
            observation = []
            action = []
            reward = []
            done = []
            for (_step_idx, _trial_id, _trial_idx, sample,) in run_session.start_and_await_trials(
                trials_id_and_params=[
                    (
                        f"{run_session.run_id}_{epoch_idx}_{trial_idx}",
                        cogment.TrialParameters(
                            cog_settings,
                            environment_name="env",
                            environment_implementation=self._environment_specs.implementation,
                            environment_config=EnvironmentConfig(
                                run_id=run_session.run_id,
                                render=False,
                                seed=self._cfg.seed + trial_idx + epoch_idx * self._cfg.epoch_num_trials,
                            ),
                            actors=[
                                cogment.ActorParameters(
                                    cog_settings,
                                    name="player",
                                    class_name=PLAYER_ACTOR_CLASS,
                                    implementation="actors.simple_a2c.SimpleA2CActor",
                                    config=AgentConfig(
                                        run_id=run_session.run_id,
                                        model_id=model_id,
                                        model_version=version_info["version_number"],
                                        environment_specs=self._environment_specs,
                                    ),
                                )
                            ],
                        ),
                    )
                    for trial_idx in range(self._cfg.epoch_num_trials)
                ],
                sample_producer_impl=self.trial_sample_sequences_producer_impl,
                num_parallel_trials=self._cfg.num_parallel_trials,
            ):
                (trial_observation, trial_action, trial_reward, trial_done) = sample
                observation.extend(trial_observation)
                action.extend(trial_action)
                reward.extend(trial_reward)
                done.extend(trial_done)

                run_session.log_metrics(total_reward=sum(r.item() for r in trial_reward))

            if len(observation) == 0:
                log.warning(
                    f"[SimpleA2CTraining/{run_session.run_id}] epoch #{epoch_idx + 1}/{self._cfg.num_epochs} finished without generating any sample (every trial ended at the first tick), skipping training."
                )
                continue

            total_samples += len(observation)

            # Convert the accumulated observation/action/reward over the epoch to tensors
            observation = torch.vstack(observation)
            action = torch.vstack(action)
            reward = torch.vstack(reward)
            done = torch.vstack(done)

            # Compute the action probability and the critic value over the epoch
            action_probs = torch.softmax(model.actor_network(observation), dim=-1)
            critic = model.critic_network(observation).squeeze(-1)

            # Compute the estimated advantage over the epoch
            advantage = (
                reward[1:] + self._cfg.discount_factor * critic[1:].detach() * (1.0 - done[1:].float()) - critic[:-1]
            )

            # Compute critic loss
            value_loss = advantage.pow(2).mean()

            # Compute entropy loss
            entropy_loss = torch.distributions.Categorical(action_probs).entropy().mean()

            # Compute A2C loss
            action_log_probs = torch.gather(action_probs, -1, action.long()).log()
            action_loss = -(action_log_probs[:-1] * advantage.detach()).mean()

            # Compute the complete loss
            loss = (
                -self._cfg.entropy_loss_coef * entropy_loss
                + self._cfg.value_loss_coef * value_loss
                + self._cfg.action_loss_coef * action_loss
            )

            # Backprop!
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Publish the newly trained version
            last_epoch = (epoch_idx + 1) == self._cfg.num_epochs

            model.epoch_idx = epoch_idx
            model.total_samples = total_samples

            version_info = await run_session.model_registry.publish_version(model, archived=last_epoch)
            run_session.log_metrics(
                model_version_number=version_info["version_number"],
                epoch_idx=epoch_idx,
                entropy_loss=entropy_loss.item(),
                value_loss=value_loss.item(),
                action_loss=action_loss.item(),
                loss=loss.item(),
                total_samples=total_samples,
            )
            log.info(
                f"[SimpleA2CTraining/{run_session.run_id}] epoch #{epoch_idx + 1}/{self._cfg.num_epochs} finished ({total_samples} samples seen)"
            )
