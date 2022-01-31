# Copyright 2021 AI Redefined Inc. <dev+cogment@ai-r.com>
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
from collections import namedtuple

import cogment
import torch
from cogment.api.common_pb2 import TrialState
from cogment_verse import AgentAdapter, MlflowExperimentTracker
from cogment_verse_torch_agents.utils.tensors import cog_action_from_tensor, tensor_from_cog_action, tensor_from_cog_obs
from data_pb2 import (
    AgentConfig,
    ActorParams,
    EnvironmentConfig,
    EnvironmentParams,
    EnvironmentSpecs,
    MLPNetworkConfig,
    SimpleA2CTrainingConfig,
    SimpleA2CTrainingRunConfig,
    TrialConfig,
)

log = logging.getLogger(__name__)

SimpleA2CModel = namedtuple("SimpleA2CModel", ["model_id", "version_number", "actor_network", "critic_network"])

# pylint: disable=arguments-differ


class SimpleA2CAgentAdapter(AgentAdapter):
    def __init__(self):
        super().__init__()
        self._dtype = torch.float

    def _create(
        self,
        model_id,
        environment_specs,
        actor_network_hidden_size=64,
        critic_network_hidden_size=64,
        **kwargs,
    ):
        model = SimpleA2CModel(
            model_id=model_id,
            version_number=1,
            actor_network=torch.nn.Sequential(
                torch.nn.Linear(environment_specs.num_input, actor_network_hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(actor_network_hidden_size, actor_network_hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(actor_network_hidden_size, environment_specs.num_action),
            ).to(self._dtype),
            critic_network=torch.nn.Sequential(
                torch.nn.Linear(environment_specs.num_input, critic_network_hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(critic_network_hidden_size, critic_network_hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(critic_network_hidden_size, 1),
            ).to(self._dtype),
        )

        model_user_data = {
            "environment_implementation": environment_specs.implementation,
            "num_input": environment_specs.num_input,
            "num_action": environment_specs.num_action,
        }

        return model, model_user_data

    def _load(
        self,
        model_id,
        version_number,
        model_user_data,
        version_user_data,
        model_data_f,
        environment_specs,
        **kwargs,
    ):
        (actor_network, critic_network) = torch.load(model_data_f)
        assert model_user_data["environment_implementation"] == environment_specs.implementation
        assert isinstance(actor_network, torch.nn.Sequential)
        assert isinstance(critic_network, torch.nn.Sequential)
        return SimpleA2CModel(
            model_id=model_id, version_number=version_number, actor_network=actor_network, critic_network=critic_network
        )

    def _save(self, model, model_user_data, model_data_f, environment_specs, epoch_idx=-1, total_samples=0, **kwargs):
        assert model_user_data["environment_implementation"] == environment_specs.implementation
        assert isinstance(model, SimpleA2CModel)
        torch.save((model.actor_network, model.critic_network), model_data_f)
        return {"epoch_idx": epoch_idx, "total_samples": total_samples}

    def _create_actor_implementations(self):
        async def impl(actor_session):
            actor_session.start()

            config = actor_session.config

            model, _, _ = await self.retrieve_version(
                config.model_id, config.model_version, environment_specs=config.environment_specs
            )

            async for event in actor_session.event_loop():
                if event.observation and event.type == cogment.EventType.ACTIVE:
                    obs = tensor_from_cog_obs(event.observation.snapshot, dtype=self._dtype)
                    scores = model.actor_network(obs)
                    probs = torch.softmax(scores, dim=-1)
                    action = torch.distributions.Categorical(probs).sample()
                    actor_session.do_action(cog_action_from_tensor(action))

        return {
            "simple_a2c": (impl, ["agent"]),
        }

    def _create_run_implementations(self):
        async def sample_producer_impl(run_sample_producer_session):
            assert run_sample_producer_session.count_actors() == 1
            observation = []
            action = []
            reward = []
            done = []
            async for sample in run_sample_producer_session.get_all_samples():
                if sample.get_trial_state() == TrialState.ENDED:
                    # This sample includes the last observation and no action
                    # The last sample was the last useful one
                    done[-1] = torch.ones(1, dtype=self._dtype)
                    break
                observation.append(tensor_from_cog_obs(sample.get_actor_observation(0), dtype=self._dtype))
                action.append(tensor_from_cog_action(sample.get_actor_action(0)))
                reward.append(torch.tensor(sample.get_actor_reward(0), dtype=self._dtype))
                done.append(torch.zeros(1, dtype=self._dtype))

            # Keeping the samples grouped by trial by emitting only one grouped sample at the end of the trial
            run_sample_producer_session.produce_training_sample((observation, action, reward, done))

        async def run_impl(run_session):
            xp_tracker = MlflowExperimentTracker(run_session.params_name, run_session.run_id)

            # Initializing a model
            model_id = f"{run_session.run_id}_model"

            config = run_session.config
            assert config.environment.specs.num_players == 1

            model, _ = await self.create_and_publish_initial_version(
                model_id,
                environment_specs=config.environment.specs,
                actor_network_hidden_size=config.actor_network.hidden_size,
                critic_network_hidden_size=config.critic_network.hidden_size,
            )
            model_version_number = 1

            xp_tracker.log_params(
                config.training,
                config.environment.config,
                environment_implementation=config.environment.specs.implementation,
                actor_network_hidden_size=config.actor_network.hidden_size,
                critic_network_hidden_size=config.critic_network.hidden_size,
            )

            # Configure the optimizer over the two models
            optimizer = torch.optim.Adam(
                torch.nn.Sequential(model.actor_network, model.critic_network).parameters(),
                lr=config.training.learning_rate,
            )

            total_samples = 0
            for epoch_idx in range(config.training.epoch_count):
                # Rollout a bunch of trials
                observation = []
                action = []
                reward = []
                done = []
                epoch_last_step_idx = None
                epoch_last_step_timestamp = None
                async for (
                    step_idx,
                    step_timestamp,
                    _trial_id,
                    _tick_id,
                    sample,
                ) in run_session.start_trials_and_wait_for_termination(
                    trial_configs=[
                        TrialConfig(
                            run_id=run_session.run_id,
                            environment=config.environment,
                            actors=[
                                ActorParams(
                                    name="agent_1",
                                    actor_class="agent",
                                    implementation="simple_a2c",
                                    agent_config=AgentConfig(
                                        run_id=run_session.run_id,
                                        model_id=model_id,
                                        model_version=model_version_number,
                                        environment_specs=config.environment.specs,
                                    ),
                                )
                            ],
                        )
                        for trial_idx in range(config.training.epoch_trial_count)
                    ],
                    max_parallel_trials=config.training.max_parallel_trials,
                ):
                    (trial_observation, trial_action, trial_reward, trial_done) = sample
                    observation.extend(trial_observation)
                    action.extend(trial_action)
                    reward.extend(trial_reward)
                    done.extend(trial_done)
                    epoch_last_step_idx = step_idx
                    epoch_last_step_timestamp = step_timestamp

                    xp_tracker.log_metrics(step_timestamp, step_idx, total_reward=sum([r.item() for r in trial_reward]))

                if len(observation) == 0:
                    log.warning(
                        f"[{run_session.params_name}/{run_session.run_id}] epoch #{epoch_idx + 1}/{config.training.epoch_count} finished without generating any sample (every trial ended at the first tick), skipping training."
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
                    reward[1:] + config.training.discount_factor * critic[1:].detach() * (1 - done[1:]) - critic[:-1]
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
                    -config.training.entropy_coef * entropy_loss
                    + config.training.value_loss_coef * value_loss
                    + config.training.action_loss_coef * action_loss
                )

                # Backprop!
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Publish the newly trained version
                last_epoch = epoch_idx + 1 == config.training.epoch_count
                version_info = await self.publish_version(
                    model_id,
                    model,
                    archived=last_epoch,
                    epoch_idx=epoch_idx,
                    total_samples=total_samples,
                    environment_specs=config.environment.specs,
                )
                model_version_number = version_info["version_number"]
                xp_tracker.log_metrics(
                    epoch_last_step_timestamp,
                    epoch_last_step_idx,
                    model_version_number=model_version_number,
                    epoch_idx=epoch_idx,
                    entropy_loss=entropy_loss.item(),
                    value_loss=value_loss.item(),
                    action_loss=action_loss.item(),
                    loss=loss.item(),
                    total_samples=total_samples,
                )
                log.info(
                    f"[{run_session.params_name}/{run_session.run_id}] epoch #{epoch_idx + 1}/{config.training.epoch_count} finished ({total_samples} samples seen)"
                )

        return {
            "simple_a2c_training": (
                sample_producer_impl,
                run_impl,
                SimpleA2CTrainingRunConfig(
                    environment=EnvironmentParams(
                        specs=EnvironmentSpecs(
                            implementation="gym/CartPole-v0",
                            num_input=4,
                            num_action=2,
                        ),
                        config=EnvironmentConfig(seed=12, framestack=1),
                    ),
                    training=SimpleA2CTrainingConfig(
                        epoch_count=100,
                        epoch_trial_count=15,
                        max_parallel_trials=8,
                        discount_factor=0.95,
                        entropy_coef=0.01,
                        value_loss_coef=0.5,
                        action_loss_coef=1.0,
                        learning_rate=0.01,
                    ),
                    actor_network=MLPNetworkConfig(hidden_size=64),
                    critic_network=MLPNetworkConfig(hidden_size=64),
                ),
            )
        }
