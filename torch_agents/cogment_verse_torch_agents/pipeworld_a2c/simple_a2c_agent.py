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

from data_pb2 import (
    SimpleA2CTrainingRunConfig,
    SimpleA2CTrainingConfig,
    AgentAction,
    TrialConfig,
    TrialActor,
    EnvConfig,
    ActorConfig,
    MLPNetworkConfig,
)

from cogment_verse import AgentAdapter
from cogment_verse import MlflowExperimentTracker

from cogment_verse_torch_agents.simple_a2c.simple_a2c_agent import SimpleA2CAgentAdapter

from cogment.api.common_pb2 import TrialState
import cogment

import logging
import torch
import numpy as np

from collections import namedtuple

log = logging.getLogger(__name__)

SimpleA2CModel = namedtuple("SimpleA2CModel", ["model_id", "version_number", "actor_network", "critic_network"])

# pylint: disable=arguments-differ


def heuristic_policy(obs, n=30):
    obs = obs[:-1].view(n, 3)
    fail_prob = obs[:, 0]
    fail_cost = obs[:, 1]
    is_fail = obs[:, 2]

    policy = torch.zeros(2 * n + 1, dtype=torch.float32)
    policy[:n] = fail_prob * fail_cost
    policy[:n] += is_fail
    policy[n:] = 0.01

    policy = torch.clamp(policy, 0, 10)

    return torch.nn.functional.softmax(policy, dim=0)


EPSILON = 1.0


class PipeWorldA2CAgentAdapter(SimpleA2CAgentAdapter):
    def _create_actor_implementations(self):
        async def impl(actor_session):
            global EPSILON
            actor_session.start()

            config = actor_session.config

            model, _ = await self.retrieve_version(config.model_id, config.model_version)

            async for event in actor_session.event_loop():
                if event.observation and event.type == cogment.EventType.ACTIVE:
                    obs = self.tensor_from_cog_obs(event.observation.snapshot)
                    scores = model.actor_network(obs)
                    probs = torch.softmax(scores, dim=-1)
                    action = torch.distributions.Categorical(probs).sample()
                    # heuristic
                    # if np.random.rand() < EPSILON:
                    #    policy = heuristic_policy(obs)
                    #    action = torch.distributions.Categorical(policy).sample()

                    actor_session.do_action(self.cog_action_from_tensor(action))

        return {
            "pipeworld_a2c": (impl, ["agent"]),
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
                observation.append(self.tensor_from_cog_obs(sample.get_actor_observation(0)))
                action.append(self.tensor_from_cog_action(sample.get_actor_action(0)))
                reward.append(torch.tensor(sample.get_actor_reward(0), dtype=self._dtype))
                done.append(torch.zeros(1, dtype=self._dtype))

            # Keeping the samples grouped by trial by emitting only one grouped sample at the end of the trial
            run_sample_producer_session.produce_training_sample((observation, action, reward, done))

        async def run_impl(run_session):
            global EPSILON

            xp_tracker = MlflowExperimentTracker(run_session.params_name, run_session.run_id)
            model_id = f"{run_session.run_id}_model"

            config = run_session.config
            assert config.environment.player_count == 1

            model, _ = await self.create_and_publish_initial_version(
                model_id,
                observation_size=config.actor.num_input,
                action_count=config.actor.num_action,
                actor_network_hidden_size=config.actor_network.hidden_size,
                critic_network_hidden_size=config.critic_network.hidden_size,
            )
            model_version_number = 1

            xp_tracker.log_params(
                config.training,
                config.environment,
                actor_network_hidden_size=config.actor_network.hidden_size,
                critic_network_hidden_size=config.critic_network.hidden_size,
            )

            # Configure the optimizer over the two models
            optimizer = torch.optim.Adam(
                torch.nn.Sequential(model.actor_network, model.critic_network).parameters(),
                lr=config.training.learning_rate,
            )

            total_samples = 0
            for epoch in range(config.training.epoch_count):
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
                            environment_config=config.environment,
                            actors=[
                                TrialActor(
                                    name="agent_1",
                                    actor_class="agent",
                                    implementation="pipeworld_a2c",
                                    config=ActorConfig(
                                        model_id=model_id,
                                        model_version=model_version_number,
                                        num_input=config.actor.num_input,
                                        num_action=config.actor.num_action,
                                        env_type=config.environment.env_type,
                                        env_name=config.environment.env_name,
                                    ),
                                )
                            ],
                        )
                        for trial_ids in range(config.training.epoch_trial_count)
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

                # debug/testing: add a term for behavioural cloning of a heuristic rule
                # print("ACTION_LOG_PROBS", action_probs.shape)
                # print("OBS", observation.shape)
                target_policy = torch.stack([heuristic_policy(obs) for obs in observation], dim=0)
                heuristic_loss = torch.mean(torch.sum(-action_probs * torch.log(target_policy), dim=1))

                action_loss = (1 - EPSILON) * action_loss + EPSILON * heuristic_loss

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

                EPSILON *= 0.999

                # Publish the newly trained version
                version_info = await self.publish_version(model_id, model)
                model_version_number = version_info["version_number"]
                xp_tracker.log_metrics(
                    epoch_last_step_timestamp,
                    epoch_last_step_idx,
                    model_version_number=model_version_number,
                    epoch=epoch,
                    entropy_loss=entropy_loss.item(),
                    value_loss=value_loss.item(),
                    action_loss=action_loss.item(),
                    loss=loss.item(),
                    total_samples=total_samples,
                    epsilon=EPSILON,
                )
                log.info(
                    f"[{run_session.params_name}/{run_session.run_id}] epoch #{epoch} finished ({total_samples} samples seen)"
                )

        return {
            "pipeworld_a2c_training": (
                sample_producer_impl,
                run_impl,
                SimpleA2CTrainingRunConfig(
                    environment=EnvConfig(
                        seed=12, env_type="gym", env_name="CartPole-v0", player_count=1, framestack=1
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
