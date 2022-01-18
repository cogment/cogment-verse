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
    ActorConfig,
    ActorParams,
    AgentAction,
    EnvironmentConfig,
    EnvironmentParams,
    MLPNetworkConfig,
    SimpleBCTrainingConfig,
    SimpleBCTrainingRunConfig,
    TrialConfig,
)

from cogment_verse import AgentAdapter
from cogment_verse import MlflowExperimentTracker

from cogment.api.common_pb2 import TrialState
import cogment

import asyncio
import logging
import torch
import numpy as np
import copy

from collections import namedtuple

log = logging.getLogger(__name__)

SimpleBCModel = namedtuple("SimpleBCModel", ["model_id", "version_number", "policy_network"])

# pylint: disable=arguments-differ


class SimpleBCAgentAdapter(AgentAdapter):
    def __init__(self):
        super().__init__()
        self._dtype = torch.float

    def tensor_from_cog_obs(self, cog_obs, device=None):
        pb_array = cog_obs.vectorized
        np_array = np.frombuffer(pb_array.data, dtype=pb_array.dtype).reshape(*pb_array.shape)
        return torch.tensor(np_array, dtype=self._dtype, device=device)

    @staticmethod
    def tensor_from_cog_action(cog_action, device=None):
        return torch.tensor(cog_action.discrete_action, dtype=torch.long, device=device)

    @staticmethod
    def cog_action_from_tensor(tensor):
        return AgentAction(discrete_action=tensor.item())

    def _create(
        self,
        model_id,
        observation_size,
        action_count,
        policy_network_hidden_size=64,
        **kwargs,
    ):
        return SimpleBCModel(
            model_id=model_id,
            version_number=1,
            policy_network=torch.nn.Sequential(
                torch.nn.Linear(observation_size, policy_network_hidden_size),
                torch.nn.BatchNorm1d(policy_network_hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(policy_network_hidden_size, policy_network_hidden_size),
                torch.nn.BatchNorm1d(policy_network_hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(policy_network_hidden_size, action_count),
            ).to(self._dtype),
        )

    def _load(self, model_id, version_number, version_user_data, model_data_f):
        policy_network = torch.load(model_data_f)
        assert isinstance(policy_network, torch.nn.Sequential)
        return SimpleBCModel(model_id=model_id, version_number=version_number, policy_network=policy_network)

    def _save(self, model, model_data_f):
        assert isinstance(model, SimpleBCModel)
        torch.save(model.policy_network, model_data_f)
        return {}

    def _create_actor_implementations(self):
        async def impl(actor_session):
            actor_session.start()

            config = actor_session.config

            model, _ = await self.retrieve_version(config.model_id, config.model_version)
            event_loop = asyncio.get_running_loop()
            policy_network = copy.deepcopy(model.policy_network)
            policy_network.eval()

            @torch.no_grad()
            def act(event):
                obs = self.tensor_from_cog_obs(event.observation.snapshot)
                scores = policy_network(obs.view(1, -1))
                probs = torch.softmax(scores, dim=-1)
                action = torch.distributions.Categorical(probs).sample()
                return action

            async for event in actor_session.event_loop():
                if event.observation and event.type == cogment.EventType.ACTIVE:
                    action = await event_loop.run_in_executor(None, act, event)
                    actor_session.do_action(self.cog_action_from_tensor(action))

        return {
            "simple_bc": (impl, ["agent"]),
        }

    def _create_run_implementations(self):
        async def sample_producer_impl(run_sample_producer_session):
            assert run_sample_producer_session.count_actors() == 2
            observation = None
            action = None

            async for sample in run_sample_producer_session.get_all_samples():
                if sample.get_trial_state() == TrialState.ENDED:
                    break

                agent_action = sample.get_actor_action(0)
                teacher_action = sample.get_actor_action(1)

                # Check for teacher override.
                # Teacher action -1 corresponeds to teacher approval,
                # i.e. the teacher considers the action taken by the agent to be correct
                if teacher_action.discrete_action != -1:
                    action = teacher_action
                else:
                    action = agent_action

                action = self.tensor_from_cog_action(action)
                observation = self.tensor_from_cog_obs(sample.get_actor_observation(0))
                run_sample_producer_session.produce_training_sample((observation, action))

        async def run_impl(run_session):
            xp_tracker = MlflowExperimentTracker(run_session.params_name, run_session.run_id)

            # Initializing a model
            model_id = f"{run_session.run_id}_model"

            config = run_session.config
            assert config.environment.config.player_count == 1

            model, _ = await self.create_and_publish_initial_version(
                model_id,
                observation_size=config.actor.num_input,
                action_count=config.actor.num_action,
                policy_network_hidden_size=config.policy_network.hidden_size,
            )
            model_version_number = 1

            xp_tracker.log_params(
                config.training,
                config.environment,
                policy_network_hidden_size=config.policy_network.hidden_size,
            )

            # Configure the optimizer
            optimizer = torch.optim.Adam(
                model.policy_network.parameters(),
                lr=config.training.learning_rate,
            )

            agent_actor_config = ActorParams(
                name="agent_1",
                actor_class="agent",
                implementation="simple_bc",
                config=ActorConfig(
                    model_id=model_id,
                    model_version=model_version_number,
                    num_input=config.actor.num_input,
                    num_action=config.actor.num_action,
                    environment_implementation=config.environment.implementation,
                ),
            )

            teacher_actor_config = ActorParams(
                name="web_actor",
                actor_class="teacher_agent",
                implementation="client",
                config=ActorConfig(
                    model_id=model_id,
                    model_version=model_version_number,
                    num_input=config.actor.num_input,
                    num_action=config.actor.num_action,
                    environment_implementation=config.environment.implementation,
                ),
            )

            batch_size = config.training.batch_size
            total_samples = 0
            observations = []
            actions = []

            loss_fn = torch.nn.CrossEntropyLoss()

            # Rollout a bunch of trials
            def modify_seed(env_config, offset):
                env_config = copy.deepcopy(env_config)
                env_config.config.seed = env_config.config.seed + offset
                env_config.config.render = True
                env_config.config.render_width = 256
                return env_config

            def train_step():
                batch_idx = np.random.randint(0, len(observations), batch_size)
                batch_obs = torch.vstack([observations[i] for i in batch_idx])
                batch_act = torch.vstack([actions[i] for i in batch_idx]).view(-1)

                model.policy_network.train()
                pred_policy = model.policy_network(batch_obs)
                loss = loss_fn(pred_policy, batch_act)

                # Backprop!
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                return loss.item()

            event_loop = asyncio.get_running_loop()

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
                        environment=modify_seed(config.environment, trial_ids),
                        actors=[agent_actor_config, teacher_actor_config],
                    )
                    for trial_ids in range(config.training.trial_count)
                ],
                max_parallel_trials=config.training.max_parallel_trials,
            ):
                (observation, action) = sample
                observations.append(observation)
                actions.append(action)

                if len(observations) < 2 * batch_size:
                    continue

                loss = await event_loop.run_in_executor(None, train_step)

                # Publish the newly trained version
                if step_idx % 100 == 0:
                    version_info = await self.publish_version(model_id, model)
                    model_version_number = version_info["version_number"]

                    xp_tracker.log_metrics(
                        step_timestamp,
                        step_idx,
                        model_version_number=model_version_number,
                        loss=loss,
                        total_samples=total_samples,
                    )

        return {
            "simple_bc_training": (
                sample_producer_impl,
                run_impl,
                SimpleBCTrainingRunConfig(
                    environment=EnvironmentParams(
                        implementation="gym/LunarLander-v2",
                        config=EnvironmentConfig(seed=12, player_count=1, framestack=1),
                    ),
                    training=SimpleBCTrainingConfig(
                        trial_count=100,
                        max_parallel_trials=1,
                        discount_factor=0.95,
                        learning_rate=0.01,
                    ),
                    policy_network=MLPNetworkConfig(hidden_size=64),
                ),
            )
        }
