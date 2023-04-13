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

# pylint: disable=invalid-name

import copy
import io
import logging
import time

import cogment
import numpy as np
import torch
import torch.nn.functional as F
from gym.spaces import Box, utils
from torch import nn

from cogment_verse import Model, TorchReplayBuffer
from cogment_verse.specs import PLAYER_ACTOR_CLASS, AgentConfig, EnvironmentConfig, EnvironmentSpecs, cog_settings

torch.multiprocessing.set_sharing_strategy("file_system")

log = logging.getLogger(__name__)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class TD3Model(Model):
    def __init__(
        self,
        model_id,
        environment_implementation,
        num_input,
        num_output,
        max_action,
        time_steps,
        expl_noise,
        random_steps,
        dtype=torch.float,
        version_number=0,
    ):
        super().__init__(model_id, version_number)
        self._dtype = dtype
        self._environment_implementation = environment_implementation
        self._num_input = num_input
        self._num_output = num_output
        self.max_action = max_action
        self.time_steps = time_steps
        self.expl_noise = expl_noise
        self.random_steps = random_steps

        self.actor = Actor(state_dim=num_input, action_dim=num_output, max_action=max_action)
        self.actor_target = copy.deepcopy(self.actor)

        self.critic = Critic(state_dim=num_input, action_dim=num_output)
        self.critic_target = copy.deepcopy(self.critic)

        # version user data
        self.epoch_idx = 0
        self.total_samples = 0

    def get_model_user_data(self):
        return {
            "environment_implementation": self._environment_implementation,
            "num_input": self._num_input,
            "num_output": self._num_output,
            "max_action": self.max_action,
            "expl_noise": self.expl_noise,
            "random_steps": self.random_steps,
            "epoch_idx": self.epoch_idx,
            "total_samples": self.total_samples,
        }

    @staticmethod
    def serialize_model(model):
        stream = io.BytesIO()
        torch.save(
            (
                model.actor.state_dict(),
                model.actor_target.state_dict(),
                model.critic.state_dict(),
                model.critic_target.state_dict(),
                model.time_steps,
                model.get_model_user_data(),
            ),
            stream,
        )
        return stream.getvalue()

    @classmethod
    def deserialize_model(cls, serialized_model, model_id, version_number):
        stream = io.BytesIO(serialized_model)
        (
            actor_state_dict,
            actor_target_state_dict,
            critic_state_dict,
            critic_target_state_dict,
            time_steps,
            model_user_data,
        ) = torch.load(stream)

        model = cls(
            model_id=model_id,
            version_number=version_number,
            environment_implementation=model_user_data["environment_implementation"],
            num_input=int(model_user_data["num_input"]),
            num_output=int(model_user_data["num_output"]),
            max_action=float(model_user_data["max_action"]),
            expl_noise=float(model_user_data["expl_noise"]),
            random_steps=int(model_user_data["random_steps"]),
            time_steps=0,
        )
        model.actor.load_state_dict(actor_state_dict)
        model.actor_target.load_state_dict(actor_target_state_dict)
        model.critic.load_state_dict(critic_state_dict)
        model.critic_target.load_state_dict(critic_target_state_dict)
        model.time_steps = time_steps
        model.epoch_idx = model_user_data["epoch_idx"]
        model.total_samples = model_user_data["total_samples"]

        return model


class TD3Actor:
    def __init__(self, _cfg):
        self._dtype = torch.float

    def get_actor_classes(self):
        return [PLAYER_ACTOR_CLASS]

    async def impl(self, actor_session):
        actor_session.start()

        config = actor_session.config

        environment_specs = EnvironmentSpecs.deserialize(config.environment_specs)
        observation_space = environment_specs.get_observation_space()
        action_space = environment_specs.get_action_space(seed=config.seed)

        assert isinstance(action_space.gym_space, Box)

        serialized_model = await actor_session.model_registry.retrieve_model(config.model_id, config.model_version)
        model = TD3Model.deserialize_model(serialized_model, config.model_id, config.model_version)

        async for event in actor_session.all_events():
            if event.observation and event.type == cogment.EventType.ACTIVE:
                observation = observation_space.deserialize(event.observation.observation)

                if model.time_steps < model.random_steps:
                    action = action_space.sample()
                else:
                    observation = torch.tensor(observation.value, dtype=self._dtype)
                    action = model.actor(observation)
                    action = action.cpu().detach().numpy()
                    # adding exploration noise
                    action = action + np.random.normal(0, model.max_action * model.expl_noise, size=2)

                    action = action_space.create(value=action)

                actor_session.do_action(action_space.serialize(action))


class TD3Training:
    default_cfg = {
        "seed": 10,
        "num_trials": 10000,
        "num_parallel_trials": 1,
        "discount": 0.99,
        "tau": 0.005,
        "policy_noise": 0.2,
        "noise_clip": 0.5,
        "policy_freq": 2,
        "expl_noise": 0.1,
        "random_steps": 25000,
    }

    def __init__(self, environment_specs, cfg):
        super().__init__()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._dtype = torch.float
        self._environment_specs = environment_specs
        self._cfg = cfg

    async def sample_producer_impl(self, sample_producer_session):
        player_actor_params = sample_producer_session.trial_info.parameters.actors[0]

        player_actor_name = player_actor_params.name
        player_environment_specs = EnvironmentSpecs.deserialize(player_actor_params.config.environment_specs)
        player_observation_space = player_environment_specs.get_observation_space()
        player_action_space = player_environment_specs.get_action_space()

        observation = None
        action = None
        reward = None

        total_reward = 0
        async for sample in sample_producer_session.all_trial_samples():
            actor_sample = sample.actors_data[player_actor_name]
            if actor_sample.observation is None:
                # This can happen when there is several "end-of-trial" samples
                continue

            next_observation = torch.tensor(
                player_observation_space.deserialize(actor_sample.observation).flat_value, dtype=self._dtype
            )

            if observation is not None:
                # It's not the first sample, let's check if it is the last
                done = sample.trial_state == cogment.TrialState.ENDED
                sample_producer_session.produce_sample(
                    (
                        observation,
                        next_observation,
                        action,
                        reward,
                        torch.ones(1, dtype=torch.int8) if done else torch.zeros(1, dtype=torch.int8),
                        total_reward,
                    )
                )
                if done:
                    break

            observation = next_observation
            action = torch.tensor(player_action_space.deserialize(actor_sample.action).value, dtype=self._dtype)
            reward = torch.tensor(actor_sample.reward if actor_sample.reward is not None else 0, dtype=self._dtype)
            total_reward += reward.item()

    async def impl(self, run_session):
        # Initializing a model
        model_id = f"{run_session.run_id}_model"

        assert self._environment_specs.num_players == 1
        assert isinstance(self._environment_specs.get_action_space().gym_space, Box)

        model = TD3Model(
            model_id,
            environment_implementation=self._environment_specs.implementation,
            num_input=utils.flatdim(self._environment_specs.get_observation_space().gym_space),
            num_output=utils.flatdim(self._environment_specs.get_action_space().gym_space),
            max_action=float(self._environment_specs.get_action_space().gym_space.high[0]),
            expl_noise=float(self._cfg.expl_noise),
            random_steps=int(self._cfg.random_steps),
            time_steps=0,
            dtype=self._dtype,
        )

        serialized_model = TD3Model.serialize_model(model)
        iteration_info = await run_session.model_registry.publish_model(
            name=model_id,
            model=serialized_model,
        )

        run_session.log_params(
            self._cfg,
            model_id=model_id,
            environment_implementation=self._environment_specs.implementation,
        )

        # Configure the optimizer over the two models
        actor_optimizer = torch.optim.Adam(model.actor.parameters(), lr=3e-4)
        critic_optimizer = torch.optim.Adam(model.critic.parameters(), lr=3e-4)

        replay_buffer = TorchReplayBuffer(
            capacity=self._cfg.buffer_size,
            observation_shape=(utils.flatdim(self._environment_specs.get_observation_space().gym_space),),
            observation_dtype=self._dtype,
            action_shape=(utils.flatdim(self._environment_specs.get_action_space().gym_space),),
            action_dtype=self._dtype,  # check
            reward_dtype=self._dtype,
            seed=self._cfg.seed,
        )

        start_time = time.time()
        total_reward_cum = 0

        for (step_idx, _trial_id, trial_idx, sample,) in run_session.start_and_await_trials(
            trials_id_and_params=[
                (
                    f"{run_session.run_id}_{trial_idx}",
                    cogment.TrialParameters(
                        cog_settings,
                        environment_name="env",
                        environment_implementation=self._environment_specs.implementation,
                        environment_config=EnvironmentConfig(
                            run_id=run_session.run_id,
                            render=False,
                            seed=self._cfg.seed + trial_idx,
                        ),
                        actors=[
                            cogment.ActorParameters(
                                cog_settings,
                                name="player",
                                class_name=PLAYER_ACTOR_CLASS,
                                implementation="actors.td3.TD3Actor",
                                config=AgentConfig(
                                    run_id=run_session.run_id,
                                    seed=self._cfg.seed + trial_idx,
                                    model_id=model_id,
                                    model_version=-1,
                                    model_update_frequency=self._cfg.policy_freq,
                                    environment_specs=self._environment_specs.serialize(),
                                ),
                            )
                        ],
                    ),
                )
                for trial_idx in range(self._cfg.num_trials)
            ],
            sample_producer_impl=self.sample_producer_impl,
            num_parallel_trials=self._cfg.num_parallel_trials,
        ):
            (observation, next_observation, action, reward, done, total_reward) = sample
            replay_buffer.add(
                observation=observation, next_observation=next_observation, action=action, reward=reward, done=done
            )

            trial_done = done.item() == 1
            if trial_done:
                run_session.log_metrics(trial_idx=trial_idx, total_reward=total_reward)
                total_reward_cum += total_reward
                if (trial_idx + 1) % 100 == 0:
                    total_reward_avg = total_reward_cum / 100
                    run_session.log_metrics(total_reward_avg=total_reward_avg)
                    total_reward_cum = 0
                    log.info(
                        f"[TD3/{run_session.run_id}] trial #{trial_idx + 1}/{self._cfg.num_trials} done (average total reward = {total_reward_avg})."
                    )

            if step_idx > model.random_steps and replay_buffer.size() > self._cfg.batch_size:
                data = replay_buffer.sample(self._cfg.batch_size)
                with torch.no_grad():
                    # Select action according to policy and add clipped noise
                    noise = (torch.randn_like(data.action) * self._cfg.policy_noise).clamp(
                        -self._cfg.noise_clip, self._cfg.noise_clip
                    )
                    next_action = (model.actor_target(data.next_observation) + noise).clamp(
                        -model.max_action, model.max_action
                    )

                    # Compute the target Q value
                    target_Q1, target_Q2 = model.critic_target(data.next_observation, next_action)
                    target_Q = torch.min(target_Q1, target_Q2)
                    target_Q = (
                        torch.unsqueeze(data.reward, dim=1)
                        + torch.unsqueeze((1 - data.done.flatten()), dim=1) * self._cfg.discount * target_Q
                    )

                    # Get current Q estimates
                current_Q1, current_Q2 = model.critic(data.observation, data.action)

                # Compute critic loss
                critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

                # Optimize the critic
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()

                if step_idx % self._cfg.policy_freq == 0:
                    # Compute actor losse
                    actor_loss = -model.critic.Q1(data.observation, model.actor(data.observation)).mean()

                    # Optimize the actor
                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    # Update the frozen target models
                    for param, target_param in zip(model.critic.parameters(), model.critic_target.parameters()):
                        target_param.data.copy_(self._cfg.tau * param.data + (1 - self._cfg.tau) * target_param.data)

                    for param, target_param in zip(model.actor.parameters(), model.actor_target.parameters()):
                        target_param.data.copy_(self._cfg.tau * param.data + (1 - self._cfg.tau) * target_param.data)

                    run_session.log_metrics(loss=actor_loss.item())

                # Update the version info
                model.total_samples += data.size()

                if step_idx % 100 == 0:
                    run_session.log_metrics(
                        batch_avg_reward=data.reward.mean().item(),
                    )

            model.time_steps += 1
            serialized_model = TD3Model.serialize_model(model)
            iteration_info = await run_session.model_registry.publish_model(
                name=model_id,
                model=serialized_model,
            )

            if step_idx % 100 == 0:
                end_time = time.time()
                steps_per_seconds = 100 / (end_time - start_time)
                start_time = end_time
                run_session.log_metrics(
                    model_version_number=iteration_info.iteration,
                    steps_per_seconds=steps_per_seconds,
                )

        serialized_model = TD3Model.serialize_model(model)
        iteration_info = await run_session.model_registry.store_model(
            name=model_id,
            model=serialized_model,
        )
