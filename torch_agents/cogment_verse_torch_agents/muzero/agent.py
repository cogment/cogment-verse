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
    MuZeroTrainingRunConfig,
    MuZeroTrainingConfig,
    AgentAction,
    TrialConfig,
    TrialActor,
    EnvConfig,
    ActorConfig,
    MLPNetworkConfig,
)

from cogment_verse import AgentAdapter
from cogment_verse import MlflowExperimentTracker

from cogment.api.common_pb2 import TrialState
import cogment

import logging
import torch
import numpy as np
import copy

from collections import namedtuple

log = logging.getLogger(__name__)

from .networks import MuZero, reward_transform, reward_tansform_inverse, Distributional, DynamicsAdapter, resnet, mlp
from .replay_buffer import ConcurrentTrialReplayBuffer, EpisodeBatch, TrialReplayBuffer
from cogment_verse_torch_agents.wrapper import proto_array_from_np_array

import itertools

# pylint: disable=arguments-differ


class LinearScheduleWithWarmup:
    """Defines a linear schedule between two values over some number of steps.

    If updated more than the defined number of steps, the schedule stays at the
    end value.
    """

    def __init__(self, init_value, end_value, total_steps, warmup_steps):
        """
        Args:
            init_value (Union[int, float]): starting value for schedule.
            end_value (Union[int, float]): end value for schedule.
            steps (int): Number of steps for schedule. Should be positive.
        """
        self._warmup_steps = max(warmup_steps, 0)
        self._total_steps = max(total_steps, self._warmup_steps)
        self._init_value = init_value
        self._end_value = end_value
        self._current_step = 0
        self._value = 0

    def get_value(self):
        return self._value

    def update(self):
        if self._current_step < self._warmup_steps:
            t = np.clip(self._current_step / (self._warmup_steps + 1), 0, 1)
            self._value = self._init_value * t
        else:
            t = np.clip((self._current_step - self._warmup_steps) / (self._total_steps - self._warmup_steps), 0, 1)
            self._value = self._init_value + t * (self._end_value - self._init_value)

        self._current_step += 1
        return self._value


class MuZeroAgent:
    """
    MuZero implementation
    """

    def __init__(self, *, obs_dim, act_dim, device, training_config: MuZeroTrainingConfig):
        self._obs_dim = obs_dim
        self._act_dim = act_dim
        self._params = training_config
        self._device = torch.device(device)
        self._make_networks()
        self._make_replay_buffer()

    def _make_replay_buffer(self):
        self._replay_buffer = TrialReplayBuffer(max_size=1000, discount_rate=0.99, bootstrap_steps=10)

    def _create_replay_buffer(self):
        # due to pickling issues for multiprocessing, we create the replay buffer lazily
        self._replay_buffer = None

    def consume_training_sample(self, state, action, reward, next_state, done, policy, value):
        state = state.unsqueeze(0)
        next_state = next_state.unsqueeze(0)
        self._replay_buffer.add_sample(state, action, reward, next_state, done, policy, value)

    def sample_training_batch(self, batch_size):
        return self._replay_buffer.sample(self._params.rollout_length, batch_size)

    def _make_networks(self):
        value_distribution = Distributional(
            self._params.vmin,
            self._params.vmax,
            self._params.hidden_dim,
            self._params.vbins,
            reward_transform,
            reward_tansform_inverse,
        )

        reward_distribution = Distributional(
            self._params.rmin,
            self._params.rmax,
            self._params.hidden_dim,
            self._params.rbins,
            reward_transform,
            reward_tansform_inverse,
        )

        representation = resnet(
            self._obs_dim,
            self._params.hidden_dim,
            self._params.representation_dim,
            self._params.hidden_layers,
            # final_act=torch.nn.BatchNorm1d(self._params.num_latent"]),  # normalize for input to subsequent networks
        )

        dynamics = DynamicsAdapter(
            resnet(
                self._params.representation_dim + self._act_dim,
                self._params.hidden_dim,
                self._params.hidden_dim,
                self._params.hidden_layers - 1,
                final_act=torch.nn.LeakyReLU(),
            ),
            self._act_dim,
            self._params.hidden_dim,
            self._params.representation_dim,
            reward_dist=reward_distribution,
        )
        policy = resnet(
            self._params.representation_dim,
            self._params.hidden_dim,
            self._act_dim,
            self._params.hidden_layers,
            final_act=torch.nn.Softmax(dim=1),
        )
        value = resnet(
            self._params.representation_dim,
            self._params.hidden_dim,
            self._params.hidden_dim,
            self._params.hidden_layers - 1,
            final_act=value_distribution,
        )
        projector = mlp(
            self._params.representation_dim,
            self._params.projector_hidden_dim,
            self._params.projector_dim,
            hidden_layers=self._params.projector_hidden_layers,
        )
        # todo: check number of hidden layers used in predictor (same as projector??)
        predictor = mlp(
            self._params.projector_dim, self._params.projector_hidden_dim, self._params.projector_dim, hidden_layers=1
        )

        self._muzero = MuZero(
            representation, dynamics, policy, value, projector, predictor, reward_distribution, value_distribution
        ).to(self._device)

        self._optimizer = torch.optim.AdamW(
            self._muzero.parameters(),
            lr=1e-3,
            weight_decay=self._params.weight_decay,
        )

    def act(self, observation):
        self._muzero.eval()
        obs = observation.float().to(self._device)
        action, policy, value = self._muzero.act(
            obs, self._params.exploration_epsilon, self._params.exploration_alpha, self._params.mcts_temperature
        )
        policy = policy.cpu().numpy()
        value = value.cpu().numpy().item()
        return action, policy, value

    def learn(self, batch, update_schedule=True):
        self._muzero.train()

        # todo: use schedule
        lr = 1e-3
        for grp in self._optimizer.param_groups:
            grp["lr"] = lr

        batch_tensors = []
        for i, tensor in enumerate(batch):
            if isinstance(tensor, np.ndarray):
                tensor = torch.from_numpy(tensor)
            batch_tensors.append(tensor.to(self._device))
        batch = EpisodeBatch(*batch_tensors)

        #
        priority = batch.priority[:, 0].view(-1)
        # todo: update priorities!!
        # self._replay_buffer.update_priorities(list(zip(episode, step, priority)))
        importance_weight = 1 / (priority + 1e-6) / self._replay_buffer.size()

        target_value = torch.clamp(batch.target_value, self._params.vmin, self._params.vmax)
        target_reward = torch.clamp(batch.rewards, self._params.rmin, self._params.rmax)

        priority, info = self._muzero.train_step(
            self._optimizer,
            batch.state,
            batch.action,
            target_reward,
            batch.next_state,
            batch.target_policy,
            target_value,
            importance_weight,
        )

        for key, val in info.items():
            if isinstance(val, torch.Tensor):
                info[key] = val.detach().cpu().numpy().item()

        # Return loss
        return priority, info

    @torch.no_grad()
    def reanalyze_replay_buffer(self, fraction):
        self.eval()
        total_steps = np.sum(self._replay_buffer._priority)
        current_steps = 0
        ctx = mp.get_context("spawn")
        # import multiprocessing
        # ctx = multiprocessing.get_context("spawn")
        # nworkers = cpu_count()
        nworkers = 2
        input_queue = ctx.JoinableQueue(nworkers)
        output_queue = ctx.JoinableQueue(nworkers)
        end_process = ctx.Value(ctypes.c_bool, False)

        pool = [
            ctx.Process(
                target=reanalyze_queue,
                args=(
                    self._representation,
                    self._dynamics,
                    self._policy,
                    self._value,
                    input_queue,
                    output_queue,
                    self._params.mcts_count,
                    self._params.mcts_depth,
                    self._params.ucb_c1,
                    self._params.ucb_c2,
                    end_process,
                    self._device,
                ),
            )
            for _ in range(nworkers)
        ]
        for process in pool:
            process.start()

        consumed_steps = 0

        def consume_output():
            nonlocal consumed_steps
            try:
                sample = output_queue.get(timeout=1.0)
            except queue.Empty:
                return False

            episode_idx, step, policy, value = sample
            self._replay_buffer._episodes[episode_idx]._policy[step] = policy
            self._replay_buffer._episodes[episode_idx]._value[step] = value
            consumed_steps += 1
            output_queue.task_done()
            return True

        for episode_idx, episode in enumerate(self._replay_buffer._episodes):
            for step, state in enumerate(episode._states[:-1]):
                while True:
                    try:
                        input_queue.put((episode_idx, step, state), timeout=1.0)
                        break
                    except queue.Full:
                        while not consume_output():
                            pass

                        continue

                consume_output()

                # p, v = self.reanalyze(state)
                # episode._policy[step] = p
                # episode._value[step] = v
            current_steps += len(episode)
            if current_steps / total_steps > fraction:
                break

        while consumed_steps < current_steps:
            consume_output()

        end_process.value = True
        input_queue.join()
        output_queue.join()

        for process in pool:
            process.join()

    def _update_target(self):
        pass

    def save(self, f):
        torch.save(
            {
                "obs_dim": self._obs_dim,
                "act_dim": self._act_dim,
                "training_config": self._params,
                "muzero": self._muzero.state_dict(),
            },
            f,
        )

    @staticmethod
    def load(f, device):
        checkpoint = torch.load(f, map_location=device)
        muzero_state_dict = checkpoint.pop("muzero")
        agent = MuZeroAgent(device=device, **checkpoint)
        agent._muzero.load_state_dict(muzero_state_dict)
        return agent
