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

    def __init__(self, *, id, obs_dim, act_dim, device):
        # debug/testing
        self._params = dict(
            id=id,
            obs_dim=obs_dim,
            act_dim=act_dim,
            discount_rate=0.99,
            weight_decay=1e-3,
            device=device,
            value_bootstrap_steps=20,
            num_latent=128,
            num_hidden=256,
            num_hidden_layers=4,
            projector_hidden=128,
            projector_dim=64,
            mcts_depth=3,
            mcts_count=8,
            ucb_c1=1.25,
            ucb_c2=10000.0,
            batch_size=16,
            dirichlet_alpha=0.5,
            rollout_length=4,
            rmin=-100.0,
            rmax=100.0,
            vmin=-300.0,
            vmax=300.0,
            r_bins=128,
            v_bins=128,
        )

        self._device = torch.device(device)
        self._make_networks()
        self._replay_buffer = TrialReplayBuffer(max_size=1000, discount_rate=0.99, bootstrap_steps=10)

    def _create_replay_buffer(self):
        # due to pickling issues for multiprocessing, we create the replay buffer lazily
        self._replay_buffer = None

    def consume_training_sample(self, state, action, reward, next_state, done, policy, value):
        self._replay_buffer.add_sample(state, action, reward, next_state, done, policy, value)

    def sample_training_batch(self, batch_size):
        return self._replay_buffer.sample(self._params["rollout_length"], batch_size)

    def _make_networks(self):
        value_distribution = Distributional(
            self._params["vmin"],
            self._params["vmax"],
            self._params["num_hidden"],
            self._params["v_bins"],
            reward_transform,
            reward_tansform_inverse,
        )

        reward_distribution = Distributional(
            self._params["rmin"],
            self._params["rmax"],
            self._params["num_hidden"],
            self._params["r_bins"],
            reward_transform,
            reward_tansform_inverse,
        )

        representation = resnet(
            self._params["obs_dim"],
            self._params["num_hidden"],
            self._params["num_latent"],
            self._params["num_hidden_layers"],
            # final_act=torch.nn.BatchNorm1d(self._params["num_latent"]),  # normalize for input to subsequent networks
        )

        dynamics = DynamicsAdapter(
            resnet(
                self._params["num_latent"] + self._params["act_dim"],
                self._params["num_hidden"],
                self._params["num_hidden"],
                self._params["num_hidden_layers"] - 1,
                final_act=torch.nn.LeakyReLU(),
            ),
            self._params["act_dim"],
            self._params["num_hidden"],
            self._params["num_latent"],
            reward_dist=reward_distribution,
        )
        policy = resnet(
            self._params["num_latent"],
            self._params["num_hidden"],
            self._params["act_dim"],
            self._params["num_hidden_layers"],
            final_act=torch.nn.Softmax(dim=1),
        )
        value = resnet(
            self._params["num_latent"],
            self._params["num_hidden"],
            self._params["num_hidden"],
            self._params["num_hidden_layers"] - 1,
            final_act=value_distribution,
        )
        projector = mlp(self._params["num_latent"], self._params["projector_hidden"], self._params["projector_dim"])
        predictor = mlp(self._params["projector_dim"], self._params["projector_hidden"], self._params["projector_dim"])

        self._muzero = MuZero(
            representation, dynamics, policy, value, projector, predictor, reward_distribution, value_distribution
        ).to(self._device)

        self._optimizer = torch.optim.AdamW(
            self._muzero.parameters(),
            lr=1e-3,
            weight_decay=self._params["weight_decay"],
        )

    def act(self, cog_obs):
        self._muzero.eval()
        obs = torch.from_numpy(cog_obs.observation).float().unsqueeze(0).to(self._device)
        action, policy, value = self._muzero.act(observation)
        action = action.unsqueeze(0).cpu().numpy().item()
        policy = policy.unsqueeze(0).cpu().numpy()
        value = value.unsqueeze(0).cpu().numpy().item()
        cog_action = AgentAction(discrete_action=action, policy=proto_array_from_np_array(policy), value=value)
        return cog_action

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

        target_value = torch.clamp(batch.target_value, self._params["vmin"], self._params["vmax"])
        target_reward = torch.clamp(batch.rewards, self._params["rmin"], self._params["rmax"])
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
                    self._params["mcts_count"],
                    self._params["mcts_depth"],
                    self._params["ucb_c1"],
                    self._params["ucb_c2"],
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
            # debug
            # print("CONSUME_OUTPUT", policy, value)
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
                "params": self._params,
                "muzero": self._muzero.state_dict(),
            },
            f,
        )

    def load(self, f):
        checkpoint = torch.load(f, map_location=self._device)
        checkpoint["device"] = self._params["device"]

        self._id = checkpoint["id"]
        self._params = checkpoint["params"]
        self._make_networks()

        self._representation.load_state_dict(checkpoint["representation"])
        self._dynamics.load_state_dict(checkpoint["dynamics"])
        self._policy.load_state_dict(checkpoint["policy"])
        self._value.load_state_dict(checkpoint["value"])
        self._projector.load_state_dict(checkpoint["projector"])
        self._predictor.load_state_dict(checkpoint["predictor"])

        self._optimizer.load_state_dict(checkpoint["optimizer"])

        self._learn_schedule = checkpoint["learn_schedule"]
        self._epsilon_schedule = checkpoint["epsilon_schedule"]
        self._target_net_update_schedule = checkpoint["target_net_update_schedule"]
        self._rng = checkpoint["rng"]
        self._lr_schedule = checkpoint["lr_schedule"]

        self._replay_buffer = None
