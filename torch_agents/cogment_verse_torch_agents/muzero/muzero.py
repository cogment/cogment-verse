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

import ctypes
import copy
import itertools
import numpy as np
from multiprocessing import cpu_count
import queue

import torch
import torch.multiprocessing as mp
from torch.nn.modules.loss import SmoothL1Loss

from .replay_buffer import ConcurrentTrialReplayBuffer, Episode, TrialReplayBuffer
from cogment_verse_torch_agents.wrapper import proto_array_from_np_array

from cogment_verse_torch_agents.third_party.hive.utils.schedule import LinearSchedule, get_schedule, PeriodicSchedule
from cogment_verse_torch_agents.third_party.hive.utils.utils import get_optimizer_fn
from cogment_verse_torch_agents.third_party.hive.dqn import legal_moves_adapter
from cogment_verse_torch_agents.third_party.hive.agent import Agent

from .mcts import MCTS

from data_pb2 import AgentAction


class LinearScheduleWithWarmup(LinearSchedule):
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


@torch.no_grad()
def parameters_l2(model):
    l2 = 0
    for p in model.parameters():
        l2 += torch.sum(torch.square(p))
    return l2


def relative_error(u, v, eps=1e-6):
    # clamp for NaN issue??
    return torch.clamp(torch.abs(u - v) / torch.sqrt(u ** 2 + v ** 2 + eps), 0, 1)


def cosine_similarity_loss(u, v, dim=1, weights=None, eps=1e-6):
    if weights is None:
        weights = 1.0
    uv = torch.sum(u * v, dim=dim)
    u2 = torch.sum(u * u, dim=dim)
    v2 = torch.sum(v * v, dim=dim)
    cosine = uv / torch.sqrt(u2 * v2 + eps)
    return torch.mean((1.0 - cosine) * weights)


def lin_bn_act(num_in, num_out, bn=True, act=None):
    layers = [torch.nn.Linear(num_in, num_out)]
    if bn:
        layers.append(torch.nn.BatchNorm1d(num_out))
    if act:
        layers.append(act)
    return torch.nn.Sequential(*layers)


def mlp(
    num_in,
    num_hidden,
    num_out,
    hidden_layers=1,
    bn=True,
    act=torch.nn.LeakyReLU(),
    final_act=None,
):
    act = torch.nn.LeakyReLU()
    stem = lin_bn_act(num_in, num_hidden, bn, act)
    hiddens = [lin_bn_act(num_hidden, num_hidden, bn, act) for _ in range(hidden_layers - 1)]
    output = lin_bn_act(num_hidden, num_out, False, final_act)
    layers = [stem] + hiddens + [output]
    return torch.nn.Sequential(*layers)


class ResidualBlock(torch.nn.Module):
    def __init__(self, channels_in, channels_out, activation=None):
        super().__init__()

        if channels_in != channels_out:
            self._prepare_input = torch.nn.Linear(channels_in, channels_out)
        else:
            self._prepare_input = torch.nn.Identity()

        self._fc1 = torch.nn.Linear(channels_out, channels_out)
        self._fc2 = torch.nn.Linear(channels_out, channels_out)
        self._bn1 = torch.nn.BatchNorm1d(channels_out)
        self._bn2 = torch.nn.BatchNorm1d(channels_out)
        self._act = torch.nn.LeakyReLU()
        self._final_act = activation or torch.nn.Identity()

    def forward(self, x):
        x = self._prepare_input(x)
        y = self._fc1(x)
        y = self._bn1(y)
        y = self._act(y)
        y = self._fc2(y)
        y = self._bn2(y)
        return self._final_act(x + y)


class DynamicsAdapter(torch.nn.Module):
    def __init__(self, net, act_dim, num_input, num_latent, reward_dist):
        super().__init__()
        self._net = net
        self._act_dim = act_dim
        self._num_latent = num_latent
        self._reward_dist = reward_dist
        self._state_pred = torch.nn.Linear(num_input, num_latent)

    def forward(self, representation, action, return_probs=False):
        """
        Returns tuple (next_state, reward)
        """
        action = torch.nn.functional.one_hot(action, self._act_dim)
        intermediate = self._net(torch.cat([representation, action], dim=1))
        state = self._state_pred(intermediate)
        reward_probs, reward = self._reward_dist(intermediate)

        return state, reward_probs, reward


def resnet(
    num_in,
    num_hidden,
    num_out,
    hidden_layers=1,
    bn=True,
    act=torch.nn.LeakyReLU(),
    final_act=None,
):
    act = torch.nn.LeakyReLU()
    stem = ResidualBlock(num_in, num_hidden, act)
    hiddens = [ResidualBlock(num_hidden, num_hidden, act) for _ in range(hidden_layers - 1)]
    output = ResidualBlock(num_hidden, num_out, final_act)
    layers = [stem] + hiddens + [output]
    return torch.nn.Sequential(*layers)


@torch.no_grad()
def reanalyze(representation, dynamics, policy, value, state, mcts_count, mcts_depth, c1, c2, device):
    representation.eval()
    dynamics.eval()
    policy.eval()
    value.eval()

    if isinstance(state, np.ndarray):
        state = torch.from_numpy(state)
        state = state.to(device).unsqueeze(0)

    # get policy/val training targets via MCTS improvement
    mcts = MCTS(
        policy=policy,
        value=value,
        dynamics=dynamics,
        representation=representation(state),
        max_depth=mcts_depth,
    )
    mcts.build_search_tree(mcts_count)
    return mcts.improved_targets()


def reanalyze_queue(
    representation,
    dynamics,
    policy,
    value,
    input_queue,
    output_queue,
    mcts_count,
    mcts_depth,
    c1,
    c2,
    end_process,
    device,
):
    while not end_process.value:
        try:
            sample = input_queue.get(timeout=1.0)
        except queue.Empty:
            # print("input queue is empty")
            continue

        # print("processing item")
        episode_idx, step, state = sample
        target_policy, target_value = reanalyze(
            representation, dynamics, policy, value, state, mcts_count, mcts_depth, c1, c2, device
        )
        target_policy = target_policy.detach().cpu()
        target_value = target_value.detach().cpu()
        input_queue.task_done()
        # print("putting item in output queue")
        while True:
            try:
                # target_policy.share_memory_()
                # target_value.share_memory_()
                # target_policy = target_policy.clone()
                # target_value = target_value.clone()
                output_queue.put((episode_idx, step, target_policy, target_value), timeout=1.0)
                # print("item placed in output queue")
                break
            except queue.Full:
                # print("output_queue.put timed out")
                continue


def reward_transform(reward, eps=0.001):
    # 1911.08265 appendix F (NB: there is a typo in the formula!)
    # See the original paper 1805.11593 for the correct formula
    return torch.sign(reward) * torch.sqrt(1 + torch.abs(reward)) - 1 + eps * reward


def reward_tansform_inverse(transformed_reward, eps=0.001):
    s = torch.sign(transformed_reward)
    y = torch.abs(transformed_reward)

    a = eps ** 2
    b = -(2 * eps * (y + 1) + 1)
    c = y ** 2 + 2 * y

    d = torch.sqrt(b ** 2 - 4 * a * c)
    e = 2 * a
    return s * torch.abs((-b - d) / e)


class Distributional(torch.nn.Module):
    def __init__(self, vmin, vmax, num_input, count, transform=None, inverse_transform=None):
        super().__init__()
        assert count >= 2
        self._transform = transform or (lambda x: x)
        self._inverse_transform = inverse_transform or (lambda x: x)
        self._act = torch.nn.Softmax(dim=1)
        self._vmin = self._transform(torch.tensor(vmin))
        self._vmax = self._transform(torch.tensor(vmax))
        self._bins = torch.nn.parameter.Parameter(
            torch.from_numpy(np.linspace(self._vmin, self._vmax, num=count, dtype=np.float32)), requires_grad=False
        )
        self._fc = torch.nn.Linear(num_input, count)
        self._count = count

    def forward(self, x):
        probs = self._act(self._fc(x))
        val = torch.sum(probs * self._bins.unsqueeze(0), dim=1)
        return probs, self._inverse_transform(val)

    def compute_target(self, vals):
        shape = vals.shape
        target_shape = list(shape) + [self._count]

        vals = vals.view(-1)
        target = torch.zeros(target_shape, dtype=torch.float32).view(-1)

        vals = self._transform(vals)
        vals = torch.clamp(vals, self._vmin, self._vmax)
        vals = (vals - self._vmin) / (self._vmax - self._vmin) * (self._count - 1)
        bin_low = torch.clamp(vals.type((torch.long)), 0, self._count - 1)
        bin_high = torch.clamp(bin_low + 1, 0, self._count - 1)
        t = bin_high - vals

        for i in range(len(vals)):
            target[bin_low[i]] = t[i]
            target[bin_high[i]] = 1 - t[i]

        return target.view(*target_shape)


def cross_entropy(pred, target, weights=None, dim=1):

    return torch.mean(weights * torch.sum(-target * torch.log(pred + 1e-6), dim=dim))


class LambdaModule(torch.nn.Module):
    def __init__(self, fn, context=None):
        super().__init__()
        self._fn = fn
        self._context = context

    def forward(self, *args, **kwargs):
        return self._fn(*args, context=self._context, **kwargs)


class MuZeroMLP(Agent):
    """
    MuZero implementation
    """

    def __init__(
        self,
        *,
        obs_dim,
        act_dim,
        optimizer_fn=None,
        id=0,
        discount_rate=0.997,
        target_net_soft_update=False,
        target_net_update_fraction=0.05,
        target_net_update_schedule=None,
        epsilon_schedule=None,
        learn_schedule=None,
        lr_schedule=None,
        seed=42,
        device="cuda" if torch.cuda.is_available() else "cpu",
        weight_decay=1e-2,
        mcts_depth=4,
        mcts_count=16,
        rollout_length=4,
        max_replay_buffer_size=50000,
        reanalyze_fraction=0.0,
        reanalyze_period=1e100,
        batch_size=128,
    ):
        # debug/testing
        lr_schedule = LinearScheduleWithWarmup(1e-3, 1e-6, 100000, 1000)
        self._temperature_schedule = LinearSchedule(1.0, 0.25, 100000)
        super().__init__(
            id=id,
            seed=seed,
            obs_dim=obs_dim,
            act_dim=act_dim,
            learn_schedule=learn_schedule,
            epsilon_schedule=epsilon_schedule,
            lr_schedule=lr_schedule,
            max_replay_buffer_size=max_replay_buffer_size,
            discount_rate=discount_rate,
            target_net_soft_update=target_net_soft_update,
            target_net_update_fraction=target_net_update_fraction,
            weight_decay=weight_decay,
            target_label_smoothing_factor=0.01,
            rollout_length=rollout_length,
            reanalyze_fraction=reanalyze_fraction,
            reanalyze_period=reanalyze_period,
            device=device,
            value_bootstrap_steps=20,
            num_latent=128,
            num_hidden=256,
            num_hidden_layers=4,
            projector_hidden=128,
            projector_dim=64,
            mcts_depth=mcts_depth,
            mcts_count=mcts_count,
            ucb_c1=1.25,
            ucb_c2=10000.0,
            batch_size=batch_size,
            dirichlet_alpha=0.5,
            rmin=-100.0,
            rmax=100.0,
            vmin=-300.0,
            vmax=300.0,
            r_bins=128,
            v_bins=128,
        )

        self._device = torch.device(device)
        self._id = id

        self._target_net_update_schedule = get_schedule(target_net_update_schedule)
        if self._target_net_update_schedule is None:
            self._target_net_update_schedule = PeriodicSchedule(False, True, 10000)

        self._reanalyze_schedule = PeriodicSchedule(False, True, reanalyze_period)

        self._training = True
        self._default_policy = torch.ones(act_dim, dtype=torch.float32).unsqueeze(0) / act_dim

        self._make_networks()

        self._replay_buffer = None

        self._representation.share_memory()
        self._dynamics.share_memory()
        self._policy.share_memory()
        self._value.share_memory()

    @torch.no_grad()
    def reanalyze(self, state, epsilon, alpha):
        self.eval()

        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state)

        state = state.to(self._device).unsqueeze(0)

        def _dynamics_deterministic(rep, action, *, context):
            distributional_dynamics = context
            rep, probs, reward = distributional_dynamics(rep, action)
            return rep, reward

        def _value_deterministic(rep, *, context):
            distributional_value = context
            probs, val = distributional_value(rep)
            return val

        dynamics = LambdaModule(_dynamics_deterministic, context=self._dynamics)
        value = LambdaModule(_value_deterministic, context=self._value)

        # get policy/val training targets via MCTS improvement
        representation = self._representation(state)
        mcts = MCTS(
            policy=self._policy,
            value=value,
            dynamics=dynamics,
            representation=representation,
            max_depth=self._params["mcts_depth"],
            epsilon=epsilon,
            alpha=alpha,
            c1=self._params["ucb_c1"],
            c2=self._params["ucb_c2"],
            discount=self._params["discount_rate"],
        )

        mcts.build_search_tree(self._params["mcts_count"])
        return mcts.improved_targets()

    def _create_replay_buffer(self):
        # due to pickling issues for multiprocessing, we create the replay buffer lazily
        self._replay_buffer = None

    def _ensure_replay_buffer(self):
        if self._replay_buffer is None:
            self._replay_buffer = ConcurrentTrialReplayBuffer(
                max_size=self._params["max_replay_buffer_size"],
                discount_rate=self._params["discount_rate"],
                bootstrap_steps=self._params["value_bootstrap_steps"],
                # min_size=self._params["batch_size"],
                min_size=1000,
                rollout_length=self._params["rollout_length"],
                batch_size=self._params["batch_size"],
            )
            self._replay_buffer.start()
        assert self._replay_buffer.is_alive()

    @torch.no_grad()
    def consume_training_sample(self, sample):
        self._ensure_replay_buffer()
        self._replay_buffer.add_sample(sample)

    def sample_training_batch(self, batch_size):
        self._ensure_replay_buffer()
        # print("Model::sample_training_batch called")
        return self._replay_buffer.sample(self._params["rollout_length"], batch_size)

    def _make_networks(self):
        self._value_distribution = Distributional(
            self._params["vmin"],
            self._params["vmax"],
            self._params["num_hidden"],
            self._params["v_bins"],
            reward_transform,
            reward_tansform_inverse,
        ).to(self._device)
        self._reward_distribution = Distributional(
            self._params["rmin"],
            self._params["rmax"],
            self._params["num_hidden"],
            self._params["r_bins"],
            reward_transform,
            reward_tansform_inverse,
        ).to(self._device)

        self._representation = resnet(
            self._params["obs_dim"],
            self._params["num_hidden"],
            self._params["num_latent"],
            self._params["num_hidden_layers"],
            # final_act=torch.nn.BatchNorm1d(self._params["num_latent"]),  # normalize for input to subsequent networks
        ).to(self._device)

        # debugging
        # self._representation = torch.nn.Identity()

        self._dynamics = DynamicsAdapter(
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
            reward_dist=self._reward_distribution,
        ).to(self._device)
        self._policy = resnet(
            self._params["num_latent"],
            self._params["num_hidden"],
            self._params["act_dim"],
            self._params["num_hidden_layers"],
            final_act=torch.nn.Softmax(dim=1),
        ).to(self._device)
        self._value = resnet(
            self._params["num_latent"],
            self._params["num_hidden"],
            self._params["num_hidden"],
            self._params["num_hidden_layers"] - 1,
            final_act=self._value_distribution,
        ).to(self._device)
        self._projector = mlp(
            self._params["num_latent"], self._params["projector_hidden"], self._params["projector_dim"]
        ).to(self._device)
        self._predictor = mlp(
            self._params["projector_dim"], self._params["projector_hidden"], self._params["projector_dim"]
        ).to(self._device)

        self._optimizer = torch.optim.AdamW(
            itertools.chain(
                self._representation.parameters(),
                self._dynamics.parameters(),
                self._policy.parameters(),
                self._value.parameters(),
                self._projector.parameters(),
                self._predictor.parameters(),
            ),
            lr=1e-3,
            weight_decay=self._params["weight_decay"],
        )

    def train(self):
        """Changes the agent to training mode."""
        super().train()
        self._representation.train()
        self._dynamics.train()
        self._policy.train()
        self._value.train()
        self._projector.train()
        self._predictor.train()

    def eval(self):
        """Changes the agent to evaluation mode."""
        super().eval()
        self._representation.eval()
        self._dynamics.eval()
        self._policy.eval()
        self._value.eval()
        self._projector.eval()
        self._predictor.eval()

    @torch.no_grad()
    def act(self, observation, formatted_legal_moves, update_schedule=True):
        self.eval()

        formatted_legal_moves = torch.tensor(formatted_legal_moves).to(self._device).float()
        # observation = torch.tensor(np.expand_dims(observation, axis=0)).to(self._device).float()
        observation = torch.tensor(observation).to(self._device).float()

        # if not self._params["distributional"]:
        epsilon = self.get_epsilon_schedule(update_schedule)
        policy, value = self.reanalyze(observation, epsilon, self._params["dirichlet_alpha"])

        # apply temperature equal to epsilon**2 (todo: decouple these)
        # temperature = torch.tensor(self._params["temperature"])
        temperature = torch.tensor(self._temperature_schedule.get_value())
        policy = torch.pow(policy, 1 / temperature)
        policy /= torch.sum(policy, dim=1)

        try:
            action = torch.distributions.Categorical(policy).sample()
            action = action.cpu().numpy().item()
        except:
            print(policy)
            raise

        cog_action = AgentAction(
            discrete_action=action, policy=proto_array_from_np_array(policy.cpu().numpy()), value=value
        )
        # print("acting with policy/value", policy.cpu().numpy(), value.cpu().numpy().item())
        return cog_action

    def learn(self, batch, update_schedule=True):

        lr = self._lr_schedule.update()
        for grp in self._optimizer.param_groups:
            grp["lr"] = lr

        # do not modify batch in-place
        batch = copy.copy(batch)
        for key, tensor in batch.items():
            if isinstance(tensor, np.ndarray):
                tensor = torch.from_numpy(tensor)
            batch[key] = tensor.to(self._device)

        self.train()

        batch_size, rollout_length = batch["state"].shape[:2]
        observation_shape = batch["state"].shape[2:]
        current_representation = self._representation(batch["state"][:, 0, :])
        representation_shape = current_representation.shape[1:]

        pred_representation = torch.zeros((batch_size, rollout_length + 1, *representation_shape)).to(self._device)
        pred_representation[:, 0, :] = current_representation
        pred_reward = torch.zeros((batch_size, rollout_length)).to(self._device)
        pred_reward_probs = torch.zeros((batch_size, rollout_length, self._params["r_bins"])).to(self._device)

        for k in range(rollout_length):
            next_representation, next_reward_probs, next_reward = self._dynamics(
                current_representation, batch["action"][:, k], return_probs=True
            )
            pred_representation[:, k + 1] = next_representation
            pred_reward[:, k] = next_reward
            pred_reward_probs[:, k, :] = next_reward_probs
            current_representation = next_representation

        del current_representation
        del next_representation
        del next_reward_probs
        del next_reward

        states = pred_representation[:, :-1, :].reshape(batch_size * rollout_length, -1)

        next_observations = batch["next_state"].view(batch_size * rollout_length, *observation_shape)

        pred_policy = self._policy(states)
        pred_value_probs, pred_value = self._value(states)
        pred_reward = pred_reward.view(batch_size * rollout_length)
        pred_reward_probs = pred_reward_probs.view(batch_size * rollout_length, self._params["r_bins"])
        pred_next_states = pred_representation[:, 1:, :].reshape(
            batch_size * rollout_length, self._params["num_latent"]
        )
        pred_projection = self._predictor(self._projector(pred_next_states))

        target_policy = batch["target_policy"].view(*pred_policy.shape)
        target_policy = torch.clamp(target_policy, self._params["target_label_smoothing_factor"], 1.0)
        target_policy /= torch.sum(target_policy, dim=1, keepdim=True)
        target_value = torch.clamp(
            batch["target_value"].view(*pred_value.shape), self._params["vmin"], self._params["vmax"]
        )
        target_reward = torch.clamp(
            batch["rewards"].view(*pred_value.shape), self._params["rmin"], self._params["rmax"]
        )

        with torch.no_grad():
            target_value_probs = self._value_distribution.compute_target(target_value).to(self._device)
            target_reward_probs = self._reward_distribution.compute_target(target_reward).to(self._device)
            target_next_state = self._representation(next_observations)
            target_projection = self._projector(target_next_state)

        episode = batch["episode"].view(-1)
        step = batch["step"].view(-1)
        priority = batch["priority"][:, 0].view(-1)
        importance_weight = 1 / (priority + 1e-6) / self._replay_buffer.size()
        importance_weight = torch.stack([importance_weight] * rollout_length, dim=-1).view(batch_size * rollout_length)

        loss_p = cross_entropy(pred_policy, target_policy, importance_weight)
        loss_r = cross_entropy(pred_reward_probs, target_reward_probs, importance_weight)
        loss_v = cross_entropy(pred_value_probs, target_value_probs, importance_weight)
        loss_s = cosine_similarity_loss(pred_projection, target_projection, weights=importance_weight)

        with torch.no_grad():
            entropy_target = cross_entropy(target_policy, target_policy, importance_weight)
            entropy_pred = cross_entropy(pred_policy, pred_policy, importance_weight)
            priority = (
                torch.abs(
                    pred_value.view(batch_size, rollout_length)[:, 0]
                    - target_value.view(batch_size, rollout_length)[:, 0]
                )
                .cpu()
                .detach()
                .numpy()
            )

        self._replay_buffer.update_priorities(list(zip(episode, step, priority)))

        self._optimizer.zero_grad()

        s_weight = 1.0
        total_loss = loss_p + loss_r + loss_v + s_weight * loss_s
        total_loss.backward()
        self._optimizer.step()

        info = dict(
            lr=lr,
            loss_r=loss_r,
            loss_v=loss_v,
            loss_p=loss_p,
            loss_s=loss_s,
            loss_kl=(loss_p - entropy_target),
            total_loss=total_loss,
            entropy_target=entropy_target,
            entropy_pred=entropy_pred,
            r_min=torch.min(target_reward),
            r_max=torch.max(target_reward),
            r_mean=torch.mean(target_reward),
            v_min=torch.min(target_value),
            v_max=torch.max(target_value),
            v_mean=torch.mean(target_value),
            # reward_error=reward_delta,
            # value_error=value_delta,
            representation_l2=parameters_l2(self._representation),
            dynamics_l2=parameters_l2(self._dynamics),
            policy_l2=parameters_l2(self._policy),
            value_l2=parameters_l2(self._value),
            projector_l2=parameters_l2(self._projector),
            predictor_l2=parameters_l2(self._predictor),
            temperature=self._temperature_schedule.get_value(),
        )

        for key, val in info.items():
            if isinstance(val, torch.Tensor):
                info[key] = val.detach().cpu().numpy().item()

        # Update target network
        if self._training and self._target_net_update_schedule.update():
            self._update_target()

        if update_schedule:
            self.get_epsilon_schedule(update_schedule)
            self._temperature_schedule.update()

        # Reanalyze old data
        if self._training and self._reanalyze_schedule.update():
            self.reanalyze_replay_buffer(self._params["reanalyze_fraction"])

        # Return loss
        return info

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
                "id": self._id,
                "params": self._params,
                "representation": self._representation.state_dict(),
                "dynamics": self._dynamics.state_dict(),
                "policy": self._policy.state_dict(),
                "value": self._value.state_dict(),
                "projector": self._projector.state_dict(),
                "predictor": self._predictor.state_dict(),
                "optimizer": self._optimizer.state_dict(),
                "learn_schedule": self._learn_schedule,
                "epsilon_schedule": self._epsilon_schedule,
                "target_net_update_schedule": self._target_net_update_schedule,
                "rng": self._rng,
                "lr_schedule": self._lr_schedule,
            },
            f,
        )

    def load(self, f):
        super().load(f)
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
