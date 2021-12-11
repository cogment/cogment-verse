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

import numpy as np

import torch

from .mcts import MCTS


def expect_equal_shape(a, b):
    assert a.shape == b.shape, f"{a.shape} == {b.shape}"


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


def mean_error_loss(u, v, dim=1, weights=None, eps=1e-6):
    if weights is None:
        weights = 1.0
    return torch.mean(weights * torch.mean((u - v) ** 2, dim=dim))  # , dim=dim)


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
    act=torch.nn.ReLU(),
    final_bn=False,
    final_act=None,
):
    act = torch.nn.ReLU()
    stem = lin_bn_act(num_in, num_hidden, bn, act)
    hiddens = [lin_bn_act(num_hidden, num_hidden, bn, act) for _ in range(hidden_layers - 1)]
    output = lin_bn_act(num_hidden, num_out, final_bn, final_act)
    layers = [stem] + hiddens + [output]
    return torch.nn.Sequential(*layers)


class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self._fc1 = torch.nn.Linear(channels, channels)
        self._fc2 = torch.nn.Linear(channels, channels)
        self._bn1 = torch.nn.BatchNorm1d(channels)
        self._bn2 = torch.nn.BatchNorm1d(channels)
        self._act = torch.nn.ReLU()

    def forward(self, x):
        y = self._fc1(x)
        y = self._bn1(y)
        y = self._act(y)
        y = self._fc2(y)
        y = self._bn2(y)
        return self._act(x + y)


class DynamicsAdapter(torch.nn.Module):
    def __init__(self, net, act_dim, num_input, num_latent, reward_dist):
        super().__init__()
        self._net = net
        self._act_dim = act_dim
        self._num_latent = num_latent
        self._reward_dist = reward_dist
        self._state_pred = torch.nn.Linear(num_input, num_latent)

    def forward(self, state, action):
        """
        Returns tuple (next_state, reward)
        """
        action = torch.nn.functional.one_hot(action, self._act_dim)
        intermediate = self._net(torch.cat([state, action], dim=1))
        next_state = self._state_pred(intermediate)
        reward_probs, reward = self._reward_dist(intermediate)

        return next_state, reward_probs, reward


def resnet(num_in, num_hidden, num_out, hidden_layers=1):
    layers = []
    if num_in != num_hidden:
        layers.append(lin_bn_act(num_in, num_hidden, bn=True, act=torch.nn.ReLU()))

    layers.extend([ResidualBlock(num_hidden) for _ in range(hidden_layers)])
    if num_hidden != num_out:
        layers.append(lin_bn_act(num_hidden, num_out, bn=True, act=torch.nn.ReLU()))
    return torch.nn.Sequential(*layers)


def reward_transform(reward, eps=0.001):
    # 1911.08265 appendix F (NB: there is a typo in the formula!)
    # See the original paper 1805.11593 for the correct formula
    return torch.sign(reward) * (torch.sqrt(1 + torch.abs(reward)) - 1) + eps * reward


def reward_transform_inverse(transformed_reward, eps=0.001):
    s = torch.sign(transformed_reward)
    y = torch.abs(transformed_reward)

    a = eps ** 2
    b = -2 * eps * (y + 1) - 1
    c = y ** 2 + 2 * y

    d = torch.sqrt(b ** 2 - 4 * a * c)
    e = 2 * a

    x = torch.abs((-b - d) / e)
    # 1 step of Newton's method for numerical stability
    x = -c / (a * x + b)

    return s * x


class Distributional(torch.nn.Module):
    def __init__(self, vmin, vmax, num_input, count, transform=None, inverse_transform=None):
        super().__init__()
        assert count >= 2
        self._transform = transform or torch.nn.Identity()
        self._inverse_transform = inverse_transform or torch.nn.Identity()
        self._act = torch.nn.Softmax(dim=1)
        self._vmin = self._transform(torch.tensor(vmin)).detach().cpu().item()
        self._vmax = self._transform(torch.tensor(vmax)).detach().cpu().item()

        self.register_buffer(
            "_bins", torch.from_numpy(np.linspace(self._vmin, self._vmax, num=count, dtype=np.float32))
        )
        self._fc = torch.nn.Linear(num_input, count)

    def forward(self, x):
        probs = self._act(self._fc(x))
        val = torch.sum(probs * self._bins.unsqueeze(0), dim=1)
        return probs, self._inverse_transform(val)

    def compute_value(self, probs):
        assert probs.shape[-1] == self._bins.shape[0]
        bin_shape = [1 for _ in range(len(probs.shape))]
        bin_shape[-1] = self._bins.shape[0]
        val = torch.sum(probs * self._bins.view(*bin_shape), dim=-1)
        return self._inverse_transform(val)

    @torch.no_grad()
    def compute_target(self, vals):
        count = self._bins.shape[-1]
        shape = vals.shape
        target_shape = list(shape) + [count]

        # print("DEBUG vals", vals.shape)
        vals = vals.view(-1)
        target = torch.zeros(target_shape, dtype=torch.float32).view(-1, count)

        vals = self._transform(vals)
        vals = torch.clamp(vals, self._vmin, self._vmax)
        vals = (vals - self._vmin) / (self._vmax - self._vmin) * (count - 1)
        bin_low = torch.clamp(vals.type((torch.long)), 0, count - 1)
        bin_high = torch.clamp(bin_low + 1, 0, count - 1)
        t = bin_high - vals

        for i in range(len(vals)):
            target[i, bin_low[i]] = t[i]
            target[i, bin_high[i]] = 1 - t[i]

        return target.view(*target_shape)


def cross_entropy(pred, target, weights=None, dim=None):
    return torch.mean(weights * torch.sum(-target * torch.log(pred + 1e-12), dim=dim))


class LambdaModule(torch.nn.Module):
    def __init__(self, fn, context=None):
        super().__init__()
        self._fn = fn
        self._context = context

    def forward(self, *args, **kwargs):
        return self._fn(*args, context=self._context, **kwargs)


def normalize_scale(state):
    max_state = torch.max(state, dim=1, keepdim=True)[0]
    min_state = torch.min(state, dim=1, keepdim=True)[0]
    return (state - min_state) / (max_state - min_state)


class RepresentationNetwork(torch.nn.Module):
    def __init__(self, stem, num_hidden, num_hidden_layers):
        super().__init__()
        self.stem = stem
        self.blocks = torch.nn.ModuleList([ResidualBlock(num_hidden) for _ in range(num_hidden_layers)])

    def forward(self, x):
        features = self.stem(x)
        for block in self.blocks:
            features = block(features)
        return normalize_scale(features)


class PolicyNetwork(torch.nn.Module):
    def __init__(self, num_hidden, num_hidden_layers, num_act):
        super().__init__()
        self.blocks = torch.nn.ModuleList([ResidualBlock(num_hidden) for _ in range(num_hidden_layers)])
        self.fc = torch.nn.Linear(num_hidden, num_act)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.fc(x)
        return torch.nn.functional.softmax(x, dim=1)


class ValueNetwork(torch.nn.Module):
    def __init__(self, num_hidden, num_hidden_layers, vmin, vmax, vbins):
        super().__init__()
        self.blocks = torch.nn.ModuleList([ResidualBlock(num_hidden) for _ in range(num_hidden_layers)])
        self.distribution = Distributional(vmin, vmax, num_hidden, vbins)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.distribution(x)


class QNetwork(torch.nn.Module):
    def __init__(self, num_act, num_hidden, num_hidden_layers, vmin, vmax, vbins):
        super().__init__()
        self.blocks = torch.nn.ModuleList([ResidualBlock(num_hidden) for _ in range(num_hidden_layers)])
        self.distributions = torch.nn.ModuleList(
            [Distributional(vmin, vmax, num_hidden, vbins) for _ in range(num_act)]
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)

        probs, vals = zip(*[dist(x) for dist in self.distributions])
        return torch.stack(probs, dim=1), torch.stack(vals, dim=1)


class DynamicsNetwork(torch.nn.Module):
    def __init__(self, num_action, num_hidden, num_hidden_layers, rmin, rmax, rbins):
        super().__init__()
        self.num_action = num_action

        self.encoding = mlp(num_action + num_hidden, num_hidden, num_hidden)
        self.blocks = torch.nn.ModuleList([ResidualBlock(num_hidden) for _ in range(num_hidden_layers)])
        self.distribution = Distributional(rmin, rmax, num_hidden, rbins)
        self.state_predictor = torch.nn.Linear(num_hidden, num_hidden)

    def forward(self, state, action):
        action_one_hot = torch.nn.functional.one_hot(action, self.num_action)
        encoded_state = self.encoding(torch.cat((state, action_one_hot), dim=1))

        for block in self.blocks:
            encoded_state = block(encoded_state)

        reward_probs, reward = self.distribution(encoded_state)
        next_state = self.state_predictor(encoded_state)
        return normalize_scale(next_state), reward_probs, reward


class MuZero(torch.nn.Module):
    def __init__(
        self,
        representation,
        dynamics,
        policy,
        value,
        projector,
        predictor,
        reward_distribution,
        value_distribution,
        dqn,
        similarity_loss=cosine_similarity_loss,
        # similarity_loss=mean_error_loss,
    ):
        super().__init__()
        self._representation = representation
        self._dynamics = dynamics
        self._policy = policy
        self._value = value
        self._projector = projector
        self._predictor = predictor
        self._reward_distribution = reward_distribution
        self._value_distribution = value_distribution
        self._similarity_loss = similarity_loss
        self._dqn = dqn

    @torch.no_grad()
    def act(self, observation, epsilon, alpha, temperature, discount_rate, mcts_depth, mcts_count, ucb_c1, ucb_c2):
        self.eval()

        if True:
            policy, q, value = self.reanalyze(
                observation, epsilon, alpha, discount_rate, mcts_depth, mcts_count, ucb_c1, ucb_c2, temperature
            )
            action = torch.distributions.Categorical(policy).sample()
            action = action.cpu().numpy().item()
            return action, policy, q, value

    def rollout(self, state, actions, length):
        bsz, nlatent = state.shape
        states = torch.zeros((length, bsz, nlatent), device=state.device)
        next_states = torch.zeros_like(states)
        rewards = torch.zeros((length, bsz), device=state.device)
        reward_probs = None
        states[0] = state
        for k in range(length):
            next_state, probs, reward = self._dynamics(states[k], actions[k])

            if reward_probs is None:
                reward_probs = torch.zeros((length, bsz, probs.shape[1]), device=state.device)

            next_states[k] = next_state
            rewards[k] = reward
            reward_probs[k] = probs
            if k < length - 1:
                states[k + 1] = next_state

        return states, reward_probs, rewards, next_states

    def train_step(
        self,
        optimizer,
        observation,
        action,
        target_reward_probs,
        reward,
        next_observation,
        done,
        target_policy,
        target_value_probs,
        target_value,
        importance_weight,
        max_norm,
        target_label_smoothing_factor,
        s_weight,
        v_weight,
        discount_factor,
        target_muzero,
    ):
        """ """
        self.train()
        target_muzero.eval()

        rollout_length, batch_size = observation.shape[:2]
        initial_state = self._representation(observation[0])
        device = initial_state.device
        priority = torch.zeros((rollout_length, batch_size), dtype=initial_state.dtype, device=device)

        loss_kl = 0
        loss_v = 0
        loss_r = 0
        loss_s = 0

        pred_states, pred_reward_probs, pred_rewards, pred_next_states = self.rollout(
            initial_state, action, rollout_length
        )

        # muzero model predictions
        pred_states = pred_states.view(batch_size * rollout_length, -1)
        pred_next_states = pred_next_states.view(batch_size * rollout_length, -1)

        pred_reward_probs = pred_reward_probs.view(batch_size * rollout_length, -1)
        pred_value_probs, pred_value = self._value(pred_states)
        pred_policy = self._policy(pred_states)

        pred_projection = self._predictor(self._projector(pred_next_states))

        # muzero training targets
        with torch.no_grad():
            target_policy = target_policy.reshape(rollout_length * batch_size, -1)
            target_reward_probs = target_reward_probs.reshape(rollout_length * batch_size, -1)
            target_value_probs = target_value_probs.reshape(rollout_length * batch_size, -1)
            entropy_target = cross_entropy(target_policy, target_policy, weights=1, dim=1)
            entropy_pred = cross_entropy(pred_policy, pred_policy, weights=1, dim=1)
            target_value = target_value.reshape(-1)
            reward = reward.reshape(-1)

            target_projection = target_muzero._projector(
                target_muzero._representation(next_observation.reshape(rollout_length * batch_size, -1))
            )

        # muzero loss calculation
        loss_kl = cross_entropy(pred_policy, target_policy, weights=1, dim=1) - entropy_target
        loss_r = cross_entropy(pred_reward_probs, target_reward_probs, weights=1, dim=1)
        loss_v = cross_entropy(pred_value_probs, target_value_probs, weights=1, dim=1)
        loss_s = self._similarity_loss(pred_projection, target_projection, weights=1, dim=1)

        # testing; disable priority replay
        priority = torch.ones_like(priority)
        priority = priority.detach().cpu().numpy()

        # muzero optimizer step
        optimizer.zero_grad()
        total_loss = loss_kl + loss_r + v_weight * loss_v + s_weight * loss_s
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=max_norm)
        optimizer.step()

        info = dict(
            loss_r=loss_r,
            loss_v=loss_v,
            loss_s=loss_s,
            loss_kl=loss_kl,
            total_loss=total_loss,
            entropy_target=entropy_target,
            entropy_pred=entropy_pred,
            reward_min=torch.min(reward),
            reward_max=torch.max(reward),
            reward_mean=torch.mean(reward),
            value_min=torch.min(target_value),
            value_mean=torch.mean(target_value),
            value_max=torch.max(target_value),
        )
        return priority, info

    @torch.no_grad()
    def reanalyze(self, state, epsilon, alpha, discount_rate, mcts_depth, mcts_count, ucb_c1, ucb_c2, temperature):
        self.eval()

        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state)

        state = state.unsqueeze(0)

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
            max_depth=mcts_depth,
            epsilon=epsilon,
            alpha=alpha,
            c1=ucb_c1,
            c2=ucb_c2,
            discount=discount_rate,
        )

        mcts.build_search_tree(mcts_count)
        return mcts.improved_targets(temperature)
