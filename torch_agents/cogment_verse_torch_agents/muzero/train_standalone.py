import muzero.mcts
from muzero.networks import (
    RepresentationNetwork,
    PolicyNetwork,
    ValueNetwork,
    DynamicsNetwork,
    QNetwork,
    MuZero,
    lin_bn_act,
    mlp,
)
from muzero.replay_buffer import TrialReplayBuffer, Episode, EpisodeBatch

import torch
import copy
import gym
import numpy as np
import itertools


def make_networks(
    obs_dim,
    act_dim,
    hidden_dim,
    hidden_layers,
    rmin,
    rmax,
    rbins,
    vmin,
    vmax,
    vbins,
    projector_dim,
    projector_hidden_dim,
):
    stem = lin_bn_act(obs_dim, hidden_dim, bn=True, act=torch.nn.ReLU())
    representation = RepresentationNetwork(stem, hidden_dim, hidden_layers)

    policy = PolicyNetwork(hidden_dim, hidden_layers, act_dim)

    value = ValueNetwork(
        hidden_dim,
        hidden_layers,
        vmin,
        vmax,
        vbins,
    )

    dynamics = DynamicsNetwork(
        act_dim,
        hidden_dim,
        hidden_layers,
        rmin,
        rmax,
        rbins,
    )

    projector = mlp(
        hidden_dim,
        projector_hidden_dim,
        projector_dim,
        hidden_layers=1,
    )
    # todo: check number of hidden layers used in predictor (same as projector??)
    predictor = mlp(projector_dim, projector_hidden_dim, projector_dim, hidden_layers=1)

    muzero = MuZero(
        representation,
        dynamics,
        policy,
        value,
        projector,
        predictor,
        dynamics.distribution,
        value.distribution,
        QNetwork(
            act_dim,
            hidden_dim,
            hidden_layers,
            vmin,
            vmax,
            vbins,
        ),
    )

    return muzero


def main():
    env = gym.make("CartPole-v0")
    muzero = make_networks(
        obs_dim=4,
        act_dim=2,
        hidden_dim=64,
        hidden_layers=2,
        rmin=-1.0,
        rmax=0.0,
        rbins=16,
        vmin=0.0,
        vmax=200.0,
        vbins=32,
        projector_dim=16,
        projector_hidden_dim=32,
    )
    optimizer = torch.optim.AdamW(muzero.parameters(), lr=1e-3, weight_decay=0.01)
    min_replay_buffer_size = 1000
    batch_size = 64
    gamma = 0.9
    rollout_length = 2
    temperature = 1.0

    target_muzero = copy.deepcopy(muzero)
    total_trial_reward = 0
    total_steps = 0

    replay_buffer = TrialReplayBuffer(max_size=100000, discount_rate=0.95, bootstrap_steps=10)

    for episode in range(1000):
        observation = env.reset()
        done = False
        trial_reward = 0
        while not done:
            target_muzero.eval()
            with torch.no_grad():
                action, policy, q, value = target_muzero.act(
                    torch.from_numpy(observation).float(),
                    epsilon=0.25,
                    alpha=0.1,
                    temperature=temperature,
                    discount_rate=0.95,
                    mcts_depth=3,
                    mcts_count=16,
                    ucb_c1=1.25,
                    ucb_c2=15000.0,
                )
            next_observation, reward, done, _ = env.step(action)
            replay_buffer.add_sample(observation, action, reward, next_observation, done, policy, value)
            observation = next_observation
            total_steps += 1
            total_trial_reward += reward
            trial_reward += reward

            env.render()

            if replay_buffer.size() > min_replay_buffer_size:
                muzero.train()
                temperature = max(0.25, temperature * 0.99)
                batch = replay_buffer.sample(rollout_length, batch_size)

                batch_tensors = []
                for _, tensor in enumerate(batch):
                    if isinstance(tensor, np.ndarray):
                        tensor = torch.from_numpy(tensor)
                    batch_tensors.append(tensor)
                batch = EpisodeBatch(*batch_tensors)

                priority, info = muzero.train_step(
                    optimizer=optimizer,
                    observation=batch.state,
                    action=batch.action,
                    reward=batch.rewards,
                    next_observation=batch.next_state,
                    done=batch.done,
                    target_policy=batch.target_policy,
                    target_value=batch.target_value,
                    importance_weight=batch.importance_weight,
                    max_norm=100.0,
                    target_label_smoothing_factor=0.01,
                    s_weight=0.01,
                    v_weight=1.0,
                    discount_factor=0.95,
                    target_muzero=target_muzero,
                )

                online_params = itertools.chain(muzero.parameters(), muzero.buffers())
                target_params = itertools.chain(target_muzero.parameters(), target_muzero.buffers())

                for pt, po in zip(target_params, online_params):
                    pt.data = gamma * pt.data + (1 - gamma) * po.data

        print(f"Episode {episode + 1} reward {trial_reward} mean reward {total_trial_reward/(episode+1)}")


if __name__ == "__main__":
    main()
