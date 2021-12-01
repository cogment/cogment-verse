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
import muzero.replay_buffer

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
        rmin=0.0,
        rmax=1.0,
        rbins=4,
        vmin=0.0,
        vmax=200.0,
        vbins=16,
        projector_dim=16,
        projector_hidden_dim=32,
    )
    optimizer = torch.optim.AdamW(muzero.parameters(), lr=1e-3, weight_decay=0.01)
    min_replay_buffer_size = 200
    batch_size = 32
    gamma = 0.9

    target_muzero = copy.deepcopy(muzero)

    observations = []
    next_observations = []
    actions = []
    rewards = []
    dones = []
    policies = []
    values = []

    total_trial_reward = 0
    total_steps = 0

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
                    alpha=1.0,
                    temperature=1.0,
                    discount_rate=0.95,
                    mcts_depth=3,
                    mcts_count=32,
                    ucb_c1=1.25,
                    ucb_c2=15000.0,
                )
            next_observation, reward, done, _ = env.step(action)
            observations.append(observation)
            next_observations.append(next_observation)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            policies.append(policy.squeeze(0).detach().numpy())
            values.append(value)
            observation = next_observation
            total_steps += 1
            total_trial_reward += reward
            trial_reward += reward

            if len(actions) > min_replay_buffer_size:
                muzero.train()
                idx = np.random.randint(0, len(actions), batch_size)
                batch_observations = torch.tensor([observations[i] for i in idx]).float().unsqueeze(1)
                batch_next_observations = torch.tensor([next_observations[i] for i in idx]).float().unsqueeze(1)
                batch_actions = torch.tensor([actions[i] for i in idx]).long().unsqueeze(1)
                batch_rewards = torch.tensor([rewards[i] for i in idx]).float().unsqueeze(1)
                batch_dones = torch.tensor([dones[i] for i in idx]).float().unsqueeze(1)
                batch_policies = torch.tensor([policies[i] for i in idx]).float().unsqueeze(1)
                batch_values = torch.tensor([values[i] for i in idx]).float().unsqueeze(1)

                priority, info = muzero.train_step(
                    optimizer=optimizer,
                    observation=batch_observations,
                    action=batch_actions,
                    reward=batch_rewards,
                    next_observation=batch_next_observations,
                    done=batch_dones,
                    target_policy=batch_policies,
                    target_value=batch_values,
                    importance_weight=torch.ones((batch_size,)),
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
