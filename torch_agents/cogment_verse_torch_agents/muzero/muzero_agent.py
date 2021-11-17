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

from collections import namedtuple

log = logging.getLogger(__name__)

from .muzero import MuZero

# pylint: disable=arguments-differ


class MuZeroAgentAdapter(AgentAdapter):
    def __init__(self):
        super().__init__()
        self._dtype = torch.float

    def tensor_from_cog_obs(self, cog_obs, device=None):
        pb_array = cog_obs.vectorized
        np_array = np.frombuffer(pb_array.data, dtype=pb_array.dtype).reshape(*pb_array.shape)
        return torch.tensor(np_array, dtype=self._dtype, device=device)

    def tensor_from_cog_action(self, cog_action, device=None):
        return torch.tensor(cog_action.discrete_action, dtype=self._dtype, device=device)

    @staticmethod
    def cog_action_from_tensor(tensor):
        return AgentAction(discrete_action=tensor.item())

    def _create(
        self,
        model_id,
        observation_size,
        action_count,
        actor_network_hidden_size=64,
        critic_network_hidden_size=64,
        **kwargs,
    ):
        return MuZeroModel(
            model_id=model_id,
            version_number=1,
            actor_network=torch.nn.Sequential(
                torch.nn.Linear(observation_size, actor_network_hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(actor_network_hidden_size, actor_network_hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(actor_network_hidden_size, action_count),
            ).to(self._dtype),
            critic_network=torch.nn.Sequential(
                torch.nn.Linear(observation_size, critic_network_hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(critic_network_hidden_size, critic_network_hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(critic_network_hidden_size, 1),
            ).to(self._dtype),
        )

    def _load(self, model_id, version_number, version_user_data, model_data_f):
        (actor_network, critic_network) = torch.load(model_data_f)
        assert isinstance(actor_network, torch.nn.Sequential)
        assert isinstance(critic_network, torch.nn.Sequential)
        return MuZeroModel(
            model_id=model_id, version_number=version_number, actor_network=actor_network, critic_network=critic_network
        )

    def _save(self, model, model_data_f):
        assert isinstance(model, MuZeroModel)
        torch.save((model.actor_network, model.critic_network), model_data_f)
        return {}

    def _create_actor_implementations(self):
        async def impl(actor_session):
            actor_session.start()

            config = actor_session.config

            model, _ = await self.retrieve_version(config.model_id, config.model_version)

            async for event in actor_session.event_loop():
                if event.observation and event.type == cogment.EventType.ACTIVE:
                    obs = self.tensor_from_cog_obs(event.observation.snapshot)
                    scores = model.actor_network(obs)
                    probs = torch.softmax(scores, dim=-1)
                    action = torch.distributions.Categorical(probs).sample()
                    actor_session.do_action(self.cog_action_from_tensor(action))

        return {
            "muzero_mlp": (impl, ["agent"]),
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
            xp_tracker = MlflowExperimentTracker(run_session.params_name, run_session.run_id)

            # Initializing a model
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
                                    implementation="muzero_mlp",
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
                )
                log.info(
                    f"[{run_session.params_name}/{run_session.run_id}] epoch #{epoch} finished ({total_samples} samples seen)"
                )

        return {
            "muzero_mlp_training": (
                sample_producer_impl,
                run_impl,
                MuZeroTrainingRunConfig(
                    environment=EnvConfig(
                        seed=12, env_type="gym", env_name="CartPole-v0", player_count=1, framestack=1
                    ),
                    training=MuZeroTrainingConfig(
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


class NewMuZeroAgentAdapter(AgentAdapter):
    def __init__(self):
        super().__init__()

    def _create(self, model_id, **kwargs):
        """
        Create and return a model instance
        Parameters:
            model_id (string): unique identifier for the model
            kwargs: any number of key/values paramters
        Returns:
            model: the created model
        """
        raise NotImplementedError

    def _load(self, model_id, version_number, version_user_data, model_data_f):
        """
        Load a serialized model instance and return it
        Args:
            model_id (string): unique identifier for the model
            version_number (int): version number of the data
            version_user_data (dict[str, str]): version user data
            model_data_f: file object that will be used to load the version model data
        Returns:
            model: the loaded model
        """
        raise NotImplementedError

    def _save(self, model, model_data_f):
        """
        Serialize and save a model
        Args:
            model: a model, as returned by method of this class
            model_data_f: file object that will be used to save the version model data
        Returns:
            version_user_data (dict[str, str]): version user data
        """
        raise NotImplementedError

    def _create_actor_implementations(self):
        """
        Create all the available actor implementation for this adapter
        Returns:
            dict[impl_name: string, (actor_impl: Callable, actor_classes: []string)]: key/value definition for the available actor implementations.
        """
        raise NotImplementedError

    def _create_run_implementations(self):
        """
        Create all the available run implementation for this adapter
        Returns:
            dict[impl_name: string, (sample_producer_impl: Callable, run_impl: Callable, default_run_config)]: key/value definition for the available run implementations.
        """
        raise NotImplementedError


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


class MuZeroAgent:
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

        action, policy, value = self.muzero.act(observation)

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

        #
        priority = batch["priority"][:, 0].view(-1)
        self._replay_buffer.update_priorities(list(zip(episode, step, priority)))
        importance_weight = 1 / (priority + 1e-6) / self._replay_buffer.size()
        target_value = torch.clamp(target_value.view(*pred_value.shape), self._params["vmin"], self._params["vmax"])
        target_reward = torch.clamp(reward.view(*pred_value.shape), self._params["rmin"], self._params["rmax"])
        info = self.muzero.train_step()

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
