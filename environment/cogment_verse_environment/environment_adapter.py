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

import logging

import cv2
import numpy as np
from cogment_verse_environment.atari import AtariEnv
from cogment_verse_environment.gym_env import GymEnv
from cogment_verse_environment.minatarenv import MinAtarEnv
from cogment_verse_environment.procgen_env import ProcGenEnv
from cogment_verse_environment.tetris import TetrisEnv
from cogment_verse_environment.utils.serialization_helpers import serialize_img, serialize_np_array
from cogment_verse_environment.zoo_env import PettingZooEnv
from cogment_verse_environment.pybullet_driving import DrivingEnv
from data_pb2 import Observation

ENVIRONMENT_CONSTRUCTORS = {
    "gym": GymEnv,
    "atari": AtariEnv,
    "minatar": MinAtarEnv,
    "tetris": TetrisEnv,
    "pettingzoo": PettingZooEnv,
    "procgen": ProcGenEnv,
    "driving": DrivingEnv,
}

log = logging.getLogger(__name__)


def gym_action_from_cog_action(cog_action):
    which_action = cog_action.WhichOneof("action")
    if which_action == "continuous_action":
        return cog_action.continuous_action.data
    # else
    return getattr(cog_action, cog_action.WhichOneof("action"))


def cog_obs_from_gym_obs(gym_obs, pixels, current_player, legal_moves_as_int, player_override=-1):
    cog_obs = Observation(
        vectorized=serialize_np_array(gym_obs),
        legal_moves_as_int=legal_moves_as_int,
        current_player=current_player,
        player_override=player_override,
        pixel_data=serialize_img(pixels),
    )
    return cog_obs


# pylint: disable=dangerous-default-value
def draw_border(pixels, width=10, color=[255, 0, 0], inplace=True):
    if not inplace:
        pixels = np.array(pixels, copy=True)

    color = np.array(color, dtype=np.uint8).reshape(1, 1, 3)
    pixels[:width, :, :] = color
    pixels[-width:, :, :] = color
    pixels[:, :width, :] = color
    pixels[:, -width:, :] = color

    return pixels


def shrink_image(pixels, max_size):
    # GRPC max message size hack
    height, width = pixels.shape[:2]
    if max(height, width) > max_size:
        if height > width:
            new_height = max_size
            new_width = int(new_height / height * width)
        else:
            new_width = max_size
            new_height = int(height / width * new_width)
        pixels = cv2.resize(pixels, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return pixels


class EnvironmentAdapter:
    def __init__(self):
        self._environments = [
            "atari/Breakout",
            "atari/Pitfall",
            "atari/TetrisALE",
            "gym/BipedalWalker-v3",
            "gym/CartPole-v0",
            "gym/LunarLander-v2",
            "gym/MountainCar-v0",
            "gym/LunarLanderContinuous-v2",
            "gym/Pendulum-v0",
            "minatar/breakout",
            "pettingzoo/backgammon_v3",
            "pettingzoo/connect_four_v3",
            "tetris/TetrisA-v0",
            "procgen/bigfish",
            "procgen/bossfight",
            "procgen/caveflyer",
            "procgen/chaser",
            "procgen/climber",
            "procgen/coinrun",
            "procgen/dodgeball",
            "procgen/fruitbot",
            "procgen/heist",
            "procgen/jumper",
            "procgen/leaper",
            "procgen/maze",
            "procgen/miner",
            "procgen/ninja",
            "procgen/plunder",
            "procgen/starpilot",
            "driving/SimpleDriving-v0",
        ]

    def _create_implementations(self):
        def create_implementation(env_impl_name):
            [env_type, env_name] = env_impl_name.split("/", maxsplit=1)
            if env_type not in ENVIRONMENT_CONSTRUCTORS:
                raise RuntimeError(f"Unknown environment [{env_type}/...]")

            def make_environment(env_config):
                env = ENVIRONMENT_CONSTRUCTORS[env_type](
                    env_type=env_type,
                    env_name=env_name,
                    flatten=env_config.flatten,
                    framestack=env_config.framestack,
                    mode=env_config.mode,
                )
                env.seed(env_config.seed)
                return env

            async def environment_implementation(environment_session):
                actors = environment_session.get_active_actors()

                env_config = environment_session.config
                env = make_environment(env_config)

                max_size = env_config.render_width or 256

                steerable = False
                teacher_idx = -1

                # If there is an extra player, it must be the teacher/expert
                # and it must be the _last_ player this is only supported form of HILL at the moment)
                # todo: Make this more general and configurable (via TrialConfig)
                if len(actors) != env.num_players:
                    log.debug(len(actors), env.num_players)
                    assert len(actors) == env.num_players + 1
                    for idx, actor in enumerate(actors):
                        log.debug(idx, actor.actor_name, actor.actor_class_name)
                    steerable = True
                    teacher_idx = len(actors) - 1
                    assert actors[teacher_idx].actor_class_name == "teacher_agent"

                env_spec = env.env_spec
                assert len(env_spec.act_dim) == 1, "only a single action space is currently supported"
                act_shape = env_spec.act_shape[0]

                gym_obs = env.reset()
                render = environment_session.config.render

                if render:
                    pixels = shrink_image(env.render(mode="rgb_array"), max_size)
                else:
                    pixels = np.array([[[0, 0, 0]]], dtype=np.uint8)

                assert len(pixels.tobytes()) == np.prod(pixels.shape)

                cog_obs = cog_obs_from_gym_obs(
                    gym_obs.observation, pixels, gym_obs.current_player, gym_obs.legal_moves_as_int
                )
                environment_session.start([("*", cog_obs)])

                async for event in environment_session.event_loop():
                    if event.actions:
                        player_override = -1
                        # special handling of human intervention
                        if steerable and event.actions[teacher_idx].action.discrete_action != -1:
                            gym_action = gym_action_from_cog_action(event.actions[teacher_idx].action)
                            player_override = teacher_idx
                            current_player = teacher_idx
                        else:
                            gym_action = gym_action_from_cog_action(event.actions[gym_obs.current_player].action)
                            current_player = gym_obs.current_player

                        gym_action = np.array(gym_action).reshape(act_shape)
                        gym_obs = env.step(gym_action)
                        if render:
                            pixels = shrink_image(env.render(mode="rgb_array"), max_size)
                            if player_override != -1:
                                pixels = draw_border(pixels)

                        assert len(pixels.tobytes()) == np.prod(pixels.shape)

                        for idx, reward in enumerate(gym_obs.rewards):
                            if player_override != -1 and idx == current_player:
                                environment_session.add_reward(
                                    value=reward,
                                    confidence=1.0,
                                    to=[actors[player_override].actor_name],
                                )
                            else:
                                environment_session.add_reward(
                                    value=reward, confidence=1.0, to=[actors[idx].actor_name]
                                )

                        cog_obs = cog_obs_from_gym_obs(
                            gym_obs.observation,
                            pixels,
                            gym_obs.current_player,
                            gym_obs.legal_moves_as_int,
                            player_override=player_override,
                        )
                        observations = [("*", cog_obs)]

                        if environment_session.get_tick_id() >= 1e1000 or gym_obs.done:
                            log.debug(
                                f"[Environment] ending trial [{environment_session.get_trial_id()}] @ tick #{environment_session.get_tick_id()}..."
                            )
                            environment_session.end(observations)
                        else:
                            environment_session.produce_observations(observations=observations)

                env.close()

            return environment_implementation

        return {env_impl_name: create_implementation(env_impl_name) for env_impl_name in self._environments}

    def register_implementations(self, context):
        """
        Register all the implementations defined in this adapter
        Parameters:
            context: Cogment context with which the implementations are adapted
        """
        for env_impl_name, env_impl in self._create_implementations().items():
            log.info(f"Registering environment implementation [{env_impl_name}]")
            context.register_environment(impl=env_impl, impl_name=env_impl_name)
