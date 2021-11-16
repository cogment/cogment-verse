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

import cog_settings

from cogment_verse_environment.factory import make_environment
from data_pb2 import Observation, NDArray

import numpy as np

import cv2

import cogment

from dotenv import load_dotenv

import asyncio
import logging
import os

load_dotenv()

PORT = int(os.getenv("COGMENT_VERSE_ENVIRONMENT_PORT", "9000"))
PROMETHEUS_PORT = int(os.getenv("COGMENT_VERSE_ENVIRONMENT_PROMETHEUS_PORT", "8000"))

logging.basicConfig(level=logging.INFO)

log = logging.getLogger(__name__)


def np_array_from_proto_array(arr):
    return np.frombuffer(arr.data, dtype=arr.dtype).reshape(*arr.shape)


def proto_array_from_np_array(arr):
    # arr = np.array(arr)
    return NDArray(shape=arr.shape, dtype=str(arr.dtype), data=arr.tobytes())


def img_encode(img):
    # note rgb -> bgr for cv2
    result, data = cv2.imencode(".jpg", img[:, :, ::-1])
    assert result
    return data.tobytes()


def gym_action_from_cog_action(cog_action):
    which_action = cog_action.WhichOneof("action")
    if which_action == "continuous_action":
        return cog_action.continuous_action.data
    # else
    return getattr(cog_action, cog_action.WhichOneof("action"))


def cog_obs_from_gym_obs(gym_obs, pixels, current_player, legal_moves_as_int, player_override=-1):
    cog_obs = Observation(
        vectorized=proto_array_from_np_array(gym_obs),
        legal_moves_as_int=legal_moves_as_int,
        current_player=current_player,
        player_override=player_override,
        pixel_data=img_encode(pixels),
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


async def environment(environment_session):
    print("************Envronment !!! ")
    log.info(f" LOG ************------>>>   Envronment !!! ")
    actors = environment_session.get_active_actors()

    env_config = environment_session.config
    env_kwargs = {
        "num_players": environment_session.config.player_count,
        "flatten": env_config.flatten,  # todo: we should have env_kwargs in proto
        "framestack": env_config.framestack,
    }
    print("env_kwargs: ", env_kwargs)
    print("make_environment", env_config.env_name)
    log.warning(f"make_environment {env_config.env_type}  ---    {env_config.env_name} ")
    env = make_environment(env_config.env_type, env_config.env_name, **env_kwargs)
    env.seed(env_config.seed)

    print("Env: ", env)

    max_size = env_config.render_width or 256

    steerable = False
    teacher_idx = -1

    # If there is an extra player, it must be the teacher/expert
    # and it must be the _last_ player this is only supported form of HILL at the moment)
    # todo: Make this more general and configurable (via TrialConfig)
    if len(actors) != env_kwargs["num_players"]:
        log.debug(len(actors), env_kwargs["num_players"])
        assert len(actors) == env_kwargs["num_players"] + 1
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

    # Hack here
    render = False
    if render:
        pixels = shrink_image(env.render(mode="rgb_array"), max_size)
    else:
        pixels = np.array([[[0, 0, 0]]], dtype=np.uint8)

    assert len(pixels.tobytes()) == np.prod(pixels.shape)

    print("before Cog Obs ", gym_obs.observation)
    cog_obs = cog_obs_from_gym_obs(gym_obs.observation, pixels, gym_obs.current_player, gym_obs.legal_moves_as_int)
    print("after Cog Obs ", cog_obs)
    environment_session.start([("*", cog_obs)])
    print("start session ")
    async for event in environment_session.event_loop():
        print("event ", event)
        if event.actions:
            player_override = -1
            # special handling of human intervention
            print(steerable, " and ", event.actions[teacher_idx].action.discrete_action != -1)
            if steerable and event.actions[teacher_idx].action.discrete_action != -1:
                gym_action = gym_action_from_cog_action(event.actions[teacher_idx].action)
                player_override = teacher_idx
                current_player = teacher_idx
            else:
                print("Else action")
                gym_action = gym_action_from_cog_action(event.actions[gym_obs.current_player].action)
                current_player = gym_obs.current_player

            print("gym action ", gym_action)
            gym_action = np.array(gym_action).reshape(act_shape)
            print("gym action reshaped ", gym_action)
            gym_obs = env.step(gym_action)
            print("gym obs ", gym_obs)
            if render:
                print("render")
                pixels = shrink_image(env.render(mode="rgb_array"), max_size)
                if player_override != -1:
                    pixels = draw_border(pixels)

            assert len(pixels.tobytes()) == np.prod(pixels.shape)

            print("sending rewards")
            for idx, reward in enumerate(gym_obs.rewards):
                if player_override != -1 and idx == current_player:
                    environment_session.add_reward(
                        value=reward,
                        confidence=1.0,
                        to=[actors[player_override].actor_name],
                    )
                else:
                    environment_session.add_reward(value=reward, confidence=1.0, to=[actors[idx].actor_name])

            cog_obs = cog_obs_from_gym_obs(
                gym_obs.observation,
                pixels,
                gym_obs.current_player,
                gym_obs.legal_moves_as_int,
                player_override=player_override,
            )
            observations = [("*", cog_obs)]

            print("sending cog obs ", cog_obs)

            if environment_session.get_tick_id() >= 1e1000 or gym_obs.done:
                log.debug(
                    f"[Environment] ending trial [{environment_session.get_trial_id()}] @ tick #{environment_session.get_tick_id()}..."
                )
                environment_session.end(observations=observations)
            else:
                environment_session.produce_observations(observations=observations)

    env.close()


async def main():
    context = cogment.Context(cog_settings=cog_settings, user_id="cogment_verse_environment")
    context.register_environment(impl=environment)
    log.info(f"Environment service starting on port {PORT}...")
    await context.serve_all_registered(cogment.ServedEndpoint(port=PORT), prometheus_port=PROMETHEUS_PORT)


if __name__ == "__main__":
    asyncio.run(main())
