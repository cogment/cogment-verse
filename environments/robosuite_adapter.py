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

import os

import cogment
import gym

from robosuite.controllers import load_controller_config
from robosuite.utils.input_utils import *

from cogment_verse.specs import (
    encode_rendered_frame,
    EnvironmentSpecs,
    Observation,
    space_from_gym_space,
    gym_action_from_action,
    observation_from_gym_observation,
)
from cogment_verse.constants import PLAYER_ACTOR_CLASS, TEACHER_ACTOR_CLASS

# configure pygame to use a dummy video server to be able to render headlessly
os.environ["SDL_VIDEODRIVER"] = "dummy"

import os
import shutil

import hydra
from omegaconf import OmegaConf
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig


def build_feature_obs_robosuite_env(cfg):
    # Acknowledgement: https://github.com/melfm/simrealjaco
    import robosuite
    from robosuite.wrappers.gym_wrapper import GymWrapper
    robo_cfg = cfg.robosuite.params

    if 'controller_config_file' in robo_cfg.keys():
        control_cfg_file = robo_cfg['controller_config_file']
        assert control_cfg_file.endswith('.json')
        print('json file required')
        control_cfg_filepath = os.path.join(
            os.path.split(robosuite.__file__)[0], 'controllers', 'config',
            control_cfg_file)
        print('loading controller from', control_cfg_filepath)
        controller_configs = robosuite.load_controller_config(
            custom_fpath=control_cfg_filepath)
    else:
        controller_configs = robosuite.load_controller_config(
            default_controller=robo_cfg['controller_name'])
    options = {}

    # Keep a copy of the controller config
    robo_path = os.path.dirname(robosuite.__file__)
    config_path = robo_path + '/controllers/config/'
    config_file = config_path + robo_cfg.controller_config_file
    shutil.copy(config_file, os.getcwd())

    # not all configs are meant to go to robosuite
    for key in robo_cfg.keys():
        if key not in ['controller_name', 'controller_config_file']:
            options[key] = robo_cfg[key]

    # Sanity check to make sure controller_name and controller_config_file
    # are a match.
    if robo_cfg['controller_name'] != controller_configs['type']:
        raise ValueError('Controller spec mismatch!')

    env = robosuite.make(
        controller_configs=controller_configs,
        use_camera_obs=False,
        use_object_obs=True,
        ignore_done=False,
        hard_reset=True,
        has_offscreen_renderer=True,
        has_renderer=False,
        reward_scale=1.0,
        camera_names='frontview',
        **options,
    )

    class CustomRobosuiteWrapper(GymWrapper):

        def render(self, mode=''):
            """ make robosuite compatible with sac video render code """
            # camera_widths is 256
            return self.env.sim.render(
                camera_name=self.env.camera_names[0],
                width=self.env.camera_widths[0],
                height=self.env.camera_heights[0])[::-1, :]

    env = CustomRobosuiteWrapper(env)
    env.max_path_length = robo_cfg.horizon
    env._max_episode_steps = robo_cfg.horizon
    return env


class EnvSampler():
    # Acknowledgement: https://github.com/melfm/simrealjaco
    """Env Class used for sampling environments with different
        simulation parameters.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.env_name = self.cfg.env_name
        # self.env_type = self.cfg.env_type

    def sample_env(self):
        env = build_feature_obs_robosuite_env(self.cfg)
        env.reset()
        env._max_episode_steps = env.max_path_length
        return env


class Environment:
    def __init__(self, cfg):
        self.gym_env_name = cfg.env_name
        print("cfg = ", cfg)
        # print("aftrer config")
        # print()
        # self.work_dir = os.getcwd()
        # print(f'workspace: {self.work_dir}')
        # hydra_cfg = HydraConfig.get()
        # config_file = self.work_dir.split('runs')[0] + 'config/' \
        #               + hydra_cfg.job.config_name + '.yaml'
        # shutil.copy(config_file, self.work_dir)
        #
        # self.cfg = cfg
        self.env = EnvSampler(cfg).sample_env()
        print(" wohooo ")
        # options = {}
        # options["env_name"] = choose_environment() #ToDo: replace with gym_env_name
        # if "TwoArm" in options["env_name"]:
        #     # Choose env config and add it to options
        #     options["env_configuration"] = choose_multi_arm_config()
        #
        #     # If chosen configuration was bimanual, the corresponding robot must be Baxter. Else, have user choose robots
        #     if options["env_configuration"] == "bimanual":
        #         options["robots"] = "Baxter"
        #     else:
        #         options["robots"] = []
        #
        #         # Have user choose two robots
        #         print("A multiple single-arm configuration was chosen.\n")
        #
        #         for i in range(2):
        #             print("Please choose Robot {}...\n".format(i))
        #             options["robots"].append(choose_robots(exclude_bimanual=True))
        #
        #     # Else, we simply choose a single (single-armed) robot to instantiate in the environment
        # else:
        #     options["robots"] = choose_robots(exclude_bimanual=True)
        #
        #     # Choose controller
        # controller_name = choose_controller()
        #
        # # Load the desired controller
        # options["controller_configs"] = load_controller_config(default_controller=controller_name)
        #
        # # Help message to user
        # print()
        # print('Press "H" to show the viewer control panel.')
        #
        # # initialize the task
        # env = suite.make(
        #     **options,
        #     has_renderer=True,
        #     has_offscreen_renderer=False,
        #     ignore_done=True,
        #     use_camera_obs=False,
        #     control_freq=20,
        # )
        # env.reset()
        # env.viewer.set_camera(camera_id=0)
        # print(" ****** env ****** ")
        # print(env)
        # print(" ****** env ****** ")

        # gym_env = gym.make(self.gym_env_name)
        self.env_specs = EnvironmentSpecs(
            num_players=1,
            turn_based=False,
            observation_space=space_from_gym_space(self.env.observation_space),
            action_space=space_from_gym_space(self.env.action_space),
        )

    def get_implementation_name(self):
        return self.gym_env_name

    def get_environment_specs(self):
        return self.env_specs

    async def impl(self, environment_session):
        actors = environment_session.get_active_actors()
        player_actors = [
            (actor_idx, actor.actor_name)
            for (actor_idx, actor) in enumerate(actors)
            if actor.actor_class_name == PLAYER_ACTOR_CLASS
        ]
        assert len(player_actors) == 1
        [(player_actor_idx, player_actor_name)] = player_actors

        teacher_actors = [
            (actor_idx, actor.actor_name)
            for (actor_idx, actor) in enumerate(actors)
            if actor.actor_class_name == TEACHER_ACTOR_CLASS
        ]
        assert len(teacher_actors) <= 1
        has_teacher = len(teacher_actors) == 1
        if has_teacher:
            [(teacher_actor_idx, _teacher_actor_name)] = teacher_actors

        session_cfg = environment_session.config

        gym_env = gym.make(self.gym_env_name, render_mode="single_rgb_array" if session_cfg.render else None)

        gym_observation, _info = gym_env.reset(seed=session_cfg.seed, return_info=True)
        observation_value = observation_from_gym_observation(gym_env.observation_space, gym_observation)

        rendered_frame = None
        if session_cfg.render:
            rendered_frame = encode_rendered_frame(gym_env.render(), session_cfg.render_width)

        environment_session.start([("*", Observation(value=observation_value, rendered_frame=rendered_frame))])

        async for event in environment_session.all_events():
            if event.actions:
                player_action_value = event.actions[player_actor_idx].action.value
                action_value = player_action_value
                overridden_players = []
                if has_teacher and event.actions[teacher_actor_idx].action.HasField("value"):
                    teacher_action_value = event.actions[teacher_actor_idx].action.value
                    action_value = teacher_action_value
                    overridden_players = [player_actor_name]

                gym_action = gym_action_from_action(
                    self.env_specs.action_space, action_value  # pylint: disable=no-member
                )

                gym_observation, reward, done, _info = gym_env.step(gym_action)
                observation_value = observation_from_gym_observation(gym_env.observation_space, gym_observation)

                rendered_frame = None
                if session_cfg.render:
                    rendered_frame = encode_rendered_frame(gym_env.render(), session_cfg.render_width)

                observations = [
                    (
                        "*",
                        Observation(
                            value=observation_value,
                            rendered_frame=rendered_frame,
                            overridden_players=overridden_players,
                        ),
                    )
                ]

                if reward is not None:
                    environment_session.add_reward(
                        value=reward,
                        confidence=1.0,
                        to=[player_actor_name],
                    )

                if done:
                    # The trial ended
                    environment_session.end(observations)
                elif event.type != cogment.EventType.ACTIVE:
                    # The trial termination has been requested
                    environment_session.end(observations)
                else:
                    # The trial is active
                    environment_session.produce_observations(observations)

        gym_env.close()
