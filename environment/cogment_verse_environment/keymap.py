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

from cogment_verse_environment.factory import make_environment


def get_pressed_keys(key_state):
    keys = []
    for code, pressed in key_state.items():
        if pressed:
            keys.append(code)
    keys.sort()
    return tuple(keys)


def get_keymap(env_type, env_name):
    # pylint: disable=import-outside-toplevel
    from pyglet.window import key

    minatar_keymap = {
        (): 0,  # NOp
        (key.A,): 1,  # Left
        (key.W,): 2,  # Up
        (key.D,): 3,  # Right
        (key.S,): 4,  # Down
        (key.SPACE,): 5,  # Fire
    }
    keymaps = {
        "LunarLander-v2": {
            (): 0,  # nop
            (key.A,): 1,  # left
            (key.S,): 2,  # right
            (key.D,): 3,  # right
        },
        "CartPole-v0": {
            (key.A,): 0,  # accelerate left
            (key.D,): 1,  # accelerate right
        },
        # specific atari games
        # "Breakout": {
        #    (): 0, # Nop
        #    (key.SPACE,): 1, # Fire
        #    (key.D,): 2, # Right
        #    (key.A,): 3, # Left
        # },
        # minatar environments
        "asterix": minatar_keymap,
        "breakout": minatar_keymap,
        "freeway": minatar_keymap,
        "seaquest": minatar_keymap,
        "space_invaders": minatar_keymap,
        # Petting Zoo
        "connect_four_v3": {
            (key.NUM_1,): 0,
            (key.NUM_2,): 1,
            (key.NUM_3,): 2,
            (key.NUM_4,): 3,
            (key.NUM_5,): 4,
            (key.NUM_6,): 5,
            (key.NUM_7,): 6,
        },
        "backgammon_v3": {},
    }

    env = make_environment(env_type, env_name)
    keymap = None
    # pylint: disable=protected-access
    if hasattr(env, "_env"):
        if hasattr(env._env, "get_keys_to_action"):
            keymap = env._env.get_keys_to_action()

    # For some environments (e.g. CartPole) the keys_to_action method returns None?!
    if keymap is None:
        keymap = keymaps[env_name]

    # Special case for steerable agents: if there is an explicit NOP, we should
    # also map it to another key so that the expert can send NOP actions
    if () in keymap:
        nop_action = keymap[()]
        keymap[(key.ESCAPE,)] = nop_action

    return keymap
