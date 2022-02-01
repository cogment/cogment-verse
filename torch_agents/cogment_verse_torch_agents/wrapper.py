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

import cv2
import numpy as np
from data_pb2 import AgentAction, ContinuousAction, Observation, NDArray

# TODO directly use torch tensors


def np_array_from_proto_array(arr):
    dtype = arr.dtype or "int8"  # default type for empty array
    return np.frombuffer(arr.data, dtype=dtype).reshape(*arr.shape)


def proto_array_from_np_array(arr):
    # arr = np.array(arr)
    return NDArray(shape=arr.shape, dtype=str(arr.dtype), data=arr.tobytes())


def img_encode(img):
    # note rgb -> bgr for cv2
    result, data = cv2.imencode(".jpg", img[:, :, ::-1])
    assert result
    return data.tobytes()


def torch_action_from_cog_action(cog_action):
    which_action = cog_action.WhichOneof("action")
    if which_action == "continuous_action":
        return cog_action.continuous_action.data
    # else
    return getattr(cog_action, cog_action.WhichOneof("action"))


def torch_obs_from_cog_obs(cog_obs):
    torch_obs = {}
    torch_obs["current_player"] = cog_obs.current_player
    torch_obs["legal_moves_as_int"] = cog_obs.legal_moves_as_int
    torch_obs["vectorized"] = np_array_from_proto_array(cog_obs.vectorized)
    img = cv2.imdecode(np.frombuffer(cog_obs.pixel_data, dtype=np.uint8), cv2.IMREAD_COLOR)
    torch_obs["pixels"] = img[:, :, ::-1]  # bgr to rgb
    return torch_obs


def format_legal_moves(legal_moves, action_dim):
    """Returns formatted legal moves.
    This function takes a list of actions and converts it into a fixed size vector
    of size action_dim. If an action is legal, its position is set to 0 and -Inf
    otherwise.
    Ex: legal_moves = [0, 1, 3], action_dim = 5
        returns [0, 0, -Inf, 0, -Inf]
    Args:
      legal_moves: list of legal actions.
      action_dim: int, number of actions.
    Returns:
      a vector of size action_dim.
    """
    if legal_moves:
        new_legal_moves = np.full(action_dim, -float("inf"))
        new_legal_moves[legal_moves] = 0
    else:
        # special case: if passed list is empty, assume there are no move constraints
        new_legal_moves = np.full(action_dim, 0.0)

    return new_legal_moves


def cog_obs_from_gym_obs(gym_obs, pixels, current_player, legal_moves_as_int, player_override=-1):
    cog_obs = Observation(
        vectorized=proto_array_from_np_array(gym_obs),
        legal_moves_as_int=legal_moves_as_int,
        current_player=current_player,
        player_override=player_override,
        pixel_data=img_encode(pixels),
    )
    return cog_obs


def cog_action_from_torch_action(action):
    # todo: move this logic? or make it more robust
    if isinstance(action, np.ndarray):
        dtype = action.dtype
    elif isinstance(action, list):
        dtype = type(action[0])
    else:
        dtype = type(action)

    if dtype in (int, np.int32, np.int64):
        field = "discrete_action"
        kwargs = {field: action}
        return AgentAction(**kwargs)

    # else
    agent_action = AgentAction(continuous_action=ContinuousAction())
    action = np.squeeze(action)
    if action.shape == ():
        agent_action.continuous_action.data.append(action)
    else:
        for act in action:
            agent_action.continuous_action.data.append(act)

    return agent_action
