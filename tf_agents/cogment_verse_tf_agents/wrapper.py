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
from data_pb2 import AgentAction, ContinuousAction

# TODO directly use tf tensors


def np_array_from_proto_array(arr):
    return np.frombuffer(arr.data, dtype=arr.dtype).reshape(*arr.shape)


def tf_action_from_cog_action(cog_action):
    which_action = cog_action.WhichOneof("action")
    if which_action == "continuous_action":
        return cog_action.continuous_action.data
    # else
    return getattr(cog_action, cog_action.WhichOneof("action"))


def tf_obs_from_cog_obs(cog_obs):
    tf_obs = {}
    tf_obs["current_player"] = cog_obs.current_player
    tf_obs["legal_moves_as_int"] = cog_obs.legal_moves_as_int
    tf_obs["vectorized"] = np_array_from_proto_array(cog_obs.vectorized)
    img = cv2.imdecode(np.frombuffer(cog_obs.pixel_data, dtype=np.uint8), cv2.IMREAD_COLOR)
    tf_obs["pixels"] = img[:, :, ::-1]  # bgr to rgb
    return tf_obs


def cog_action_from_tf_action(action):
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
