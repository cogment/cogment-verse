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
from data_pb2 import AgentAction


def tensor_from_cog_obs(cog_obs, device=None, dtype=torch.float):
    pb_array = cog_obs.vectorized
    np_array = np.frombuffer(pb_array.data, dtype=pb_array.dtype).reshape(*pb_array.shape)
    return torch.tensor(np_array, dtype=dtype, device=device)


def tensor_from_cog_action(cog_action, device=None, dtype=torch.long):
    # TODO also support continuous actions and merge with wrapper.torch_action_from_cog_action
    return torch.tensor(cog_action.discrete_action, dtype=dtype, device=device)


def cog_action_from_tensor(tensor):
    # TODO also support continuous actions and merge with wrapper.cog_action_from_torch_action
    return AgentAction(discrete_action=tensor.item())
