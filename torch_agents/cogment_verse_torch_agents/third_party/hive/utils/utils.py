import inspect
import os
import pickle
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import torch

from cogment_verse_torch_agents.third_party.hive.utils import registry
from cogment_verse_torch_agents.third_party.hive.utils.registry import CallableType, Registry
from torch import optim
registry = Registry()

def legal_moves_adapter(cls):
    class _Adapted(cls):
        def forward(self, *args, legal_moves=None, **kwargs):
            val = super().forward(*args, **kwargs)
            if legal_moves is not None:
                val += legal_moves
            return val

    return _Adapted


def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def numpify(t):
    if isinstance(t, np.ndarray):
        return t
    elif isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    else:
        return np.array(t)


class Chomp(dict):
    """An extension of the dictionary class that allows for accessing through dot
    notation and easy saving/loading.
    """

    def __getattr__(self, k):
        if k not in self:
            raise AttributeError()
        return self.__getitem__(k)

    def __setattr__(self, k, v):
        self.__setitem__(k, v)

    def save(self, filename):
        pickle.dump(self, open(filename, "wb"))

    def load(self, filename):
        self.clear()
        self.update(pickle.load(open(filename, "rb")))


class OptimizerFn(CallableType):
    @classmethod
    def type_name(cls):
        return "optimizer_fn"


registry.register_all(
    OptimizerFn,
    {
        "Adadelta": OptimizerFn(optim.Adadelta),
        "Adagrad": OptimizerFn(optim.Adagrad),
        "Adam": OptimizerFn(optim.Adam),
        "Adamax": OptimizerFn(optim.Adamax),
        "AdamW": OptimizerFn(optim.AdamW),
        "ASGD": OptimizerFn(optim.ASGD),
        "LBFGS": OptimizerFn(optim.LBFGS),
        "RMSprop": OptimizerFn(optim.RMSprop),
        "Rprop": OptimizerFn(optim.Rprop),
        "SGD": OptimizerFn(optim.SGD),
        "SparseAdam": OptimizerFn(optim.SparseAdam),
    },
)

get_optimizer_fn = getattr(registry, f"get_{OptimizerFn.type_name()}")
