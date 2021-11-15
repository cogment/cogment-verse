from abc import ABC
from cogment_verse_torch_agents.third_party.hive.utils.registry import CallableType


class FunctionApproximator(CallableType):
    @classmethod
    def type_name(cls):
        return "function"
