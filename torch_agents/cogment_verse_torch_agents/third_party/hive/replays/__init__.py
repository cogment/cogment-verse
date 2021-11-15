from cogment_verse_torch_agents.third_party.hive.utils import registry
from cogment_verse_torch_agents.third_party.hive.replays.efficient_replay import EfficientCircularBuffer
from cogment_verse_torch_agents.third_party.hive.replays.prioritized_replay import PrioritizedReplayBuffer
from cogment_verse_torch_agents.third_party.hive.replays.replay_buffer import BaseReplayBuffer, CircularReplayBuffer
from cogment_verse_torch_agents.third_party.hive.replays.hanabi_buffer import HanabiBuffer

registry = registry.Registry()
registry.register_all(
    BaseReplayBuffer,
    {
        "CircularReplayBuffer": CircularReplayBuffer,
        "EfficientCircularBuffer": EfficientCircularBuffer,
        "PrioritizedReplayBuffer": PrioritizedReplayBuffer,
        "HanabiBuffer": HanabiBuffer,
    },
)

get_replay = getattr(registry, f"get_{BaseReplayBuffer.type_name()}")
