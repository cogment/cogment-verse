from cogment_verse_torch_agents.third_party.hive.replays.circular_replay import CircularReplayBuffer, SimpleReplayBuffer
from cogment_verse_torch_agents.third_party.hive.replays.legal_moves_replay import LegalMovesBuffer
from cogment_verse_torch_agents.third_party.hive.replays.prioritized_replay import PrioritizedReplayBuffer
from cogment_verse_torch_agents.third_party.hive.replays.replay_buffer import BaseReplayBuffer
from cogment_verse_torch_agents.third_party.hive.utils.registry import registry

registry.register_all(
    BaseReplayBuffer,
    {
        "CircularReplayBuffer": CircularReplayBuffer,
        "SimpleReplayBuffer": SimpleReplayBuffer,
        "PrioritizedReplayBuffer": PrioritizedReplayBuffer,
        "LegalMovesBuffer": LegalMovesBuffer,
    },
)

get_replay = getattr(registry, f"get_{BaseReplayBuffer.type_name()}")
