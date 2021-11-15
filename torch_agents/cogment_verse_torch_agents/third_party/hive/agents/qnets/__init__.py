from cogment_verse_torch_agents.third_party.hive.utils import registry
from cogment_verse_torch_agents.third_party.hive.agents.qnets.atari import NatureAtariDQNModel
from cogment_verse_torch_agents.third_party.hive.agents.qnets.base import FunctionApproximator
from cogment_verse_torch_agents.third_party.hive.agents.qnets.conv import ConvNetwork
from cogment_verse_torch_agents.third_party.hive.agents.qnets.mlp import MLPNetwork

registry = registry.Registry()
registry.register_all(
    FunctionApproximator,
    {
        "MLPNetwork": FunctionApproximator(MLPNetwork),
        "ConvNetwork": FunctionApproximator(ConvNetwork),
        "NatureAtariDQNModel": FunctionApproximator(NatureAtariDQNModel),
    },
)

get_qnet = getattr(registry, f"get_{FunctionApproximator.type_name()}")
