from ..utils.registry import registry
from ..agents.agent import Agent
from ..agents.dqn import DQNAgent
from ..agents.rainbow import RainbowDQNAgent
from ..agents.random import RandomAgent
from ..agents.legal_moves_rainbow import LegalMovesRainbowAgent


registry.register_all(
    Agent,
    {
        "DQNAgent": DQNAgent,
        "RandomAgent": RandomAgent,
        "RainbowDQNAgent": RainbowDQNAgent,
        "LegalMovesRainbowAgent": LegalMovesRainbowAgent,
    },
)

get_agent = getattr(registry, f"get_{Agent.type_name()}")
