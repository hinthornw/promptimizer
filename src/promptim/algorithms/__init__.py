from promptim.algorithms.base import BaseAlgorithm
from promptim.algorithms.minibatch import MinibatchAlgorithm
from promptim.algorithms.phaseevo import PhaseEvoAlgorithm
from promptim.algorithms.mipro import MIPROAlgorithm
from langchain_core.language_models import BaseChatModel

_MAP = {
    "minibatch": MinibatchAlgorithm,
    "phaseevo": PhaseEvoAlgorithm,
    "mipro": MIPROAlgorithm,
}


def load_algorithm(config: dict, optimizer_model: BaseChatModel) -> BaseAlgorithm:
    """Load an algorithm from its config dictionary."""
    config = config.copy()
    kind = config.pop("kind", "minibatch")
    if kind not in _MAP:
        raise ValueError(
            f"Unknown algorithm kind: {kind}. Available kinds: {list(_MAP.keys())}"
        )

    return _MAP[kind](config, optimizer_model)


__all__ = [
    "MinibatchAlgorithm",
    "PhaseEvoAlgorithm",
    "MIPROAlgorithm",
    "load_algorithm",
]
