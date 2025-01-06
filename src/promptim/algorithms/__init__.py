from promptim.algorithms.base import BaseAlgorithm
from promptim.algorithms.minibatch import MinibatchAlgorithm
from promptim.algorithms.phaseevo import PhaseEvoAlgorithm
from langchain_core.language_models import BaseChatModel

_MAP = {
    "minibatch": MinibatchAlgorithm,
    "phaseevo": PhaseEvoAlgorithm,
}


def load_algorithm(config: dict, optimizer_model: BaseChatModel) -> BaseAlgorithm:
    """Load an algorithm from its config dictionary."""
    kind = config["kind"]
    if kind not in _MAP:
        raise ValueError(
            f"Unknown algorithm kind: {kind}. Available kinds: {list(_MAP.keys())}"
        )

    return _MAP[kind].from_config({**config, "model": optimizer_model})


__all__ = ["MinibatchAlgorithm", "PhaseEvoAlgorithm"]
