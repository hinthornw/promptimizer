from promptim.optimizers.fewshot import FewShotOptimizer
from promptim.optimizers.metaprompt import MetaPromptOptimizer
from promptim.optimizers.feedback_guided import FeedbackGuidedOptimizer
from promptim.optimizers.base import BaseOptimizer

# Use the config_cls.kind.default to get the map keys

_MAP = {
    "metaprompt": MetaPromptOptimizer,
    "fewshot": FewShotOptimizer,
    "feedback_guided": FeedbackGuidedOptimizer,
}


def load_optimizer(config: dict) -> BaseOptimizer:
    """Load an optimizer from its config dictionary."""
    kind = config["kind"]
    if kind not in _MAP:
        raise ValueError(
            f"Unknown optimizer kind: {kind}. Available kinds: {list(_MAP.keys())}"
        )

    return _MAP[kind].from_config(config)


__all__ = [
    "MetaPromptOptimizer",
    "FewShotOptimizer",
    "FeedbackGuidedOptimizer",
    "BaseOptimizer",
    "load_optimizer",
]
