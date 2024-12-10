from dataclasses import dataclass, field
from typing import Optional, Union


# Import optimizer configs
from promptim.optimizers.metaprompt import Config as MetaPromptConfig
from promptim.optimizers.fewshot import Config as FewShotConfig
from promptim.optimizers.feedback_guided import Config as FeedbackGuidedConfig
from promptim.types import TaskLike


OptimizerConfig = Union[MetaPromptConfig, FewShotConfig, FeedbackGuidedConfig]


@dataclass(kw_only=True)
class Config(TaskLike):
    optimizer: OptimizerConfig | None = field(
        default=None,
        metadata={
            "description": "Optimization configuration specifying model settings and hyperparameters. If None, default configuration will be used."
        },
    )
    evaluators: str = field(
        metadata={
            "description": (
                "Import path to evaluator functions in format 'file_path:variable_name'. The functions should evaluate prompt quality.\n\n"
                "Example:\n    ./task/evaluators.py:evaluators"
            )
        }
    )
    system: Optional[str] = field(
        default=None,
        metadata={
            "description": (
                "Import path to system configuration in format 'file_path:variable_name'. Defines how prompts are executed.\n\n"
                "Example:\n    ./task/my_system.py:chain"
            )
        },
    )
