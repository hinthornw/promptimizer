from dataclasses import dataclass, field
from typing import List, Literal, Callable

from langsmith.evaluation._arunner import ExperimentResultRow
from promptim import _utils as pm_utils
from promptim import types as pm_types
from promptim.optimizers import base as optimizers


@dataclass(kw_only=True)
class Config(optimizers.Config):
    """Configuration for the few-shot optimization algorithm."""

    kind: Literal["fewshot"] = field(
        default="fewshot",
        metadata={
            "description": "The fewshot optimizer that selects few-shot examples and inserts them into the prompt."
        },
    )
    few_shot_selector: dict = field(
        metadata={"description": "Configuration for the few-shot example selector."}
    )


class FewShotOptimizer(optimizers.BaseOptimizer):
    """
    A simple example of an algorithm that selects few-shot examples and inserts them into the prompt.
    This might integrate with a separate FewShotSelector class.
    """

    config_cls = Config

    def __init__(
        self,
        *,
        model: optimizers.MODEL_TYPE | None = None,
        few_shot_selector: Callable | None = None,
    ):
        super().__init__(model=model)
        self.few_shot_selector = few_shot_selector

    async def improve_prompt(
        self,
        current_prompt: pm_types.PromptWrapper,
        results: List[ExperimentResultRow],
        task: pm_types.Task,
        other_attempts: List[pm_types.PromptWrapper],
    ) -> pm_types.PromptWrapper:
        raise NotImplementedError("The few-shot optimizer is not yet implemented.")
