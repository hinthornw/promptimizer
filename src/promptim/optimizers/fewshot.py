from dataclasses import dataclass, field
from typing import List, Literal, Callable

from langsmith.evaluation._arunner import ExperimentResultRow
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
    max_k: int = field(
        default=32,
        metadata={"description": "Maximum number of few-shot examples to select."},
    )
    score_threshold: float = field(
        default=0.8,
        metadata={"description": "Threshold for passing examples."},
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
        max_k: int = 32,
        few_shot_selector: Callable | None = None,
        score_threshold: float = 0.8,
    ):
        super().__init__(model=model)
        self.max_k = max_k
        self.few_shot_selector = few_shot_selector or _default_few_shot_selector
        self.score_threshold = score_threshold

    async def improve_prompt(
        self,
        current_prompt: pm_types.PromptWrapper,
        results: List[ExperimentResultRow],
        task: pm_types.Task,
        other_attempts: List[pm_types.PromptWrapper],
    ) -> pm_types.PromptWrapper:
        raise NotImplementedError()


def _default_few_shot_selector(
    results: List[ExperimentResultRow], max_k: int = 32, score_threshold: float = 0.8
):
    # Default is to just randomly select up to max_k examples from the passing examples
    selected = []
    for result in results:
        if any(
            (eval_result.score is not None and eval_result.score < score_threshold)
            for eval_result in result["evaluation_results"]["results"]
        ):
            selected.append(result)
        if len(selected) >= max_k:
            break
