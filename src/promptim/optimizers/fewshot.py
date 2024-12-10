from typing import List
from langsmith.evaluation._arunner import ExperimentResultRow
from promptim import types as pm_types, _utils as pm_utils
from promptim.optimizers.base import BaseOptimizationAlgorithm


class FewShotInsertionAlgorithm(BaseOptimizationAlgorithm):
    """
    A simple example of an algorithm that selects few-shot examples and inserts them into the prompt.
    This might integrate with a separate FewShotSelector class.
    """

    def __init__(self, model, few_shot_selector, meta_prompt: str):
        self.model = model
        self.few_shot_selector = few_shot_selector
        self.meta_prompt = meta_prompt

    async def improve_prompt(
        self,
        current_prompt: pm_types.PromptWrapper,
        results: List[ExperimentResultRow],
        task: pm_types.Task,
        other_attempts: List[pm_types.PromptWrapper],
    ) -> pm_types.PromptWrapper:
        # 1. Use the few_shot_selector to pick examples
        few_shots = self.few_shot_selector.select_examples(results, task)

        # 2. Insert these examples into the prompt:
        #    For simplicity, assume the prompt has a <FEW_SHOT_EXAMPLES> placeholder
        current_str = current_prompt.get_prompt_str()
        improved_str = current_str.replace("<FEW_SHOT_EXAMPLES>", few_shots)

        candidate = pm_types.PromptWrapper.from_prior(current_prompt, improved_str)
        pm_utils.print_rich_diff(
            current_prompt.get_prompt_str_in_context(),
            candidate.get_prompt_str_in_context(),
            "Updated Prompt with Few-Shot Examples",
        )
        return candidate
