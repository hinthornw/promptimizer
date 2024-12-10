from typing import List
from langsmith.evaluation._arunner import ExperimentResultRow
from dataclasses import dataclass
from promptim import types as pm_types
from promptim.optimizers.base import BaseOptimizationAlgorithm


@dataclass
class MetaPromptConfig:
    meta_prompt: str
    model: any


class MetaPromptImprovementAlgorithm(BaseOptimizationAlgorithm):
    """
    This is the original style meta-prompt algorithm:
    It takes the current results and uses the meta-prompt to propose a new prompt.
    """

    def __init__(
        self,
        model,
        meta_prompt: str,
    ):
        self.model = model
        self.meta_prompt = meta_prompt

    def _format_results(self, results: List[ExperimentResultRow]) -> str:
        formatted = []
        for i, r in enumerate(results):
            formatted.append(f"Example {i+1}:")
            formatted.append(f'Input: {r["run"].inputs}')
            formatted.append(f'Output: {r["run"].outputs}')
            formatted.append("Evaluations:")
            for eval_result in r["evaluation_results"]["results"]:
                formatted.append(f"- {eval_result.key}: {eval_result.score}")
                if eval_result.comment:
                    formatted.append(f"  Comment: {eval_result.comment}")
            formatted.append("")
        return "\n".join(formatted)

    async def improve_prompt(
        self,
        current_prompt: pm_types.PromptWrapper,
        results: List[ExperimentResultRow],
        task: pm_types.Task,
        other_attempts: List[pm_types.PromptWrapper],
    ) -> pm_types.PromptWrapper:
        annotated_results = self._format_results(results)

        chain = self.model.with_structured_output(pm_types.OptimizedPromptOutput)
        inputs = self.meta_prompt.format(
            current_prompt=current_prompt.get_prompt_str_in_context(),
            annotated_results=annotated_results,
            task_description=task.describe(),
            other_attempts=(
                "\n\n---".join([p.get_prompt_str() for p in other_attempts])
                if other_attempts
                else "N/A"
            ),
        )
        prompt_output: pm_types.OptimizedPromptOutput = await chain.ainvoke(inputs)
        candidate = pm_types.PromptWrapper.from_prior(
            current_prompt, prompt_output.improved_prompt
        )

        pm_types._print_rich_diff(
            current_prompt.get_prompt_str_in_context(),
            candidate.get_prompt_str_in_context(),
            "Updated Prompt",
        )

        return candidate
