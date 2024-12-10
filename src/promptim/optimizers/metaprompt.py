from typing import List
from langsmith.evaluation._arunner import ExperimentResultRow
from dataclasses import dataclass, field
from promptim import types as pm_types
from promptim import _utils as pm_utils
from promptim.optimizers import base as optimizers
from typing_extensions import Literal


DEFAULT_METAPROMPT = """You are an expert prompt engineer tasked with improving prompts for AI tasks.
You will use all means necessary to optimize the scores for the provided prompt so that the resulting model can
perform well on the target task.

## Current prompt

The following is the current best-performing prompt:

<current_prompt>
{current_prompt}
</current_prompt>

Your generations will replace the content within the <TO_OPTIMIZE></TO_OPTIMIZE> tags. The rest is fixed context over which you have no control. The TO_OPTIMIZE and CONTEXT\
 tags are provided here to help you disambiguateand not present in the prompt itself.

## Previous Prompt Attempts

You previously attempted to use the following prompts, but they earned worse scores than the current one:
<other_attempts>
{other_attempts}
</other_attempts>

Reflect on your previous attempts to ensure you search for and identify better patterns.

## Annotated results:
<results>
{annotated_results}
</results>

## Task description:
<task_description>
{task_description}
</task_description>

Unless otherwise specified, higher scores are better (try to maximize scores). Aim for perfect scores across all examples.

In your head, search through all edits, planning the optimization step-by-step:
1. Analyze the current results and where they fall short
2. Identify patterns in successful vs unsuccessful cases
3. Propose specific improvements to address the shortcomings
4. Generate an improved prompt that maintains all required formatting

The improved prompt must:
- Keep all original input variables
- Maintain any special formatting or delimiters
- Focus on improving the specified metrics
- Be clear and concise.
- Avoid repeating mistakes.

Use prompting strategies as appropriate for the task. For logic and math, consider encourage more chain-of-thought reasoning, 
or include reasoning trajectories to induce better performance. For creative tasks, consider adding style guidelines.
Or consider including exemplars.

Output your response in this format:
<analysis>
Your step-by-step analysis here...
</analysis>

<improved_prompt>
Your improved prompt here...
</improved_prompt>"""


@dataclass(kw_only=True)
class Config(optimizers.Config):
    """Configuration for the metaprompt optimization algorithm."""

    kind: Literal["metaprompt"] = field(
        default="metaprompt",
        metadata={
            "description": "The meta-prompt optimizer that uses an LLM to analyze and improve prompts."
        },
    )
    meta_prompt: str = field(
        default=DEFAULT_METAPROMPT,
        metadata={
            "description": "The meta-prompt to use for analyzing and improving prompts."
        },
    )


class MetaPromptOptimizer(optimizers.BaseOptimizer):
    """
    This is the original style meta-prompt algorithm:
    It takes the current results and uses the meta-prompt to propose a new prompt.
    """

    config_cls = Config

    def __init__(
        self,
        *,
        model: optimizers.MODEL_TYPE | None = None,
        meta_prompt: str | None = None,
    ):
        super().__init__(model=model)
        self.meta_prompt = meta_prompt or DEFAULT_METAPROMPT

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

        pm_utils.print_rich_diff(
            current_prompt.get_prompt_str_in_context(),
            candidate.get_prompt_str_in_context(),
            "Updated Prompt",
        )

        return candidate
