from typing import List, Optional, Literal
from langsmith.evaluation._arunner import ExperimentResultRow
from dataclasses import dataclass, field
from promptim import types as pm_types, _utils as pm_utils
from promptim.optimizers import base as optimizers

_DEFAULT_RECOMMENDATION_PROMPT = """You are an expert prompt engineer. 
Given the following examples that scored below our target threshold of {score_threshold}, 
analyze their inputs/outputs and suggest specific improvements to the prompt.
Focus on patterns in the failures and propose concrete changes."""


@dataclass
class Config(optimizers.BaseConfig):
    kind: Literal["feedback_guided"] = field(
        default="feedback_guided",
        metadata={
            "description": "The feedback_guided optimizer  that iteratively improves"
            " prompts based on feedback from evaluation results, focusing on examples that fall below a specified performance threshold."
        },
    )
    recommendation_prompt: str = field(
        default=_DEFAULT_RECOMMENDATION_PROMPT,
    )
    score_threshold: float = 0.8


class FeedbackGuidedOptimizer(optimizers.BaseOptimizer):
    """
    A two-phase optimization algorithm that:
    1. Identifies examples with scores below a threshold
    2. Generates targeted recommendations for improvements
    3. Uses these recommendations to guide prompt refinement

    The algorithm is particularly effective when you want to focus
    optimization efforts on specific failure cases while maintaining
    overall prompt quality.
    """

    config_cls = Config

    def __init__(
        self,
        *,
        model: optimizers.MODEL_TYPE,
        meta_prompt: str,
        score_threshold: float = 0.8,
        recommendation_prompt: Optional[str] = None,
    ):
        self.model = model
        self.meta_prompt = meta_prompt
        self.score_threshold = score_threshold
        self.recommendation_prompt = (
            recommendation_prompt or _DEFAULT_RECOMMENDATION_PROMPT
        )

    def _format_failing_examples(
        self, results: List[ExperimentResultRow]
    ) -> List[dict]:
        """Identify and format examples that fall below the score threshold."""
        failing = []
        for r in results:
            # Consider "failing" if any evaluation score is below threshold
            if any(
                (
                    eval_result.score is not None
                    and eval_result.score < self.score_threshold
                )
                for eval_result in r["evaluation_results"]["results"]
            ):
                failing.append(r)
        return failing

    def _format_example_for_analysis(
        self, failing_examples: List[ExperimentResultRow]
    ) -> str:
        """Format failing examples into a string for analysis."""
        formatted = []
        for i, example in enumerate(failing_examples):
            formatted.append(f"Failing Example {i+1}:")
            formatted.append(f"Input: {example['run'].inputs}")
            formatted.append(f"Output: {example['run'].outputs}")
            formatted.append("Scores:")
            for eval_result in example["evaluation_results"]["results"]:
                formatted.append(
                    f"- {eval_result.key}: {eval_result.score}"
                    f"{f' (Comment: {eval_result.comment})' if eval_result.comment else ''}"
                )
            formatted.append("")
        return "\n".join(formatted)

    async def improve_prompt(
        self,
        current_prompt: pm_types.PromptWrapper,
        results: List[ExperimentResultRow],
        task: pm_types.Task,
        other_attempts: List[pm_types.PromptWrapper],
    ) -> pm_types.PromptWrapper:
        # 1. Identify failing examples
        failing_examples = self._format_failing_examples(results)

        # If no failing examples, return current prompt unchanged
        if not failing_examples:
            return current_prompt

        # 2. Generate targeted recommendations
        formatted_examples = self._format_example_for_analysis(failing_examples)
        rec_input = (
            f"{self.recommendation_prompt.format(score_threshold=self.score_threshold)}"
            f"\n\n{formatted_examples}\n\nProvide a list of recommended changes."
        )
        recommendations = await self.model.ainvoke(rec_input)

        # 3. Use recommendations to guide prompt improvement
        chain = self.model.with_structured_output(pm_types.OptimizedPromptOutput)
        inputs = self.meta_prompt.format(
            current_prompt=current_prompt.get_prompt_str_in_context(),
            annotated_results=formatted_examples
            + "\n\nRecommended Fixes:\n"
            + recommendations,
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
            "Updated Prompt with Targeted Improvements",
        )
        return candidate
