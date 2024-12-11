from typing import Optional, Literal
from langsmith.evaluation._arunner import ExperimentResultRow
from dataclasses import dataclass, field
from promptim import types as pm_types, _utils as pm_utils
from promptim.optimizers import base as optimizers
from pydantic import BaseModel, Field
import langsmith as ls
import random
from promptim.optimizers.metaprompt import DEFAULT_METAPROMPT

_DEFAULT_RECOMMENDATION_PROMPT = """You are giving feedback on the performance of an AI model.

Analyze the test case, along with the prompt and any evaluation scores. Based on those results,
develop a theory of why the model failed. Perform a detailed analysis, commensurate to the complexity of the task.
Then provide targeted recommendations for improvements.

The current prompt is:

<current_prompt>
{prompt}
</current_prompt>
Another AI will optimize the above prompt based on your recommendations. Be as clear and specific as possible.
"""


@dataclass(kw_only=True)
class Config(optimizers.Config):
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
    max_batch_size: Optional[int] = 20


class Advise(BaseModel):
    """Think step-by-step, analyzing the task and test results. Provide a clear recommendation on why the prompt failed this
    test case, and what it should do to succeed next time for this type of input. Focus on the test metrics and expected output (if available).
    """

    analysis: str = Field(
        description="First, analyze why the prompt failed for this example. Think of what instructions in the prompt were poorly defined or missing."
    )
    recommended_changes: str = Field(
        description="Second, provide targeted recommendations for improvements."
    )


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
        model: optimizers.MODEL_TYPE | None = None,
        score_threshold: float = 0.8,
        recommendation_prompt: Optional[str] = None,
        meta_prompt: Optional[str] = None,
        max_batch_size: Optional[int] = 20,
    ):
        super().__init__(model=model)
        self.score_threshold = score_threshold
        self.recommendation_prompt = (
            recommendation_prompt or _DEFAULT_RECOMMENDATION_PROMPT
        )
        self.meta_prompt = meta_prompt or DEFAULT_METAPROMPT
        self.max_batch_size = max_batch_size

    def _format_failing_examples(
        self, results: list[ExperimentResultRow]
    ) -> list[dict]:
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
                failing.append(self._format_example(r))
        return failing

    def _format_example(self, example: ExperimentResultRow) -> str:
        """Format failing examples into a string for analysis."""
        outputs = example["example"].outputs

        if outputs:
            ref_output = f"But we expected: {outputs}"
        else:
            ref_output = ""
        scores = []
        for eval_result in example["evaluation_results"]["results"]:
            scores.append(
                f"- {eval_result.key}: {eval_result.score}"
                f"{f' (Comment: {eval_result.comment})' if eval_result.comment else ''}"
            )

        scores = "\n".join(scores)
        if scores:
            scores = f"\n\nTest results:\n{scores}"

        return f"""Failing Example:
For input: {example['example'].inputs}
The prompt predicted: {example['run'].outputs}
{ref_output}
{scores}
"""

    async def improve_prompt(
        self,
        current_prompt: pm_types.PromptWrapper,
        results: list[ExperimentResultRow],
        task: pm_types.Task,
        other_attempts: list[pm_types.PromptWrapper],
    ) -> pm_types.PromptWrapper:
        """Improve prompt using feedback from failing examples.

        1. Select failing examples
        2. If no failing examples, return current prompt
        3. Batch advisor over failing examples
        4. Format advisor responses into a string
        5. Run metaprompt over formatted advice
        """
        # 1. Identify failing examples
        failing_examples = self._format_failing_examples(results)

        # 2. If no failing examples, return current prompt unchanged
        if not failing_examples:
            return current_prompt
        if self.max_batch_size and len(failing_examples) > self.max_batch_size:
            random.shuffle(failing_examples)
            failing_examples = failing_examples[: self.max_batch_size]
        # 3. Generate targeted recommendations for each failing example
        advisor = self.model.with_structured_output(Advise)
        advisor_inputs = [
            [
                (
                    "system",
                    self.recommendation_prompt.format(
                        prompt=current_prompt.get_prompt_str_in_context()
                    ),
                ),
                ("user", example),
            ]
            for example in failing_examples
        ]
        with ls.trace(
            name="Analyze examples", inputs={"num_examples": len(failing_examples)}
        ):
            recommendations: list[Advise] = await advisor.abatch(advisor_inputs)

        # 4. Format recommendations into a consolidated string
        formatted_recommendations = []
        for i, (example, rec) in enumerate(zip(failing_examples, recommendations)):
            formatted_recommendations.append("Recommended changes for example {i+1}:")
            formatted_recommendations.append(rec.recommended_changes)
            formatted_recommendations.append("-" * 40 + "\n")

        all_recommendations = "\n".join(formatted_recommendations)

        # 5. Use consolidated recommendations to guide final prompt improvement
        chain = self.model.with_structured_output(pm_types.OptimizedPromptOutput)
        inputs = {
            "current_prompt": current_prompt.get_prompt_str_in_context(),
            "task_description": task.describe(),
            "annotated_results": all_recommendations,
            "other_attempts": (
                "\n\n---".join([p.get_prompt_str() for p in other_attempts])
                if other_attempts
                else "N/A"
            ),
        }

        prompt_output: pm_types.OptimizedPromptOutput = await chain.ainvoke(
            self.meta_prompt.format(**inputs)
        )

        candidate = pm_types.PromptWrapper.from_prior(
            current_prompt, prompt_output.improved_prompt
        )

        pm_utils.print_rich_diff(
            current_prompt.get_prompt_str_in_context(),
            candidate.get_prompt_str_in_context(),
            "Updated Prompt with Targeted Improvements",
        )
        return candidate
