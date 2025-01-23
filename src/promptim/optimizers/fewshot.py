from typing import List
from dataclasses import dataclass, field
from typing_extensions import Literal
import random

import langsmith as ls
from promptim.optimizers import base as optimizers
from promptim import types as pm_types
from langsmith.evaluation._arunner import ExperimentResultRow
from promptim import _utils as pm_utils


@dataclass(kw_only=True)
class Config(optimizers.Config):
    kind: Literal["fewshot"] = field(
        default="fewshot",
        metadata={
            "description": "The few-shot optimizer that uses TPE to select optimal example combinations"
        },
    )
    max_examples: int = field(
        default=50,
        metadata={"description": "Maximum number of few-shot examples in the pool"},
    )
    n_trials: int = field(
        default=5, metadata={"description": "Number of TPE optimization trials"}
    )
    minibatch_size: int = field(
        default=10,
        metadata={"description": "Number of few-shot examples per minibatch"},
    )


class FewShotOptimizer(optimizers.BaseOptimizer):
    config_cls = Config

    def __init__(
        self,
        *,
        model: optimizers.MODEL_TYPE | None = None,
        max_examples: int = 50,
        minibatch_size: int = 10,
        n_trials: int = 5,
    ):
        super().__init__(model=model)
        self.max_examples = max_examples
        self.n_trials = n_trials
        self.minibatch_size = minibatch_size
        self._rng = random.Random(42)  # Just for any extra randomization you might do
        from promptim.algorithms.tpe_sampler import TPESampler

        self.sampler = TPESampler(seed=42)

    @ls.traceable
    async def improve_prompt(
        self,
        history: List[List[pm_types.PromptWrapper]],
        results: List[ExperimentResultRow],
        task: pm_types.Task,
        trainer: "PromptTrainer" = None,
        **kwargs,
    ) -> List[pm_types.PromptWrapper]:
        """
        Improve the prompt by picking an optimal subset of few-shot examples
        that yields the highest average evaluation score.
        """
        try:
            url_ = ls.get_current_run_tree().get_url()
            print(f"See optimization run: {url_}")
        except Exception:
            pass
        from promptim.algorithms.tpe_sampler import Trial

        if not results:
            # No data to optimize with
            return list(history[-1])

        current_prompt = history[-1][-1]
        train_examples = [r["example"] for r in results]
        best_score = float("-inf")
        best_prompt = current_prompt
        n_examples = len(train_examples)

        async with ls.trace("FewShotOptimization") as run_tree:

            # This objective is called once per trial, with a new "Trial" each time
            @ls.traceable
            async def objective(trial: Trial) -> float:
                nonlocal best_prompt, best_score

                example_mask = []
                for i in range(n_examples):
                    # We want to MAXIMIZE, so set lower_is_better=False
                    include_flag = trial.suggest_categorical(
                        f"include_example_{i}",
                        choices=[0, 1],
                        n_candidates=10,
                        gamma=0.2,
                        lower_is_better=False,
                    )
                    example_mask.append(bool(include_flag))

                while sum(example_mask) > self.max_examples:
                    chosen_to_remove = self._rng.choice(
                        [k for k, inc in enumerate(example_mask) if inc]
                    )
                    example_mask[chosen_to_remove] = False

                # Construct new prompt with selected few-shot examples
                selected_examples = [
                    ex for ex, inc in zip(train_examples, example_mask) if inc
                ]

                if not selected_examples:
                    score = float("-inf")
                else:
                    shuffled = self._rng.sample(
                        selected_examples, len(selected_examples)
                    )
                    candidate = self._create_prompt_with_examples(
                        current_prompt, shuffled
                    )
                    other_examples = [
                        ex for ex in train_examples if ex not in selected_examples
                    ][: self.minibatch_size]
                    if other_examples:
                        results = await trainer._evaluate_prompt(
                            candidate, task, other_examples, upload_results=False
                        )
                        score = self._calculate_score(results)
                        # Keep track of best
                        if score > best_score:
                            best_score = score
                            best_prompt = candidate
                            # For logging
                            rt = ls.get_current_run_tree()
                            rt.metadata["best_score"] = score
                    else:
                        if best_score > float("-inf"):
                            score = best_score
                        else:
                            best_prompt = candidate
                            score = -99999.0
                            best_score = score

                # Manually register each param w.r.t. outcome
                # We pass objective=score for each parameter we suggested
                for i, inc_val in enumerate(example_mask):
                    self.sampler.register(
                        param_name=f"include_example_{i}",
                        value=int(inc_val),
                        objective=score,
                    )

                return score

            # Actually run TPE optimization for the given number of trials
            try:
                best_trial = await self.sampler.optimize(
                    objective, n_trials=self.n_trials
                )
                run_tree.add_outputs(
                    {
                        "best_score": best_score,
                        "n_trials": self.n_trials,
                        "best_params": best_trial.params if best_trial else {},
                    }
                )
            except Exception as e:
                print(f"TPE optimization failed: {e}")
                # If it fails, just fall back to last prompt
                return list(history[-1])

            # Print a side-by-side difference of the improved prompt
            pm_utils.print_rich_diff(
                current_prompt.get_prompt_str_in_context(),
                best_prompt.get_prompt_str_in_context(),
                "Updated Prompt with Optimized Few-Shot Examples",
            )

            return [best_prompt]

    def _calculate_score(self, results: List[ExperimentResultRow]) -> float:
        # We average over all "score" values that exist.
        # If none exist, returns -∞ to discourage such combos.
        scores = []
        for result in results:
            for eval_result in result["evaluation_results"]["results"]:
                if eval_result.score is not None:
                    scores.append(eval_result.score)
        if not scores:
            return float("-inf")
        return sum(scores) / len(scores)

    def _create_prompt_with_examples(
        self, base_prompt: pm_types.PromptWrapper, examples: List[pm_types.Example]
    ) -> pm_types.PromptWrapper:
        """Create a new prompt with the given few-shot examples."""

        def sanitize(s: str) -> str:
            # Guard braces for typical format-string safety
            return str(s).replace("{", "{{").replace("}", "}}")

        few_shot_text = "\n\n# Few-Shot Examples:\n"
        for ex in examples:
            outputs = ex.outputs
            if isinstance(outputs, dict) and len(outputs) == 1 and "output" in outputs:
                outputs = outputs["output"]

            few_shot_text += (
                f"Input: {sanitize(ex.inputs)}\n"
                f"Output: {sanitize(outputs)}\n"
                "---\n"
            )

        new_prompt_text = base_prompt.get_prompt_str() + few_shot_text
        return pm_types.PromptWrapper.from_prior(
            base_prompt,
            new_prompt_text,
            extra_info={
                "num_fewshot": len(examples),
                "fewshot_indices": [ex.id for ex in examples],
                "optimizer": "fewshot",
            },
        )
