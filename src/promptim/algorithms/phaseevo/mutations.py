from abc import abstractmethod
import math
import promptim.types as pm_types
from collections import deque
from typing import Optional, Literal, TypedDict, cast
from promptim.optimizers import base as optimizers
from langsmith.evaluation._arunner import ExperimentResultRow
from promptim import _utils as pm_utils
import asyncio
import random
import langsmith as ls
from trustcall import create_extractor
import logging

logger = logging.getLogger(__name__)


class Variant:
    def __init__(
        self,
        prompt: pm_types.PromptWrapper,
        results: list[ExperimentResultRow],
    ):
        if not results:
            raise ValueError("No results provided")
        rows = [
            eval_result
            for row in sorted(results, key=lambda x: x["example"].id)
            for eval_result in sorted(
                (
                    row["evaluation_results"]["results"]
                    if "evaluation_results" in row
                    and "results" in row["evaluation_results"]
                    else []
                ),
                key=lambda x: x.key,
            )
        ]
        defined_scores = [
            eval_result.score for eval_result in rows if eval_result.score is not None
        ]
        fitness = sum(defined_scores) / len(defined_scores)
        self.prompt = prompt
        self.fitness = fitness
        self.vector = [
            float(eval_result.score if eval_result.score is not None else float("-inf"))
            for row in results
            for eval_result in (
                row["evaluation_results"]["results"]
                if "evaluation_results" in row
                and "results" in row["evaluation_results"]
                else []
            )
        ]
        self.results = results

    def __repr__(self) -> str:
        return f"Variant(fitness={self.fitness}, id={self.prompt.identifier}, lineage_depth={len(self.prompt.lineage) if self.prompt.lineage else 0}, prompt={self.prompt.get_prompt_str_in_context()})"


class Mutation(optimizers.BaseMutator):
    def __init__(self, *, model: optimizers.MODEL_TYPE, **kwargs):
        super().__init__(model=model)

    @abstractmethod
    async def mutate(
        self, population: list[Variant], train_examples: list[pm_types.Example]
    ) -> list[pm_types.PromptWrapper]: ...


GRADIENT_DESCENT_GENERATION_PROMPT = """You are a prompt optimizer. Given an existing prompt and cases where it made mistakes, analyze what's causing the failures and how the prompt could be improved.

## Existing Prompt ##
{existing_prompt}

{previous_analysis}

## Cases where it gets wrong: ##
{failing_examples}{passing_examples}

Carefully analyze the test cases to interpret the test criteria/rubric. Check your work. Review each test case one-by-one. Label all reasons it failed. Identify all changes needed in the output to pass. Consider both what the prompt is doing and what it isn't doing, and how the language model is responding to (or ignoring) the existing content and structure. You might explore techniques like:
- Few-shot examples showing ideal behavior
- Chain-of-thought paths for complex cases
- Counter-examples demonstrating failure modes
- Strategic structure (phases, markers, roles)
- Task-specific patterns for this domain

Or other techniques to induce better performance from the language model for these test cases.

Detail your suggested improvements, being specific about what to change and why those changes would help prevent the observed errors."""

GRADIENT_DESCENT_APPLICATION_PROMPT = """You are a prompt optimizer. Transform this prompt based on the provided feedback, feeling free to use any effective techniques that will improve performance.

## Existing Prompt ##
{existing_prompt}

## Feedback ##
{feedback}

## Improved Prompt ##
"""

LAMARCKIAN_MUTATION_PROMPT = """I gave a friend an instruction and some inputs. The friend read the instruction and wrote an output for every one of the inputs. Here are the input-output pairs:

## Examples ##
{examples}

The instruction was:"""


class LamarckianMutation(Mutation):
    def __init__(
        self,
        *,
        model: optimizers.MODEL_TYPE,
        population_size: int = 15,
        batch_size: int = 30,
        **kwargs,
    ):
        super().__init__(model=model)
        self.population_size = population_size
        self.batch_size = batch_size

    @ls.traceable
    async def mutate(
        self, population: list[Variant], train_examples: list[pm_types.Example]
    ) -> list[pm_types.PromptWrapper]:
        contain_outputs = [e for e in train_examples if e.outputs]
        if not contain_outputs:
            logger.warning("No examples contain outputs. Skipping Lamarckian mutation.")
            return []
        N = self.population_size - len(population)
        batches = []
        for _ in range(N):
            batches.append(random.sample(contain_outputs, self.batch_size))
        return [
            p
            for p in (
                await asyncio.gather(
                    *[
                        self.mutate_single(population[0].prompt, batch)
                        for batch in batches
                    ],
                    return_exceptions=True,
                )
            )
            if isinstance(p, pm_types.PromptWrapper)
        ]

    def _format_examples(self, examples: list[pm_types.Example]) -> str:
        return "\n".join(
            f"Input: {example.inputs}\nOutput: {example.outputs}\n"
            for example in examples
        )

    @ls.traceable
    async def mutate_single(
        self, prompt: pm_types.PromptWrapper, examples: list[pm_types.Example]
    ) -> pm_types.PromptWrapper:
        formatted = self._format_examples(examples)
        with ls.trace(name="Lamarckian Mutation", inputs={"examples": formatted}) as rt:
            prompt_response = await create_extractor(
                self.model,
                tools=[pm_types.prompt_schema(prompt)],
                tool_choice="OptimizedPromptOutput",
            ).ainvoke(
                [
                    (
                        "system",
                        "You are a prompt generator. "
                        "Write an f-string prompt based on the provided examples. Every input key should be included in the prompt in brackets.",
                    ),
                    ("user", LAMARCKIAN_MUTATION_PROMPT.format(examples=formatted)),
                ]
            )
            prompt_output = cast(
                pm_types.OptimizedPromptOutput, prompt_response["responses"][0]
            )
            rt.add_outputs({"output": prompt_output})
        return pm_types.PromptWrapper.from_prior(prompt, prompt_output.improved_prompt)


class GradientDescentMutation(Mutation):
    def __init__(
        self,
        *,
        model: optimizers.MODEL_TYPE | None = None,
        score_threshold: float = 0.8,
        max_batch_size: Optional[int] = 20,
        **kwargs,
    ):
        super().__init__(model=model)
        self.score_threshold = score_threshold
        self.max_batch_size = max_batch_size
        self.passing_num = 5

    @ls.traceable
    async def mutate(
        self, population: list[Variant], train_examples: list[pm_types.Example]
    ) -> list[pm_types.PromptWrapper]:
        """Improve prompt using "gradient descent".

        AKA feedback from failing examples.

        1. Select failing examples
        2. If no failing examples, return current prompt
        3. Batch advisor over failing examples
        4. Format advisor responses into a string
        5. Run metaprompt over formatted advice
        """
        return [
            p
            for p in await asyncio.gather(
                *(self.mutate_single(v) for v in population), return_exceptions=True
            )
            if isinstance(p, pm_types.PromptWrapper)
        ]

    @ls.traceable
    async def mutate_single(self, variant: Variant) -> pm_types.PromptWrapper:
        failing_examples = self._format_failing_examples(variant.results)
        passing_examples = self._format_passing_examples(variant.results)
        previous_analysis_list = None
        if variant.prompt.extra and (
            previous_analysis_list := variant.prompt.extra.get(
                "previous_gradient_analysis"
            )
        ):
            previous_analysis = self._format_previous_analysis(previous_analysis_list)
        else:
            previous_analysis = ""
        if not failing_examples:
            return variant.prompt
        if self.max_batch_size and len(failing_examples) > self.max_batch_size:
            random.shuffle(failing_examples)
            failing_examples = failing_examples[: self.max_batch_size]
        existing_prompt = variant.prompt.get_prompt_str_in_context()
        with ls.trace(
            name="Compute Gradient",
            inputs={
                "failing_examples": "\n".join(failing_examples),
                "passing_examples": passing_examples,
                "existing_prompt": existing_prompt,
                "previous_analysis": previous_analysis,
            },
        ) as rt:
            advice_msg = await self.model.ainvoke(
                GRADIENT_DESCENT_GENERATION_PROMPT.format(
                    existing_prompt=existing_prompt,
                    failing_examples="\n".join(failing_examples),
                    passing_examples=passing_examples,
                    previous_analysis=previous_analysis,
                )
            )
            rt.add_outputs({"output": advice_msg})
        with ls.trace(
            name="Apply Gradient",
            inputs={"existing_prompt": existing_prompt, "feedback": advice_msg.content},
        ):
            chain = create_extractor(
                self.model,
                tools=[pm_types.prompt_schema(variant.prompt)],
                tool_choice="OptimizedPromptOutput",
            )
            prompt_output = await chain.ainvoke(
                GRADIENT_DESCENT_APPLICATION_PROMPT.format(
                    existing_prompt=existing_prompt,
                    feedback=advice_msg.content,
                )
            )
            prompt_output = cast(
                pm_types.OptimizedPromptOutput, prompt_output["responses"][0]
            )
            rt.add_outputs({"output": prompt_output})

        previous_analysis = previous_analysis_list or []
        previous_analysis.append(prompt_output.analysis)
        candidate = pm_types.PromptWrapper.from_prior(
            variant.prompt,
            prompt_output.improved_prompt,
            extra_info={"previous_gradient_analysis": previous_analysis},
        )

        pm_utils.print_rich_diff(
            variant.prompt.get_prompt_str_in_context(),
            candidate.get_prompt_str_in_context(),
            f"{self.__class__.__name__} Mutated Prompt",
        )
        return candidate

    def _format_previous_analysis(self, previous_analysis: list[str]):
        linear = "\n".join([f"{i}: {a}" for i, a in enumerate(previous_analysis)])
        return f"""## Previous Gradient Analysis ##

You analyzed previous variants of this prompt with the following conclusions:
<analysis>
{linear}
</analysis>

Correct errors in your analysis. What are you misunderstanding or miscommunicating?\
Your previous analysis was either wrong, inadequate, or ignored. Consider this before analyzing the test results."""

    def _format_failing_examples(self, results: list[ExperimentResultRow]) -> list[str]:
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
            scores = f"<Test results>\n{scores}\n</Test results>"

        return f"""<failing example>
<input>
{example['example'].inputs}
</input>
<prompt_prediction>
{example['run'].outputs}
</prompt_prediction>
{ref_output}
{scores}
</failing example>"""

    def _format_passing_examples(self, results: list[ExperimentResultRow]) -> str:
        """Format examples that pass the score threshold into a string for analysis."""
        passing = []
        for r in results:
            # Consider "passing" if all evaluation scores meet or exceed threshold
            if all(
                (
                    eval_result.score is not None
                    and eval_result.score >= self.score_threshold
                )
                for eval_result in r["evaluation_results"]["results"]
            ):
                passing.append(self._format_passing_example(r))

        if not passing:
            return ""
        random.shuffle(passing)
        # Only include the section header if we have examples
        return "\n\n## Cases it gets right: ##\n" + "\n".join(
            passing[: self.passing_num]
        )

    def _format_passing_example(self, example: ExperimentResultRow) -> str:
        """Format a single passing example into a string."""
        outputs = example["example"].outputs
        scores = []
        for eval_result in example["evaluation_results"]["results"]:
            scores.append(
                f"- {eval_result.key}: {eval_result.score}"
                f"{f' (Comment: {eval_result.comment})' if eval_result.comment else ''}"
            )

        scores = "\n".join(scores)
        if scores:
            scores = f"<test results>\n{scores}\n</test results>"

        return f"""
<passing example>
<input>
{example['example'].inputs}
</input>
<prompt_prediction>
{example['run'].outputs}
</prompt_prediction>
<expected>
{outputs if outputs else ''}
</expected>
{scores}
</passing example>"""


PROMPT_TECHNIQUES = [
    "Rephrase using different but equivalent words.",
    "Restructure using a different format.",
    "Add challenging, distinct few-shot examples.",
    "Add thought trajectories over challenging examples.",
    "Try to simplify the prompt without losing meaning or quality.",
    "",
]

SEMANTIC_MUTATION_PROMPT = """Given a prompt, your task is to generate another prompt with the same semantic meaning. {technique}

# Example:
current prompt: Classify the sentiment of the following sentence as either negative or positive:
mutated prompt: Determine the sentiment of the given sentence and assign a label from ['negative', 'positive'].

Given:
current prompt: {existing_prompt}
mutated prompt:
"""


class SemanticMutation(Mutation):
    def __init__(
        self,
        *,
        model: optimizers.MODEL_TYPE | None = None,
        stop_at_population_limit: bool = False,
        population_limit: int = 10,
        **kwargs,
    ):
        super().__init__(model=model)
        self.stop_at_population_limit = stop_at_population_limit
        self.population_limit = population_limit
        self.techniques = PROMPT_TECHNIQUES

    @ls.traceable
    async def mutate(
        self, population: list[Variant], train_examples: list[pm_types.Example]
    ) -> list[pm_types.PromptWrapper]:
        if self.stop_at_population_limit:
            remaining = self.population_limit - len(population)
            multiply = math.ceil(remaining / len(population))
            to_process = random.sample(population * multiply, remaining)
        else:
            to_process = population
        techniques = self.techniques.copy()
        random.shuffle(techniques)
        results = await asyncio.gather(
            *(
                self.mutate_single(v, techniques[i % len(techniques)])
                for i, v in enumerate(to_process)
            ),
            return_exceptions=True,
        )
        return [
            result for result in results if isinstance(result, pm_types.PromptWrapper)
        ]

    @ls.traceable
    async def mutate_single(
        self, variant: Variant, technique: str
    ) -> pm_types.PromptWrapper:
        existing_prompt = variant.prompt.get_prompt_str_in_context()

        with ls.trace(
            name="Semantic Mutation",
            inputs={"existing_prompt": existing_prompt, "technique": technique},
        ) as rt:
            prompt_output = await create_extractor(
                self.model,
                tools=[pm_types.prompt_schema(variant.prompt)],
                tool_choice="OptimizedPromptOutput",
            ).ainvoke(
                SEMANTIC_MUTATION_PROMPT.format(
                    existing_prompt=existing_prompt, technique=technique
                )
            )
            prompt_output = cast(
                pm_types.OptimizedPromptOutput, prompt_output["responses"][0]
            )
            rt.add_outputs({"output": prompt_output})

        candidate = pm_types.PromptWrapper.from_prior(
            variant.prompt, prompt_output.improved_prompt
        )

        pm_utils.print_rich_diff(
            variant.prompt.get_prompt_str_in_context(),
            candidate.get_prompt_str_in_context(),
            f"{self.__class__.__name__} Mutated Prompt",
        )
        return candidate


EDA_PROMPT = """Given a series of prompts, your task is to generate another prompt with the same semantic meaning and intentions.

## Existing Prompts ##
{existing_prompts}

The newly mutated prompt is:
"""

EDA_INDEX_PROMPT = """Given a series of prompts, your task is to generate another prompt with the same semantic meaning and intentions. The series of prompts are ranked by their quality from best to worst.

## Existing Prompts ##
{existing_prompts}

The newly mutated prompt is:
"""


class EdaMutation(Mutation):
    def __init__(
        self,
        *,
        model: optimizers.MODEL_TYPE | None = None,
        prompt: str = EDA_PROMPT,
        **kwargs,
    ):
        super().__init__(model=model)
        self.prompt = prompt

    def _prepare_cluster(self, cluster: list[Variant]) -> list[Variant]:
        cluster = cluster.copy()
        random.shuffle(cluster)
        return cluster

    @ls.traceable
    async def mutate(
        self, population: list[Variant], train_examples: list[pm_types.Example]
    ) -> list[pm_types.PromptWrapper]:
        src = deque(sorted(population, key=lambda v: v.fitness, reverse=True))
        clusters = []
        while len(src) > 1:
            best = src.popleft()
            clusters.append([best])
            for v in list(src):
                if manhattan_distance(v.vector, best.vector) < 0.1:
                    clusters.append([v])
                    src.remove(v)
                    break
        return await asyncio.gather(
            *(self.distill_cluster(cluster) for cluster in clusters)
        )

    async def distill_cluster(self, cluster: list[Variant]) -> pm_types.PromptWrapper:
        cluster = self._prepare_cluster(cluster)
        cluster_prompts = [v.prompt.get_prompt_str_in_context() for v in cluster]
        existing_prompts = "\n".join(cluster_prompts)
        with ls.trace(
            name="Semantic Mutation",
            inputs={"existing_prompts": existing_prompts},
        ) as rt:
            prompt_output = await create_extractor(
                self.model,
                tools=[pm_types.prompt_schema(cluster[0].prompt)],
                tool_choice="OptimizedPromptOutput",
            ).ainvoke(self.prompt.format(existing_prompts=existing_prompts))
            prompt_output = cast(
                pm_types.OptimizedPromptOutput, prompt_output["responses"][0]
            )
            rt.add_outputs({"output": prompt_output})

        candidate = pm_types.PromptWrapper.from_prior(
            cluster[0].prompt, prompt_output.improved_prompt
        )

        pm_utils.print_rich_diff(
            cluster[0].prompt.get_prompt_str_in_context(),
            candidate.get_prompt_str_in_context(),
            f"{self.__class__.__name__} Distilled Prompt",
        )
        return candidate


class EDAIndexMutation(EdaMutation):
    def __init__(
        self,
        *,
        model: optimizers.MODEL_TYPE | None = None,
        prompt: str = EDA_INDEX_PROMPT,
        **kwargs,
    ):
        super().__init__(model=model, prompt=prompt)

    def _prepare_cluster(self, cluster: list[Variant]) -> list[Variant]:
        # Note: we are DELIBERATELY lying to the LLM saying the most fit prompt is first.
        # This is mean to counter-act the position bias inherent in some LLMs.
        # Need to benchmark further whether this actually has the desired effect.
        return sorted(cluster, key=lambda v: v.fitness, reverse=False)


CROSS_OVER_PROMPT = """You are a mutator who is familiar with the concept of cross-over in genetic algorithms, namely combining the genetic information of two parents to generate new offspring. Given two parent prompts, you will perform a cross-over to generate an offspring prompt that covers the same semantic meaning as both parents.

## Given ##
Parent prompt 1: {prompt_1}
Parent prompt 2: {prompt_2}
Offspring prompt:
"""


class CrossoverMutation(Mutation):
    def __init__(
        self,
        *,
        model: optimizers.MODEL_TYPE | None = None,
        prompt: str = CROSS_OVER_PROMPT,
        **kwargs,
    ):
        super().__init__(model=model)
        self.prompt = prompt

    def produce_pairs(self, population: list[Variant]) -> list[tuple[Variant, Variant]]:
        src = sorted(population, key=lambda v: v.fitness, reverse=True)
        return [(src[0], src[2])]

    @ls.traceable
    async def mutate(
        self, population: list[Variant], train_examples: list[pm_types.Example]
    ) -> list[pm_types.PromptWrapper]:
        pairs = self.produce_pairs(population)
        return await asyncio.gather(*(self.merge(pair) for pair in pairs))

    @ls.traceable
    async def merge(self, pair: tuple[Variant, Variant]) -> pm_types.PromptWrapper:
        cluster_prompts = [v.prompt.get_prompt_str_in_context() for v in pair]
        existing_prompts = "\n".join(cluster_prompts)
        with ls.trace(
            name="Semantic Mutation",
            inputs={"existing_prompts": existing_prompts},
        ) as rt:
            prompt_output = await create_extractor(
                self.model,
                tools=[pm_types.prompt_schema(pair[0].prompt)],
                tool_choice="OptimizedPromptOutput",
            ).ainvoke(
                self.prompt.format(
                    prompt_1=cluster_prompts[0], prompt_2=cluster_prompts[1]
                )
            )
            prompt_output = cast(
                pm_types.OptimizedPromptOutput, prompt_output["responses"][0]
            )
            rt.add_outputs({"output": prompt_output})

        candidate = pm_types.PromptWrapper.from_prior(
            pair[0].prompt, prompt_output.improved_prompt
        )

        pm_utils.print_rich_diff(
            pair[0].prompt.get_prompt_str_in_context(),
            candidate.get_prompt_str_in_context(),
            f"{self.__class__.__name__} Distilled Prompt",
        )
        return candidate


class CrossoverDistinctMutation(CrossoverMutation):
    def produce_pairs(self, population: list[Variant]) -> list[tuple[Variant, Variant]]:
        src = deque(sorted(population, key=lambda v: v.fitness, reverse=True))
        pairs = []
        while len(src) > 1:
            best = src.popleft()
            other = sorted(
                src,
                key=lambda v: manhattan_distance(v.vector, best.vector),
                reverse=True,
            )[0]
            src.remove(other)
            pairs.append((best, other))
        return pairs


def manhattan_distance(vec1: list[float], vec2: list[float]) -> float:
    return sum(abs(v1 - v2) for v1, v2 in zip(vec1, vec2))


MUTATIONS = {
    "lamarckian": LamarckianMutation,
    "gradient": GradientDescentMutation,
    "semantic": SemanticMutation,
    "eda": EdaMutation,
    "eda-index": EDAIndexMutation,
    "crossover": CrossoverMutation,
    "crossover-distinct": CrossoverDistinctMutation,
}


class PhaseConfig(TypedDict):
    mutation: Literal[
        "lamarckian",
        "gradient",
        "semantic",
        "eda",
        "eda-index",
        "crossover",
        "crossover-distinct",
    ]
    population_size: int
    improvement_threshold: float
    max_attempts: int


def ensure_phase_config(config: dict | PhaseConfig) -> PhaseConfig:
    if "mutation" not in config:
        raise ValueError(f"Phase config must specify a mutation. Got {config}")
    return cast(
        PhaseConfig,
        {
            **config,
            "mutation": config["mutation"],
            "population_limit": config.get("population_limit", 5),
            "improvement_threshold": config.get(
                "improvement_threshold", 0.6
            ),  # 90% "better" - kinda arbitrary
            "max_attempts": config.get("max_attempts", 5),
        },
    )


def load_mutation(
    config: PhaseConfig,
    model: optimizers.MODEL_TYPE,
) -> Mutation:
    config = config.copy()
    mutation_cls = MUTATIONS[config.pop("mutation")]
    return mutation_cls.from_config({**config, "model": model})
