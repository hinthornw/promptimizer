from abc import abstractmethod
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


class Mutation(optimizers.BaseMutator):
    def __init__(self, *, model: optimizers.MODEL_TYPE, **kwargs):
        super().__init__(model=model)

    @abstractmethod
    async def mutate(
        self, population: list[Variant], train_examples: list[pm_types.Example]
    ) -> list[pm_types.PromptWrapper]: ...


GRADIENT_DESCENT_GENERATION_PROMPT = """You are a quick improver. Given an existing prompt and a series of cases where it made mistakes, look through each case carefully and identify what is causing the mistakes. Based on these observations, output ways to improve the prompts based on the mistakes.
## Existing Prompt ##
{existing_prompt}

## Cases where it gets wrong: ##
{failing_examples}

ways to improve the existing prompt based on observations of the mistakes above are:
"""

GRADIENT_DESCENT_APPLICATION_PROMPT = """You are a quick improver. Given an existing prompt and feedback on how it should improve, create an improved version based on the feedback.

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

    async def mutate(
        self, population: list[Variant], train_examples: list[pm_types.Example]
    ) -> list[pm_types.PromptWrapper]:
        N = self.population_size - len(population)
        batches = []
        for _ in range(N):
            batches.append(random.sample(train_examples, self.batch_size))
        return await asyncio.gather(
            *[self.mutate_single(population[0].prompt, batch) for batch in batches]
        )

    def _format_examples(self, examples: list[pm_types.Example]) -> str:
        return "\n".join(
            f"Input: {example.inputs}\nOutput: {example.outputs}\n"
            for example in examples
        )

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
            scores = f"\n\nTest results:\n{scores}"

        return f"""Failing Example:
For input: {example['example'].inputs}
The prompt predicted: {example['run'].outputs}
{ref_output}
{scores}
"""

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
        return await asyncio.gather(*(self.mutate_single(v) for v in population))

    async def mutate_single(self, variant: Variant) -> pm_types.PromptWrapper:
        failing_examples = self._format_failing_examples(variant.results)
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
                "existing_prompt": existing_prompt,
            },
        ) as rt:
            advice_msg = await self.model.ainvoke(
                GRADIENT_DESCENT_GENERATION_PROMPT.format(
                    existing_prompt=existing_prompt,
                    failing_examples="\n".join(failing_examples),
                )
            )
            rt.add_outputs({"output": advice_msg})
        with ls.trace(
            name="Apply Gradient",
            inputs={"existing_prompt": existing_prompt, "feedback": advice_msg.content},
        ):
            prompt_output = await create_extractor(
                self.model,
                tools=[pm_types.prompt_schema(variant.prompt)],
                tool_choice="OptimizedPromptOutput",
            ).ainvoke(
                GRADIENT_DESCENT_APPLICATION_PROMPT.format(
                    existing_prompt=existing_prompt,
                    feedback=advice_msg.content,
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
            "Mutated Prompt",
        )
        return candidate


SEMANTIC_MUTATION_PROMPT = """You are a mutator. Given a prompt, your task is to generate another prompt with the same semantic meaning and intentions.

# Example:
current prompt: Classify the sentiment of the following sentence as either negative or positive:
mutated prompt: Determine the sentiment of the given sentence and assign a label from ['negative', 'positive'].

Given:
current prompt: {existing_prompt}
mutated prompt:
"""


class SemanticMutation(Mutation):
    async def mutate(
        self, population: list[Variant], train_examples: list[pm_types.Example]
    ) -> list[pm_types.PromptWrapper]:
        return await asyncio.gather(*(self.mutate_single(v) for v in population))

    async def mutate_single(self, variant: Variant) -> pm_types.PromptWrapper:
        existing_prompt = variant.prompt.get_prompt_str_in_context()
        with ls.trace(
            name="Semantic Mutation",
            inputs={"existing_prompt": existing_prompt},
        ) as rt:
            prompt_output = await create_extractor(
                self.model,
                tools=[pm_types.prompt_schema(variant.prompt)],
                tool_choice="OptimizedPromptOutput",
            ).ainvoke(SEMANTIC_MUTATION_PROMPT.format(existing_prompt=existing_prompt))
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
            "Mutated Prompt",
        )
        return candidate


EDA_PROMPT = """You are a mutator. Given a series of prompts, your task is to generate another prompt with the same semantic meaning and intentions.

## Existing Prompts ##
{existing_prompts}

The newly mutated prompt is:
"""

EDA_INDEX_PROMPT = """You are a mutator. Given a series of prompts, your task is to generate another prompt with the same semantic meaning and intentions. The series of prompts are ranked by their quality from best to worst.

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
            "Distilled Prompt",
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
        return sorted(cluster, key=lambda v: v.fitness, reverse=True)


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

    async def mutate(
        self, population: list[Variant], train_examples: list[pm_types.Example]
    ) -> list[pm_types.PromptWrapper]:
        pairs = self.produce_pairs(population)
        return await asyncio.gather(*(self.merge(pair) for pair in pairs))

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
            "Distilled Prompt",
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
    return PhaseConfig(
        mutation=config["mutation"],
        population_size=config.get("population_size", 5),
        improvement_threshold=config.get(
            "improvement_threshold", 0.2
        ),  # 20% "better" - kinda arbitrary
        max_attempts=config.get("max_attempts", 5),
    )


def load_mutation(
    config: PhaseConfig,
    model: optimizers.MODEL_TYPE,
) -> Mutation:
    config = config.copy()
    mutation_cls = MUTATIONS[config.pop("mutation")]
    return mutation_cls.from_config({**config, "model": model})
