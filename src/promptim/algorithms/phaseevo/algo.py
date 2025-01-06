from typing import List, Union, Optional, cast

from promptim import types as pm_types
from promptim.trainer import PromptTrainer

from promptim.optimizers import base as optimizers
from promptim.algorithms.base import BaseAlgorithm, AlgorithmConfig
from promptim.algorithms.phaseevo import mutations
from dataclasses import dataclass, field


def default_curriculum():
    return [
        mutations.ensure_phase_config(config)
        for config in [
            {"mutation": "lamarckian", "max_attempts": 1, "population_size": 15},
            {"mutation": "gradient", "max_attempts": 1},
            {"mutation": "eda-index"},
            {"mutation": "crossover-distinct"},
            {"mutation": "semantic"},
        ]
    ]


@dataclass(kw_only=True)
class EvolutionaryConfig(AlgorithmConfig):
    """Configuration for evolutionary algorithms."""

    phases: list[mutations.PhaseConfig] = field(default_factory=default_curriculum)


class PhaseEvoAlgorithm(BaseAlgorithm[EvolutionaryConfig]):
    """
    Population-based optimization using evolutionary principles.
    """

    config_cls = EvolutionaryConfig

    async def run(
        self,
        trainer: PromptTrainer,
        task: pm_types.Task,
        initial_population: Union[pm_types.PromptWrapper, List[pm_types.PromptWrapper]],
        train_examples: list[pm_types.Example],
        dev_examples: list[pm_types.Example],
        *,
        system_config: Optional[dict] = None,
        annotation_queue: Optional[str] = None,
        commit_prompts: bool = False,
        experiment_name: str = "Prompt Optimization",
        baseline_scores: Optional[dict] = None,
        baseline_experiment_results: Optional[list] = None,
    ) -> tuple[pm_types.PromptWrapper, float]:
        """
        Runs evolutionary optimization:
        1. Evaluate population
        2. Select parents
        3. Create offspring through crossover/mutation
        4. Optionally apply local improvements
        5. Repeat for N generations
        """
        if isinstance(initial_population, pm_types.PromptWrapper):
            initial_population = [initial_population]
        if not baseline_experiment_results:
            raise ValueError("baseline_experiment_results is required")

        config = cast(EvolutionaryConfig, self.config)
        phases = [PhaseRunner(phase, self.model) for phase in config.phases]

        # The initial population stuff is wrong; assumes single
        population = [
            mutations.Variant(prompt=prompt, results=baseline_experiment_results)
            for prompt in initial_population
        ]
        for phase in phases:
            population = await phase.run(
                population,
                train_examples,
                dev_examples,
                trainer,
                task,
                debug=self.config.debug,
                system_config=system_config,
            )
        best_prompt = population[0]
        return best_prompt.prompt, best_prompt.fitness


class PhaseRunner:
    def __init__(
        self,
        phase: mutations.PhaseConfig,
        model: optimizers.MODEL_TYPE,
    ):
        self.phase = phase
        self.mutation = mutations.load_mutation(phase, model=model)

    async def run(
        self,
        population: list[mutations.Variant],
        train_examples: list[pm_types.Example],
        dev_examples: list[pm_types.Example],
        trainer: PromptTrainer,
        task: pm_types.Task,
        debug: bool = False,
        system_config: Optional[dict] = None,
    ) -> list[mutations.Variant]:
        retained = population.copy()
        starting_fitness = sum(v.fitness for v in population) / len(population)
        for _ in range(self.phase["max_attempts"]):
            generated = await self.mutation.mutate(population, train_examples)
            candidate_variants = []
            for prompt in generated:
                results = await trainer._evaluate_prompt(
                    prompt,
                    task,
                    dev_examples,
                    debug=debug,
                    system_config=system_config,
                )
                candidate_variants.append(
                    mutations.Variant(prompt=prompt, results=results)
                )

            # Prune to top N
            retained = sorted(
                retained + candidate_variants, key=lambda v: v.fitness, reverse=True
            )[: self.phase["population_size"]]
            improvement = (
                sum(v.fitness - starting_fitness for v in retained) / len(retained)
            ) / starting_fitness - 1.0
            if improvement > self.phase["improvement_threshold"]:
                break
        return retained
