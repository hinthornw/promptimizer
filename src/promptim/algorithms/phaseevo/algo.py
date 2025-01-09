from typing import List, Union, Optional, cast
from rich.progress import Progress, Live

from promptim import types as pm_types
from promptim.trainer import PromptTrainer

from promptim.optimizers import base as optimizers
from promptim import _utils as pm_utils
from promptim.algorithms.base import BaseAlgorithm, AlgorithmConfig
from promptim.algorithms.phaseevo import mutations
from dataclasses import dataclass, field
import langsmith as ls


def default_curriculum():
    return [
        mutations.ensure_phase_config(config)
        for config in [
            {
                "mutation": "lamarckian",
                "max_attempts": 1,
                "population_limit": 15,
            },
            {
                "mutation": "semantic",
                "max_attempts": 1,
                "population_limit": 15,
                "stop_at_population_limit": True,
            },
            {"mutation": "gradient", "max_attempts": 4, "improvement_threshold": 1.0},
            {"mutation": "eda-index"},
            {"mutation": "crossover-distinct"},
            {"mutation": "gradient", "max_attempts": 2, "improvement_threshold": 1.0},
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
    ) -> tuple[
        pm_types.PromptWrapper,
        float,
    ]:
        from rich.console import Group
        from rich.panel import Panel
        from rich import box
        from rich.columns import Columns

        fitness_history = []
        phase_points = {}  # Track which points belong to which phase

        with ls.trace(
            "PhaseEvo",
            inputs={
                "task": task.name,
                "algorithm": "phaseevo",
                "initial_prompt": initial_population[0].prompt_str,
            },
        ) as rt:
            print(f"View Optimizers run: {rt.get_url()}")
            config = cast(EvolutionaryConfig, self.config)
            phase_names = [phase["mutation"] for phase in config.phases]

            # Create progress instance for metrics
            metrics_progress = Progress()

            # Create separate progress for best/avg/min fitness and population
            best_fitness_task = metrics_progress.add_task(
                "[green]Best Fitness", total=config.max_score
            )
            avg_fitness_task = metrics_progress.add_task(
                "[yellow]Average Fitness", total=config.max_score
            )
            min_fitness_task = metrics_progress.add_task(
                "[red]Min Fitness", total=config.max_score
            )
            population_task = metrics_progress.add_task(
                "[blue]Population Size", total=None, visible=True
            )

            if isinstance(initial_population, pm_types.PromptWrapper):
                initial_population = [initial_population]
            if not baseline_experiment_results:
                raise ValueError("baseline_experiment_results is required")

            phases = [PhaseRunner(phase, self.model) for phase in config.phases]

            population = [
                mutations.Variant(prompt=prompt, results=baseline_experiment_results)
                for prompt in initial_population
            ]

            # Initialize population metrics
            metrics_progress.update(population_task, total=len(population))
            initial_fitness = population[0].fitness
            fitness_history.append(initial_fitness)
            token_usage = [0]
            phase_points["initial"] = 0  # Mark initial point

            metrics_progress.update(best_fitness_task, completed=initial_fitness)
            metrics_progress.update(avg_fitness_task, completed=initial_fitness)
            metrics_progress.update(min_fitness_task, completed=initial_fitness)

            def generate_chart():
                from rich.text import Text

                if len(fitness_history) < 2:
                    return Panel("Collecting fitness data...", title="Fitness History")

                # Create ASCII chart with phase markers
                width = 60  # Full width since we're using vertical layout
                height = 12  # Slightly taller for better visibility
                max_val = max(fitness_history)
                min_val = min(fitness_history)
                range_val = max_val - min_val if max_val != min_val else 1

                # Initialize chart with spaces
                chart = [[" " for _ in range(width)] for _ in range(height)]

                # Plot points with phase markers
                for x, value in enumerate(fitness_history):
                    if x >= width:
                        break
                    y = int(
                        (height - 2) * (value - min_val) / range_val
                    )  # Leave room for x-axis

                    # Use different markers for different phases
                    marker = "•"
                    for phase_name, phase_idx in phase_points.items():
                        if x == phase_idx and phase_name != "initial":
                            marker = "◆"  # Diamond marker for phase starts (except initial point)

                    chart[height - 2 - y][x] = marker

                # Add x-axis with phase markers
                x_axis = ["-" for _ in range(width)]
                for phase_name, phase_idx in phase_points.items():
                    if phase_idx < width:
                        x_axis[phase_idx] = "┼"
                chart[-2] = x_axis

                # Add phase labels on bottom
                phase_label = [" " for _ in range(width)]
                for phase_name, phase_idx in phase_points.items():
                    if phase_idx < width and phase_name != "initial":
                        # Get first letter of phase name
                        label = phase_name[0].lower()
                        phase_label[phase_idx] = label
                chart[-1] = phase_label

                # Convert to text with colors
                chart_text = Text()
                for row in chart:
                    for char in row:
                        if char == "◆":
                            chart_text.append(char, style="bold yellow")
                        elif char == "•":
                            chart_text.append(char, style="green")
                        else:
                            chart_text.append(char)
                    chart_text.append("\n")

                # Add y-axis labels
                chart_text.append(f"Range: {min_val:.2f} - {max_val:.2f}")

                return Panel(
                    chart_text,
                    title="[yellow]Fitness History by Phase",
                    box=box.ROUNDED,
                )

            # Create a function to render both progress and chart side by side
            def get_renderable():
                # Create phase progress display
                phase_progress = Progress()

                main_task = phase_progress.add_task(
                    "[cyan]Phase Progress",
                    total=len(phase_names),
                    completed=len(phase_points) - 1,  # Subtract 1 for initial point
                )

                # Update phase progress description
                phase_indicators = []
                current_phase_idx = len(phase_points) - 1
                for j, name in enumerate(phase_names):
                    if j < current_phase_idx:
                        indicator = f"[green]✓ {name}[/green]"
                    elif j == current_phase_idx:
                        indicator = f"[bold blue]► {name}[/bold blue]"
                    else:
                        indicator = f"[gray]{name}[/gray]"
                    phase_indicators.append(indicator)

                phase_progress.update(
                    main_task,
                    description=f"[blue]{' → '.join(phase_indicators)}",
                )

                # Arrange elements side by side
                return Columns(
                    [
                        Panel(
                            Group(phase_progress, metrics_progress),
                            title="Progress",
                            width=60,
                        ),
                        generate_chart(),
                    ],
                    equal=True,
                    expand=True,
                )

            # Single Live display for everything
            with Live(get_renderable(), refresh_per_second=4) as live:
                for i, phase in enumerate(phases):
                    # Run the phase
                    population = await phase.run(
                        population,
                        train_examples,
                        dev_examples,
                        trainer,
                        task,
                        debug=self.config.debug,
                        system_config=system_config,
                    )

                    # Calculate and update population statistics
                    fitness_scores = [variant.fitness for variant in population]
                    avg_fitness = sum(fitness_scores) / len(fitness_scores)
                    min_fitness = min(fitness_scores)
                    max_fitness = max(fitness_scores)
                    best_prompt = population[0]
                    trainer.log_metric(
                        "score",
                        value=max_fitness,
                        x=i,
                        x_label="epoch",
                        split="dev",
                        prompt=best_prompt.prompt,
                    )
                    tokens_used = pm_utils.get_token_usage()
                    if tokens_used is not None:
                        token_usage.append(tokens_used)
                        trainer.log_metric(
                            "score",
                            value=max_fitness,
                            x=tokens_used,
                            x_label="total tokens",
                            split="dev",
                            prompt=best_prompt.prompt,
                        )

                    # Update fitness history and phase points
                    fitness_history.append(max_fitness)
                    phase_points[phase_names[i]] = len(fitness_history) - 1

                    # Update progress bars
                    metrics_progress.update(
                        best_fitness_task,
                        completed=max_fitness,
                        description=f"[green]Best Fitness: {max_fitness:.2f}",
                    )
                    metrics_progress.update(
                        avg_fitness_task,
                        completed=avg_fitness,
                        description=f"[yellow]Avg Fitness: {avg_fitness:.2f}",
                    )
                    metrics_progress.update(
                        min_fitness_task,
                        completed=min_fitness,
                        description=f"[red]Min Fitness: {min_fitness:.2f}",
                    )
                    metrics_progress.update(
                        population_task,
                        completed=len(population),
                        description=f"[blue]Population Size: {len(population)}",
                    )

                    # Refresh the display
                    live.update(get_renderable())

                    # Log statistics for this phase
                    rt.add_metadata(
                        {
                            f"phase_{i}_{phase_names[i]}": {
                                "avg_fitness": avg_fitness,
                                "min_fitness": min_fitness,
                                "max_fitness": max_fitness,
                                "population_size": len(population),
                            }
                        }
                    )

                    best_prompt = population[0]
                    # Check for early completion
                    if best_prompt.fitness == config.max_score:
                        print(
                            f"[bold green]✓ Optimization complete![/bold green]\n"
                            f"Final Population Stats:\n"
                            f"- Best Fitness: {max_fitness:.2f}\n"
                            f"- Avg Fitness: {avg_fitness:.2f}\n"
                            f"- Min Fitness: {min_fitness:.2f}\n"
                            f"- Population Size: {len(population)}"
                        )
                        break

            best_prompt = population[0]
            rt.add_outputs(
                {
                    "best_prompt": best_prompt.prompt.get_prompt_str_in_context(),
                    "best_score": best_prompt.fitness,
                    "final_population_size": len(population),
                    "final_avg_fitness": avg_fitness,
                    "final_min_fitness": min_fitness,
                    "fitness_history": fitness_history,
                    "phase_points": phase_points,
                }
            )

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
        async with ls.trace(
            self.mutation.__class__.__name__,
            inputs={"starting_fitness": starting_fitness},
        ) as rt:
            improvement = None
            retained_fitness = None
            for attempt in range(self.phase["max_attempts"]):
                with ls.trace(f"Step {attempt}") as step_rt:
                    generated = await self.mutation.mutate(retained, train_examples)
                    candidate_variants = []
                    for prompt in generated:
                        with ls.tracing_context(tags=["langsmith:hidden"]):
                            with ls.trace(
                                "Evaluate Prompt",
                                inputs={"prompt": prompt.get_prompt_str_in_context()},
                            ) as rt:
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
                            rt.add_outputs({"fitness": candidate_variants[-1].fitness})
                    # Prune to top N
                    retained = sorted(
                        retained + candidate_variants,
                        key=lambda v: v.fitness,
                        reverse=True,
                    )[: self.phase["population_limit"]]
                    retained_fitness = sum(v.fitness for v in retained) / len(retained)
                    improvement = retained_fitness - starting_fitness
                    if improvement > self.phase["improvement_threshold"]:
                        break
                    step_rt.add_outputs(
                        {
                            "avg": retained_fitness,
                            "max": max(v.fitness for v in retained),
                            "min": min(v.fitness for v in retained),
                            "improvement": improvement,
                        }
                    )
            rt.add_outputs(
                {
                    "improvement": improvement,
                    "avg": retained_fitness,
                    "max": max(v.fitness for v in retained),
                    "min": min(v.fitness for v in retained),
                }
            )

        return retained
