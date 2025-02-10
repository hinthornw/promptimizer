import asyncio
from contextlib import AsyncExitStack
import csv
import datetime
import functools
import math
import os
import random
import statistics
import time
import weakref
import uuid
from collections import defaultdict
from dataclasses import asdict, is_dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Coroutine,
    Literal,
    Optional,
    TypeVar,
    Union,
    overload,
)
from uuid import UUID

import langsmith as ls
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langsmith.evaluation import _arunner, _runner
from langsmith.evaluation._arunner import ExperimentResultRow
from langsmith.schemas import Example, TracerSession
from langsmith.utils import ContextThreadPoolExecutor
from promptim import _utils as pm_utils
from promptim import config as pm_config
from promptim import optimizers as pm_optimizers
from promptim import types as pm_types
from rich import print as richprint
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress
from rich.table import Table
import logging

if TYPE_CHECKING:
    from promptim.algorithms import BaseAlgorithm


def ltq():
    return lambda x: x


_runner._load_tqdm = ltq
_arunner._load_tqdm = ltq


def _noop(*args, **kwargs):
    pass


_runner.print = _noop  # type: ignore
_arunner.print = _noop  # type: ignore
logger = logging.getLogger(__name__)


class PromptTrainer:
    """A framework for optimizing prompts through multi-task evaluation."""

    def __init__(
        self,
        optimizer: pm_optimizers.BaseOptimizer,
        algorithm: Optional["BaseAlgorithm"] = None,
        client: Optional[ls.Client] = None,
        seed: int = 43,
        experiment_dir: str = "~",
    ):
        """Initialize the trainer with a specific optimization algorithm.

        Args:
            optimizer: The optimization algorithm to use for improving prompts
            client: Optional LangSmith client. If not provided, a new one will be created
            seed: Random seed for reproducibility
        """
        self.optimizer = optimizer
        self.client = client or ls.Client()
        random.seed(seed)
        self.rng = random.Random(seed)
        self.algorithm = algorithm
        self._loop = asyncio.get_running_loop()
        self._aqueue: dict[asyncio.Future, Callable[[], Any]] = {}
        self._task = self._loop.create_task(_run(self._aqueue, weakref.ref(self)))
        self._experiment_dir = experiment_dir
        self._threadpool = ContextThreadPoolExecutor(max_workers=1)

    async def wait_for_all(self) -> None:
        """Wait for all queued tasks to complete."""
        if self._aqueue:
            await asyncio.gather(*self._aqueue.keys())
        self._threadpool.shutdown(wait=True)

    @classmethod
    def from_config(
        cls, config: dict | pm_config.Config, algo_config: dict, experiment_dir: str
    ):
        """Create a PromptTrainer from a configuration dictionary.

        Args:
            config: Either a MetaPromptOptimizerConfig or FewShotOptimizerConfig
        """
        from promptim import algorithms as pm_algorithms

        if is_dataclass(config):
            cp = asdict(config)
        else:
            cp = config.copy()
        kind = cp.get("kind") or "metaprompt"
        optimizer_config = {**cp, "kind": kind}
        if "model" not in optimizer_config:
            optimizer_config["model"] = cp.get(
                "model", pm_types.DEFAULT_OPTIMIZER_MODEL_CONFIG
            )
        optimizer = pm_optimizers.load_optimizer(optimizer_config)
        algorithm = pm_algorithms.load_algorithm(
            algo_config, optimizer_model=optimizer.model
        )
        return cls(
            optimizer=optimizer, algorithm=algorithm, experiment_dir=experiment_dir
        )

    def save_config(self) -> dict:
        return {
            "optimizer": self.optimizer.config,
            "algorithm": self.algorithm.config,
        }

    @ls.traceable(name="Train Prompt")
    async def train(
        self,
        task: pm_types.Task,
        *,
        initial_population: Union[
            pm_types.PromptWrapper,
            list[pm_types.PromptWrapper],
            None,
            str,
            dict,
            pm_types.PromptConfig,
        ] = None,
        system_config: Optional[dict] = None,
        annotation_queue: Optional[str] = None,
        commit_prompts: bool = False,
        experiment_name: Optional[str] = None,
    ) -> tuple[pm_types.PromptWrapper, float]:
        """
        Delegates the macro-level training flow to the specified algorithm.
        The trainer still handles data loading, concurrency, experiment creation, etc.
        """
        if initial_population is None:
            initial_population = task.initial_prompt
        if isinstance(initial_population, pm_types.PromptWrapper):
            initial_population = [initial_population]
        elif isinstance(initial_population, (str, dict, pm_types.PromptConfig)):
            initial_population = pm_types.PromptWrapper.from_config(initial_population)

        # Initialize the prompt(s)
        for prompt in initial_population:
            if prompt.prompt_str and commit_prompts:
                richprint(
                    "[yellow]Warning: No prompt identifier is configured for this run. "
                    "Prompts will not be committed.[/yellow]"
                )
                commit_prompts = False
            if task.system is None:
                task.system = task.get_prompt_system(prompt)
            prompt.load(self.client)

        train_examples, dev_examples, test_examples = await self._get_data(
            initial_population[0], task
        )
        experiment_name = experiment_name or f"Prompt Optimization - {task.name}"

        (
            baseline_scores,
            baseline_experiment_results,
        ) = await self._get_baseline_scores(
            task,
            train_examples,
            dev_examples,
            initial_population[-1],
            experiment_name=experiment_name,
            debug=self.algorithm.config.debug,
            system_config=system_config,
        )
        # TODO: fix potential baseline <> population mismatch
        best_prompt, _ = await self.algorithm.run(
            trainer=self,
            task=task,
            initial_population=initial_population,
            system_config=system_config,
            annotation_queue=annotation_queue,
            train_examples=train_examples,
            dev_examples=dev_examples,
            commit_prompts=commit_prompts,
            experiment_name=experiment_name,
            baseline_scores=baseline_scores,
            baseline_experiment_results=baseline_experiment_results,
        )
        _, final_test_scores = await self._get_test_scores(
            task,
            test_examples,
            initial_population[0],
            best_prompt,
            experiment_name,
            self.algorithm.config.debug,
            system_config,
        )
        pm_utils.print_rich_diff(
            initial_population[0].get_prompt_str_in_context(),
            best_prompt.get_prompt_str_in_context(),
            title="Final Prompt Updates",
        )
        await self.wait_for_all()
        test_score = statistics.mean(final_test_scores.values())
        return best_prompt, test_score

    async def optimize_prompt(
        self,
        task: pm_types.Task,
        initial_prompt: Optional[Union[str, dict, pm_types.PromptWrapper]] = None,
        train_size: Optional[int] = None,
        batch_size: int = 40,
        epochs: int = 5,
        debug: bool = False,
        system_config: Optional[dict] = None,
        annotation_queue: Optional[str] = None,
        commit_prompts: bool = False,
        experiment_name: Optional[str] = None,
    ) -> tuple[pm_types.PromptWrapper, float]:
        """Legacy method that uses MinibatchAlgorithm for backward compatibility."""
        from .algorithms import AlgorithmConfig, MinibatchAlgorithm

        algo_config = AlgorithmConfig(
            train_size=train_size,
            batch_size=batch_size,
            epochs=epochs,
            debug=debug,
        )
        algo = MinibatchAlgorithm(algo_config)

        if initial_prompt is None:
            initial_prompt = task.initial_prompt
        if isinstance(initial_prompt, (str, dict)):
            initial_prompt = pm_types.PromptWrapper.from_config(initial_prompt)

        return await self.train(
            task=task,
            algorithm=algo,
            initial_population=[initial_prompt],
            system_config=system_config,
            annotation_queue=annotation_queue,
            commit_prompts=commit_prompts,
            experiment_name=experiment_name,
        )

    async def _get_data(
        self, initial_prompt: pm_types.PromptWrapper, task: pm_types.Task
    ):
        # Print the original prompt
        richprint(
            Panel.fit(
                f"[bold cyan]Original Prompt:[/bold cyan]\n\n{initial_prompt.get_prompt_str_in_context(self.client)}",
                title="Initial Prompt to optimize:",
                border_style="bold",
            )
        )
        splits = {
            split
            for split in self.client.list_dataset_splits(dataset_name=task.dataset)
        }
        whole_banana = (
            "train" not in splits and "dev" not in splits and "test" not in splits
        )
        with Progress() as progress:
            ptsk = progress.add_task("[cyan]Loading data...", total=1)
            if whole_banana:
                progress.console.print(
                    "[yellow]No splits found! "
                    "We'll train on the test set, but remember: a split dataset is appealing![/yellow]"
                )
                all_examples = sorted(
                    self.client.list_examples(dataset_name=task.dataset),
                    key=lambda x: x.id,
                )
                if not all_examples:
                    raise ValueError(
                        "The dataset is empty. Please provide a non-empty dataset. "
                        "Ensure that you have correctly specified the dataset name in your config file, "
                        "and that the dataset has been properly uploaded to LangSmith. "
                        f"Current dataset name: '{task.dataset}'. "
                    )
                train_examples = all_examples.copy()
                dev_examples = all_examples.copy()
                test_examples = all_examples.copy()
                progress.console.print(
                    "[yellow]Warning: Using the same examples for train, dev, and test may lead to overfitting.[/yellow]"
                )
                if not train_examples:
                    raise ValueError(
                        "The dataset is empty. Please provide a non-empty dataset. "
                        "Ensure that you have correctly specified the dataset name in your config file, "
                        "and that the dataset has been properly uploaded to LangSmith. "
                        f"Current dataset name: '{task.dataset}'. "
                    )
            else:
                train_examples = sorted(
                    self.client.list_examples(
                        dataset_name=task.dataset, splits=["train"]
                    ),
                    key=lambda x: x.id,
                )
                dev_examples = sorted(
                    self.client.list_examples(
                        dataset_name=task.dataset, splits=["dev"]
                    ),
                    key=lambda x: x.id,
                )
                test_examples = sorted(
                    self.client.list_examples(
                        dataset_name=task.dataset, splits=["test"]
                    ),
                    key=lambda x: x.id,
                )
                if not train_examples:
                    ids_ = {example.id for example in dev_examples + test_examples}
                    train_examples = sorted(
                        [
                            example
                            for example in self.client.list_examples(
                                dataset_name=task.dataset
                            )
                            if example.id not in ids_
                        ],
                        key=lambda x: x.id,
                    )
                    del ids_
            train_examples, dev_examples, test_examples = self._validate_split_examples(
                train_examples, dev_examples, test_examples, progress.console
            )

            progress.update(ptsk, advance=1)

        return train_examples, dev_examples, test_examples

    async def _get_baseline_scores(
        self,
        task: pm_types.Task,
        train_examples: list[Example],
        dev_examples: list[Example],
        current_prompt: pm_types.PromptWrapper,
        experiment_name: str,
        debug: bool,
        system_config: Optional[dict] = None,
    ) -> tuple[dict[str, float] | None, list[ExperimentResultRow] | None]:
        # Step 1: Get baseline scores
        if task.baseline_experiment:
            return (await self._fetch_baseline_metrics(task.baseline_experiment)), None
        else:
            baseline_session = await _queue(
                self,
                _create_experiment,
                client=self.client,
                dataset_id=train_examples[0].dataset_id,
                experiment_name=experiment_name + f"- baseline {uuid.uuid4().hex[:6]}",
            )
            baseline_experiment_results = await self._evaluate_prompt(
                current_prompt,
                task,
                dev_examples,
                debug=debug,
                system_config=system_config,
                experiment=baseline_session,
            )
            if experiment_url := _get_url(baseline_session, dev_examples[0].dataset_id):
                print(f"See baseline experiment at: {experiment_url}")

            return (
                await self.calculate_scores(baseline_experiment_results)
            ), baseline_experiment_results

    async def _get_test_scores(
        self,
        task: pm_types.Task,
        test_examples: list[Example],
        initial_prompt: pm_types.PromptWrapper,
        best_prompt: pm_types.PromptWrapper,
        experiment_name: str,
        debug: bool,
        system_config: Optional[dict] = None,
        num_repetitions: int = 5,
    ):
        test_baseline_session_fut = self._enqueue_experiment(
            experiment_name=experiment_name,
            examples=test_examples,
            split="test",
            metadata={"which": "baseline"},
        )
        test_final_session_fut = self._enqueue_experiment(
            experiment_name=experiment_name,
            examples=test_examples,
            split="test",
            metadata={"which": "final"},
        )

        initial_test_results = await self._evaluate_prompt(
            initial_prompt,
            task,
            test_examples,
            debug=debug,
            system_config=system_config,
            experiment=await test_baseline_session_fut,
            num_repetitions=num_repetitions,
        )
        final_test_results = await self._evaluate_prompt(
            best_prompt,
            task,
            test_examples,
            debug=debug,
            experiment=await test_final_session_fut,
            system_config=system_config,
            num_repetitions=num_repetitions,
        )
        # Print final report
        initial_scores, initial_stdev, initial_ci = await self.calculate_scores(
            initial_test_results, ci=True
        )
        final_scores, final_stdev, final_ci = await self.calculate_scores(
            final_test_results, ci=True
        )

        table = Table(
            title="Optimization Results Test",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Initial Score", justify="right", style="green")
        table.add_column("Final Score", justify="right", style="green")

        for metric in initial_scores.keys():
            initial_std = initial_stdev.get(metric, (None, None))
            final_std = final_stdev.get(metric, (None, None))
            table.add_row(
                metric,
                f"{initial_scores[metric]:.4f} ({initial_std[0]:.4f}-{initial_std[1]:.4f})",
                f"{final_scores[metric]:.4f} ({final_std[0]:.4f}-{final_std[1]:.4f})",
            )
        initial_average = sum(initial_scores.values()) / len(initial_scores)
        final_average = sum(final_scores.values()) / len(final_scores)
        initial_lower, initial_upper = initial_ci
        final_lower, final_upper = final_ci
        table.add_row(
            "Average",
            f"{initial_average:.4f} ({initial_lower:.4f}-{initial_upper:.4f})",
            f"{final_average:.4f} ({final_lower:.4f}-{final_upper:.4f})",
        )
        richprint(Panel(table, title="Final Report", border_style="bold"))

        self.log_metric(
            "score",
            value=initial_average,
            x=0,
            x_label="base",
            split="test",
            bounds=(initial_ci[0], initial_ci[1]),
            prompt=initial_prompt,
        )
        self.log_metric(
            "score",
            value=final_average,
            x=0,
            x_label="final",
            split="test",
            bounds=(final_ci[0], final_ci[1]),
            prompt=best_prompt,
        )
        tokens_used = pm_utils.get_token_usage()
        if tokens_used is not None:
            self.log_metric(
                "score",
                value=final_average,
                x=tokens_used,
                x_label="total tokens",
                split="dev",
                bounds=(final_ci[0], final_ci[1]),
                prompt=best_prompt,
            )
        return (
            initial_scores,
            final_scores,
        )

    async def _wait_for_annotation_queue(
        self,
        results: list[ExperimentResultRow],
        queue_name: str,
        task: pm_types.Task,
        progress: Progress,
    ) -> tuple[list[ExperimentResultRow], Literal["continue"]]:
        """Add runs to the queue and block to let a reviewer check the outputs and leave feedback."""
        # Clear the queue of old things and add the new ones on.
        queues = list(self.client.list_annotation_queues(name=queue_name))
        if queues:
            q = queues[0]
            while True:
                try:
                    r = self.client.get_run_from_annotation_queue(q.id, index=0)
                    self.client.delete_run_from_annotation_queue(q.id, run_id=r.id)
                except Exception:
                    break
        else:
            q = self.client.create_annotation_queue(
                name=queue_name,
                description=f"Annotation queue used for prompt optimization on {task.name}",
            )
        runs = [str(r["run"].id) for r in results]
        N = 10
        for i in range(N):
            try:
                self.client.add_runs_to_annotation_queue(str(q.id), run_ids=runs)
                break
            except Exception:
                if i == N - 1:
                    raise
                time.sleep(i)

        # Now, log instructions and await user input in the terminal.
        # User input can either continue or break the loop
        richprint(
            Panel.fit(
                f"[bold cyan]Annotation Queue Instructions:[/bold cyan]\n\n"
                f"1. Go to {self.client._host_url}/o/{self.client._get_optional_tenant_id()}/annotation-queues/{q.id}/?runIndex=0\n"
                f"2. Review the outputs and leave feedback on the runs.\n"
                f"3. When finished, return here and enter 'c'/'continue' to proceed or 'q'/'quit' to exit.\n",
                title="Manual Review Required",
                border_style="bold",
            )
        )
        # Wait for the user to annotate some runs
        user_input = "continue"
        progress.stop()
        console = progress.console
        while True:
            try:
                user_input = (
                    console.input(
                        "\n\n[bold]Enter 'c'/'continue' to proceed, or 'q' to exit:[/bold] "
                    )
                    .strip()
                    .lower()
                )
                if user_input in ["c", "continue", "q", "quit"]:
                    break
                elif user_input == "":  # Handle EOF (Ctrl+D on Unix, Ctrl+Z on Windows)
                    console.print("\n[yellow]EOF detected. Exiting...[/yellow]")
                    user_input = "q"
                    break
                else:
                    console.print(
                        "[red]Invalid input. Please enter 'continue', 'break', or 'q'.[/red]"
                    )
            except KeyboardInterrupt:
                console.print(
                    "[yellow]Ctrl+C detected. Please enter 'continue', 'break', or 'q'.[/yellow]"
                )
            except EOFError:
                console.print("\n[yellow]EOF detected. Exiting...[/yellow]")
                user_input = "q"
                break
            except Exception as e:
                console.print(f"[red]An error occurred: {e}. Please try again.[/red]")

        if user_input == "q":
            console.print("[bold red]Exiting the whole process...[/bold red]")
            import sys

            sys.exit(0)
        progress.start()
        # Merge the user feedback in with the model feedback (stored locally)
        feedback = list(
            self.client.list_feedback(run_ids=runs, feedback_source_type="app")
        )
        results_dict = {r["run"].id: r for r in results}
        for f in feedback:
            results_dict[f.run_id]["evaluation_results"]["results"].append(
                ls.EvaluationResult(key=f.key, score=f.score, comment=f.comment)
            )

        return list(results_dict.values()), user_input

    @ls.traceable(process_outputs=lambda _: {})
    async def _evaluate_prompt(
        self,
        prompt_config: pm_types.PromptWrapper,
        task: pm_types.Task,
        data: str | list,
        debug: bool = False,
        experiment: str | TracerSession | None = None,
        system_config: dict | None = None,
        num_repetitions: int = 1,
        upload_results: bool = True,
    ) -> list[ExperimentResultRow]:
        """Evaluates a prompt against a task's dataset and evaluators."""
        prompt = prompt_config.load(self.client)
        metadata = {
            "prompt": prompt_config.identifier if prompt_config.identifier else "local"
        }
        rt = ls.get_current_run_tree()

        async def predict(inputs: dict):
            stack = AsyncExitStack()
            prt = None
            if not upload_results:
                prt = await stack.enter_async_context(
                    ls.trace("predict", inputs=inputs, parent=rt)
                )
            async with stack:
                if system_config:
                    result = await task.system_safe(prompt, inputs, **system_config)
                else:
                    result = await task.system_safe(prompt, inputs)
                if prt is not None:
                    if isinstance(result, dict):
                        prt.add_outputs(result)
                    else:
                        prt.add_outputs({"output": result})
            return result

        with ls.tracing_context(parent={"langsmith-trace": ""}):
            results = await ls.aevaluate(
                predict,
                data=data,
                evaluators=task.evaluators,
                max_concurrency=0 if debug else None,
                experiment=experiment,
                experiment_prefix="Optimizer" if not experiment else None,
                num_repetitions=num_repetitions,
                metadata=metadata,
                upload_results=upload_results,
            )
        now = datetime.datetime.now(datetime.timezone.utc)
        if results._manager._experiment is not None:
            await _queue(
                self,
                self.client.update_project,
                project_id=results._manager._experiment.id,
                end_time=now + datetime.timedelta(days=999),
            )

        return [r async for r in results]

    @overload
    async def calculate_scores(
        self, results: list[ExperimentResultRow], ci: Literal[True] = True
    ) -> tuple[dict[str, float], dict[str, tuple[float, float]], tuple[float, float]]:
        """Calculates aggregate scores from evaluation results, grouped by key."""

    @overload
    async def calculate_scores(
        self, results: list[ExperimentResultRow], ci: Literal[False] = False
    ) -> dict[str, float]:
        """Calculates aggregate scores from evaluation results, grouped by key."""

    @ls.traceable(process_inputs=lambda _: {})
    async def calculate_scores(
        self,
        results: list[ExperimentResultRow],
        ci: bool = False,
    ) -> Union[
        dict[str, float],
        tuple[dict[str, float], dict[str, tuple[float, float]], tuple[float, float]],
    ]:
        """
        Calculate and return:
          - If ci=False: {key: mean_score}
          - If ci=True : ( {key: mean_score},
                           {key: (lower, upper)},
                           (global_lower, global_upper) )
        Where "global" means combining all data across all keys into a single distribution or proportion.
        """

        scores_dict = defaultdict(list)
        all_scores = []

        for result in results:
            for res in result["evaluation_results"]["results"]:
                if res.score is not None:
                    scores_dict[res.key].append(res.score)
                    all_scores.append(res.score)

        means = {}
        cintervals = {}

        for key, vals in scores_dict.items():
            if len(vals) == 0:
                means[key] = 0.0
                if ci:
                    cintervals[key] = (0.0, 0.0)
                continue

            is_binomial = all(v in (0, 1) for v in vals)
            mean_ = statistics.mean(vals)
            means[key] = mean_

            if ci:
                if is_binomial:
                    x = sum(vals)
                    n = len(vals)
                    lower, upper = _wilson_score_interval(x, n)
                else:
                    lower, upper = _continuous_confidence_interval(vals)
                cintervals[key] = (lower, upper)

        if not ci:
            return means

        if len(all_scores) == 0:
            global_ci = (0.0, 0.0)
        else:
            all_binomial = all(v in (0, 1) for v in all_scores)
            if all_binomial:
                x = sum(all_scores)
                n = len(all_scores)
                global_ci = _wilson_score_interval(x, n)
            else:
                global_ci = _continuous_confidence_interval(all_scores)

        return (means, cintervals, global_ci)

    def log_metric(
        self,
        metric_name: str,
        /,
        value: float,
        x: int | float,
        x_label: str = "epoch",
        split: str = "dev",
        bounds: tuple[float, float] | None = None,
        prompt: pm_types.PromptWrapper | None = None,
    ):
        self._threadpool.submit(
            self._log_metric, metric_name, value, x, x_label, split, bounds, prompt
        )

    def _log_metric(
        self,
        metric_name: str,
        value: float,
        x: int | float,
        x_label: str = "epoch",
        split: str = "dev",
        bounds: tuple[float, float] | None = None,
        prompt: pm_types.PromptWrapper | None = None,
    ):
        fname = os.path.join(self._experiment_dir, "metrics.csv")
        nexist = not os.path.exists(fname)
        with open(fname, "a", newline="") as f:
            writer = csv.writer(f)
            if nexist:
                writer.writerow(
                    ["x", "y", "x_label", "metric", "split", "lower", "upper", "prompt"]
                )
            ps = prompt.dumps(push=True) if prompt else ""
            lower, upper = bounds or ("", "")
            writer.writerow([x, value, x_label, metric_name, split, lower, upper, ps])

    async def _fetch_baseline_metrics(self, experiment_id: UUID) -> dict:
        """Fetches metrics for a baseline experiment."""
        # Implementation to fetch metrics from LangSmith using the experiment ID
        test_results = self.client.get_test_results(project_id=experiment_id)
        metric_cols = [
            col for col in test_results.columns if col.startswith("feedback.")
        ]
        return {col: test_results[col].mean() for col in metric_cols}

    def _enqueue_experiment(
        self,
        experiment_name: str,
        examples: list[Example],
        split: str,
        epoch: int | None = None,
        metadata: dict | None = None,
    ):
        metadata = metadata or {}
        if epoch is not None:
            metadata["epoch"] = epoch
        return _queue(
            self,
            _create_experiment,
            client=self.client,
            dataset_id=examples[0].dataset_id,
            experiment_name=experiment_name + f" - {split} [epoch {epoch}]",
            metadata={
                "split": split,
                **metadata,
            },
        )

    @staticmethod
    def _validate_split_examples(
        train_examples: list[Example],
        dev_examples: list[Example],
        test_examples: list[Example],
        console: Console,
    ) -> tuple[list[Example], list[Example], list[Example]]:
        """Validate and potentially adjust the split examples."""
        if not train_examples:
            raise ValueError(
                "Train examples list is empty. Please provide training data."
            )

        if not dev_examples:
            console.log(
                "[yellow]Warning: Dev examples list is empty. Using train examples for dev set.[/yellow]"
            )
            dev_examples = train_examples

        if not test_examples:
            console.log(
                "[yellow]Warning: Test examples list is empty. Using dev examples for test set.[/yellow]"
            )
            test_examples = dev_examples

        return train_examples, dev_examples, test_examples


class PromptOptimizer(PromptTrainer):
    """Legacy wrapper for backward compatibility that uses the MetaPromptOptimizer."""

    def __init__(
        self,
        model: BaseChatModel,
        meta_prompt: Optional[str] = None,
        seed: int = 42,
    ):
        optimizer = pm_optimizers.MetaPromptOptimizer(
            model=model,
            meta_prompt=meta_prompt or pm_types.DEFAULT_METAPROMPT,
        )
        super().__init__(optimizer=optimizer, seed=seed)
        self.model = model  # For backward compatibility
        self.meta_prompt = meta_prompt or pm_types.DEFAULT_METAPROMPT

    @classmethod
    def from_config(cls, config: dict):
        """Legacy config method that assumes metaprompt optimizer."""
        cp = config.copy()
        model_config = cp.pop("model", pm_types.DEFAULT_OPTIMIZER_MODEL_CONFIG)
        model = init_chat_model(**model_config)
        meta_prompt = cp.pop("meta_prompt", None)
        return cls(model=model, meta_prompt=meta_prompt, **cp)

    async def apply_metaprompt(
        self,
        current_prompt: pm_types.PromptWrapper,
        meta_prompt: str,
        task: pm_types.Task,
        results: list[ExperimentResultRow],
        other_attempts: list | None = None,
    ) -> pm_types.PromptWrapper:
        """Legacy method for backward compatibility."""
        return await self.optimizer.improve_prompt(
            current_prompt=current_prompt,
            results=results,
            task=task,
            other_attempts=other_attempts or [],
        )


T = TypeVar("T")
SyncOrAsyncCallable = Union[Callable[[], T], Callable[[], Coroutine[Any, Any, T]]]


def _get_url(project: Any, dataset_id: str) -> None | str:
    if project and project.url:
        project_url = project.url.split("?")[0]
        base_url = project_url.split("/projects/p/")[0]
        return (
            f"{base_url}/datasets/{dataset_id}/compare?"
            f"selectedSessions={project.id}"
        )


def _create_experiment(
    client: ls.Client,
    dataset_id: str,
    experiment_name: str,
    description: str | None = None,
    metadata: dict | None = None,
):
    """Create a new experiment with an incrementing index in the name.

    The naming scheme is: "{experiment_name} [idx]" where idx starts at 1
    and increments for each existing experiment with the same base name.
    """
    from langsmith import utils as ls_utils

    # Find existing experimens with the same base name
    existing = list(
        client.list_projects(
            reference_dataset_id=dataset_id,
            name_contains=experiment_name,
            metadata={"__creation_source": "promptim"},
        )
    )

    # Extract indices from existing names
    indices = []
    for p in existing:
        try:
            if p.name.startswith(experiment_name):
                # Extract [idx] from the end if it exists
                if match := p.name.strip().split(" ")[-1]:
                    if match.startswith("[") and match.endswith("]"):
                        try:
                            idx = int(match[1:-1])
                            indices.append(idx)
                        except ValueError:
                            continue
        except (ValueError, IndexError):
            continue

    # Get next available index
    next_idx = max(indices, default=-1) + 1

    # Create new experiment name with index
    new_name = f"{experiment_name} [{next_idx}]"

    num_attempts = 10
    for attempt in range(num_attempts):
        try:
            return client.create_project(
                new_name,
                description=description,
                reference_dataset_id=dataset_id,
                metadata={"__creation_source": "promptim", **(metadata or {})},
            )
        except ls_utils.LangSmithConflictError:
            # If there's a conflict, increment the index and try again
            next_idx += 1
            new_name = f"{experiment_name} [{next_idx}-{uuid.uuid4().hex[:4]}]"
    raise ValueError(
        f"Could not find a unique experiment name ({experiment_name})in {num_attempts} attempts."
        " Please try again with a different experiment name."
    )


def _queue(trainer, func: SyncOrAsyncCallable[T], *args, **kwargs) -> asyncio.Future[T]:
    """Queue a function to be executed by the trainer.

    Args:
        func: Function to execute, can be sync or async

    Returns:
        Future that will contain the result of the function
    """
    fut = trainer._loop.create_future()
    trainer._aqueue[fut] = functools.partial(func, *args, **kwargs)
    return fut


async def _run(
    aqueue: dict[asyncio.Future[T], SyncOrAsyncCallable[T]],
    trainer_ref: weakref.ReferenceType["PromptTrainer"],
) -> None:
    """Run queued functions, either sync or async, in order."""
    loop = asyncio.get_running_loop()

    while True:
        await asyncio.sleep(0)

        if not aqueue:
            continue

        trainer = trainer_ref() if trainer_ref is not None else None
        if trainer is None:
            break

        current_batch = aqueue.copy()

        try:
            results = []
            for fut, func in current_batch.items():
                try:
                    if asyncio.iscoroutinefunction(func):
                        result = await func()
                    else:
                        result = await loop.run_in_executor(None, func)
                    results.append((fut, result))
                except Exception as e:
                    fut.set_exception(e)
                    del aqueue[fut]

            for fut, result in results:
                fut.set_result(result)
                del aqueue[fut]

        except Exception as e:
            for fut in current_batch:
                if not fut.done():
                    fut.set_exception(e)
                if fut in aqueue:
                    del aqueue[fut]


def _z_value_for_conf() -> float:
    # For alpha 0.05 lol
    return 1.96


T_CRITICAL_VALUES = {
    2: 12.706,
    3: 4.303,
    4: 3.182,
    5: 2.776,
    6: 2.571,
    7: 2.447,
    8: 2.365,
    9: 2.306,
    10: 2.262,
    11: 2.228,
    12: 2.201,
    13: 2.179,
    14: 2.160,
    15: 2.145,
    16: 2.131,
    17: 2.120,
    18: 2.110,
    19: 2.101,
    20: 2.093,
    21: 2.086,
    22: 2.080,
    23: 2.074,
    24: 2.069,
    25: 2.064,
    26: 2.060,
    27: 2.056,
    28: 2.052,
    29: 2.048,
    30: 2.045,
}


def _t_value_for_conf(n: int) -> float:
    """
    Returns the t critical value for alpha (default=0.05 => ~95% CI)
    with df = n - 1. Uses precomputed lookup for n <= 30.
    """
    if n <= 1:
        return float("inf")
    if n <= 30:
        return T_CRITICAL_VALUES.get(n, float("inf"))
    # Fallback approximation for n > 30 (as t approaches z value ~1.96 for 95% CI)
    return 1.96


def _continuous_confidence_interval(values: list[float]) -> tuple[float, float]:
    """
    Uses a t-based approach if n <= 30, else z-based for "large" n.
    Returns (lower, upper).
    """

    n = len(values)
    if n == 0:
        return (0.0, 0.0)
    mean_ = statistics.mean(values)
    if n == 1:
        return (mean_, mean_)  # Degenerate case

    stdev_ = statistics.stdev(values)
    t_crit = _t_value_for_conf(n)
    margin = t_crit * (stdev_ / math.sqrt(n))
    return (mean_ - margin, mean_ + margin)


def _wilson_score_interval(x: int, n: int) -> tuple[float, float]:
    """
    Wilson interval for binomial data, default 95% confidence (alpha=0.05).
    """
    import math

    if n == 0:
        return (0.0, 0.0)
    z = 1.96
    p_hat = x / n
    denom = 1 + z**2 / n
    center = p_hat + z**2 / (2 * n)
    main = z * math.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n)
    lower = (center - main) / denom
    upper = (center + main) / denom
    return (lower, upper)
