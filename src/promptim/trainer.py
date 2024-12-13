import random
import time
from collections import defaultdict
from typing import Literal, Optional, Callable, Coroutine, TypeVar, Union, Any
from uuid import UUID
from dataclasses import asdict, is_dataclass
import datetime
import functools

import langsmith as ls
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langsmith.evaluation import _arunner, _runner
from langsmith.evaluation._arunner import ExperimentResultRow
from langsmith.schemas import Example, TracerSession
from promptim import types as pm_types
from promptim import _utils as pm_utils
from promptim import optimizers as pm_optimizers
from promptim import config as pm_config
from rich import print as richprint
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress
from rich.table import Table
import asyncio
import weakref


def ltq():
    return lambda x: x


_runner._load_tqdm = ltq
_arunner._load_tqdm = ltq


def _noop(*args, **kwargs):
    pass


_runner.print = _noop  # type: ignore
_arunner.print = _noop  # type: ignore


class PromptTrainer:
    """A framework for optimizing prompts through multi-task evaluation."""

    def __init__(
        self,
        optimizer: pm_optimizers.BaseOptimizer,
        client: Optional[ls.Client] = None,
        seed: int = 42,
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
        self._loop = asyncio.get_running_loop()
        self._aqueue: dict[asyncio.Future, Callable[[], Any]] = {}
        self._task = self._loop.create_task(_run(self._aqueue, weakref.ref(self)))

    async def wait_for_all(self) -> None:
        """Wait for all queued tasks to complete."""
        if self._aqueue:
            await asyncio.gather(*self._aqueue.keys())

    @classmethod
    def from_config(cls, config: dict | pm_config.Config):
        """Create a PromptTrainer from a configuration dictionary.

        Args:
            config: Either a MetaPromptOptimizerConfig or FewShotOptimizerConfig
        """
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

        return cls(optimizer=optimizer)

    async def optimize_prompt(
        self,
        task: pm_types.Task,
        *,
        system_config: dict | None = None,
        train_size: Optional[int] = None,
        batch_size: int = 40,
        epochs: int = 1,
        annotation_queue: str | None = None,
        debug: bool = False,
        commit_prompts: bool = False,
        experiment_name: str | None = None,
    ) -> tuple[pm_types.PromptWrapper, float]:
        """Optimizes a prompt for a specific task through multiple iterations."""
        experiment_name = experiment_name or f"Prompt Optimization - {task.name}"
        initial_prompt = pm_types.PromptWrapper.from_config(task.initial_prompt)
        if initial_prompt.prompt_str:
            if commit_prompts:
                richprint(
                    "[yellow]Warning: No prompt identifier is configured for this run. "
                    "Prompts will not be committed.[/yellow]"
                )
                commit_prompts = False
        if task.system is None:
            task.system = task.get_prompt_system(initial_prompt)
        initial_prompt.load(self.client)  # check
        current_prompt = initial_prompt
        best_score = 0
        best_prompt = initial_prompt
        other_attempts = []
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

        with Progress() as progress:
            main_task = progress.add_task(
                "[cyan]Optimizing prompt...", total=epochs + 2
            )

            # Step 1: Get baseline scores
            progress.update(
                main_task, advance=10, description="[cyan]Getting baseline scores..."
            )
            if task.baseline_experiment:
                baseline_scores = await self._fetch_baseline_metrics(
                    task.baseline_experiment
                )
            else:
                baseline_session = await _queue(
                    self,
                    _create_experiment,
                    client=self.client,
                    dataset_id=train_examples[0].dataset_id,
                    experiment_name=experiment_name + "- baseline",
                )
                baseline_experiment_results = await self._evaluate_prompt(
                    current_prompt,
                    task,
                    dev_examples,
                    debug=debug,
                    system_config=system_config,
                    experiment=baseline_session,
                )
                if experiment_url := _get_url(
                    baseline_session, dev_examples[0].dataset_id
                ):
                    progress.console.print(
                        f"See baseline experiment at: {experiment_url}"
                    )

                baseline_scores = await self.calculate_scores(
                    baseline_experiment_results
                )
            best_score = (
                sum(baseline_scores.values()) / len(baseline_scores)
                if baseline_scores
                else None
            )

            table = Table(
                title="Baseline Scores (Dev Set)",
                show_header=True,
                header_style="bold magenta",
            )
            table.add_column("Metric", style="cyan", no_wrap=True)
            table.add_column("Score", justify="right", style="green")

            for metric, score in baseline_scores.items():
                table.add_row(metric, f"{score:.4f}")

            table.add_row("Average", f"{best_score:.4f}", style="bold")

            progress.console.print(
                Panel(
                    table,
                    title="[bold]Initial Prompt Evaluation[/bold]",
                    border_style="cyan",
                )
            )
            progress.console.print("\n[bold cyan]Beginning optimization.[/bold cyan]")
            progress.console.print()

            # Step 2: Train
            progress.update(
                main_task,
                advance=1,
                description="[cyan]Optimizing prompt on epoch 1...",
            )
            training_session_fut = _queue(
                self,
                _create_experiment,
                client=self.client,
                dataset_id=train_examples[0].dataset_id,
                experiment_name=experiment_name + "- train [epoch 0]",
                metadata={
                    "split": "train",
                    "epoch": 0,
                },
            )
            for epoch in range(epochs):
                training_session = await training_session_fut
                if experiment_url := _get_url(
                    training_session, train_examples[0].dataset_id
                ):
                    progress.console.print(
                        f"See experiment for epoch {epoch} at: {experiment_url}"
                    )

                self.optimizer.on_epoch_start(epoch, task)
                self.rng.shuffle(train_examples)
                if train_size:
                    train_examples = train_examples[:train_size]

                batches = [
                    train_examples[i : i + batch_size]
                    for i in range(0, len(train_examples), batch_size)
                ]

                batch_task = progress.add_task(
                    f"[yellow]Epoch {epoch+1} batches", total=len(batches)
                )
                all_train_scores = []
                avg_score = -1
                for bix, batch in enumerate(batches):
                    if (
                        bix == 0
                        and epoch == 0
                        and whole_banana
                        and baseline_experiment_results
                    ):
                        bindices = {e.id for e in batch}
                        results = [
                            r
                            for r in baseline_experiment_results
                            if r["example"].id in bindices
                        ]
                    else:
                        results = await self._evaluate_prompt(
                            current_prompt,
                            task,
                            batch,
                            debug=debug,
                            experiment=training_session,
                            system_config=system_config,
                        )
                    training_session_fut = _queue(
                        self,
                        _create_experiment,
                        client=self.client,
                        dataset_id=train_examples[0].dataset_id,
                        experiment_name=experiment_name + f" - train [epoch {epoch+1}]",
                        metadata={"epoch": epoch + 1, "split": "train"},
                    )
                    next_action = "continue"
                    if annotation_queue:
                        results, next_action = await self._wait_for_annotation_queue(
                            results,
                            annotation_queue,
                            task,
                            progress,
                        )
                    train_scores = await self.calculate_scores(results)
                    train_score = (
                        sum(train_scores.values()) / len(train_scores)
                        if train_scores
                        else None
                    )
                    all_train_scores.append(train_score)
                    avg_score = sum(all_train_scores) / len(all_train_scores)
                    progress.update(
                        batch_task,
                        description=f"[yellow]Epoch {epoch+1} (Avg training score: {avg_score:.4f})",
                    )
                    with ls.tracing_context(project_name=training_session.name):
                        with ls.trace(
                            name="improve_prompt",
                            inputs={
                                "current": current_prompt.get_prompt_str(),
                            },
                        ) as rt:
                            setattr(rt, "session_id", training_session.id)
                            run_url = self.client.get_run_url(
                                run=rt, project_id=training_session.id
                            )
                            progress.console.print(f"See optimizer trace: {run_url}")
                            improved = await self.optimizer.improve_prompt(
                                current_prompt=current_prompt,
                                results=results,
                                task=task,
                                other_attempts=other_attempts,
                            )
                            rt.add_outputs({"improved": improved.get_prompt_str()})
                            current_prompt = improved
                    if commit_prompts:
                        pushed_id = current_prompt.push_prompt(client=self.client)
                        progress.console.print(f"See prompt checkpoint: {pushed_id}")

                    progress.update(
                        batch_task,
                        advance=1,
                        description=f"[yellow]Epoch {epoch+1} (Avg Score: {avg_score:.4f})",
                    )
                    if next_action != "continue":
                        break
                # Evaluate on dev set after each epoch
                progress.update(main_task, description="[cyan]Evaluating on dev set...")
                dev_results = await self._evaluate_prompt(
                    current_prompt,
                    task,
                    dev_examples,
                    debug=debug,
                    system_config=system_config,
                )
                dev_scores = await self.calculate_scores(dev_results)
                dev_score = (
                    sum(dev_scores.values()) / len(dev_scores) if dev_scores else None
                )
                progress.update(
                    batch_task,
                    description=f'[yellow]Epoch {epoch+1} (Dev: {f"{dev_score:.4f}" if dev_score is not None else "-"}, Train: {f"{avg_score:.4f}" if avg_score is not None else "-"})',
                )

                if dev_score is not None and dev_score > best_score:
                    if best_prompt not in other_attempts:
                        other_attempts.append(best_prompt)
                    best_score = dev_score
                    best_prompt = current_prompt
                    progress.console.print(
                        f"New best score: {best_score:.4f} (surpassed previous best)"
                    )
                    progress.console.print("Average of:")
                    for metric, score in dev_scores.items():
                        progress.console.print(f"  {metric}: {score:.4f}")
                else:
                    other_attempts.append(current_prompt)
                    current_prompt = best_prompt
                    progress.console.print(
                        f"Score {dev_score:.4f} did not surpass best score {best_score:.4f}"
                    )
                progress.console.print()

                progress.console.print(
                    Panel(
                        f"[bold]Epoch {epoch+1}[/bold]\n"
                        f"Dev score: [cyan]{dev_score:.4f}[/cyan]\n"
                        f"Best score: [green]{best_score:.4f}[/green]",
                        title="Training Progress",
                        expand=False,
                        border_style="bold",
                    )
                )
                progress.console.print()
                progress.update(
                    main_task,
                    advance=1,
                    description="[cyan]Optimizing prompt...",
                )

            test_baseline_session_fut = _queue(
                self,
                _create_experiment,
                client=self.client,
                dataset_id=train_examples[0].dataset_id,
                experiment_name=experiment_name + "- test baseline",
                metadata={
                    "split": "test",
                    "which": "baseline",
                },
            )
            test_final_session_fut = _queue(
                self,
                _create_experiment,
                client=self.client,
                dataset_id=train_examples[0].dataset_id,
                experiment_name=experiment_name + "- test final",
                metadata={
                    "split": "test",
                    "which": "final",
                },
            )

            # Step 3: Test
            progress.update(
                main_task, advance=10, description="[cyan]Running final tests..."
            )
            del train_examples
            del dev_examples

            initial_test_results = await self._evaluate_prompt(
                initial_prompt,
                task,
                test_examples,
                debug=debug,
                system_config=system_config,
                experiment=await test_baseline_session_fut,
            )
            final_test_results = await self._evaluate_prompt(
                best_prompt,
                task,
                test_examples,
                debug=debug,
                experiment=await test_final_session_fut,
                system_config=system_config,
            )
            progress.update(
                main_task, advance=10, description="[cyan]Optimization complete!"
            )
        # Print final report
        initial_scores = await self.calculate_scores(initial_test_results)
        final_scores = await self.calculate_scores(final_test_results)

        table = Table(
            title="Optimization Results", show_header=True, header_style="bold magenta"
        )
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Initial Score", justify="right", style="green")
        table.add_column("Final Score", justify="right", style="green")

        for metric in initial_scores.keys():
            table.add_row(
                metric, f"{initial_scores[metric]:.4f}", f"{final_scores[metric]:.4f}"
            )

        richprint(Panel(table, title="Final Report", border_style="bold"))

        pm_utils.print_rich_diff(
            initial_prompt.get_prompt_str_in_context(),
            best_prompt.get_prompt_str_in_context(),
            title="Final Prompt Updates",
        )
        await self.wait_for_all()
        return best_prompt, best_score

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

    async def _evaluate_prompt(
        self,
        prompt_config: pm_types.PromptWrapper,
        task: pm_types.Task,
        data: str | list,
        debug: bool = False,
        experiment: str | TracerSession | None = None,
        system_config: dict | None = None,
    ) -> list[ExperimentResultRow]:
        """Evaluates a prompt against a task's dataset and evaluators."""
        prompt = prompt_config.load(self.client)
        metadata = {
            "prompt": prompt_config.identifier if prompt_config.identifier else "local"
        }

        async def predict(inputs: dict):
            if system_config:
                return await task.system_safe(prompt, inputs, **system_config)
            else:
                return await task.system_safe(prompt, inputs)

        results = await ls.aevaluate(
            predict,
            data=data,
            evaluators=task.evaluators,
            max_concurrency=0 if debug else None,
            experiment=experiment,
            experiment_prefix="Optimizer" if not experiment else None,
            metadata=metadata,
        )
        now = datetime.datetime.now(datetime.timezone.utc)
        # Temporary: permit run ingestion to extend existing experiment
        await _queue(
            self,
            self.client.update_project,
            project_id=results._manager._experiment.id,
            end_time=now + datetime.timedelta(days=999),
        )
        return [r async for r in results]

    async def calculate_scores(
        self, results: list[ExperimentResultRow]
    ) -> dict[str, float]:
        """Calculates aggregate scores from evaluation results, grouped by key."""

        scores = defaultdict(list)
        for result in results:
            for res in result["evaluation_results"]["results"]:
                if res.score is not None:
                    scores[res.key].append(res.score)

        return {
            key: sum(values) / len(values) if values else 0.0
            for key, values in scores.items()
        }

    async def _fetch_baseline_metrics(self, experiment_id: UUID) -> dict:
        """Fetches metrics for a baseline experiment."""
        # Implementation to fetch metrics from LangSmith using the experiment ID
        test_results = self.client.get_test_results(project_id=experiment_id)
        metric_cols = [
            col for col in test_results.columns if col.startswith("feedback.")
        ]
        return {col: test_results[col].mean() for col in metric_cols}

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
            new_name = f"{experiment_name} [{next_idx}]"

    raise ValueError(
        f"Could not find a unique experiment name in {num_attempts} attempts."
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
