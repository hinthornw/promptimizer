from typing import List, Union, Optional
from rich.panel import Panel
from rich.progress import Progress
from rich.table import Table

from promptim import types as pm_types
from promptim.trainer import PromptTrainer

from promptim.algorithms.base import BaseAlgorithm, AlgorithmConfig
from promptim import _utils as pm_utils
import langsmith as ls


class MinibatchAlgorithm(BaseAlgorithm[AlgorithmConfig]):
    """
    Classic epoch-based training that processes data in minibatches.
    This preserves the original optimize_prompt behavior.
    """

    @ls.traceable(name="MinibatchAlgorithm.run")
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
        """Implementation of the original optimize_prompt flow."""
        if isinstance(initial_population, pm_types.PromptWrapper):
            initial_population = [initial_population]
        history = [initial_population]
        best_score = float("-inf")
        best_prompt = initial_population[0]
        with Progress() as progress:
            main_task = progress.add_task(
                "[cyan]Optimizing prompt...", total=self.config.epochs + 2
            )
            progress.update(
                main_task, advance=10, description="[cyan]Getting baseline scores..."
            )
            baseline_scores = baseline_scores or {}
            best_score = (
                sum(baseline_scores.values()) / len(baseline_scores)
                if baseline_scores
                else float("-inf")
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
            training_session_fut = trainer._enqueue_experiment(
                experiment_name=experiment_name,
                examples=train_examples,
                split="train",
                epoch=0,
            )
            for epoch in range(self.config.epochs):
                training_session = await training_session_fut

                trainer.optimizer.on_epoch_start(epoch, task)
                trainer.rng.shuffle(train_examples)
                if self.config.train_size:
                    train_examples = train_examples[: self.config.train_size]

                batches = [
                    train_examples[i : i + self.config.batch_size]
                    for i in range(0, len(train_examples), self.config.batch_size)
                ]

                batch_task = progress.add_task(
                    f"[yellow]Epoch {epoch+1} batches", total=len(batches)
                )
                all_train_scores = []
                avg_score = float("-inf")
                training_session_fut = trainer._enqueue_experiment(
                    experiment_name=experiment_name,
                    examples=train_examples,
                    split="train",
                    epoch=epoch + 1,
                )
                for bix, batch in enumerate(batches):
                    results = None
                    if bix == 0 and epoch == 0 and baseline_experiment_results:
                        bindices = {e.id for e in batch}
                        results = [
                            r
                            for r in baseline_experiment_results
                            if r["example"].id in bindices
                        ]
                        if len(results) != len(batch):
                            results = None
                    if results is None:
                        results = await trainer._evaluate_prompt(
                            history[-1][-1],
                            task,
                            batch,
                            debug=self.config.debug,
                            experiment=training_session,
                            system_config=system_config,
                        )
                    next_action = "continue"

                    if annotation_queue:
                        results, next_action = await trainer._wait_for_annotation_queue(
                            results,
                            annotation_queue,
                            task,
                            progress,
                        )
                        if next_action != "continue":
                            break
                    train_scores = await trainer.calculate_scores(results)
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
                    # Get improved population
                    try:
                        improved = await trainer.optimizer.improve_prompt(
                            history=history,
                            results=results,
                            task=task,
                            trainer=trainer,
                        )
                        history[-1].extend(improved)

                        if commit_prompts:
                            for prompt in improved:
                                prompt.push_prompt(client=trainer.client)
                    except Exception as e:
                        progress.console.print(
                            f"Failed to improve prompt: {e}", style="red"
                        )
                        break

                # Evaluate on dev set
                progress.update(main_task, description="[cyan]Evaluating on dev set...")
                dev_results = await trainer._evaluate_prompt(
                    history[-1][-1],
                    task,
                    dev_examples,
                    debug=self.config.debug,
                    system_config=system_config,
                )
                dev_scores = await trainer.calculate_scores(dev_results)
                dev_score = (
                    sum(dev_scores.values()) / len(dev_scores) if dev_scores else None
                )
                progress.update(
                    batch_task,
                    description=f'[yellow]Epoch {epoch+1} (Dev: {f"{dev_score:.4f}" if dev_score is not None else "-"}, Train: {f"{avg_score:.4f}" if avg_score is not None else "-"})',
                )

                if dev_score is not None and dev_score > best_score:
                    best_score = dev_score
                    best_prompt = history[-1][-1]
                    progress.console.print(
                        f"New best score: {best_score:.4f} (surpassed previous best)"
                    )
                    progress.console.print("Average of:")
                    for metric, score in dev_scores.items():
                        progress.console.print(f"  {metric}: {score:.4f}")
                else:
                    progress.console.print(
                        f"Score {dev_score:.4f} did not surpass best score {best_score:.4f}"
                    )

                trainer.log_metric(
                    "score",
                    value=best_score,
                    x=epoch,
                    x_label="epoch",
                    split="dev",
                    prompt=best_prompt,
                )

                tokens_used = pm_utils.get_token_usage()
                if tokens_used is not None:
                    trainer.log_metric(
                        "score",
                        value=best_score,
                        x=tokens_used,
                        x_label="total tokens",
                        split="dev",
                        prompt=best_prompt,
                    )
                history.append([best_prompt])

            return best_prompt, best_score
