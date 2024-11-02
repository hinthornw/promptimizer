import copy
import random
from collections import defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Callable, Literal, Optional
from uuid import UUID

import langsmith as ls
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.structured import StructuredPrompt
from langchain_core.runnables import RunnableBinding, RunnableSequence
from langsmith.evaluation import _arunner, _runner
from langsmith.evaluation._arunner import ExperimentResultRow
from langsmith.schemas import Example, Run
from pydantic import BaseModel, Field
from rich import print as richprint
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress
from rich.text import Text


def ltq():
    return lambda x: x


_runner._load_tqdm = ltq
_arunner._load_tqdm = ltq


def _noop(*args, **kwargs):
    pass


_runner.print = _noop  # type: ignore
_arunner.print = _noop  # type: ignore


DEFAULT_METAPROMPT = """You are an expert prompt engineer tasked with improving prompts for AI tasks.
You will use all means necessary to optimize the scores for the provided prompt so that the resulting model can
perform well on the target task.

## Current prompt

The following is the current best-performing prompt:

<current_prompt>
{current_prompt}
</current_prompt>

Your generations will replace the content within the <TO_OPTIMIZE></TO_OPTIMIZE> tags. The rest is fixed context over which you have no control. The TO_OPTIMIZE and CONTEXT\
 tags are provided here to help you disambiguateand not present in the prompt itself.

## Previous Prompt Attempts

You previously attempted to use the following prompts, but they earned worse scores than the current one:
<other_attempts>
{other_attempts}
</other_attempts>

Reflect on your previous attempts to ensure you search for and identify better patterns.

## Annotated results:
<results>
{annotated_results}
</results>

## Task description:
<task_description>
{task_description}
</task_description>

Unless otherwise specified, higher scores are better (try to maximize scores). Aim for perfect scores across all examples.

In your head, search through all edits, planning the optimization step-by-step:
1. Analyze the current results and where they fall short
2. Identify patterns in successful vs unsuccessful cases
3. Propose specific improvements to address the shortcomings
4. Generate an improved prompt that maintains all required formatting

The improved prompt must:
- Keep all original input variables
- Maintain any special formatting or delimiters
- Focus on improving the specified metrics
- Be clear and concise.
- Avoid repeating mistakes.

Use prompting strategies as appropriate for the task. For logic and math, consider encourage more chain-of-thought reasoning, 
or include reasoning trajectories to induce better performance. For creative tasks, consider adding style guidelines.
Or consider including examplars.

Output your response in this format:
<analysis>
Your step-by-step analysis here...
</analysis>

<improved_prompt>
Your improved prompt here...
</improved_prompt>"""

SystemType = Callable[[ChatPromptTemplate, dict], dict]
"""Takes the current prompt and the example inputs and returns the results."""


# TODO: split user facing from the loadable one
@dataclass(kw_only=True)
class PromptConfig:
    identifier: str | None = None
    """If optimizing a prompt in the hub. Do not provide if providing a prompt string directly."""
    prompt_str: str | None = None
    """If optimizing a local string. Do not provide if providing a repo."""
    model_config: dict | None = None
    # TODO: Support tool / schema optimization
    which: int = 0
    """Which message to optimize."""
    _cached: ChatPromptTemplate | None = None
    _postlude: RunnableSequence | None = None

    def __post_init__(self):
        if self.identifier and self.prompt_str:
            raise ValueError(
                "Cannot provide both identifier and prompt_str. Choose one."
            )
        elif not self.identifier and not self.prompt_str:
            raise ValueError("Must provide either identifier or prompt_str.")

    def load(self, client: ls.Client | None = None) -> ChatPromptTemplate:
        if self._cached is None:
            if self.prompt_str:
                self._cached = ChatPromptTemplate.from_messages(
                    [("user", self.prompt_str)]
                )
            else:
                client = client or ls.Client()
                postlude = None
                prompt = client.pull_prompt(self.identifier, include_model=True)
                if isinstance(prompt, RunnableSequence):
                    # I hate this
                    prompt, bound_llm = prompt.first, prompt.steps[1]
                    if isinstance(prompt, StructuredPrompt) and isinstance(
                        bound_llm, RunnableBinding
                    ):
                        # seq.steps[1].bind(**tmpl.last.dict(exclude={"bound"}))
                        seq: RunnableSequence = prompt | bound_llm.bound
                        rebound_llm = seq.steps[1]
                        parser = seq.steps[2]
                        postlude = RunnableSequence(
                            rebound_llm.bind(
                                **{**bound_llm.kwargs, **(self.model_config or {})}
                            ),
                            parser,
                        )
                    else:
                        postlude = bound_llm
                else:
                    # Default to gpt-4o-mini
                    postlude = init_chat_model(**self.model_config)
                self._cached = prompt
                self._postlude = postlude
        return self._cached

    def get_prompt_str(self, client: ls.Client | None = None) -> str:
        tmpl = self.load(client)
        msg = tmpl.messages[self.which]
        try:
            return msg.prompt.template  # type: ignore
        except Exception as e:
            raise NotImplementedError(
                f"Unsupported message template format. {msg}"
            ) from e

    def get_prompt_str_in_context(self, client: ls.Client | None = None) -> str:
        tmpl = self.load(client)
        formatted = []
        for i, msg in enumerate(tmpl.messages):
            kind = msg.__class__.__name__.replace("MessagePromptTemplate", "").replace(
                "Human", "User"
            )
            if i == self.which:
                formatted.append(
                    f"""<TO_OPTIMIZE kind="{kind}">
{msg.prompt.template}
</TO_OPTIMIZE>"""
                )
            else:
                formatted.append(
                    f"""<CONTEXT kind="{kind}">
{msg.prompt.template}
</CONTEXT>
"""
                )
        return "\n".join(formatted)

    @classmethod
    def from_prior(cls, prior: "PromptConfig", output: str):
        copied = prior._cached
        if not copied:
            raise ValueError("Cannot load from unloaded prior.")
        copied = copy.deepcopy(copied)
        tmpl = copied.messages[prior.which]
        tmpl.prompt.template = output  # type: ignore
        return cls(
            identifier=prior.identifier,
            prompt_str=prior.prompt_str,
            which=prior.which,
            _cached=copied,
            _postlude=prior._postlude,
        )


@dataclass(kw_only=True)
class Task:
    """Represents a specific task for prompt optimization."""

    name: str
    description: str = ""
    evaluator_descriptions: dict = field(default_factory=dict)
    dataset: str
    initial_prompt: PromptConfig
    evaluators: list[Callable[[Run, Example], dict]]
    system: Optional[SystemType] = None
    baseline_experiment: UUID | None = None

    @classmethod
    def from_dict(cls, d: dict):
        d_ = d.copy()
        return cls(**{"initial_prompt": PromptConfig(**d_.pop("initial_prompt")), **d_})

    def describe(self):
        descript = self.description if self.description else self.name
        evaluator_desc = "\n".join(
            [f"- {key}: {value}" for key, value in self.evaluator_descriptions.items()]
        )
        return f"{descript}\n\nDescription of scores:\n{evaluator_desc}"

    @property
    def system_safe(self) -> SystemType:
        if self.system:
            return self.system

        async def prompt_system(prompt: ChatPromptTemplate, inputs: dict):
            return await self.initial_prompt._postlude.ainvoke(prompt.invoke(inputs))

        return prompt_system


class OptimizedPromptOutput(BaseModel):
    """Schema for the optimized prompt output."""

    analysis: str = Field(
        description="First, analyze the current results and plan improvements to reconcile them."
    )
    improved_prompt: str = Field(description="The improved prompt text")


class PromptOptimizer:
    """A framework for optimizing meta-prompts through multi-task evaluation."""

    def __init__(
        self,
        model: BaseChatModel,
        meta_prompt: Optional[str] = None,
        seed: int = 42,
    ):
        self.model = model
        self.client = ls.Client()
        self.meta_prompt = meta_prompt or DEFAULT_METAPROMPT
        random.seed(seed)
        self.rng = random.Random(seed)

    @classmethod
    def from_config(cls, config: dict):
        cp = config.copy()
        model_config = cp.pop(
            "model", dict(model="claude-3-5-sonnet-20241022", max_tokens_to_sample=8192)
        )
        model = init_chat_model(**model_config)
        return cls(model, **cp)

    async def optimize_prompt(
        self,
        task: Task,
        *,
        system_config: dict | None = None,
        train_size: Optional[int] = None,
        batch_size: int = 40,
        epochs: int = 1,
        use_annotation_queue: str | None = None,
        debug: bool = False,
    ) -> tuple[PromptConfig, float]:
        """Optimizes a prompt for a specific task through multiple iterations."""
        task.initial_prompt.load(self.client)  # check
        current_prompt = task.initial_prompt
        best_score = 0
        best_prompt = task.initial_prompt
        other_attempts = []
        # Print the original prompt
        richprint(
            Panel.fit(
                f"[bold cyan]Original Prompt:[/bold cyan]\n\n{task.initial_prompt.get_prompt_str_in_context(self.client)}",
                title="Starting Prompt",
                border_style="bold",
            )
        )
        dev_examples = list(
            self.client.list_examples(dataset_name=task.dataset, splits=["dev"])
        )
        with Progress() as progress:
            main_task = progress.add_task("[cyan]Optimizing prompt...", total=100)

            # Step 1: Get baseline scores
            progress.update(
                main_task, advance=10, description="[cyan]Getting baseline scores..."
            )
            if task.baseline_experiment:
                baseline_scores = await self._fetch_baseline_metrics(
                    task.baseline_experiment
                )
            else:
                baseline_experiment_results = await self._evaluate_prompt(
                    current_prompt,
                    task,
                    dev_examples,
                    debug=debug,
                    system_config=system_config,
                )
                baseline_scores = await self.calculate_scores(
                    baseline_experiment_results
                )
            best_score = (
                sum(baseline_scores.values()) / len(baseline_scores)
                if baseline_scores
                else None
            )
            baseline_scores_output = "[cyan]Baseline scores:\n"
            for metric, score in baseline_scores.items():
                baseline_scores_output += f"  {metric}: {score:.4f}\n"
            baseline_scores_output += f"Overall baseline score: {best_score:.4f}"
            progress.console.print(baseline_scores_output)
            progress.console.print()

            # Step 2: Train
            progress.update(
                main_task, advance=10, description="[cyan]Training prompt..."
            )
            train_examples = list(
                self.client.list_examples(dataset_name=task.dataset, splits=["train"])
            )

            epoch_task = progress.add_task("[green]Epochs", total=epochs)
            for epoch in range(epochs):
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
                experiment_name = None
                for batch in batches:
                    results = await self._evaluate_prompt(
                        current_prompt,
                        task,
                        batch,
                        debug=debug,
                        experiment_name=experiment_name,
                        system_config=system_config,
                    )
                    next_action = "continue"
                    if use_annotation_queue:
                        results, next_action = await self._wait_for_annotation_queue(
                            results,
                            use_annotation_queue,
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
                    progress.console.print(
                        f"Batch train score: {train_score:.4f}", end="\n"
                    )
                    progress.console.print()
                    improved = await self.apply_metaprompt(
                        current_prompt=current_prompt,
                        other_attempts=other_attempts,
                        meta_prompt=self.meta_prompt,
                        task=task,
                        results=results,
                    )
                    current_prompt = improved
                    progress.update(batch_task, advance=1)
                    if next_action != "continue":
                        break

                console = Console()

                train_scores_panel = Panel(
                    Text(", ".join(f"{score:.4f}" for score in all_train_scores)),
                    title="Train Scores",
                    expand=False,
                    border_style="bold",
                )
                console.print(train_scores_panel)
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
                progress.update(epoch_task, advance=1)

            # Step 3: Test
            progress.update(
                main_task, advance=10, description="[cyan]Running final tests..."
            )
            del train_examples
            del dev_examples
            test_examples = list(
                self.client.list_examples(dataset_name=task.dataset, splits=["test"])
            )
            initial_test_results = await self._evaluate_prompt(
                task.initial_prompt,
                task,
                test_examples,
                debug=debug,
                system_config=system_config,
            )
            final_test_results = await self._evaluate_prompt(
                best_prompt,
                task,
                test_examples,
                debug=debug,
                system_config=system_config,
            )
            progress.update(
                main_task, advance=10, description="[cyan]Optimization complete!"
            )
        # Print final report
        richprint(
            Panel.fit(
                f"[bold green]Optimization Results:[/bold green]\n\n"
                f"[cyan]Initial Prompt Performance:[/cyan]\n"
                f"{await self.calculate_scores(initial_test_results)}\n\n"
                f"[cyan]Optimized Prompt Performance:[/cyan]\n"
                f"{await self.calculate_scores(final_test_results)}",
                title="Final Report",
                border_style="bold",
            )
        )

        # Print prompt diff
        _print_rich_diff(
            task.initial_prompt.get_prompt_str_in_context(),
            best_prompt.get_prompt_str_in_context(),
            title="Final Prompt Updates",
        )
        return best_prompt, best_score

    async def _wait_for_annotation_queue(
        self,
        results: list[ExperimentResultRow],
        queue_name: str,
        task: Task,
        progress: Progress,
    ) -> tuple[list[ExperimentResultRow], Literal["continue", "break"]]:
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
        runs = [r["run"].id for r in results]
        self.client.add_runs_to_annotation_queue(q.id, run_ids=runs)

        # Now, log instrutions and await user input in the terminal.
        # User input can either continue or break the loop
        richprint(
            Panel.fit(
                f"[bold cyan]Annotation Queue Instructions:[/bold cyan]\n\n"
                f"1. Go to {self.client._host_url}/o/{self.client._get_optional_tenant_id()}/annotation-queues/{q.id}/?runIndex=0\n"
                f"2. Review the outputs and leave feedback on the runs.\n"
                f"3. When finished, return here and enter 'continue' to proceed or 'break' to stop this epoch (continue to next epoch).\n",
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
                        "\n\n[bold]Enter 'continue' to proceed, 'break' to stop, or 'q' to exit:[/bold] "
                    )
                    .strip()
                    .lower()
                )
                if user_input in ["continue", "break", "q"]:
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
        prompt_config: PromptConfig,
        task: Task,
        data: str | list,
        debug: bool = False,
        experiment_name: str | None = None,
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
            experiment=experiment_name,
            experiment_prefix="Optimizer",
            metadata=metadata,
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

    async def apply_metaprompt(
        self,
        current_prompt: PromptConfig,
        meta_prompt: str,
        task: Task,
        results: list[ExperimentResultRow],
        other_attempts: list | None = None,
    ) -> PromptConfig:
        annotated_results = self._format_results(results)
        return await self._generate_improved_prompt(
            current_prompt,
            meta_prompt,
            annotated_results,
            task,
            other_attempts=other_attempts,
        )

    def _format_results(self, results: list[ExperimentResultRow]) -> str:
        """Formats evaluation results for inclusion in the meta-prompt."""
        formatted = []
        i = 0
        for result in results:
            formatted.append(f"Example {i+1}:")
            formatted.append(f'Input: {result["run"].inputs}')
            formatted.append(f'Output: {result["run"].outputs}')
            formatted.append("Evaluations:")
            for eval in result["evaluation_results"]["results"]:
                formatted.append(f"- {eval.key}: {eval.score}")
                if eval.comment:
                    formatted.append(f"  Comment: {eval.comment}")
            formatted.append("")
            i += 1
        return "\n".join(formatted)

    async def _generate_improved_prompt(
        self,
        current_prompt: PromptConfig,
        meta_prompt: str,
        annotated_results: str,
        task: Task,
        other_attempts: list | None = None,
    ) -> PromptConfig:
        """Generates an improved prompt using the meta-prompt."""
        chain = self.model.with_structured_output(OptimizedPromptOutput)
        inputs = meta_prompt.format(
            current_prompt=current_prompt.get_prompt_str_in_context(self.client),
            annotated_results=annotated_results,
            task_description=task.describe(),
            other_attempts=(
                "\n\n---".join([p.get_prompt_str() for p in other_attempts])
                if other_attempts
                else "N/A"
            ),
        )
        prompt_output: OptimizedPromptOutput = await chain.ainvoke(inputs)
        candidate = PromptConfig.from_prior(
            current_prompt, prompt_output.improved_prompt
        )

        _print_rich_diff(
            current_prompt.get_prompt_str_in_context(self.client),
            candidate.get_prompt_str_in_context(self.client),
            "Updated Prompt",
        )

        return candidate

    async def _fetch_baseline_metrics(self, experiment_id: UUID) -> dict:
        """Fetches metrics for a baseline experiment."""
        # Implementation to fetch metrics from LangSmith using the experiment ID
        test_results = self.client.get_test_results(project_id=experiment_id)
        metric_cols = [
            col for col in test_results.columns if col.startswith("feedback.")
        ]
        return {col: test_results[col].mean() for col in metric_cols}


def _colorize_diff(diff):
    for op, i1, i2, j1, j2 in diff.get_opcodes():
        if op == "equal":
            yield diff.a[i1:i2]
        elif op == "insert":
            yield f"[green]{diff.b[j1:j2]}[/green]"
        elif op == "delete":
            yield f"[red]{diff.a[i1:i2]}[/red]"
        elif op == "replace":
            yield f"[red]{diff.a[i1:i2]}[/red][green]{diff.b[j1:j2]}[/green]"


def _print_rich_diff(original: str, updated: str, title: str = ""):
    diff = SequenceMatcher(None, original, updated)
    colorized_diff = "".join(_colorize_diff(diff))
    panel = Panel(
        colorized_diff, title=title or "Prompt Diff", expand=False, border_style="bold"
    )
    richprint(panel)
