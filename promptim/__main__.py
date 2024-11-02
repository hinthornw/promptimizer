import asyncio
import importlib.util

import click
import langsmith as ls
from langchain_anthropic import ChatAnthropic
from promptim.tasks.metaprompt import metaprompt_task
from promptim.tasks.scone import scone_task
from promptim.tasks.simpleqa import simpleqa_task
from promptim.tasks.ticket_classification import ticket_classification_task
from promptim.tasks.tweet_generator import tweet_task
from promptim.trainer import PromptOptimizer, Task

tasks = {
    "scone": scone_task,
    "tweet": tweet_task,
    "metaprompt": metaprompt_task,
    "simpleqa": simpleqa_task,
    "ticket-classification": ticket_classification_task,
}


optimizer = PromptOptimizer(
    model=ChatAnthropic(model="claude-3-5-sonnet-20241022", max_tokens_to_sample=8192),
)


def load_task(name_or_path: str) -> Task:
    task = tasks.get(name_or_path)
    if task:
        return task
    # If task is not in predefined tasks, try to load from file
    try:
        module_path, task_variable = [part for part in name_or_path.split(":") if part]

        spec = importlib.util.spec_from_file_location("task_module", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        task = getattr(module, task_variable)
        if not isinstance(task, Task):
            raise ValueError
        return task
    except Exception as e:
        raise ValueError(f"Could not load task from {name_or_path}: {e}")


async def run(
    task_name: str,
    batch_size: int,
    train_size: int,
    epochs: int,
    use_annotation_queue: str | None = None,
    debug: bool = False,
    commit: bool = True,
):
    task = load_task(task_name)

    with ls.tracing_context(project_name="Optim"):
        prompt, score = await optimizer.optimize_prompt(
            task,
            batch_size=batch_size,
            train_size=train_size,
            epochs=epochs,
            use_annotation_queue=use_annotation_queue,
            debug=debug,
        )
    if commit and task.initial_prompt.identifier is not None:
        optimizer.client.push_prompt(
            task.initial_prompt.identifier.rsplit(":", maxsplit=1)[0],
            object=prompt.load(optimizer.client),
        )

    return prompt, score


@click.command(help="Optimize prompts for different tasks.")
@click.option("--version", type=click.Choice(["1"]), required=True)
@click.option(
    "--task",
    help="Task to optimize. You can pick one off the shelf or select a path "
    "(e.g., '/path/to/task.py:TaskClass'). Off-the-shelf options"
    f" include: {', '.join([t for t in tasks if t not in ('ticket-classification', 'metaprompt')])}.",
)
@click.option("--batch-size", type=int, default=40, help="Batch size for optimization")
@click.option(
    "--train-size", type=int, default=40, help="Training size for optimization"
)
@click.option("--epochs", type=int, default=2, help="Number of epochs for optimization")
@click.option("--debug", is_flag=True, help="Enable debug mode")
@click.option(
    "--use-annotation-queue",
    type=str,
    default=None,
    help="The name of the annotation queue to use. Note: we will delete the queue whenever you resume training (on every batch).",
)
@click.option(
    "--no-commit",
    is_flag=True,
    help="Do not commit the optimized prompt to the hub",
)
def main(
    version,
    task,
    batch_size,
    train_size,
    epochs,
    debug,
    use_annotation_queue,
    no_commit,
):
    results = asyncio.run(
        run(
            task,
            batch_size,
            train_size,
            epochs,
            use_annotation_queue,
            debug,
            commit=not no_commit,
        )
    )
    print(results)


if __name__ == "__main__":
    main()
