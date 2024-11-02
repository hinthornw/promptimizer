import asyncio
import importlib.util
import json

import click
import langsmith as ls


def get_tasks(task_name: str):
    from promptim.tasks.metaprompt import metaprompt_task
    from promptim.tasks.scone import scone_task
    from promptim.tasks.simpleqa import simpleqa_task
    from promptim.tasks.ticket_classification import ticket_classification_task
    from promptim.tasks.tweet_generator import tweet_task

    tasks = {
        "scone": scone_task,
        "tweet": tweet_task,
        "metaprompt": metaprompt_task,
        "simpleqa": simpleqa_task,
        "ticket-classification": ticket_classification_task,
    }
    return tasks.get(task_name)


def load_task(name_or_path: str):
    from promptim.trainer import Task

    task = get_tasks(name_or_path)
    if task:
        return task, {}
    # If task is not in predefined tasks, try to load from file
    try:
        with open(name_or_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        task_path = config["task"]
        module_path, task_variable = [part for part in task_path.split(":") if part]

        spec = importlib.util.spec_from_file_location("task_module", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        task = getattr(module, task_variable)
        if not isinstance(task, Task):
            task = Task.from_dict(task)
        return task, config
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
    task, config = load_task(task_name)
    from promptim.trainer import PromptOptimizer

    optimizer = PromptOptimizer.from_config(config.get("optimizer_config", {}))

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
    help="Task to optimize. You can pick one off the shelf or select a path to a config file. "
    "Example: 'examples/tweet_writer/config.json",
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
    version: str,
    task: str,
    batch_size: int,
    train_size: int,
    epochs: int,
    debug: bool,
    use_annotation_queue: str | None,
    no_commit: bool,
):
    """Optimize prompts for different tasks."""

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
