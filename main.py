import argparse
import asyncio
import importlib.util

import langsmith as ls
from langchain_anthropic import ChatAnthropic
from prompt_optimizer.tasks.metaprompt import metaprompt_task
from prompt_optimizer.tasks.scone import scone_task
from prompt_optimizer.tasks.simpleqa import simpleqa_task
from prompt_optimizer.tasks.ticket_classification import ticket_classification_task
from prompt_optimizer.tasks.tweet_generator import tweet_task
from prompt_optimizer.trainer import PromptOptimizer, Task

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Optimize prompts for different tasks."
    )
    parser.add_argument(
        "--task",
        help=f"Task to optimize. You can pick one off the shelf or select select a path. Off-the-shelf options include: {', '.join(tasks)}.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=40, help="Batch size for optimization"
    )
    parser.add_argument(
        "--train-size", type=int, default=40, help="Training size for optimization"
    )
    parser.add_argument(
        "--epochs", type=int, default=2, help="Number of epochs for optimization"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument(
        "--use-annotation-queue",
        type=str,
        default=None,
        help="The name of the annotation queue to use. Note: we will delete the queue whenever you resume training (on every batch).",
    )
    parser.add_argument(
        "--no-commit",
        action="store_true",
        help="Do not commit the optimized prompt to the hub",
    )

    args = parser.parse_args()

    results = asyncio.run(
        run(
            args.task,
            args.batch_size,
            args.train_size,
            args.epochs,
            args.use_annotation_queue,
            args.debug,
            commit=not args.no_commit,
        )
    )
    print(results)
