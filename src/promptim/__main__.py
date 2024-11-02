import asyncio
import importlib.util
import json
import os
from typing import Optional

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
        # First try to load it relative to the config path
        config_dir = os.path.dirname(name_or_path)
        relative_module_path = os.path.join(config_dir, module_path)
        if os.path.exists(relative_module_path):
            module_path = relative_module_path
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
    use_annotation_queue: Optional[str] = None,
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


@click.group()
@click.version_option(version="1")
def cli():
    """Optimize prompts for different tasks."""
    pass


@cli.command()
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
def train(
    task: str,
    batch_size: int,
    train_size: int,
    epochs: int,
    debug: bool,
    use_annotation_queue: Optional[str],
    no_commit: bool,
):
    """Train and optimize prompts for different tasks."""
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


@cli.command()
@click.argument("path", type=click.Path(file_okay=False, dir_okay=True))
@click.argument("name", type=str)
def create_task(path: str, name: str):
    """Create a new task directory with config.json, task file, and example dataset."""
    # Create example dataset
    from langsmith import Client

    client = Client()
    prompt = client.pull_prompt("langchain-ai/tweet-generator-example:c39837bd")
    identifier = client.push_prompt(f"{name}-starter", object=prompt)
    identifier = identifier.split("?")[0].replace(
        "https://smith.langchain.com/prompts/", ""
    )
    identifier = ":".join(identifier.rsplit("/", maxsplit=1))
    dataset = client.create_dataset(name)
    topics = [
        "NBA",
        "NFL",
        "Movies",
        "Taylor Swift",
        "Artificial Intelligence",
        "Climate Change",
        "Space Exploration",
        "Cryptocurrency",
        "Healthy Living",
        "Travel Destinations",
        "Technology",
        "Fashion",
        "Music",
        "Politics",
        "Food",
        "Education",
        "Environment",
        "Science",
        "Business",
        "Health",
    ]

    for split_name, dataset_topics in [
        ("train", topics[:10]),
        ("dev", topics[10:15]),
        ("test", topics[15:]),
    ]:
        client.create_examples(
            inputs=[{"topic": topic} for topic in dataset_topics],
            dataset_id=dataset.id,
            splits=[split_name] * len(dataset_topics),
        )

    print(f"Task directory created at {path}")
    print(f"Example dataset '{dataset.name}' created with {len(topics)} examples")
    print(f"See: {dataset.url}")
    os.makedirs(path, exist_ok=True)

    config = {
        "task": "./task.py:example_task",
        "optimizer": {
            "model": {
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens_to_sample": 8192,
            }
        },
    }

    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    task_template = f"""
def multiple_lines(run, example):
    \"\"\"Evaluate if the tweet contains multiple lines.\"\"\"
    result = run.outputs["output"].content
    score = int("\\n" in result)
    comment = "Pass" if score == 1 else "Fail"
    return {{
        "key": "multiline",
        "score": score,
        "comment": comment,
    }}

example_task = dict(
    name="{name}",
    dataset="{name}",
    initial_prompt={{
        "identifier": "{identifier}",
        "model_config": {{"model": "claude-3-5-sonnet-20241022"}}
    }},
    evaluators=[multiple_lines],
    evaluator_descriptions={{
        "multiline": "Fails if the tweet is not multiple lines. 1 if true, 0 if false. 0 is bad.",
    }},
)
"""

    with open(os.path.join(path, "task.py"), "w") as f:
        f.write(task_template.strip())


if __name__ == "__main__":
    cli()
