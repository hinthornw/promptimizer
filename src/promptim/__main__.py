from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import sys
import time
from typing import TYPE_CHECKING, Optional

import click
import langsmith as ls
from langsmith.utils import LangSmithNotFoundError

if TYPE_CHECKING:
    from langsmith import Client


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
    from promptim.types import Task

    task = get_tasks(name_or_path)
    if task:
        return task, {}
    # If task is not in predefined tasks, try to load from file
    try:
        with open(name_or_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        if "$schema" in config:
            del config["$schema"]
        evaluators_path = config["evaluators"]
        module_path, evaluators_variable = [
            part for part in evaluators_path.split(":") if part
        ]
        # First try to load it relative to the config path
        config_dir = os.path.dirname(name_or_path)
        relative_module_path = os.path.join(config_dir, module_path)
        if os.path.exists(relative_module_path):
            module_path = relative_module_path
        spec = importlib.util.spec_from_file_location("evaluators_module", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        evaluators = getattr(module, evaluators_variable)
        if not isinstance(evaluators, list):
            raise ValueError(
                f"Expected evaluators to be a list, but got {type(evaluators).__name__}"
            )
        task = Task.from_dict({**config, "evaluators": evaluators})
        return task, config
    except Exception as e:
        raise ValueError(f"Could not load task from {name_or_path}: {e}")


async def run(
    task_name: str,
    batch_size: int,
    train_size: int,
    epochs: int,
    annotation_queue: Optional[str] = None,
    debug: bool = False,
    commit: bool = True,
):
    task, config = load_task(task_name)
    from promptim.trainer import PromptTrainer

    optimizer = PromptTrainer.from_config(
        config.get("optimizer", config.get("optimizer_config", {}))
    )

    with ls.tracing_context(project_name="Optim"):
        prompt, score = await optimizer.optimize_prompt(
            task,
            batch_size=batch_size,
            train_size=train_size,
            epochs=epochs,
            annotation_queue=annotation_queue,
            debug=debug,
            commit_prompts=commit,
        )
    if commit and task.initial_prompt.identifier is not None:
        prompt.push_prompt(
            identifier=task.initial_prompt.identifier.rsplit(":", maxsplit=1)[0],
            include_model_info=True,
            client=optimizer.client,
        )

    return prompt, score


def load_environment():
    """Load environment variables from environment files.

    Attempts to load from .env file if it exists, using python-dotenv.
    Only attempts to import dotenv if a file is found.
    """
    # Check common locations first before importing anything
    for dirname in [os.getcwd()] + [os.path.dirname(p) for p in sys.path]:
        check_path = os.path.join(dirname, ".env")
        if os.path.isfile(check_path):
            try:
                from dotenv import load_dotenv
            except ImportError:
                click.secho(
                    "python-dotenv package not installed. Environment variables will not be loaded from file.",
                    fg="yellow",
                )
                return

            load_dotenv(check_path)
            click.echo(f"Loaded environment variables from {check_path}")
            return


@click.group()
@click.version_option(version="1")
def cli():
    """Optimize prompts for AI tasks using automated evaluation and feedback.

    Promptim helps improve prompts for various AI tasks by running an optimization loop.
    You provide an initial prompt, a dataset, and custom evaluators. Promptim then
    iteratively refines the prompt to improve performance on your specific task.

    To get started, create a task configuration or use a pre-defined one, then run
    the 'train' command to begin optimization.

    Example:
        promptim train --task ./my-task/config.json
    """
    load_environment()


@cli.command()
@click.option(
    "--task",
    help="Task to optimize. Specify a pre-defined task name or path to a custom config file. "
    "The task defines the dataset, evaluators, and initial prompt to optimize. "
    "Example: 'examples/tweet_writer/config.json' for a custom task, or 'sentiment_analysis' for a pre-defined task.",
)
@click.option(
    "--batch-size",
    type=int,
    default=40,
    help="Number of examples to process in each optimization iteration. "
    "Larger batches may improve stability but are limited by the metaprompter's maximum context window size.",
)
@click.option(
    "--train-size",
    type=int,
    default=40,
    help="Maximum number of training examples to use per epoch. Useful for limiting optimization time on large datasets. "
    "If smaller than total available data, a random subset will be used each epoch.",
)
@click.option(
    "--epochs",
    type=int,
    default=2,
    help="Number of complete passes through the training data. More epochs may improve results but increase runtime.",
)
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug mode for verbose logging and sequential processing.",
)
@click.option(
    "--annotation-queue",
    type=str,
    default=None,
    help="Name of the LangSmith annotation queue for manual review of optimization results. "
    "The queue will be cleared and updated on each batch.",
)
@click.option(
    "--no-commit",
    is_flag=True,
    help="Prevent committing the optimized prompt to the LangChain Hub. Use this for local experimentation.",
)
def train(
    task: str,
    batch_size: int,
    train_size: int,
    epochs: int,
    debug: bool,
    annotation_queue: Optional[str],
    no_commit: bool,
):
    """Train and optimize prompts for different tasks."""
    results = asyncio.run(
        run(
            task,
            batch_size,
            train_size,
            epochs,
            annotation_queue,
            debug,
            commit=not no_commit,
        )
    )
    prompt_config, score = results
    print("Final\n\n")
    print(prompt_config.get_prompt_str())
    print("\n\n")
    print(f"Identifier: {prompt_config.identifier}")
    print(f"Score: {score}")


@cli.group()
def create():
    """Commands for creating new tasks."""
    pass


class MissingPromptError(ValueError):
    """Error raised when a prompt is not found."""

    def __init__(self, attempted: str):
        self.attempted = attempted
        super().__init__(f"Prompt not found: {attempted}")


def _try_get_prompt(client: Client, prompt: str | None, yes: bool):
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.prompts.structured import StructuredPrompt
    from langchain_core.runnables import RunnableBinding, RunnableSequence

    from promptim.types import PromptWrapper

    expected_run_outputs = 'predicted: AIMessage = run.outputs["output"]'
    if prompt is None and not yes:
        prompt = click.prompt(
            "Enter the identifier for the initial prompt\n"
            "\tFormat: prompt-name"
            " OR <organization>/<prompt-name>:<commit-or-tag>\n"
            "\tExamples:\n"
            "\t  - langchain-ai/tweet-generator-example-with-nothing:starter\n"
            "\t  - langchain-ai/tweet-generator-example:c39837bd\n"
            "Prompt identifier"
        )
        if prompt == "q":
            click.echo("Exiting task creation.")
            sys.exit()
    elif prompt is None and yes:
        raise ValueError("Prompt identifier is required when using --yes flag")

    # Fetch prompt
    try:
        prompt_repo = client.get_prompt(prompt)
        chain = client.pull_prompt(prompt, include_model=True)
    except LangSmithNotFoundError:
        raise MissingPromptError(attempted=prompt)

    if isinstance(chain, ChatPromptTemplate):
        prompt_obj = chain
    elif isinstance(chain, RunnableSequence):
        prompt_obj = chain.first
    else:
        raise ValueError(f"Unrecognized prompt format: {chain}")
    if isinstance(prompt_obj, StructuredPrompt):
        expected_run_outputs = "predicted: Output = run.outputs"
    elif isinstance(chain, RunnableSequence):
        expected_run_outputs = 'predicted: AIMessage = run.outputs["output"]'
    elif (
        isinstance(chain, RunnableSequence)
        and isinstance(chain.steps[1], RunnableBinding)
        and chain.steps[1].kwargs.get("tools")
    ):
        tools = chain.steps[1].kwargs.get("tools")
        tool_names = [
            t.get("function", {}).get("name")
            for t in tools
            if t.get("function", {}).get("name")
        ]
        expected_run_outputs = f"# AI message contains optional tool_calls from your prompt\n    # Example tool names: {tool_names}\n    {expected_run_outputs}"
    else:
        raise ValueError(f"Unexpected prompt type: {type(prompt_obj)}\n\n{prompt_obj}")
    identifier = prompt
    if "/" in identifier:  # It may be a public prompt:
        tenant_id = None
        for _ in range(4):
            tenant_id = client._get_optional_tenant_id()
            if tenant_id:
                break
            time.sleep(0.2)

        if tenant_id is not None and prompt_repo.tenant_id != str(tenant_id):
            # Warn user and ask for confirmation to clone the prompt
            click.echo(
                f"Warning: The prompt '{identifier}' does not belong to your workspace."
            )
            truncated_identifier = identifier.split("/", maxsplit=1)[1]
            target_repo = truncated_identifier.split(":")[0]

            # Check if target repo exists
            try:
                client.pull_prompt_commit(target_repo)
                repo_exists = True
            except LangSmithNotFoundError:
                repo_exists = False

            if repo_exists:
                # Check if truncated_identifier exists
                try:
                    client.pull_prompt_commit(truncated_identifier)
                    click.echo(f"Using existing prompt: {truncated_identifier}")
                    identifier = truncated_identifier
                except LangSmithNotFoundError:
                    click.echo(
                        f"Prompt {truncated_identifier} not found. Using {target_repo} instead."
                    )
                    identifier = target_repo
            else:
                if yes:
                    clone_confirmation = True
                else:
                    clone_confirmation = click.confirm(
                        f"Would you like to clone prompt {target_repo} to your workspace before continuing?",
                        default=True,
                    )

                if clone_confirmation:
                    try:
                        if isinstance(chain, RunnableSequence):
                            cloned_prompt = PromptWrapper._push_seq(
                                client, chain, identifier=truncated_identifier
                            )
                        else:
                            cloned_prompt = client.push_prompt(
                                truncated_identifier, object=prompt_obj
                            )
                        identifier = cloned_prompt.split("?")[0].split(
                            "/prompts/", maxsplit=1
                        )[1]
                        identifier = ":".join(identifier.rsplit("/"))
                        click.echo(
                            f"Prompt cloned successfully to {cloned_prompt}. New identifier: {identifier}"
                        )
                    except Exception as e:
                        click.echo(f"Error cloning prompt: {e}")
                        click.echo(f"Continuing with the original prompt {identifier}.")
                        click.echo(
                            "You will have to clone this manually in the UI if you want to push optimized commits."
                        )
                else:
                    click.echo(f"Continuing with the original prompt {identifier}.")
                    click.echo(
                        "You will have to clone this manually in the UI if you want to push optimized commits."
                    )

    return prompt_obj, identifier, expected_run_outputs


def get_prompt(client: Client, prompt: str | None, yes: bool):
    while True:
        try:
            return _try_get_prompt(client, prompt, yes)
        except MissingPromptError as e:
            if yes:
                raise ValueError(
                    f"Prompt not found: {e.attempted}. Cannot proceed with --yes flag."
                )
            click.echo(f"Could not find prompt: {e.attempted}")
            response = client.list_prompts(
                query=e.attempted.split(":")[0].strip(),
                limit=10,
                # has_commits=True, # TODO: Use new version
            )
            matches = []
            for repo in response.repos:
                if repo.last_commit_hash:
                    matches.append(f"{repo.repo_handle}:{repo.last_commit_hash[:8]}")
            if not matches:
                prompt = None
                click.echo("Please try again or press 'q' to quit.")
            else:
                click.echo("Did you mean one of these?")
                for i, match in enumerate(matches, 1):
                    click.echo(f"{i}. {match}")
                selection = click.prompt(
                    "Enter the number of your selection or type an identifier to try again"
                )
                if selection.isdigit() and 1 <= int(selection) <= len(matches):
                    prompt = matches[int(selection) - 1]
                elif selection.strip() == "q":
                    sys.exit()
                else:
                    prompt = selection.strip() or None
                    click.echo("Please try again or press 'q' to quit.")
        except click.Abort:
            sys.exit()
        except Exception as e:
            click.echo(f"Error loading prompt: {e!r}")
            if yes:
                raise
            click.echo("Please try again or press 'q' to quit.")
            prompt = None


class MissingDatasetError(ValueError):
    """Error raised when a dataset is not found."""

    def __init__(self, attempted: str):
        self.attempted = attempted
        super().__init__(f"Dataset not found: {attempted}")


def get_dataset(client: Client, dataset: str | None, yes: bool):
    while True:
        try:
            if dataset is None and not yes:
                dataset = click.prompt(
                    "Enter the name of an existing dataset or a URL of a public dataset:\n"
                    "\tExamples:\n"
                    "\t  - my-dataset\n"
                    "\t  - https://smith.langchain.com/public/6ed521df-c0d8-42b7-a0db-48dd73a0c680/d\n"
                    "Dataset name or URL"
                )
                if dataset == "q":
                    click.echo("Exiting task creation.")
                    sys.exit()
            elif dataset is None and yes:
                raise ValueError(
                    "Dataset name or URL is required when using --yes flag"
                )

            return _try_get_dataset(client, dataset)
        except MissingDatasetError as e:
            if yes:
                raise ValueError(
                    f"Dataset not found: {e.attempted}. Cannot proceed with --yes flag."
                )
            click.echo(f"Could not find dataset: {e.attempted}")
            response = client.list_datasets(
                dataset_name_contains=e.attempted,
                limit=10,
            )
            matches = [ds.name for ds in response]
            if not matches:
                create_dataset = click.confirm(
                    f"Dataset '{e.attempted}' not found. Would you like to create it?",
                    default=False,
                )
                if create_dataset:
                    ds = client.create_dataset(dataset_name=e.attempted)
                    click.echo(f"Dataset '{e.attempted}' created successfully.")
                    return ds
                else:
                    dataset = None
                    click.echo("Please try again or press 'q' to quit.")
            else:
                click.echo("Did you mean one of these?")
                for i, match in enumerate(matches, 1):
                    click.echo(f"{i}. {match}")
                click.echo(f"{len(matches) + 1}. Create a new dataset")
                selection = click.prompt(
                    "Enter the number of your selection, type a name to try again, or choose to create a new dataset"
                )
                if selection.isdigit():
                    if 1 <= int(selection) <= len(matches):
                        dataset = matches[int(selection) - 1]
                    elif int(selection) == len(matches) + 1:
                        new_name = click.prompt("Enter the name for the new dataset")
                        ds = client.create_dataset(dataset_name=new_name)
                        click.echo(f"Dataset '{new_name}' created successfully.")
                        return ds
                elif selection.strip() == "q":
                    sys.exit()
                else:
                    dataset = selection.strip() or None
                    click.echo("Please try again or press 'q' to quit.")
        except click.Abort:
            sys.exit()
        except Exception as e:
            click.echo(f"Error loading dataset: {e!r}")
            if yes:
                raise
            click.echo("Please try again or press 'q' to quit.")
            dataset = None


def _try_get_dataset(client: Client, dataset: str):
    if dataset.startswith("https://"):
        return client.clone_public_dataset(dataset)

    try:
        ds = client.read_dataset(dataset_name=dataset)
        return ds
    except LangSmithNotFoundError:
        raise MissingDatasetError(attempted=dataset)
    except Exception as e:
        raise ValueError(f"Could not fetch dataset '{dataset}': {e}") from e


@create.command("task")
@click.argument(
    "path",
    type=click.Path(file_okay=False, dir_okay=True),
)
@click.option(
    "--name",
    required=False,
    help="Name for the task. If not provided, the directory name will be used as default. This name will be used in the config.json file.",
)
@click.option(
    "--prompt",
    required=False,
    help="Name of the prompt in LangSmith to be optimized. If not provided, you'll be prompted to select or create one. This will be used as the initial prompt for optimization.",
)
@click.option(
    "--description",
    required=False,
    help="Description of the task for the optimizer. This helps guide the optimization process by providing context about the task's objectives and constraints.",
)
@click.option(
    "--dataset",
    required=False,
    help="Name or public URL of the dataset in LangSmith to be used for training and evaluation. If not provided, you'll be prompted to select or create one. This dataset will be used to test and improve the prompt.",
)
@click.option(
    "-y",
    "--yes",
    is_flag=True,
    help="Automatically answer yes to all CLI prompts. Use with caution as it skips confirmation steps and uses defaults where applicable.",
)
def create_task(
    path: str,
    name: str | None = None,
    prompt: str | None = None,
    dataset: str | None = None,
    description: str | None = None,
    yes: bool = False,
):
    """Create a new task directory with config.json and task file for a custom prompt and dataset."""
    from langchain_core.prompts.structured import StructuredPrompt
    from langsmith import Client

    client = Client()
    if not client.api_key:
        raise ValueError("LANGSMITH_API_KEY required to create the task.")
    if name is None:
        default_name = os.path.basename(os.path.abspath(path))
        name = click.prompt(
            "Enter a name for your task",
            default=default_name,
        ).strip()
        if name == "q":
            print("Exiting task creation.")
            return

    prompt_obj, identifier, expected_run_outputs = get_prompt(client, prompt, yes)

    # Create task directory
    os.makedirs(path, exist_ok=True)
    ds = get_dataset(client, dataset, yes)
    try:
        example = next(client.list_examples(dataset_id=ds.id, limit=1))
    except Exception:
        example = None

    def json_comment(d: dict, indent: int = 4):
        return "\n".join(
            f"{' ' * indent}# {line}" for line in json.dumps(d, indent=2).splitlines()
        )

    example_content = ""
    if example is not None:
        example_inputs = json_comment(example.inputs) if example else None
        example_outputs = (
            (
                json_comment(example.outputs)
                if example.outputs is not None
                else "    # None - this example lacks expected outputs"
            )
            if example
            else None
        )
        example_content = f"""
    # We've copied the inputs & outputs for the first example in the configured dataset.
    prompt_inputs = example.inputs
{example_inputs}
    reference_outputs = example.outputs  # aka labels
{example_outputs}
    # The comments above autogenerated from example:
    # {example.url}
"""
    if description is None:
        description = click.prompt("Please enter a description for the task")

    # Create config.json
    config = {
        "name": name,
        "dataset": ds.name,
        "description": description,
        "evaluators": "./task.py:evaluators",
        "evaluator_descriptions": {
            "my_example_criterion": "CHANGEME: This is a description of what the example criterion is testing."
            " It is provided to the metaprompt "
            "to improve how it responds to different results."
        },
        "optimizer": {
            "model": {
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens_to_sample": 8192,
            }
        },
        "initial_prompt": {"identifier": identifier},
    }
    config[
        "$schema"
    ] = "https://raw.githubusercontent.com/hinthornw/promptimizer/refs/heads/main/config-schema.json"
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Create task.py with placeholder evaluators
    expected_imports = "from langchain_core.messages import AIMessage"
    output_keys = []
    expected_type = ""
    if isinstance(prompt_obj, StructuredPrompt):
        expected_imports = "from typing_extensions import TypedDict"
        try:
            properties = prompt_obj.schema_["properties"]
            output_keys = list(properties.keys())
            type_map = {
                "string": "str",
                "integer": "int",
                "number": "float",
                "array": "list",
                "object": "dict",
                "boolean": "bool",
                "null": "None",
            }
            expected_type = "\n    ".join(
                f'{k.replace(" ", "_").replace("-", "_")}: {type_map.get(properties[k].get("type", ""), "Any")}'
                for k in output_keys
            )
            expected_type = f"""

class Outputs(TypedDict):
    {expected_type}"""
        except Exception:
            expected_type = "\nOutputs = dict"
    if isinstance(prompt_obj, StructuredPrompt):
        score_str = f"score = all(key in outputs for key in {output_keys})  # Replace with the actual score calculation"
    else:
        score_str = (
            "score = len(str(predicted.content)) < 180  # Replace with actual score"
        )

    task_template = f"""\"\"\"Evaluators to optimize task: {name}.

THIS IS A TEMPLATE FOR YOU TO CHANGE!

Evaluators compute scores for prompts run over the configured dataset:
{ds.url}
\"\"\"
{expected_imports}
from langsmith.schemas import Run, Example

# Modify these evaluators to measure the requested criteria.
# For most prompt optimization tasks, run.outputs["output"] will contain an AIMessage
# (Advanced usage) If you are defining a custom system to optimize, then the outputs will contain the object returned by your system
{expected_type}

def example_evaluator(run: Run, example: Example) -> dict:
    \"\"\"An example evaluator. Larger numbers are better.\"\"\"
    # The Example object contains the inputs and reference labels from a single row in your dataset (if provided).
    {example_content}    
    # The Run object contains the full trace of your system. Typically you run checks on the outputs,
    # often comparing them to the reference_outputs 
    {"outputs: Outputs = run.outputs" if isinstance(prompt_obj, StructuredPrompt) else expected_run_outputs}

    # Implement your evaluation logic here
    {score_str}
    return {{
        # The evaluator keys here define the metric you are measuring
        # You can provide further descriptions for these in the config.json
        "key": "my_example_criterion",
        "score": score,
        "comment": (
            "CHANGEME: It's good practice to return "
            "information that can help the metaprompter fix mistakes, "
            "such as Pass/Fail or expected X but got Y, etc. "
            "This comment instructs the LLM how to improve."
            "The placeholder metric checks that the content is less than 180 in length."
        ),
    }}


evaluators = [example_evaluator]
"""

    with open(os.path.join(path, "task.py"), "w") as f:
        f.write(task_template.strip())

    print("*" * 80)
    print(f"Task '{name}' created at {path}")
    print(f"Config file created at: {os.path.join(path, 'config.json')}")
    print(f"Task file created at: {os.path.join(path, 'task.py')}")
    print(f"Using dataset: {ds.url}")
    print(f"Using prompt: {identifier}\n\n{prompt_obj.pretty_repr()}\n\n")
    print(
        f"Remember to implement your custom evaluators in {os.path.join(path, 'task.py')}"
    )


@create.command("example")
@click.argument("path", type=click.Path(file_okay=False, dir_okay=True))
@click.argument("name", type=str)
def create_example_task(path: str, name: str):
    """Create an example task directory with config.json, task file, and example dataset."""
    # Create example dataset
    from langsmith import Client

    client = Client()
    if not client.api_key:
        raise ValueError("LANGSMITH_API_KEY required to create the example tweet task.")
    prompt = client.pull_prompt("langchain-ai/tweet-generator-example:c39837bd")
    identifier = f"{name}-starter"
    try:
        identifier = client.push_prompt(identifier, object=prompt, tags=["starter"])
    except ValueError as e:
        try:
            client.pull_prompt_commit(identifier)

        except Exception:
            raise e
        print(f"Prompt {name}-starter already found. Continuing.")

    identifier = identifier.split("?")[0].replace(
        "https://smith.langchain.com/prompts/", ""
    )
    identifier = identifier.rsplit("/", maxsplit=1)[0]
    identifier = f"{identifier}:starter"
    try:
        dataset = client.create_dataset(name)
    except Exception as e:
        if dataset := client.read_dataset(dataset_name=name):
            pass
        else:
            raise e

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
        "name": "Tweet Generator",
        "dataset": name,
        "evaluators": "./task.py:evaluators",
        "optimizer": {
            "model": {
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens_to_sample": 8192,
            }
        },
        "initial_prompt": {"identifier": identifier},
        "evaluator_descriptions": {
            "under_180_chars": "Checks if the tweet is under 180 characters. 1 if true, 0 if false.",
            "no_hashtags": "Checks if the tweet contains no hashtags. 1 if true, 0 if false.",
            "multiline": "Fails if the tweet is not multiple lines. 1 if true, 0 if false. 0 is bad.",
        },
    }

    config[
        "$schema"
    ] = "https://raw.githubusercontent.com/hinthornw/promptimizer/refs/heads/main/config-schema.json"
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    task_template = """
# You can replace these evaluators with your own.
# See https://docs.smith.langchain.com/evaluation/how_to_guides/evaluation/evaluate_llm_application#custom-evaluators
# for more information
def under_180_chars(run, example):
    \"\"\"Evaluate if the tweet is under 180 characters.\"\"\"
    result = run.outputs.get("tweet", "")
    score = int(len(result) < 180)
    comment = "Pass" if score == 1 else "Fail"
    return {
        "key": "under_180_chars",
        "score": score,
        "comment": comment,
    }

def no_hashtags(run, example):
    \"\"\"Evaluate if the tweet contains no hashtags.\"\"\"
    result = run.outputs.get("tweet", "")
    score = int("#" not in result)
    comment = "Pass" if score == 1 else "Fail"
    return {
        "key": "no_hashtags",
        "score": score,
        "comment": comment,
    }

evaluators = [multiple_lines, no_hashtags, under_180_chars]
"""

    with open(os.path.join(path, "task.py"), "w") as f:
        f.write(task_template.strip())


if __name__ == "__main__":
    cli()
