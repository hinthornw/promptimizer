# promptimizer

Experimental prompt optimization library.

Example:

Clone the repo, then setup:

```shell
uv venv
source .venv/bin/activate
uv pip install -e .
python examples/tweet_writer/create_dataset.py
```

Then run prompt optimization.

```shell
promptim --task examples/tweet_writer/config.json --version 1
```

## Create a custom task

Currently, `promptim` runs over **tasks**. Each task contains the following information:

```python
    name: str  # The name of the task
    description: str = ""  # A description of the task (optional)
    evaluator_descriptions: dict = field(default_factory=dict)  # Descriptions of the evaluation metrics
    dataset: str  # The name of the dataset to use for the task
    initial_prompt: PromptConfig  # The initial prompt configuration.
    evaluators: list[Callable[[Run, Example], dict]]  # List of evaluation functions
    system: Optional[SystemType] = None  # Optional custom function with signature (current_prompt: ChatPromptTemplate, inputs: dict) -> outputs
```

Check out the example ["tweet writer"](./examples/tweet_writer/task.py) task to see what's expected.

## CLI Arguments

The CLI is experimental.

```shell
Usage: promptim [OPTIONS]

  Optimize prompts for different tasks.

Options:
  --version [1]                [required]
  --task TEXT                  Task to optimize. You can pick one off the
                               shelf or select a path to a config file.
                               Example: 'examples/tweet_writer/config.json
  --batch-size INTEGER         Batch size for optimization
  --train-size INTEGER         Training size for optimization
  --epochs INTEGER             Number of epochs for optimization
  --debug                      Enable debug mode
  --use-annotation-queue TEXT  The name of the annotation queue to use. Note:
                               we will delete the queue whenever you resume
                               training (on every batch).
  --no-commit                  Do not commit the optimized prompt to the hub
  --help                       Show this message and exit.
```

We have created a few off-the-shelf tasks:

- tweet: write tweets
- simpleqa: really hard Q&A
- scone: NLI

![run](./static/optimizer.gif)
