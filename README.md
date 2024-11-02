# Promptim

Experimental prompt optimization library.

## Example:

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

Currently, `promptim` runs over individual **tasks**. A task defines the dataset (with train/dev/test splits), initial prompt, evaluators, and other information needed to optimize your prompt.

```python
    name: str  # The name of the task
    description: str = ""  # A description of the task (optional)
    evaluator_descriptions: dict = field(default_factory=dict)  # Descriptions of the evaluation metrics
    dataset: str  # The name of the dataset to use for the task
    initial_prompt: PromptConfig  # The initial prompt configuration.
    evaluators: list[Callable[[Run, Example], dict]]  # List of evaluation functions
    system: Optional[SystemType] = None  # Optional custom function with signature (current_prompt: ChatPromptTemplate, inputs: dict) -> outputs
```

Let's walk through the example ["tweet writer"](./examples/tweet_writer/task.py) task to see what's expected. First, view the [config.json](./examples/tweet_writer/config.json) file

```json
{
  "optimizer": {
    "model": {
      "model": "claude-3-5-sonnet-20241022",
      "max_tokens_to_sample": 8192
    }
  },
  "task": "examples/tweet_writer/task.py:tweet_task"
}
```

The first part contains confgiuration for the optimizer process. For now, this is a simple configuration for the default (and only) metaprmopt optimizer. You can control which LLM is used via the `model` configuration.

The second part is the path to the task file itself. We will review this below.

```python
def multiple_lines(run, example):
    """Evaluate if the tweet contains multiple lines."""
    result = run.outputs.get("tweet", "")
    score = int("\n" in result)
    comment = "Pass" if score == 1 else "Fail"
    return {
        "key": "multiline",
        "score": score,
        "comment": comment,
    }


tweet_task = dict(
    name="Tweet Generator",
    dataset="tweet-optim",
    initial_prompt={
        "identifier": "tweet-generator-example:c39837bd",
    },
    # See the starting prompt here:
    # https://smith.langchain.com/hub/langchain-ai/tweet-generator-example/c39837bd
    evaluators=[multiple_lines],
    evaluator_descriptions={
        "under_180_chars": "Checks if the tweet is under 180 characters. 1 if true, 0 if false.",
        "no_hashtags": "Checks if the tweet contains no hashtags. 1 if true, 0 if false.",
        "multiline": "Fails if the tweet is not multiple lines. 1 if true, 0 if false. 0 is bad.",
    },
)
```

We've defined a simple [evaluator](https://docs.smith.langchain.com/evaluation/how_to_guides/evaluation/evaluate_llm_application#use-custom-evaluators) to check that the output spans multiple lines.

We have also selected an initial prompt to optimize. You can check this out [in the hub](https://smith.langchain.com/hub/langchain-ai/tweet-generator-example/c39837bd).

By modifying the above values, you can configure your own task.

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
