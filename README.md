# promptimizer

Prompt optimization trainer.

Example:

```shell
uv venv
source .venv/bin/activate
uv pip install -e .
python main.py tweet
```

Script:

```shell
Usage:  [OPTIONS]

  Optimize prompts for different tasks.

Options:
  --version [1]                [required]
  --task TEXT                  Task to optimize. You can pick one off the
                               shelf or  path. Off-the-shelf options
                               include: scone, tweet, metaprompt, simpleqa,
                               ticket-classification.
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

Currently has ~ 4 tasks:

- tweet: write tweets
- simpleqa: really hard Q&A
- scone: NLI
- ticket-classification: classify & triage inbound support tickets
- metaprompt: optimize the metaprompt itself over the above 3 tasks

![run](./static/optimizer.gif)