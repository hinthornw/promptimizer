import langsmith as ls
from langchain_anthropic import ChatAnthropic
from promptim.trainer import (
    PromptWrapper,
    PromptOptimizer,
    Task,
    DEFAULT_METAPROMPT,
)
from promptim.tasks.scone import scone_task
from promptim.tasks.tweet_generator import tweet_task
from promptim.tasks.simpleqa import simpleqa_task
from promptim.tasks.ticket_classification import ticket_classification_task
from langchain_core.prompts import ChatPromptTemplate


DEFAULT_METAMETAPROMPT = """You are an expert in prompt optimization systems. Your task is to improve the effectiveness of prompt optimization prompts - the prompts used to guide the improvement of task-specific prompts.

Current Optimization Prompt:
{current_prompt}

Performance Data:
Shows how this optimization prompt performed in improving various task-specific prompts
{annotated_results}

Objective:
Improve the optimization prompt to better guide the enhancement of task-specific prompts across:
- Systematic analysis of prompt performance
- Identification of improvement opportunities
- Generation of enhanced prompts
- Validation of improvements

Analysis Steps:
1. Optimization Effectiveness
   - How well did this optimization prompt guide improvements?
   - Which aspects of prompt optimization were handled well/poorly?
   - What patterns emerge in successful vs unsuccessful optimization attempts?

2. Structural Assessment
   - How clearly does it guide the optimization process?
   - How well does it maintain prompt constraints?
   - What components are missing or ineffective?

3. Improvement Strategy
   - Which elements of the optimization process need enhancement?
   - How can we make the optimization guidance more effective?
   - What additional checks or validations would help?

Output Format:
<effectiveness_analysis>
Analysis of how well this optimization prompt guides improvements
</effectiveness_analysis>

<improvement_strategy>
Specific changes to enhance optimization capabilities
</improvement_strategy>

<improved_optimization_prompt>
The enhanced prompt for optimizing task-specific prompts
</improved_optimization_prompt>"""


class MetapromptSystem:
    """System for running the metaprompt optimization task."""

    def __init__(self, task_map: dict[str, Task], meta_prompt: PromptWrapper):
        self.task_map = task_map
        self.model = ChatAnthropic(
            model="claude-3-5-sonnet-20241022", max_tokens_to_sample=8192
        )
        self.trainer = PromptOptimizer(self.model, meta_prompt.get_prompt_str())

    async def __call__(self, prompt: ChatPromptTemplate, inputs: dict) -> dict:
        task = self.task_map[inputs["task"]]
        task.initial_prompt.load()

        # Run initial prompt on batch using aevaluate
        async def predict(example_inputs: dict):
            return await task.system_safe(task.initial_prompt.load(), example_inputs)

        train_batch = list(
            self.trainer.client.list_examples(example_ids=inputs["train_batch"])
        )
        dev_batch = list(
            self.trainer.client.list_examples(example_ids=inputs["dev_batch"])
        )
        initial_results = [
            r
            async for r in (
                await ls.aevaluate(
                    predict,
                    data=train_batch,
                    evaluators=task.evaluators,
                )
            )
        ]
        task.initial_prompt.get_prompt_str()

        # Generate new downstream task prompt
        extracted = await self.trainer.apply_metaprompt(
            current_prompt=task.initial_prompt,
            meta_prompt=prompt.messages[0].prompt.template,  # type: ignore
            task=task,
            results=initial_results,
        )

        # Now we actually evaluate based on how well the updated prompt's "improvements"
        # translate to a dev batch
        initial_dev_results = [
            r
            async for r in (
                await ls.aevaluate(
                    predict,
                    data=dev_batch,
                    evaluators=task.evaluators,
                )
            )
        ]
        initial_dev_scores = await self.trainer.calculate_scores(initial_dev_results)

        async def predict_new(example_inputs: dict):
            return await task.system_safe(extracted._cached, example_inputs)

        new_results = [
            r
            async for r in (
                await ls.aevaluate(
                    predict_new,
                    data=dev_batch,
                    evaluators=task.evaluators,
                )
            )
        ]
        new_scores = await self.trainer.calculate_scores(new_results)
        return {
            "original_prompt": task.initial_prompt,
            "new_prompt": extracted.get_prompt_str(),
            # "reasoning_for_changes": extracted.analysis,
            "initial_scores": initial_dev_scores,
            "new_scores": new_scores,
        }


def metaprompt_evaluator(run, example):
    """Evaluate the performance of the metaprompt."""
    original_score = sum(run.outputs["initial_scores"].values()) / len(
        run.outputs["initial_scores"]
    )
    new_score = sum(run.outputs["new_scores"].values()) / len(run.outputs["new_scores"])
    # Map the difference in scores to a 0 to 1 scale
    score_diff = new_score - original_score
    normalized_score = max(0, min(1, (score_diff + 1) / 2))
    if normalized_score > 0.5:
        comment = "The average scores improved after making the changes suggested by the prompt optimizer."
    elif normalized_score < 0.5:
        comment = "The average score dropped after making the changes suggested by the prompt optimizer."
    else:
        comment = "The average score remained the same after making the changes suggested by the prompt optimizer."

    return {
        "key": "metaprompt_improvement",
        "score": normalized_score,
        "comment": comment,
    }


prompt_config = PromptWrapper(prompt_str=DEFAULT_METAPROMPT)
metaprompt_task = Task(
    name="Metaprompt Optimizer",
    description="A meta-optimization task that aims to improve the prompt used for optimizing task-specific prompts. This task evaluates and enhances the effectiveness of the prompt optimization process itself, leading to better performance across various language tasks.",
    dataset="metaprompt-optim",
    initial_prompt=prompt_config,
    evaluators=[metaprompt_evaluator],
    evaluator_descriptions={
        "metaprompt_improvement": "Checks if the new prompt leads to improved scores. 1 if better, 0.5 if same, 0 if worse."
    },
    system=MetapromptSystem(
        {
            "scone": scone_task,
            "tweet": tweet_task,
            "simpleqa": simpleqa_task,
            "ticket_classification_task": ticket_classification_task,
        },
        meta_prompt=prompt_config,
    ),
)


if __name__ == "__main__":
    import argparse
    from langsmith import Client
    import random

    random.seed(42)

    def create_datasets(client, tasks, batchsize, overwrite=False):
        datasets = {}
        for split_name in ["train", "dev", "test"]:
            dataset_name = "metaprompt-optim"
            if overwrite:
                if client.has_dataset(dataset_name=dataset_name):
                    client.delete_dataset(dataset_name=dataset_name)
            datasets[split_name] = client.create_dataset(dataset_name=dataset_name)

        for task_name in tasks:
            task_datasets = {
                "train": client.list_examples(dataset_name=f"{task_name}-train"),
                "dev": client.list_examples(dataset_name=f"{task_name}-dev"),
                "test": client.list_examples(dataset_name=f"{task_name}-test"),
            }

            for split_name in ["train", "dev", "test"]:
                examples = []
                task_split_examples = list(task_datasets[split_name])
                random.shuffle(task_split_examples)
                for i in range(len(task_split_examples) // (batchsize * 2)):
                    batch = [
                        str(example.id)
                        for example in task_split_examples[i : i + batchsize * 2]
                    ]

                    examples.append(
                        {
                            "task": task_name,
                            "train_batch": batch[0 : len(batch) // 2],
                            "dev_batch": batch[len(batch) // 2 :],
                        }
                    )

                client.create_examples(
                    inputs=examples,
                    dataset_id=datasets[split_name].id,
                    splits=[split_name] * len(examples),
                )

        return datasets

    parser = argparse.ArgumentParser(
        description="Generate datasets for metaprompt optimization"
    )
    parser.add_argument(
        "--batchsize", type=int, default=5, help="Number of examples in each batch."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing datasets if they exist.",
    )
    args = parser.parse_args()

    client = Client()
    tasks = ["scone", "tweet", "simpleqa", "ticket_classification_task"]

    datasets = create_datasets(client, tasks, args.batchsize, args.overwrite)

    print("Datasets created successfully!")
    for name, dataset in datasets.items():
        print(f"{name.capitalize()} dataset ID: {dataset.id}")
