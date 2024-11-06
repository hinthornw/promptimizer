import functools
import logging

from langchain_openai import ChatOpenAI
from promptim.trainer import PromptWrapper, Task
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


class Grade(BaseModel):
    """Call to submit your grade."""

    reasoning: str = Field(
        description="First, explain your thought process on why you are giving the provided grade."
    )
    score: int = Field(
        ge=0, le=5, description="Then, submit your score on a scale from 0 to 5."
    )

    @property
    def normalized(self):
        return self.score / 5


@functools.lru_cache
def _get_judge():
    from trustcall import create_extractor

    return create_extractor(
        ChatOpenAI(model="gpt-4o-mini"), tools=[Grade], tool_choice=Grade.__name__
    )


utemplate = """Grade the following:
Predicted: {predicted}
Reference example: {reference}"""


async def summary_quality(run, example):
    predicted = run.outputs.get("summary")
    rubric = """Grade the quality of summary. If it fails any criteria, give a 0. If it's perfect, give a 5.
Criteria:
- Must not include idle words like "the email is about X"
- Preferred format is <person> from <org> needs/wants X
"""
    reference = example.outputs["summary"]
    result = await _get_judge().ainvoke(
        [
            ("system", rubric),
            ("user", utemplate.format(predicted=predicted, reference=reference)),
        ]
    )
    grade: Grade = result["responses"][0]
    pf = "Pass" if grade.score >= 4 else "Fail"
    return {"score": grade.normalized, "comment": f"{pf}: {grade.reasoning}"}


def accuracy_check(run, example, key: str):
    predicted = run.outputs.get(key)
    reference = example.outputs.get(key)
    if reference is None:
        return {
            "key": f"{key}-correctness",
            "comment": "Skipping - reference label not found.",
        }
    score = (
        predicted == reference
        if not isinstance(reference, list)
        else predicted in reference
    )
    pf = "Pass" if score else "Fail"
    return {
        "key": f"{key}-correctness",
        "score": score,
        "comment": f"{pf}",
    }  #: Expected {reference}. Got: {predicted}. Why did you get this wrong? Think deeply and update associations."}


classifiers = [
    functools.partial(accuracy_check, key=key)
    for key in [
        "category",
        "support_category",
        "ticket_status",
        "requires_response",
        "non_support_category",
    ]
]
evaluators = [summary_quality, *classifiers]


ticket_classification_task = Task(
    name="Ticket Classification",
    description="A task to classify customer support tickets",
    dataset="ticket-classification-optim",
    initial_prompt=PromptWrapper(
        identifier="langchain-ai/ticket-classifier-example:376ab5e4",
        which=1,
    ),
    evaluators=evaluators,
    evaluator_descriptions={
        "summary_quality": "Evaluates the quality of the summary",
        "category-correctness": "Checks if the category is correct",
        "support_category-correctness": "Checks if the support category is correct",
        "ticket_status-correctness": "Checks if the ticket status is correct",
        "requires_response-correctness": "Checks if the requires_response field is correct",
        "non_support_category-correctness": "Checks if the non-support category is correct",
    },
)

if __name__ == "__main__":
    import random

    import langsmith as ls

    c = ls.Client()
    examples = list(
        c.list_examples(dataset_name="customer-support-bot.test_extraction")
    )

    random.shuffle(examples)
    full = examples.copy()
    train, dev, test = [], [], []
    dname = "ticket-classification-optim"
    try:
        dataset = c.create_dataset(dataset_name=dname)
    except Exception:
        c.delete_dataset(dataset_name=dname)
        dataset = c.create_dataset(dataset_name=dname)
    for ds, size, name in zip(
        [train, dev, test], [41, 20, 20], ["train", "dev", "test"]
    ):
        for i in range(size):
            ds.append(full.pop())

        outputs = [e.outputs for e in ds]
        for o in outputs:
            for k, v in o.pop("outputs", {}).items():
                o[k] = v
        c.create_examples(
            inputs=[e.inputs for e in ds],
            outputs=outputs,
            dataset_id=dataset.id,
            splits=[name] * len(ds),
        )
