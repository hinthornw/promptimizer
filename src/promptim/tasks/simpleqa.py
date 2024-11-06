from typing import Literal

from pydantic import BaseModel, Field
from promptim.trainer import PromptWrapper, Task
from langchain_openai import ChatOpenAI

GRADER_TEMPLATE = """Your job is to look at a question, a gold target, and a predicted answer, and then assign a grade of either ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"].
First, I will give examples of each grade, and then you will grade a new example.

The following are examples of CORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia Obama and Sasha Obama
Predicted answer 1: sasha and malia obama
Predicted answer 2: most people would say Malia and Sasha, but I'm not sure and would have to double check
Predicted answer 3: Barack Obama has two daughters. Their names are Malia Ann and Natasha Marian, but they are commonly referred to as Malia Obama and Sasha Obama. Malia was born on July 4, 1998, and Sasha was born on June 10, 2001.
```
These predicted answers are all CORRECT because:
    - They fully contain the important information in the gold target.
    - They do not contain any information that contradicts the gold target.
    - Only semantic meaning matters; capitalization, punctuation, grammar, and order don't matter.
    - Hedging and guessing are permissible, provided that the gold target is fully included and the response contains no incorrect information or contradictions.

The following are examples of INCORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: Malia.
Predicted answer 2: Malia, Sasha, and Susan.
Predicted answer 3: Barack Obama does not have any children.
Predicted answer 4: I think it's either Malia and Sasha. Or it could be Malia and Jackie. Or it could be Joey and Malia.
Predicted answer 4: While I don't know their exact names, I can tell you that Barack Obama has three children.
Predicted answer 5: It's possible you may mean Betsy and Olivia. However, you should clarify further details with updated references if necessary. Is that the correct answer?
Predicted answer 6: It may be the case that Obama's child is named James. However, it's recommended to confirm the most accurate and updated information since this could change over time. This model may not always reflect the most current information.
```
These predicted answers are all INCORRECT because:
    - A factual statement in the answer contradicts the gold target. Incorrect statements that have some hedging (e.g., "it is possible that", "although i'm not sure, i think") are also considered incorrect.

The following are examples of NOT_ATTEMPTED predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: I don't know.
Predicted answer 2: I need more context about which Obama you are talking about.
Predicted answer 3: Without researching the web, I cannot answer this question. However, I can tell you that Barack Obama has two children.
Predicted answer 4: Barack Obama has two children. I know that one of them is Malia, but I'm not sure about the other one.
```
These predicted answers are all NOT_ATTEMPTED because:
    - The important information in the gold target is not included in the answer.
    - No statements in the answer contradict the gold target.

Also note the following things:
- For grading questions where the gold target is a number, the predicted answer needs to be correct to the last significant figure in the gold answer. For example, consider a question "How many citations does the Transformer Paper have?" with gold target "120k". 
    - Predicted answers "120k", "124k", and 115k" are all CORRECT. 
    - Predicted answers "100k" and "113k" are INCORRECT. 
    - Predicted answers "around 100k" and "more than 50k" are considered NOT_ATTEMPTED because they neither confirm nor contradict the gold target.
- The gold target may contain more information than the question. In such cases, the predicted answer only needs to contain the information that is in the question.
    - For example, consider the question "What episode did Derek and Meredith get legally married in Grey's Anatomy?" with gold target "Season 7, Episode 20: White Wedding". Either "Season 7, Episode 20" or "White Wedding" would be considered a CORRECT answer.
- Do not punish predicted answers if they omit information that would be clearly inferred from the question.
    - For example, consider the question "What city is OpenAI headquartered in?" and the gold target "San Francisco, California". The predicted answer "San Francisco" would be considered CORRECT, even though it does not include "California".
    - Consider the question "What award did A pretrainer's guide to training data: Measuring the effects of data age, domain coverage, quality, & toxicity win at NAACL '24?", the gold target is "Outstanding Paper Award". The predicted answer "Outstanding Paper" would be considered CORRECT, because "award" is presumed in the question.
    - For the question "What is the height of Jason Wei in meters?", the gold target is "1.73 m". The predicted answer "1.75" would be considered CORRECT, because meters is specified in the question.
    - For the question "What is the name of Barack Obama's wife?", the gold target is "Michelle Obama". The predicted answer "Michelle" would be considered CORRECT, because the last name can be presumed.
- Do not punish for typos in people's name if it's clearly the same name. 
    - For example, if the gold target is "Hyung Won Chung", you can consider the following predicted answers as correct: "Hyoong Won Choong", "Hyungwon Chung", or "Hyun Won Chung".

Here is a new example. Simply reply with either CORRECT, INCORRECT, NOT_ATTEMPTED. Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answer.
```
Question: {question}
Gold target: {target}
Predicted answer: {predicted_answer}
```

Grade the predicted answer of this new question as one of:
- CORRECT
- INCORRECT
- NOT_ATTEMPTED
"""


class GraderOutput(BaseModel):
    """Submit the grade."""

    grade: Literal["CORRECT", "INCORRECT", "NOT_ATTEMPTED"] = Field(
        description="The grade for the predicted answer"
    )


grader = ChatOpenAI(model="gpt-4o-mini").with_structured_output(GraderOutput)


async def simpleqa_evaluator(run, example):
    """Evaluate the correctness of the SimpleQA answer."""
    grader_prompt = GRADER_TEMPLATE.format(
        question=example.inputs["problem"],
        target=example.outputs["answer"],
        predicted_answer=run.outputs["output"],
    )

    grading_response = await grader.ainvoke(grader_prompt)

    grade = grading_response.grade

    score = 1 if grade == "CORRECT" else 0
    comment = grade.lower()

    return {"key": "simpleqa_score", "score": score, "comment": comment}


simpleqa_task = Task(
    name="SimpleQA",
    description="A task to measure short-form factuality in large language models",
    dataset="simpleqa-optim",
    initial_prompt=PromptWrapper(identifier="langchain-ai/simpleqa-example:43349b82"),
    evaluators=[simpleqa_evaluator],
    evaluator_descriptions={
        "simpleqa_score": "Evaluates the correctness of the answer. 1 if correct, 0 if incorrect or not attempted."
    },
)

if __name__ == "__main__":
    import random

    import langsmith as ls

    c = ls.Client()
    examples = list(c.list_examples(dataset_name="Simple QA Full"))

    random.shuffle(examples)
    full = examples.copy()
    train, dev, test = [], [], []
    dataset = c.create_dataset(dataset_name="simpleqa-optim")
    for ds, size, name in zip(
        [train, dev, test], [200, 100, 100], ["train", "dev", "test"]
    ):
        for i in range(size):
            ds.append(full.pop())

        c.create_examples(
            inputs=[{"problem": e.inputs["problem"]} for e in ds],
            outputs=[e.outputs for e in ds],
            dataset_id=dataset.id,
            metadata=[e.inputs["metadata"] for e in ds],
            splits=[name] * len(ds),
        )
