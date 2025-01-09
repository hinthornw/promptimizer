from langchain_core.messages import AIMessage
from langsmith.schemas import Run, Example


def accuracy_evaluator(run: Run, example: Example) -> dict:
    """Evaluator to check if the predicted emotion class matches the reference."""
    reference_outputs = example.outputs
    predicted: AIMessage = run.outputs["classification"]
    score = 1 if predicted == reference_outputs["classification"] else 0
    return {
        "key": "accuracy",
        "score": score,
        "comment": "Pass: triage class is correct"
        if score == 1
        else "Fail: triage class is not correct",
    }


evaluators = [accuracy_evaluator]
