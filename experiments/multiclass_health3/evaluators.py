from langchain_core.messages import AIMessage
from langsmith.schemas import Run, Example


def accuracy_evaluator(run: Run, example: Example) -> dict:
    """Evaluator to check if the predicted emotion class matches the reference."""
    reference_outputs = example.outputs
    predicted: AIMessage = run.outputs["output"]
    result = str(predicted.content)
    score = 1 if result == reference_outputs["three_class"] else 0
    return {
        "key": "accuracy",
        "score": score,
        "comment": "Pass: health disease is correct"
        if score == 1
        else "Fail: health disease is not correct",
    }


evaluators = [accuracy_evaluator]
