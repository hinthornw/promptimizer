from langsmith.schemas import Run, Example


def accuracy_evaluator(run: Run, example: Example) -> dict:
    """Evaluator to check if the predicted emotion class matches the reference."""
    score = 1 if run.outputs["action"] == example.outputs["action"] else 0
    return {
        "key": "accuracy",
        "score": score,
        "comment": (
            "Pass: triage class is correct"
            if score == 1
            else "Fail: triage class is not correct"
        ),
    }


evaluators = [accuracy_evaluator]
