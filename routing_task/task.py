from typing import TypedDict, Dict
from langsmith.schemas import Run, Example

class Outputs(TypedDict):
    is_portal_mail: bool
    is_base_po_mail: bool


def overall_accuracy(run: Run, example: Example) -> dict:
    pred: Dict[str, bool] = run.outputs
    ref: Dict[str, bool] = example.outputs

    portal_match = pred.get("is_portal_mail") == ref.get("is_portal_mail")
    base_match   = pred.get("is_base_po_mail") == ref.get("is_base_po_mail")
    both_match   = portal_match and base_match

    comment = "" if both_match else (
        f"Expected portal={ref['is_portal_mail']} base={ref['is_base_po_mail']}, "
        f"got portal={pred.get('is_portal_mail')} base={pred.get('is_base_po_mail')}"
    )

    return {
        "key": "overall_accuracy",
        "score": 1.0 if both_match else 0.0,
        "comment": comment
    }


def portal_accuracy(run: Run, example: Example) -> dict:
    pred: Dict[str, bool] = run.outputs
    ref: Dict[str, bool] = example.outputs

    match = pred.get("is_portal_mail") == ref.get("is_portal_mail")
    comment = "" if match else f"Portal expected={ref['is_portal_mail']}, got={pred.get('is_portal_mail')}"

    return {
        "key": "portal_accuracy",
        "score": 1.0 if match else 0.0,
        "comment": comment
    }


def base_accuracy(run: Run, example: Example) -> dict:
    pred: Dict[str, bool] = run.outputs
    ref: Dict[str, bool] = example.outputs

    match = pred.get("is_base_po_mail") == ref.get("is_base_po_mail")
    comment = "" if match else f"Base expected={ref['is_base_po_mail']}, got={pred.get('is_base_po_mail')}"

    return {
        "key": "base_accuracy",
        "score": 1.0 if match else 0.0,
        "comment": comment
    }


evaluators = [
    overall_accuracy,
    portal_accuracy,
    base_accuracy,
]
