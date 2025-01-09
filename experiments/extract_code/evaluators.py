from typing_extensions import TypedDict
from langsmith.schemas import Run, Example
from difflib import SequenceMatcher
import json


class Outputs(TypedDict):
    function_name: str
    parameters: list
    return_type: str


def semantic_similarity(a, b):
    """compare two values with semantic similarity, handling different data types."""
    if a is None or b is None:
        return 0

    # convert to strings for comparison
    if isinstance(a, (dict, list)):
        a = json.dumps(a, sort_keys=True)
    if isinstance(b, (dict, list)):
        b = json.dumps(b, sort_keys=True)

    a = str(a).lower()
    b = str(b).lower()
    return SequenceMatcher(None, a, b).ratio()


def accuracy_evaluator(run: Run, example: Example) -> dict:
    """evaluator for partial matching of function details."""
    try:
        # safely get reference outputs
        reference_outputs = example.outputs or {}
        extractions_str = reference_outputs.get("extractions", "{}")

        if isinstance(extractions_str, dict):
            nested_json = extractions_str
        else:
            nested_json = json.loads(extractions_str)

        # safely get run outputs
        run_outputs = run.outputs or {}

        if isinstance(run_outputs, str):
            try:
                run_outputs = json.loads(run_outputs)
            except json.JSONDecodeError:
                run_outputs = {}

        # calculate matches with semantic similarity
        matches = {
            "function_name": semantic_similarity(
                run_outputs.get("function_name"), nested_json.get("function_name")
            ),
            "parameters": semantic_similarity(
                run_outputs.get("parameters", []), nested_json.get("parameters", [])
            ),
            "return_type": semantic_similarity(
                run_outputs.get("return_type"), nested_json.get("return_type")
            ),
        }

        # calculate overall score
        score = sum(matches.values()) / len(matches)

        # generate detailed feedback
        if score == 1:
            comment = "Pass: Perfect match in function details."
        elif score > 0:
            comment = f"Partial match (score: {score:.2f})."
        else:
            comment = "Fail: No match in function details."

        # add specific field comparisons to comment
        field_feedback = [f"{k}: {v:.2f}" for k, v in matches.items()]
        comment += f"\nField-wise similarity scores:\n{', '.join(field_feedback)}"

    except Exception as e:
        # provide informative error feedback
        score = 0
        comment = (
            f"Error in evaluation: {str(e)}. Run outputs: {str(run.outputs)[:200]}..."
        )

    return {
        "key": "function_extraction_accuracy",
        "score": score,
        "comment": comment,
    }


evaluators = [accuracy_evaluator]
