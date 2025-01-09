from typing_extensions import TypedDict
from langsmith.schemas import Run, Example
from difflib import SequenceMatcher
import json


class Outputs(TypedDict):
    parties_involved: list
    effective_date: str
    termination_clauses: list
    jurisdiction: str
    governing_law: str
    payment_terms: dict
    liability_clauses: list
    confidentiality_terms: dict


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
    """evaluator for partial matching of legal document details."""
    try:
        # parse reference outputs from nested json structure
        reference_outputs = example.outputs or {}
        extractions_str = reference_outputs.get("extractions", "{}")

        if isinstance(extractions_str, dict):
            reference_data = extractions_str
        else:
            reference_data = json.loads(extractions_str)

        # parse run outputs
        run_outputs = run.outputs
        if isinstance(run_outputs, str):
            try:
                outputs = json.loads(run_outputs)
            except json.JSONDecodeError:
                outputs = {}
        else:
            outputs = run_outputs or {}

        # calculate matches with semantic similarity
        matches = {
            "parties_involved": semantic_similarity(
                outputs.get("parties_involved"), reference_data.get("parties_involved")
            ),
            "effective_date": semantic_similarity(
                outputs.get("effective_date"), reference_data.get("effective_date")
            ),
            "termination_clauses": semantic_similarity(
                outputs.get("termination_clauses"),
                reference_data.get("termination_clauses"),
            ),
            "jurisdiction": semantic_similarity(
                outputs.get("jurisdiction"), reference_data.get("jurisdiction")
            ),
            "governing_law": semantic_similarity(
                outputs.get("governing_law"), reference_data.get("governing_law")
            ),
            "payment_terms": semantic_similarity(
                outputs.get("payment_terms"), reference_data.get("payment_terms")
            ),
            "liability_clauses": semantic_similarity(
                outputs.get("liability_clauses"),
                reference_data.get("liability_clauses"),
            ),
            "confidentiality_terms": semantic_similarity(
                outputs.get("confidentiality_terms"),
                reference_data.get("confidentiality_terms"),
            ),
        }

        # calculate overall score
        score = sum(matches.values()) / len(matches)

        # generate detailed feedback
        if score > 0.9:
            comment = "Pass: Very close match in legal document details."
        elif score > 0.7:
            comment = "Good: Strong match with minor differences."
        elif score > 0.5:
            comment = "Fair: Moderate match with some differences."
        else:
            comment = "Need improvement: Significant differences found."

        # add specific field comparisons to comment
        field_feedback = [f"{k}: {v:.2f}" for k, v in matches.items()]
        comment += f"\nField-wise similarity scores:\n{', '.join(field_feedback)}"

    except Exception as e:
        score = 0
        comment = f"Error in evaluation: {str(e)}. Check JSON structure and parsing."

    return {"key": "legal_extraction_accuracy", "score": score, "comment": comment}


evaluators = [accuracy_evaluator]
