"""Evaluators to optimize task: tooluse-finance.

Evaluators compute scores for prompts run over the configured dataset:
https://smith.langchain.com/o/4b3539a7-f6b9-4950-a199-a27fd5dcbf2f/datasets/10de0a62-603f-429c-b222-e5fb624f7ca6
"""

from langsmith.schemas import Run, Example
import json


def extract_tool_call(run_outputs):
    """Helper function to extract tool call from nested run output structure."""
    try:
        if isinstance(run_outputs, dict) and "tool_calls" in run_outputs:
            tool_call = run_outputs["tool_calls"][0]
            if isinstance(tool_call, dict):
                return {"name": tool_call["name"], "args": tool_call["args"]}

        if isinstance(run_outputs, dict) and "output" in run_outputs:
            output = run_outputs["output"]

            if hasattr(output, "additional_kwargs"):
                additional_kwargs = output.additional_kwargs
                if (
                    isinstance(additional_kwargs, dict)
                    and "tool_calls" in additional_kwargs
                ):
                    tool_call = additional_kwargs["tool_calls"][0]
                    if "function" in tool_call:
                        return {
                            "name": tool_call["function"]["name"],
                            "args": json.loads(tool_call["function"]["arguments"])
                            if isinstance(tool_call["function"]["arguments"], str)
                            else tool_call["function"]["arguments"],
                        }

            if isinstance(output, dict):
                if "tool_calls" in output:
                    tool_call = output["tool_calls"][0]
                    if isinstance(tool_call, dict):
                        return {"name": tool_call["name"], "args": tool_call["args"]}

                if "additional_kwargs" in output:
                    additional_kwargs = output["additional_kwargs"]
                    if "tool_calls" in additional_kwargs:
                        tool_call = additional_kwargs["tool_calls"][0]
                        if "function" in tool_call:
                            return {
                                "name": tool_call["function"]["name"],
                                "args": json.loads(tool_call["function"]["arguments"])
                                if isinstance(tool_call["function"]["arguments"], str)
                                else tool_call["function"]["arguments"],
                            }

    except (KeyError, IndexError, json.JSONDecodeError, AttributeError):
        return None

    return None


def tool_use_evaluator(run: Run, example: Example) -> dict:
    """Evaluator for matching the correct tool and its inputs."""
    try:
        reference_outputs = example.outputs or {}
        correct_tool_str = reference_outputs.get("correct_tool", "{}")
        if isinstance(correct_tool_str, str):
            correct_tool = json.loads(correct_tool_str)
        else:
            correct_tool = correct_tool_str

        predicted_tool = extract_tool_call(run.outputs)
        if not predicted_tool:
            return {
                "key": "tool_use_accuracy",
                "score": 0,
                "comment": f"No valid tool calls found in run outputs: {str(run.outputs)[:200]}...",
            }

        tool_name_match = predicted_tool.get("name") == correct_tool.get("name")

        correct_inputs = set(correct_tool.get("inputs", {}).items())
        predicted_inputs = set(predicted_tool.get("args", {}).items())

        if correct_inputs:
            inputs_match_count = len(correct_inputs.intersection(predicted_inputs))
            tool_inputs_match = inputs_match_count / len(correct_inputs)
        else:
            tool_inputs_match = 1 if not predicted_inputs else 0

        score = (tool_name_match + tool_inputs_match) / 2

        if score == 1:
            comment = "Pass: Correct tool and inputs matched."
        elif score > 0:
            comment = (
                f"Partial match (score: {score:.2f}). Expected tool '{correct_tool.get('name')}' "
                f"with inputs {correct_tool.get('inputs')}, but got tool '{predicted_tool.get('name')}' "
                f"with inputs {predicted_tool.get('args')}."
            )
        else:
            comment = (
                f"Fail: Expected tool '{correct_tool.get('name')}' with inputs {correct_tool.get('inputs')}, "
                f"but got tool '{predicted_tool.get('name')}' with inputs {predicted_tool.get('args')}."
            )

    except Exception as e:
        score = 0
        comment = (
            f"Error in evaluation: {str(e)}. Run outputs: {str(run.outputs)[:200]}... "
            f"Reference correct tool: {str(correct_tool)}"
        )

    result = {
        "key": "tool_use_accuracy",
        "score": score,
        "comment": comment,
    }

    return result


evaluators = [tool_use_evaluator]
