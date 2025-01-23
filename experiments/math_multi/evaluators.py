from langsmith.schemas import Run, Example
from trustcall import create_extractor
from langchain_openai import ChatOpenAI
from typing import Literal


def segment_error(
    value_correctness_analysis: str,
    value_correctness: bool,
    language_correctness_analysis: str,
    language_correctness: bool,
    error_type: Literal["logic", "language", "syntax"],
):
    """Analyze the failing test case to break down **why** the prompt failed. It could fail either because the value was wrong (logic error), the response language was wrong (language error) or just spelling error (syntax error; so the value is correct and langugae is correct but there was a small spacing or punctuation error)."""
    pass


grader = create_extractor(
    ChatOpenAI(model="gpt-4o-mini"),
    tools=[segment_error],
    tool_choice="segment_error",
)


async def correctness_evaluator(run: Run, example: Example) -> dict:
    """Evaluator to check if the predicted answer matches the reference."""
    reference_outputs = example.outputs
    try:
        predicted = run.outputs["answer"]
    except KeyError:
        predicted = "Failed to generate answer"
    score = 1 if predicted.lower() == reference_outputs["answer"].lower() else 0
    if not score:
        response = await grader.ainvoke(
            "Analyze the following test case to break-down why it failed. First analyze the value in a language-agnostic sense, then analyze the language."
            " If it is in the correct language and the correct value but just written differently, then it is a syntax error. "
            "\n\nExample 1:\n\n"
            "Test case: 1 + 2\n"
            "Reference answer: 三\n"
            "Predicted answer: three\n"
            " Judgment: "
            " value_correctness_analysis: 三 and three both represent 3, so the value is correct.\n"
            " value_correctness: True\n"
            "language_correctness_analysis: three is in the incorrect language. To pass, the prompt should have responded in Mandarin.\n"
            "language_correctness: false"
            "error_type: language"
            " Example 2:"
            "Test case: 1 + 30\n"
            "Reference answer: thirty-three\n"
            "Predicted answer: thirty three\n"
            " Judgment: "
            " value_correctness_analysis: thirty-three and thirty three both represent 33, so the value is correct.\n"
            " value_correctness: True\n"
            "language_correctness_analysis: thirty three is in the correct language but written differently; the language is correct\n"
            "language_correctness: true"
            "error_type: syntax"
            "\n\n"
            "# Test:\n"
            f"Test case: {example.inputs['problem']}\n"
            f"Reference answer: {reference_outputs['answer']}\n"
            f"Predicted answer: {predicted}"
        )
        result = response["responses"][0]
        if result.error_type == "syntax":
            return {
                "key": "correctness",
                "score": 1,
                "comment": "Pass: answer is correct, modulo a small syntax error.",
            }
        return {
            "key": "correctness",
            "score": 0,
            "comment": f"Fail: answer is not correct. Error type: {result.error_type}. "
            f"Logical correctness analysis: {result.value_correctness_analysis}. Language correctness analysis: {result.language_correctness_analysis}.",
        }
    return {
        "key": "correctness",
        "score": score,
        "comment": (
            "Pass: answer is correct" if score == 1 else "Fail: answer is not correct"
        ),
    }


evaluators = [correctness_evaluator]
