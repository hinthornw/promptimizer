from promptim.trainer import PromptWrapper, Task


def exact_match(run, example):
    """Evaluate the exact match correctness of the NLI result."""
    try:
        predicted = run.outputs["is_entailed"]
        expected = example.outputs["answer"]
        score = expected.lower() == predicted.lower()
    except Exception:
        try:
            expected = example.outputs["answer"]
            expected_bool = {"no": False, "yes": True}.get(expected.strip().lower())
            score = run.outputs["output"].is_entailed == expected_bool
        except Exception:
            score = 0
    return {
        "key": "exact_match",
        "score": int(score),
    }


scone_task = Task(
    name="Scone (NLI)",
    dataset="scone-optim",
    initial_prompt=PromptWrapper(identifier="langchain-ai/scone-example:d49910d6"),
    evaluators=[exact_match],
    evaluator_descriptions={
        "exact_match": "Directly compares the expected against the predicted outputs. 1 if correct, 0 if incorrect."
    },
)
