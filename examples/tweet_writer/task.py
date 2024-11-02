def multiple_lines(run, example):
    """Evaluate if the tweet contains multiple lines."""
    result = run.outputs.get("tweet", "")
    score = int("\n" in result)
    comment = "Pass" if score == 1 else "Fail"
    return {
        "key": "multiline",
        "score": score,
        "comment": comment,
    }


tweet_task = dict(
    name="Tweet Generator",
    dataset="tweet-optim",
    initial_prompt={
        "identifier": "tweet-generator-example:c39837bd",
    },
    # See the starting prompt here:
    # https://smith.langchain.com/hub/langchain-ai/tweet-generator-example/c39837bd
    evaluators=[multiple_lines],
    evaluator_descriptions={
        "under_180_chars": "Checks if the tweet is under 180 characters. 1 if true, 0 if false.",
        "no_hashtags": "Checks if the tweet contains no hashtags. 1 if true, 0 if false.",
        "multiline": "Fails if the tweet is not multiple lines. 1 if true, 0 if false. 0 is bad.",
    },
)
