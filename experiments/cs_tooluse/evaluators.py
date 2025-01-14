from langchain_core.messages import AIMessage
import asyncio


async def check_tool_call(expected: dict, tool_calls: list) -> dict:
    matching = next((tc for tc in tool_calls if tc["name"] == expected["name"]), None)
    if matching is None:
        return {"comment": "", "score": 0.0}
    args: dict = matching["args"]
    expected_args: dict = expected["args"]
    pf = {}
    for arg_name, arg_value in expected_args.items():
        if arg_name not in args:
            return {"comment": f"missing arg: {arg_name}", "score": 0.0}
        if isinstance(arg_value, str) and len(arg_value.split(" ")) > 5:
            pf[arg_name] = (
                arg_name in args
                and isinstance(args[arg_name], str)
                and len(args[arg_name].split(" ")) > 5
            )
            continue
        pf[arg_name] = arg_value == args[arg_name]
    score = sum(pf.values()) / len(pf)
    if score < 1.0:
        score_per_arg = ", ".join(
            f"{arg_name}: {int(score)}" for arg_name, score in pf.items()
        )
        comment = f"Score per arg: {score_per_arg}"
    else:
        comment = "ok"
    return {"comment": comment, "score": score}


async def check_output(outputs: dict, reference_outputs: dict) -> dict:
    expected: dict = reference_outputs["output"]
    actual: AIMessage = outputs["output"]
    actual_tcs = actual.tool_calls or []

    coros = []
    for tc in expected.get("tool_calls", []) or []:
        coros.append(check_tool_call(tc, actual_tcs))

    results = await asyncio.gather(*coros)
    if not results:
        return {
            "key": "quality",
            "score": 1.0,
            "comment": "No tool calls found to evaluate.",
        }
    score = sum(r["score"] for r in results) / len(results)
    passed = score == 1.0
    comment = ", ".join(r["comment"] for r in results if r["comment"])
    comment = f"{'Passed' if passed else 'Failed'}: {comment}"

    return {"key": "quality", "score": score, "comment": comment}
