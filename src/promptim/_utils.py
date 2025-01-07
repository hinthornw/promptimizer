"""Utility functions."""

from difflib import SequenceMatcher
from rich import print as richprint
from rich.panel import Panel

import re
import uuid
import langsmith as ls
from collections import deque


def _colorize_diff(diff):
    for op, i1, i2, j1, j2 in diff.get_opcodes():
        if op == "equal":
            yield diff.a[i1:i2]
        elif op == "insert":
            yield f"[green]{diff.b[j1:j2]}[/green]"
        elif op == "delete":
            yield f"[red]{diff.a[i1:i2]}[/red]"
        elif op == "replace":
            yield f"[red]{diff.a[i1:i2]}[/red][green]{diff.b[j1:j2]}[/green]"


def print_rich_diff(original: str, updated: str, title: str = "") -> None:
    diff = SequenceMatcher(None, original, updated)
    colorized_diff = "".join(_colorize_diff(diff))
    panel = Panel(
        colorized_diff, title=title or "Prompt Diff", expand=False, border_style="bold"
    )
    richprint(panel)


def get_var_healer(vars: set[str], all_required: bool = False):
    var_to_uuid = {f"{{{v}}}": uuid.uuid4().hex for v in vars}
    uuid_to_var = {v: k for k, v in var_to_uuid.items()}

    def escape(input_string: str) -> str:
        result = re.sub(r"(?<!\{)\{(?!\{)", "{{", input_string)
        result = re.sub(r"(?<!\})\}(?!\})", "}}", result)
        return result

    if not vars:
        return escape

    mask_pattern = re.compile("|".join(map(re.escape, var_to_uuid.keys())))
    unmask_pattern = re.compile("|".join(map(re.escape, var_to_uuid.values())))

    strip_to_optimize_pattern = re.compile(
        r"<TO_OPTIMIZE.*?>|</TO_OPTIMIZE>", re.MULTILINE | re.DOTALL
    )

    def assert_all_required(input_string: str) -> str:
        if not all_required:
            return input_string

        missing = [var for var in vars if f"{{{var}}}" not in input_string]
        if missing:
            raise ValueError(f"Missing required variable: {', '.join(missing)}")

        return input_string

    def mask(input_string: str) -> str:
        return mask_pattern.sub(lambda m: var_to_uuid[m.group(0)], input_string)

    def unmask(input_string: str) -> str:
        return unmask_pattern.sub(lambda m: uuid_to_var[m.group(0)], input_string)

    def pipe(input_string: str) -> str:
        return unmask(
            strip_to_optimize_pattern.sub(
                "", escape(mask(assert_all_required(input_string)))
            )
        )

    return pipe


def get_token_usage() -> int | None:
    rt = ls.get_current_run_tree()
    if not rt:
        return
    runs = deque([rt])
    kept = []
    while runs:
        run = runs.popleft()
        if run.run_type == "llm":
            kept.append(run)
        runs.extend(run.child_runs)
    all_toks = []
    for r in kept:
        usage = ((r.outputs or {}).get("llm_output") or {}).get("usage")
        if not usage:
            continue
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
        if input_tokens is None or output_tokens is None:
            continue
        all_toks.append(output_tokens + input_tokens)

    if all_toks:
        return sum(all_toks)
