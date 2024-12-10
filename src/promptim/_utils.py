"""Utility functions."""
from difflib import SequenceMatcher
from rich import print as richprint
from rich.panel import Panel


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
