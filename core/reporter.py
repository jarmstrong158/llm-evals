"""
Reporter — print results to the terminal using Rich.
"""

from __future__ import annotations

from rich.console import Console
from rich.table import Table
from rich import box
from rich.text import Text

from core.runner import RunResult
from core.baseline import RegressionReport


console = Console()


def print_run(run: RunResult, show_passing: bool = False) -> None:
    """Print a full run summary with per-case table."""

    # Header
    console.rule(f"[bold cyan]Eval: {run.dataset}[/]")
    console.print(
        f"Executor: [yellow]{run.executor_name}[/]  "
        f"Scorer: [yellow]{run.scorer_name}[/]"
    )

    # Per-case table
    table = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold")
    table.add_column("ID", style="dim", width=10)
    table.add_column("Pass", width=6)
    table.add_column("Score", width=7)
    table.add_column("Reason", overflow="fold")
    table.add_column("ms", width=6, justify="right")

    for c in run.cases:
        if not show_passing and c.passed:
            continue
        status = "[green]PASS[/]" if c.passed else "[red]FAIL[/]"
        score_str = f"{c.score:.2f}"
        reason = c.reason[:120]
        if c.error:
            reason = f"[red]{c.error[:80]}[/]"
        table.add_row(c.case_id, status, score_str, reason, str(c.duration_ms))

    if table.row_count == 0 and not show_passing:
        console.print("[green]All cases passed.[/]")
    else:
        console.print(table)

    # Summary bar
    bar = _progress_bar(run.pass_rate)
    color = "green" if run.pass_rate >= 0.9 else "yellow" if run.pass_rate >= 0.7 else "red"
    console.print(
        f"\n[{color}]{bar}[/]  "
        f"[bold]{run.passed}/{run.total}[/] passed  "
        f"({run.pass_rate:.0%})  "
        f"avg score [bold]{run.avg_score:.3f}[/]"
    )


def print_regression(report: RegressionReport) -> None:
    """Print a regression comparison."""
    console.rule("[bold]Regression Check[/]")

    delta_str = f"{report.delta:+.1%}"
    if report.is_regression:
        console.print(
            f"[bold red]REGRESSION[/]  "
            f"Pass rate dropped {delta_str}  "
            f"({report.baseline_pass_rate:.1%} → {report.current_pass_rate:.1%})"
        )
        if report.newly_failing:
            console.print(f"[red]Newly failing:[/] {', '.join(report.newly_failing)}")
    else:
        console.print(
            f"[green]No regression[/]  "
            f"{delta_str} vs baseline  "
            f"({report.baseline_pass_rate:.1%} → {report.current_pass_rate:.1%})"
        )

    if report.newly_passing:
        console.print(f"[green]Newly passing:[/] {', '.join(report.newly_passing)}")


def _progress_bar(rate: float, width: int = 30) -> str:
    filled = int(rate * width)
    return "█" * filled + "░" * (width - filled)
