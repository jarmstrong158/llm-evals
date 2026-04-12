"""
llm-evals CLI

Usage:
  python cli.py run --dataset <name> --executor <module:Class> --scorer <name>
  python cli.py baseline --dataset <name>
  python cli.py list
  python cli.py add --dataset <name> --input "..." --expected "..."
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import typer
from rich.console import Console

# make sure core/ is importable
sys.path.insert(0, str(Path(__file__).parent))

from core import dataset as ds
from core import baseline as bl
from core import runner as rn
from core import reporter as rp
from core.scorers import ExactMatch, Contains, SemanticSimilarity, LLMJudge

app = typer.Typer(help="LLM eval framework — test any tool, score any output.")
console = Console()

SCORERS = {
    "exact": ExactMatch(),
    "contains": Contains(),
    "semantic": SemanticSimilarity(),
    "llm": LLMJudge(),
}


def _resolve_scorer(name: str):
    if name in SCORERS:
        return SCORERS[name]
    # allow module:Class syntax e.g. core.scorers:LLMJudge
    if ":" in name:
        mod_path, cls_name = name.rsplit(":", 1)
        mod = importlib.import_module(mod_path)
        return getattr(mod, cls_name)()
    raise typer.BadParameter(f"Unknown scorer '{name}'. Options: {list(SCORERS)}")


def _resolve_executor(spec: str):
    """
    Resolve an executor from a module:Class spec.
    e.g. core.executors:CLI or myevals.executors:MyExecutor
    """
    if ":" not in spec:
        raise typer.BadParameter(
            f"Executor must be module:Class format, e.g. 'core.executors:CLI'"
        )
    mod_path, cls_name = spec.rsplit(":", 1)
    mod = importlib.import_module(mod_path)
    return getattr(mod, cls_name)


@app.command()
def run(
    dataset: str = typer.Option(..., "--dataset", "-d", help="Dataset name"),
    executor: str = typer.Option(..., "--executor", "-e", help="module:Class for executor"),
    scorer: str = typer.Option("exact", "--scorer", "-s", help="Scorer: exact|contains|semantic|llm|module:Class"),
    show_passing: bool = typer.Option(False, "--show-passing", help="Show passing cases too"),
    save: bool = typer.Option(True, "--save/--no-save", help="Save run to runs/"),
    set_baseline: bool = typer.Option(False, "--set-baseline", help="Save this run as the new baseline"),
    regression_threshold: float = typer.Option(0.05, "--threshold", help="Regression threshold (pass rate drop)"),
):
    """Run an eval suite."""
    cases = ds.load(dataset)
    if not cases:
        console.print(f"[red]No cases found for dataset '{dataset}'[/]")
        raise typer.Exit(1)

    scorer_obj = _resolve_scorer(scorer)
    ExecutorCls = _resolve_executor(executor)
    executor_obj = ExecutorCls()

    console.print(f"Running [bold]{len(cases)}[/] cases...")

    def on_progress(i, total, result):
        icon = "[green]✓[/]" if result.passed else "[red]✗[/]"
        console.print(f"  {icon} [{i}/{total}] {result.case_id} ({result.duration_ms}ms)")

    result = rn.run(
        executor=executor_obj,
        cases=cases,
        scorer=scorer_obj,
        dataset_name=dataset,
        on_progress=on_progress,
    )

    rp.print_run(result, show_passing=show_passing)

    if save:
        path = bl.save_run(result)
        console.print(f"\nRun saved: [dim]{path}[/]")

    if set_baseline:
        path = bl.save_baseline(result)
        console.print(f"Baseline updated: [dim]{path}[/]")

    report = bl.compare(result, threshold=regression_threshold)
    if report:
        rp.print_regression(report)

    if result.failed > 0:
        raise typer.Exit(1)


@app.command()
def baseline(
    dataset: str = typer.Option(..., "--dataset", "-d", help="Dataset name"),
    executor: str = typer.Option(..., "--executor", "-e", help="module:Class for executor"),
    scorer: str = typer.Option("exact", "--scorer", "-s", help="Scorer name"),
):
    """Run the eval suite and save result as the new baseline."""
    cases = ds.load(dataset)
    if not cases:
        console.print(f"[red]No cases found for dataset '{dataset}'[/]")
        raise typer.Exit(1)

    scorer_obj = _resolve_scorer(scorer)
    ExecutorCls = _resolve_executor(executor)
    executor_obj = ExecutorCls()

    result = rn.run(executor=executor_obj, cases=cases, scorer=scorer_obj, dataset_name=dataset)
    rp.print_run(result, show_passing=True)
    path = bl.save_baseline(result)
    console.print(f"\n[green]Baseline saved:[/] {path}")


@app.command(name="list")
def list_cmd():
    """List all datasets and their case counts."""
    names = ds.list_datasets()
    if not names:
        console.print("[dim]No datasets found.[/]")
        return
    console.print(f"[bold]Datasets ({len(names)}):[/]")
    for name in names:
        cases = ds.load(name)
        console.print(f"  [cyan]{name}[/]  [dim]{len(cases)} cases[/]")


@app.command()
def add(
    dataset: str = typer.Option(..., "--dataset", "-d", help="Dataset name"),
    input: str = typer.Option(..., "--input", "-i", help="Input value (string or JSON)"),
    expected: str = typer.Option(..., "--expected", "-x", help="Expected output"),
    tags: str = typer.Option("", "--tags", "-t", help="Comma-separated tags"),
    notes: str = typer.Option("", "--notes", "-n", help="Optional notes"),
):
    """Add a test case to a dataset."""
    import json as _json
    try:
        input_val = _json.loads(input)
    except Exception:
        input_val = input

    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
    case = ds.add(dataset, input_val, expected, tags=tag_list, notes=notes)
    console.print(f"[green]Added[/] {case['id']} to [cyan]{dataset}[/]")


@app.command()
def show(
    dataset: str = typer.Option(..., "--dataset", "-d", help="Dataset name"),
):
    """Show all cases in a dataset."""
    cases = ds.load(dataset)
    if not cases:
        console.print(f"[dim]No cases in '{dataset}'[/]")
        return
    for c in cases:
        console.print(f"[bold]{c['id']}[/]  tags={c.get('tags', [])}")
        console.print(f"  input:    {str(c['input'])[:100]}")
        console.print(f"  expected: {str(c['expected'])[:100]}")
        if c.get("notes"):
            console.print(f"  notes:    {c['notes']}")


if __name__ == "__main__":
    app()
