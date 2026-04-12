"""
llm-evals CLI

Usage:
  python cli.py run   --dataset <name> --executor <module:name> --scorer <name>
  python cli.py add   --dataset <name> --input "..." --expected "..."
  python cli.py show  --dataset <name>
  python cli.py list
  python cli.py promote --dataset <name>   # mark last run as baseline

Executor format: module:name
  - If <name> resolves to a pre-instantiated object, it is used directly.
  - If <name> resolves to a class with no required args, it is instantiated.
  - Otherwise, an error explains how to pre-instantiate in a wrapper module.

Examples:
  # Use a pre-instantiated executor from your own module
  python cli.py run -d url_validation -e myproject.evals:check_url -s contains

  # Use the built-in Prompt executor (requires pre-instantiation)
  # In myproject/evals.py:
  #   from core.executors import Prompt
  #   check_grammar = Prompt("Is this grammatically correct? {input}", model="llama3.2")
  python cli.py run -d grammar -e myproject.evals:check_grammar -s llm
"""

from __future__ import annotations

import importlib
import inspect
import sys
from pathlib import Path

import typer
from rich.console import Console

sys.path.insert(0, str(Path(__file__).parent))

from core import dataset as ds
from core import baseline as bl
from core import runner as rn
from core import reporter as rp
from core.scorers import ExactMatch, Contains, SemanticSimilarity, LLMJudge

app = typer.Typer(
    help="LLM eval framework — test any tool, score any output.",
    no_args_is_help=True,
)
console = Console()

SCORERS = {
    "exact": ExactMatch(),
    "contains": Contains(),
    "semantic": SemanticSimilarity(),
    "llm": LLMJudge(),
}


def _resolve(spec: str, kind: str = "executor"):
    """
    Resolve a module:name reference to a callable object.

    Accepts:
      - A pre-instantiated object (used directly)
      - A class with no required constructor args (instantiated with no args)

    Raises a clear error if the class requires constructor args.
    """
    if ":" not in spec:
        raise typer.BadParameter(
            f"{kind} must be 'module:name', e.g. 'myproject.evals:my_executor'. "
            f"Built-in scorers: {list(SCORERS)}"
        )
    mod_path, attr_name = spec.rsplit(":", 1)
    try:
        mod = importlib.import_module(mod_path)
    except ModuleNotFoundError as e:
        raise typer.BadParameter(f"Cannot import module '{mod_path}': {e}")

    if not hasattr(mod, attr_name):
        raise typer.BadParameter(f"'{mod_path}' has no attribute '{attr_name}'")

    obj = getattr(mod, attr_name)

    if not inspect.isclass(obj):
        # Already instantiated — use directly
        if not callable(obj):
            raise typer.BadParameter(f"'{spec}' is not callable")
        return obj

    # It's a class — try no-arg instantiation
    sig = inspect.signature(obj.__init__)
    required = [
        p for name, p in sig.parameters.items()
        if name != "self" and p.default is inspect.Parameter.empty
        and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    ]
    if required:
        req_names = [p.name for p in required]
        raise typer.BadParameter(
            f"'{attr_name}' requires constructor args {req_names}. "
            f"Pre-instantiate it in a module and reference the instance:\n\n"
            f"  # In myproject/evals.py:\n"
            f"  from {mod_path} import {attr_name}\n"
            f"  my_{kind} = {attr_name}({', '.join(f'{n}=...' for n in req_names)})\n\n"
            f"  # Then run:\n"
            f"  python cli.py run ... --{kind} myproject.evals:my_{kind}"
        )
    return obj()


def _resolve_scorer(name: str):
    if name in SCORERS:
        return SCORERS[name]
    return _resolve(name, kind="scorer")


def _resolve_executor(spec: str):
    return _resolve(spec, kind="executor")


def _filter_cases(cases: list[dict], tags: str) -> list[dict]:
    if not tags:
        return cases
    required = {t.strip() for t in tags.split(",") if t.strip()}
    return [c for c in cases if required.issubset(set(c.get("tags", [])))]


@app.command()
def run(
    dataset: str = typer.Option(..., "--dataset", "-d", help="Dataset name"),
    executor: str = typer.Option(..., "--executor", "-e", help="module:name of executor (instance or no-arg class)"),
    scorer: str = typer.Option("exact", "--scorer", "-s", help="Scorer: exact|contains|semantic|llm|module:name"),
    tags: str = typer.Option("", "--tags", "-t", help="Only run cases with these tags (comma-separated)"),
    show_passing: bool = typer.Option(False, "--show-passing", help="Show passing cases in output"),
    save: bool = typer.Option(True, "--save/--no-save", help="Save run to runs/"),
    set_baseline: bool = typer.Option(False, "--set-baseline", help="Save this run as the new baseline"),
    threshold: float = typer.Option(0.05, "--threshold", help="Regression alert threshold (pass rate drop)"),
):
    """Run an eval suite against a dataset."""
    all_cases = ds.load(dataset)
    if not all_cases:
        console.print(f"[red]No cases found for dataset '{dataset}'[/]")
        raise typer.Exit(1)

    cases = _filter_cases(all_cases, tags)
    if not cases:
        console.print(f"[red]No cases match tags '{tags}'[/]")
        raise typer.Exit(1)

    if tags:
        console.print(f"Filtered to [bold]{len(cases)}/{len(all_cases)}[/] cases (tags: {tags})")

    scorer_obj = _resolve_scorer(scorer)
    executor_obj = _resolve_executor(executor)

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

    report = bl.compare(result, threshold=threshold)
    if report:
        rp.print_regression(report)

    raise typer.Exit(0 if result.failed == 0 else 1)


@app.command()
def promote(
    dataset: str = typer.Option(..., "--dataset", "-d", help="Dataset name"),
):
    """Promote the most recent saved run to the baseline (no re-run needed)."""
    runs_dir = bl.RUNS_DIR
    pattern = f"{dataset}_*.json"
    candidates = sorted(runs_dir.glob(pattern), reverse=True)
    if not candidates:
        console.print(f"[red]No saved runs found for '{dataset}'[/]")
        raise typer.Exit(1)

    import json
    latest = candidates[0]
    data = json.loads(latest.read_text(encoding="utf-8"))
    baseline_path = runs_dir / f"baseline_{dataset}.json"
    baseline_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    console.print(f"[green]Baseline set from:[/] {latest.name}")


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
        has_baseline = (bl.RUNS_DIR / f"baseline_{name}.json").exists()
        baseline_str = "  [green]baseline ✓[/]" if has_baseline else ""
        console.print(f"  [cyan]{name}[/]  [dim]{len(cases)} cases[/]{baseline_str}")


@app.command()
def add(
    dataset: str = typer.Option(..., "--dataset", "-d", help="Dataset name"),
    input: str = typer.Option(..., "--input", "-i", help="Input (string or JSON)"),
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
def remove(
    dataset: str = typer.Option(..., "--dataset", "-d", help="Dataset name"),
    case_id: str = typer.Option(..., "--id", help="Case ID to remove (e.g. case-003)"),
):
    """Remove a test case from a dataset."""
    removed = ds.remove(dataset, case_id)
    if removed:
        console.print(f"[green]Removed[/] {case_id} from [cyan]{dataset}[/]")
    else:
        console.print(f"[red]Case '{case_id}' not found in '{dataset}'[/]")
        raise typer.Exit(1)


@app.command()
def show(
    dataset: str = typer.Option(..., "--dataset", "-d", help="Dataset name"),
    tags: str = typer.Option("", "--tags", "-t", help="Filter by tags (comma-separated)"),
):
    """Show all cases in a dataset."""
    cases = ds.load(dataset)
    if not cases:
        console.print(f"[dim]No cases in '{dataset}'[/]")
        return
    cases = _filter_cases(cases, tags)
    for c in cases:
        console.print(f"[bold]{c['id']}[/]  tags={c.get('tags', [])}")
        console.print(f"  input:    {str(c['input'])[:100]}")
        console.print(f"  expected: {str(c['expected'])[:100]}")
        if c.get("notes"):
            console.print(f"  notes:    {c['notes']}")


if __name__ == "__main__":
    app()
