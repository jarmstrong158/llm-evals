"""
Baseline — save runs, load baselines, detect regressions.

A baseline is just a saved RunResult. When you run an eval, you compare
against the current baseline. If pass rate drops, it's a regression.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from core.runner import RunResult, CaseResult


RUNS_DIR = Path(__file__).parent.parent / "runs"


def _run_to_dict(run: RunResult) -> dict:
    return {
        "dataset": run.dataset,
        "executor_name": run.executor_name,
        "scorer_name": run.scorer_name,
        "pass_rate": run.pass_rate,
        "avg_score": run.avg_score,
        "total": run.total,
        "passed": run.passed,
        "failed": run.failed,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "cases": [
            {
                "case_id": c.case_id,
                "passed": c.passed,
                "score": c.score,
                "reason": c.reason,
                "actual": c.actual,
                "expected": c.expected,
                "duration_ms": c.duration_ms,
                "error": c.error,
            }
            for c in run.cases
        ],
    }


def save_run(run: RunResult, runs_dir: Path | None = None) -> Path:
    """Save a run with a timestamp filename."""
    d = runs_dir or RUNS_DIR
    d.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    path = d / f"{run.dataset}_{ts}.json"
    path.write_text(json.dumps(_run_to_dict(run), indent=2), encoding="utf-8")
    return path


def save_baseline(run: RunResult, runs_dir: Path | None = None) -> Path:
    """Save (or overwrite) the baseline for this dataset."""
    d = runs_dir or RUNS_DIR
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"baseline_{run.dataset}.json"
    path.write_text(json.dumps(_run_to_dict(run), indent=2), encoding="utf-8")
    return path


def load_baseline(dataset: str, runs_dir: Path | None = None) -> dict | None:
    """Load the baseline for a dataset. Returns None if no baseline exists."""
    path = (runs_dir or RUNS_DIR) / f"baseline_{dataset}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass
class RegressionReport:
    dataset: str
    baseline_pass_rate: float
    current_pass_rate: float
    delta: float
    is_regression: bool
    newly_failing: list[str]  # case IDs that passed in baseline but fail now
    newly_passing: list[str]  # case IDs that failed in baseline but pass now
    threshold: float


def compare(
    run: RunResult,
    threshold: float = 0.05,
    runs_dir: Path | None = None,
) -> RegressionReport | None:
    """
    Compare a run against its baseline.
    Returns None if no baseline exists.
    threshold: pass rate drop this large or more = regression.
    """
    baseline = load_baseline(run.dataset, runs_dir)
    if baseline is None:
        return None

    baseline_by_id = {c["case_id"]: c for c in baseline.get("cases", [])}
    current_by_id = {c.case_id: c for c in run.cases}

    newly_failing = [
        cid for cid, c in current_by_id.items()
        if not c.passed and baseline_by_id.get(cid, {}).get("passed", False)
    ]
    newly_passing = [
        cid for cid, c in current_by_id.items()
        if c.passed and not baseline_by_id.get(cid, {}).get("passed", True)
    ]

    baseline_rate = baseline.get("pass_rate", 0.0)
    delta = run.pass_rate - baseline_rate

    return RegressionReport(
        dataset=run.dataset,
        baseline_pass_rate=baseline_rate,
        current_pass_rate=run.pass_rate,
        delta=delta,
        is_regression=delta < -threshold,
        newly_failing=newly_failing,
        newly_passing=newly_passing,
        threshold=threshold,
    )
