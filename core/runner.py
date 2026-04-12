"""
Runner — orchestrates a full eval run.

Takes an executor, a list of test cases, and a scorer.
Returns a RunResult with per-case outcomes and aggregate stats.
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class CaseResult:
    case_id: str
    input: Any
    expected: str
    actual: str
    passed: bool
    score: float
    reason: str
    duration_ms: int
    error: str = ""


@dataclass
class RunResult:
    dataset: str
    executor_name: str
    scorer_name: str
    cases: list[CaseResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.cases)

    @property
    def passed(self) -> int:
        return sum(1 for c in self.cases if c.passed)

    @property
    def failed(self) -> int:
        return self.total - self.passed

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    @property
    def avg_score(self) -> float:
        if not self.cases:
            return 0.0
        return sum(c.score for c in self.cases) / len(self.cases)

    def failures(self) -> list[CaseResult]:
        return [c for c in self.cases if not c.passed]


def run(
    executor: Callable,
    cases: list[dict],
    scorer: Callable,
    dataset_name: str = "",
    on_progress: Callable[[int, int, CaseResult], None] | None = None,
) -> RunResult:
    """
    Run all test cases through executor, score each with scorer.

    on_progress(index, total, result) is called after each case if provided.
    """
    result = RunResult(
        dataset=dataset_name,
        executor_name=type(executor).__name__,
        scorer_name=type(scorer).__name__,
    )

    for i, case in enumerate(cases):
        case_id = case.get("id", f"case-{i+1:03d}")
        input_val = case.get("input", "")
        expected = str(case.get("expected", ""))

        actual = ""
        error = ""
        score = 0.0
        passed = False
        reason = ""
        t0 = time.monotonic()

        try:
            actual = str(executor(input_val))
            scorer_result = scorer(actual, expected)
            passed = scorer_result.passed
            score = scorer_result.score
            reason = scorer_result.reason
        except Exception as e:
            error = traceback.format_exc()
            reason = f"executor/scorer error: {e}"

        duration_ms = int((time.monotonic() - t0) * 1000)

        case_result = CaseResult(
            case_id=case_id,
            input=input_val,
            expected=expected,
            actual=actual,
            passed=passed,
            score=score,
            reason=reason,
            duration_ms=duration_ms,
            error=error,
        )
        result.cases.append(case_result)

        if on_progress:
            on_progress(i + 1, len(cases), case_result)

    return result
