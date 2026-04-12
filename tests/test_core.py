"""
Tests for llm-evals core: dataset, runner, scorers, baseline.
Does NOT test SemanticSimilarity or LLMJudge (require Ollama).
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import dataset as ds
from core import runner as rn
from core import baseline as bl
from core.scorers import ExactMatch, Contains, Custom, ScorerResult
from core.executors import PythonFunc, CLI


# ── Dataset ───────────────────────────────────────────────────────────────────

class TestDataset:
    def test_load_empty(self, tmp_path):
        assert ds.load("missing", datasets_dir=tmp_path) == []

    def test_add_and_load(self, tmp_path):
        case = ds.add("test", "hello", "world", datasets_dir=tmp_path)
        cases = ds.load("test", datasets_dir=tmp_path)
        assert len(cases) == 1
        assert cases[0]["id"] == "case-001"
        assert cases[0]["input"] == "hello"
        assert cases[0]["expected"] == "world"

    def test_sequential_ids(self, tmp_path):
        ds.add("test", "a", "b", datasets_dir=tmp_path)
        ds.add("test", "c", "d", datasets_dir=tmp_path)
        ds.add("test", "e", "f", datasets_dir=tmp_path)
        cases = ds.load("test", datasets_dir=tmp_path)
        ids = [c["id"] for c in cases]
        assert ids == ["case-001", "case-002", "case-003"]

    def test_tags_stored(self, tmp_path):
        ds.add("test", "x", "y", tags=["url", "validation"], datasets_dir=tmp_path)
        case = ds.load("test", datasets_dir=tmp_path)[0]
        assert case["tags"] == ["url", "validation"]

    def test_remove(self, tmp_path):
        ds.add("test", "a", "b", datasets_dir=tmp_path)
        ds.add("test", "c", "d", datasets_dir=tmp_path)
        removed = ds.remove("test", "case-001", datasets_dir=tmp_path)
        assert removed is True
        cases = ds.load("test", datasets_dir=tmp_path)
        assert len(cases) == 1
        assert cases[0]["id"] == "case-002"

    def test_remove_missing(self, tmp_path):
        ds.add("test", "a", "b", datasets_dir=tmp_path)
        removed = ds.remove("test", "case-999", datasets_dir=tmp_path)
        assert removed is False

    def test_list_datasets(self, tmp_path):
        ds.add("alpha", "a", "b", datasets_dir=tmp_path)
        ds.add("beta", "c", "d", datasets_dir=tmp_path)
        names = ds.list_datasets(datasets_dir=tmp_path)
        assert names == ["alpha", "beta"]

    def test_dict_input(self, tmp_path):
        ds.add("test", {"key": "value"}, "ok", datasets_dir=tmp_path)
        case = ds.load("test", datasets_dir=tmp_path)[0]
        assert case["input"] == {"key": "value"}

    def test_save_creates_dir(self, tmp_path):
        nested = tmp_path / "nested" / "dir"
        ds.save("test", [{"id": "case-001", "input": "x", "expected": "y"}], datasets_dir=nested)
        assert (nested / "test.json").exists()


# ── Scorers ───────────────────────────────────────────────────────────────────

class TestExactMatch:
    def test_pass(self):
        r = ExactMatch()("hello", "hello")
        assert r.passed is True
        assert r.score == 1.0

    def test_fail(self):
        r = ExactMatch()("hello", "world")
        assert r.passed is False
        assert r.score == 0.0

    def test_strips_whitespace(self):
        r = ExactMatch()("  hello  ", "hello")
        assert r.passed is True

    def test_case_sensitive(self):
        r = ExactMatch()("Hello", "hello")
        assert r.passed is False


class TestContains:
    def test_single_string_present(self):
        r = Contains()("the quick brown fox", "quick")
        assert r.passed is True

    def test_single_string_missing(self):
        r = Contains()("the quick brown fox", "lazy")
        assert r.passed is False

    def test_multiple_required(self):
        r = Contains()("the quick brown fox", "quick||brown")
        assert r.passed is True

    def test_one_missing(self):
        r = Contains()("the quick brown fox", "quick||lazy")
        assert r.passed is False
        assert r.score == 0.5

    def test_case_insensitive_default(self):
        r = Contains()("The Quick Brown Fox", "quick")
        assert r.passed is True

    def test_case_sensitive(self):
        r = Contains(case_sensitive=True)("The Quick Brown Fox", "quick")
        assert r.passed is False


class TestCustomScorer:
    def test_bool_return(self):
        scorer = Custom(lambda a, e: a.startswith(e), name="startswith")
        r = scorer("hello world", "hello")
        assert r.passed is True

    def test_scorer_result_return(self):
        def my_fn(actual, expected):
            return ScorerResult(passed=True, score=0.9, reason="custom check")
        scorer = Custom(my_fn)
        r = scorer("anything", "anything")
        assert r.score == 0.9
        assert r.reason == "custom check"

    def test_fail(self):
        scorer = Custom(lambda a, e: False)
        r = scorer("x", "y")
        assert r.passed is False


# ── Executors ─────────────────────────────────────────────────────────────────

class TestPythonFuncExecutor:
    def test_string_input(self):
        exc = PythonFunc(str.upper)
        assert exc("hello") == "HELLO"

    def test_dict_input_kwargs(self):
        def add(a, b):
            return a + b
        exc = PythonFunc(add)
        assert exc({"a": 1, "b": 2}) == "3"

    def test_none_output(self):
        exc = PythonFunc(lambda x: None)
        assert exc("anything") == ""

    def test_name_from_function(self):
        def my_func(x): return x
        exc = PythonFunc(my_func)
        assert exc.name == "my_func"


class TestCLIExecutor:
    def test_basic_command(self):
        exc = CLI("python -c \"print('hello')\"")
        result = exc("")
        assert result == "hello"

    def test_input_interpolation(self):
        exc = CLI("python -c \"print('{input}')\"")
        result = exc("world")
        assert "world" in result

    def test_failed_command_returns_stderr(self):
        exc = CLI("python -c \"import sys; sys.exit(1)\"")
        result = exc("")
        # Should return something (exit code message or empty)
        assert isinstance(result, str)


# ── Runner ────────────────────────────────────────────────────────────────────

class TestRunner:
    def _cases(self):
        return [
            {"id": "case-001", "input": "hello", "expected": "HELLO"},
            {"id": "case-002", "input": "world", "expected": "WORLD"},
            {"id": "case-003", "input": "fail", "expected": "WRONG"},
        ]

    def test_basic_run(self):
        result = rn.run(
            executor=PythonFunc(str.upper),
            cases=self._cases(),
            scorer=ExactMatch(),
            dataset_name="test",
        )
        assert result.total == 3
        assert result.passed == 2
        assert result.failed == 1

    def test_pass_rate(self):
        result = rn.run(
            executor=PythonFunc(str.upper),
            cases=self._cases(),
            scorer=ExactMatch(),
            dataset_name="test",
        )
        assert abs(result.pass_rate - 2/3) < 0.001

    def test_result_fields(self):
        result = rn.run(
            executor=PythonFunc(str.upper),
            cases=[{"id": "case-001", "input": "hello", "expected": "HELLO"}],
            scorer=ExactMatch(),
            dataset_name="test",
        )
        c = result.cases[0]
        assert c.actual == "HELLO"
        assert c.passed is True
        assert c.duration_ms >= 0

    def test_executor_error_captured(self):
        def boom(x):
            raise RuntimeError("oops")
        result = rn.run(
            executor=PythonFunc(boom),
            cases=[{"id": "case-001", "input": "x", "expected": "y"}],
            scorer=ExactMatch(),
            dataset_name="test",
        )
        assert result.cases[0].passed is False
        assert "oops" in result.cases[0].error

    def test_on_progress_called(self):
        calls = []
        rn.run(
            executor=PythonFunc(str.upper),
            cases=self._cases(),
            scorer=ExactMatch(),
            dataset_name="test",
            on_progress=lambda i, t, r: calls.append((i, t)),
        )
        assert calls == [(1, 3), (2, 3), (3, 3)]

    def test_failures_method(self):
        result = rn.run(
            executor=PythonFunc(str.upper),
            cases=self._cases(),
            scorer=ExactMatch(),
            dataset_name="test",
        )
        failures = result.failures()
        assert len(failures) == 1
        assert failures[0].case_id == "case-003"

    def test_empty_cases(self):
        result = rn.run(
            executor=PythonFunc(str.upper),
            cases=[],
            scorer=ExactMatch(),
            dataset_name="test",
        )
        assert result.total == 0
        assert result.pass_rate == 0.0


# ── Baseline ──────────────────────────────────────────────────────────────────

class TestBaseline:
    def _make_run(self, passed: int, total: int, dataset: str = "test") -> rn.RunResult:
        cases = []
        for i in range(total):
            cases.append(rn.CaseResult(
                case_id=f"case-{i+1:03d}",
                input=f"input-{i}",
                expected="x",
                actual="x" if i < passed else "y",
                passed=i < passed,
                score=1.0 if i < passed else 0.0,
                reason="",
                duration_ms=10,
            ))
        return rn.RunResult(dataset=dataset, executor_name="Test", scorer_name="Exact", cases=cases)

    def test_save_and_load(self, tmp_path):
        run = self._make_run(8, 10)
        bl.save_baseline(run, runs_dir=tmp_path)
        loaded = bl.load_baseline("test", runs_dir=tmp_path)
        assert loaded is not None
        assert abs(loaded["pass_rate"] - 0.8) < 0.001

    def test_no_baseline_returns_none(self, tmp_path):
        assert bl.load_baseline("nonexistent", runs_dir=tmp_path) is None

    def test_compare_no_baseline_returns_none(self, tmp_path):
        run = self._make_run(8, 10)
        assert bl.compare(run, runs_dir=tmp_path) is None

    def test_no_regression(self, tmp_path):
        old = self._make_run(8, 10)
        bl.save_baseline(old, runs_dir=tmp_path)
        new = self._make_run(9, 10)
        report = bl.compare(new, runs_dir=tmp_path)
        assert report is not None
        assert report.is_regression is False
        assert report.delta > 0

    def test_regression_detected(self, tmp_path):
        old = self._make_run(10, 10)
        bl.save_baseline(old, runs_dir=tmp_path)
        new = self._make_run(7, 10)
        report = bl.compare(new, threshold=0.05, runs_dir=tmp_path)
        assert report.is_regression is True
        assert report.delta < 0

    def test_newly_failing(self, tmp_path):
        old = self._make_run(10, 10)
        bl.save_baseline(old, runs_dir=tmp_path)
        # now make case-001 fail
        new = self._make_run(9, 10)
        new.cases[9].passed = False  # flip last case
        report = bl.compare(new, runs_dir=tmp_path)
        assert "case-010" in report.newly_failing

    def test_save_run_creates_timestamped_file(self, tmp_path):
        run = self._make_run(5, 5)
        path = bl.save_run(run, runs_dir=tmp_path)
        assert path.exists()
        assert "test_" in path.name
        assert path.suffix == ".json"
