"""
Scorers — judge whether actual output matches expected.

Each scorer is a callable: scorer(actual: str, expected: str) -> ScorerResult

ScorerResult:
  passed:  bool
  score:   float  (0.0–1.0)
  reason:  str
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from typing import Callable


@dataclass
class ScorerResult:
    passed: bool
    score: float
    reason: str


# ── ExactMatch ────────────────────────────────────────────────────────────────

class ExactMatch:
    """Passes only if actual == expected (stripped)."""

    def __call__(self, actual: str, expected: str) -> ScorerResult:
        passed = actual.strip() == expected.strip()
        return ScorerResult(
            passed=passed,
            score=1.0 if passed else 0.0,
            reason="exact match" if passed else f"got {repr(actual.strip()[:80])}, expected {repr(expected.strip()[:80])}",
        )


# ── Contains ──────────────────────────────────────────────────────────────────

class Contains:
    """Passes if all required strings appear in actual output."""

    def __init__(self, case_sensitive: bool = False):
        self.case_sensitive = case_sensitive

    def __call__(self, actual: str, expected: str) -> ScorerResult:
        # expected is treated as a comma-separated list of required substrings
        # OR as a single string to find
        needles = [s.strip() for s in expected.split("||")] if "||" in expected else [expected]
        a = actual if self.case_sensitive else actual.lower()
        missing = [n for n in needles if (n if self.case_sensitive else n.lower()) not in a]
        passed = len(missing) == 0
        return ScorerResult(
            passed=passed,
            score=1.0 - (len(missing) / len(needles)),
            reason="all present" if passed else f"missing: {missing}",
        )


# ── SemanticSimilarity ────────────────────────────────────────────────────────

class SemanticSimilarity:
    """
    Embeds actual and expected via Ollama's nomic-embed-text model,
    computes cosine similarity, passes if above threshold.

    Requires Ollama running at localhost:11434.
    """

    def __init__(self, threshold: float = 0.80, model: str = "nomic-embed-text"):
        self.threshold = threshold
        self.model = model

    @staticmethod
    def _check_ollama():
        import urllib.request as _req
        try:
            with _req.urlopen("http://localhost:11434/api/tags", timeout=3):
                pass
        except Exception:
            raise RuntimeError(
                "Ollama is not running. Start it with: ollama serve\n"
                "Then pull the model: ollama pull nomic-embed-text"
            )

    def _embed(self, text: str) -> list[float]:
        payload = json.dumps({"model": self.model, "input": text}).encode()
        req = urllib.request.Request(
            "http://localhost:11434/api/embed",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        return data["embeddings"][0]

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        mag_a = sum(x ** 2 for x in a) ** 0.5
        mag_b = sum(x ** 2 for x in b) ** 0.5
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)

    def __call__(self, actual: str, expected: str) -> ScorerResult:
        try:
            self._check_ollama()
            vec_actual = self._embed(actual)
            vec_expected = self._embed(expected)
            sim = self._cosine(vec_actual, vec_expected)
            passed = sim >= self.threshold
            return ScorerResult(
                passed=passed,
                score=round(sim, 4),
                reason=f"cosine similarity {sim:.4f} (threshold {self.threshold})",
            )
        except Exception as e:
            return ScorerResult(passed=False, score=0.0, reason=f"embedding error: {e}")


# ── LLMJudge ─────────────────────────────────────────────────────────────────

class LLMJudge:
    """
    Asks an Ollama model to judge whether the actual output satisfies the rubric.

    rubric: a plain-English description of what a correct answer looks like.
            If omitted, uses expected output as the reference.

    Requires Ollama running at localhost:11434.
    """

    SYSTEM = (
        "You are an impartial evaluator. Given an expected answer (or rubric) "
        "and an actual answer, decide if the actual answer is correct. "
        "Reply with a JSON object: {\"passed\": true/false, \"reason\": \"...\"}. "
        "Nothing else."
    )

    def __init__(self, rubric: str = "", model: str = "llama3.2"):
        self.rubric = rubric
        self.model = model

    @staticmethod
    def _check_ollama():
        import urllib.request as _req
        try:
            with _req.urlopen("http://localhost:11434/api/tags", timeout=3):
                pass
        except Exception:
            raise RuntimeError(
                "Ollama is not running. Start it with: ollama serve\n"
                "Then pull the model: ollama pull llama3.2"
            )

    def __call__(self, actual: str, expected: str) -> ScorerResult:
        self._check_ollama()
        reference = self.rubric if self.rubric else f"Expected output: {expected}"
        prompt = (
            f"{reference}\n\n"
            f"Actual output:\n{actual}\n\n"
            "Is the actual output correct? Reply with JSON only."
        )
        payload = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.SYSTEM},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
        }).encode()

        try:
            req = urllib.request.Request(
                "http://localhost:11434/api/chat",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read())
            content = data["message"]["content"].strip()
            # strip markdown fences if present
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            result = json.loads(content)
            passed = bool(result.get("passed", False))
            reason = result.get("reason", "")
            return ScorerResult(passed=passed, score=1.0 if passed else 0.0, reason=reason)
        except Exception as e:
            return ScorerResult(passed=False, score=0.0, reason=f"judge error: {e}")


# ── Custom ────────────────────────────────────────────────────────────────────

class Custom:
    """Wrap any function: fn(actual, expected) -> bool or ScorerResult."""

    def __init__(self, fn: Callable[[str, str], bool | ScorerResult], name: str = "custom"):
        self.fn = fn
        self.name = name

    def __call__(self, actual: str, expected: str) -> ScorerResult:
        result = self.fn(actual, expected)
        if isinstance(result, ScorerResult):
            return result
        passed = bool(result)
        return ScorerResult(
            passed=passed,
            score=1.0 if passed else 0.0,
            reason=f"{self.name}: {'pass' if passed else 'fail'}",
        )
