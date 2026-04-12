"""
Dataset management — load, save, and add test cases.

Each test case:
  {
    "id":       "case-001",
    "input":    <str or dict>,
    "expected": <str>,
    "tags":     ["tag1", "tag2"],
    "notes":    "optional human note",
    "added_at": "2026-04-12T..."
  }
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DATASETS_DIR = Path(__file__).parent.parent / "datasets"


def _next_id(cases: list[dict]) -> str:
    if not cases:
        return "case-001"
    nums = []
    for c in cases:
        try:
            nums.append(int(c["id"].split("-")[1]))
        except (IndexError, ValueError):
            pass
    return f"case-{(max(nums) + 1):03d}" if nums else "case-001"


def load(name: str, datasets_dir: Path | None = None) -> list[dict]:
    """Load a dataset by name (without .json extension)."""
    path = (datasets_dir or DATASETS_DIR) / f"{name}.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def save(name: str, cases: list[dict], datasets_dir: Path | None = None) -> Path:
    """Persist a dataset to disk."""
    d = datasets_dir or DATASETS_DIR
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{name}.json"
    path.write_text(json.dumps(cases, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def add(
    name: str,
    input: Any,
    expected: str,
    tags: list[str] | None = None,
    notes: str = "",
    datasets_dir: Path | None = None,
) -> dict:
    """Add a single test case to a dataset and save."""
    cases = load(name, datasets_dir)
    case = {
        "id": _next_id(cases),
        "input": input,
        "expected": expected,
        "tags": tags or [],
        "notes": notes,
        "added_at": datetime.now(timezone.utc).isoformat(),
    }
    cases.append(case)
    save(name, cases, datasets_dir)
    return case


def remove(name: str, case_id: str, datasets_dir: Path | None = None) -> bool:
    """Remove a test case by ID. Returns True if found and removed."""
    cases = load(name, datasets_dir)
    original_len = len(cases)
    cases = [c for c in cases if c["id"] != case_id]
    if len(cases) == original_len:
        return False
    save(name, cases, datasets_dir)
    return True


def list_datasets(datasets_dir: Path | None = None) -> list[str]:
    """Return names of all datasets (without .json extension)."""
    d = datasets_dir or DATASETS_DIR
    if not d.exists():
        return []
    return sorted(p.stem for p in d.glob("*.json"))
