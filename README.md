# llm-evals

Eval framework for LLM tools, agents, and programs. Test any callable — MCP tools, CLI programs, REST APIs, Python functions, or raw Ollama prompts — against a labeled dataset, score the outputs, and detect regressions automatically.

## What it does

```
input → executor → output → scorer → pass/fail + score
```

Every run is saved. Runs are compared against a baseline. If pass rate drops by more than a threshold, it flags a regression.

## Scorers

| Scorer | How it judges |
|---|---|
| `exact` | `actual == expected` (stripped) |
| `contains` | all required strings present in output (`\|\|` for multiple) |
| `semantic` | cosine similarity via Ollama `nomic-embed-text` ≥ threshold |
| `llm` | asks Ollama `llama3.2` to judge against a rubric |
| `custom` | any Python function returning `bool` or `ScorerResult` |

## Executors

| Executor | Calls |
|---|---|
| `PythonFunc` | any Python callable |
| `CLI` | subprocess, `{input}` interpolated into command |
| `HTTPEndpoint` | REST API via POST |
| `MCPTool` | MCP tool via mcp-bridge HTTP layer |
| `Prompt` | raw prompt to Ollama model |

## Install

```bash
pip install -r requirements.txt
```

Semantic and LLM judge scorers require [Ollama](https://ollama.com) running locally:

```bash
ollama pull nomic-embed-text
ollama pull llama3.2
```

## Usage

```bash
# List datasets
python cli.py list

# Add a test case
python cli.py add --dataset url_validation --input "https://greenhouse.io/company/jobs/123" --expected "valid" --tags url,greenhouse

# Run an eval
python cli.py run --dataset url_validation --executor core.executors:PythonFunc --scorer contains

# Run and set as new baseline
python cli.py run --dataset url_validation --executor core.executors:PythonFunc --scorer exact --set-baseline

# Just set the baseline without re-running
python cli.py baseline --dataset url_validation --executor core.executors:PythonFunc
```

## Writing a custom executor

```python
# myevals/executors.py
from core.executors import PythonFunc
import server  # your actual tool module

check_url = PythonFunc(server._is_valid_job_url, name="url_validator")
```

Then run:
```bash
python cli.py run --dataset url_validation --executor myevals.executors:check_url --scorer exact
```

## Project structure

```
llm-evals/
  core/
    dataset.py     # load, save, add, remove test cases
    runner.py      # run cases through executor + scorer
    scorers.py     # ExactMatch, Contains, SemanticSimilarity, LLMJudge, Custom
    executors.py   # PythonFunc, CLI, HTTPEndpoint, MCPTool, Prompt
    baseline.py    # save runs, compare, detect regressions
    reporter.py    # Rich terminal output
  datasets/        # JSON test case files (tracked in git)
  runs/            # saved run results (gitignored)
  tests/           # pytest suite for the framework itself
  cli.py           # entry point
```
